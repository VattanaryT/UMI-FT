"""r4vt reference pose is the TCP foot. Virtual target carries delta_n. No robot."""

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "surface_frames_scripts"
sys.path.insert(0, str(SCRIPTS))

from fixb_field import FixBMoldField  # noqa: E402
from fixb_labels import pose7_to_Rp  # noqa: E402
from mold_aware_runtime import (  # noqa: E402
    FOOT_CAP_M,
    VT_RAMP_S,
    MoldAwareRuntime,
    d405_housing_corners_in_tcp,
    r4_safety_limits,
    shift_fixture_up,
)

CAUSAL = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "r4vt_frame_continuity_causal/acp_replay_buffer.zarr"
)
FIELD = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "shared/fixb_field.npz"
)
EYE6 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
FEW_MM = 0.005
FEW_DEG = 5.0


class _Flat:
    """z = 0, outward normal +z. track_step snaps to the closest point."""

    def query(self, p):
        p = np.asarray(p, dtype=np.float64).reshape(-1, 3)
        foot = p.copy()
        foot[:, 2] = 0.0
        frames = np.broadcast_to(np.eye(3), (len(p), 3, 3)).copy()
        return {"x_s": foot, "R": frames, "sd": p[:, 2].copy()}

    def track_step(self, p, p_prev, x_prev, R_prev):
        hit = self.query(p)
        return hit["x_s"], hit["R"]


def _runtime(limits=None):
    return MoldAwareRuntime(
        field=_Flat(),
        R_off=np.eye(3),
        run_type="r4_world_fixb_tilt",
        limits=limits or r4_safety_limits(),
        dry_run=False,
    )


def _action(delta_n, steps=1):
    action = np.zeros((steps, 27))
    action[:, 2] = delta_n
    action[:, 18] = 200.0
    action[:, 21:27] = EYE6
    return action


def _axis_deg(offset, normal):
    length = np.linalg.norm(offset, axis=-1)
    unit = offset / np.clip(length[:, None], 1e-12, None)
    cos = np.clip(np.abs(np.sum(unit * normal, axis=-1)), 0.0, 1.0)
    return np.degrees(np.arccos(cos)), length


def test_nominal_is_foot_and_vt_carries_delta_n():
    runtime = _runtime()
    tcp = np.array([0.1, -0.2, 0.02])
    decoded = runtime.decode_chunk(
        _action(-0.15), tcp, np.eye(3), np.array([1.0 / 15.0]), apply_safety=False
    )
    nominal, virtual = runtime.decoded_to_action_mats(decoded)
    foot = runtime.field.query(tcp.reshape(1, 3))["x_s"][0]
    assert np.linalg.norm(nominal[0, :3, 3] - foot) < FEW_MM
    offset = virtual[0, :3, 3] - nominal[0, :3, 3]
    assert np.allclose(offset, [0.0, 0.0, -0.15], atol=1e-8)
    # The deep point is the attractor, not the reference.
    assert np.linalg.norm(nominal[0, :3, 3] - (foot + offset)) > 0.1


def test_bench_floor_limits_the_virtual_target_to_5mm():
    runtime = _runtime(r4_safety_limits(enable_delta_n_floor=True))
    tcp = np.array([0.0, 0.0, 0.0])
    steps = 20
    decoded = runtime.decode_chunk(
        _action(-0.15, steps=steps), tcp, np.eye(3), np.full(steps, 1.0 / 15.0)
    )
    assert np.all(decoded["delta_n"] >= -0.005 - 1e-9)
    assert decoded["delta_n"][-1] == pytest.approx(-0.005, abs=1e-6)
    nominal, virtual = runtime.decoded_to_action_mats(decoded)
    offset = np.linalg.norm(virtual[-1, :3, 3] - nominal[-1, :3, 3])
    assert offset <= 0.005 + 1e-9


def test_episode0_nominal_tracks_tcp_foot():
    zarr = pytest.importorskip("zarr")
    if not CAUSAL.exists() or not FIELD.exists():
        pytest.skip("causal r4vt zarr or mold field is not on this machine")
    from diffusion_policy.common.sampler_umift import (
        sparse_action_indices_from_rgb_timestamps,
    )

    group = zarr.open(str(CAUSAL), "r")["data"]["episode_0"]
    fb = np.asarray(group["ts_pose_fb_0"], dtype=np.float64)
    tool_r, tcp = pose7_to_Rp(fb)
    v_par = np.asarray(group["fixb_v_par_0"], dtype=np.float64)
    delta_n = np.asarray(group["fixb_delta_n_0"], dtype=np.float64).reshape(-1)
    robot_t = np.asarray(group["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
    rgb_t = np.asarray(group["rgb_time_stamps_0"], dtype=np.float64).reshape(-1)
    field = FixBMoldField(FIELD)
    runtime = MoldAwareRuntime(
        field=field,
        R_off=np.eye(3),
        run_type="r4_world_fixb_tilt",
        limits=r4_safety_limits(),
        dry_run=False,
    )
    checked = 0
    press = 0
    for rgb_id in range(0, len(rgb_t), 6):
        # Each window is a first replan. The cross-replan cap is tested below.
        runtime.reset()
        indices, _times = sparse_action_indices_from_rgb_timestamps(
            robot_t, rgb_t, rgb_id, 6, 1
        )
        if len(indices) < 6:
            break
        indices = np.asarray(indices[:6], dtype=np.int64)
        action = _action(delta_n[indices], steps=6)
        action[:, 0:2] = v_par[indices]
        decoded = runtime.decode_chunk(
            action,
            tcp[indices[0]],
            tool_r[indices[0]],
            np.full(6, 1.0 / 15.0),
            apply_safety=False,
        )
        nominal, virtual = runtime.decoded_to_action_mats(decoded)
        contact = field.query(tcp[indices[0]].reshape(1, 3))["x_s"][0]
        assert np.linalg.norm(nominal[0, :3, 3] - contact) < FEW_MM
        offset = virtual[:, :3, 3] - nominal[:, :3, 3]
        normal = decoded["R_field"][:, :, 2]
        angle, length = _axis_deg(offset, normal)
        live = np.abs(decoded["delta_n"]) > 1e-3
        assert np.all(angle[live] < FEW_DEG)
        assert np.allclose(length, np.abs(decoded["delta_n"]), atol=1e-6)
        deep = np.abs(delta_n[indices[0]]) > 0.05
        if deep:
            press += 1
            assert np.linalg.norm(nominal[0, :3, 3] - contact) < FEW_MM
        checked += 1
    assert checked > 100
    assert press > 10


def test_fast_20mm_tcp_step_is_limited_by_the_label_gate():
    """A later chunk walks toward the TCP at 6 mm per step. It does not hop 20 mm."""
    from fixb_tcp_track import GATE_M

    runtime = _runtime()
    runtime.decode_chunk(
        _action(0.0), np.zeros(3), np.eye(3), np.array([1.0 / 15.0]), apply_safety=False
    )
    steps = 4
    decoded = runtime.decode_chunk(
        _action(0.0, steps=steps),
        np.array([0.020, 0.0, 0.02]),
        np.eye(3),
        np.full(steps, 1.0 / 15.0),
        apply_safety=False,
    )
    moved = np.linalg.norm(np.diff(np.vstack([np.zeros(3), decoded["x_s"]]), axis=0), axis=1)
    assert np.all(moved <= GATE_M + 1e-6)
    assert moved[0] == pytest.approx(GATE_M, abs=1e-6)
    assert not np.allclose(decoded["x_s"][0], [0.020, 0.0, 0.0])


def test_ray_recovery_is_not_the_closest_point():
    """After more than N rejected steps the foot takes the press-axis hit."""
    from mold_aware_runtime import HOP_HOLD_LIMIT

    class _Ray(_Flat):
        def ray_hit(self, origin, direction):
            del origin, direction
            return np.array([0.05, 0.0, 0.0])

    runtime = MoldAwareRuntime(
        field=_Ray(),
        R_off=np.eye(3),
        run_type="r4_world_fixb_tilt",
        limits=r4_safety_limits(),
        dry_run=False,
    )
    runtime.decode_chunk(
        _action(0.0), np.zeros(3), np.eye(3), np.array([1.0 / 15.0]), apply_safety=False
    )
    steps = HOP_HOLD_LIMIT + 1
    decoded = runtime.decode_chunk(
        _action(0.0, steps=steps),
        np.array([0.08, 0.0, 0.03]),
        np.eye(3),
        np.full(steps, 1.0 / 15.0),
        apply_safety=False,
    )
    assert int(np.sum(decoded["hop_hold"])) == HOP_HOLD_LIMIT
    assert bool(decoded["foot_reanchor"][-1])
    assert np.allclose(decoded["x_s"][-1], [0.05, 0.0, 0.0])
    assert not np.allclose(decoded["x_s"][-1], [0.08, 0.0, 0.0])


def test_rd_slew_caps_three_degrees_across_replans():
    runtime = _runtime(r4_safety_limits())
    tcp = np.array([0.0, 0.0, 0.02])
    first = runtime.decode_chunk(
        _action(0.0, steps=2), tcp, np.eye(3), np.full(2, 1.0 / 15.0)
    )
    turned = _action(0.0, steps=2)
    angle = np.radians(30.0)
    cosine, sine = np.cos(angle), np.sin(angle)
    turned[:, 21:27] = [cosine, -sine, 0.0, sine, cosine, 0.0]
    second = runtime.decode_chunk(
        turned, tcp, np.eye(3), np.full(2, 1.0 / 15.0)
    )
    from scipy.spatial.transform import Rotation

    def step(a, b):
        return np.degrees((Rotation.from_matrix(a).inv() * Rotation.from_matrix(b)).magnitude())

    seam = step(first["R_d"][-1], second["R_d"][0])
    inside = step(second["R_d"][0], second["R_d"][1])
    assert seam <= 3.0 + 1e-6
    assert inside <= 3.0 + 1e-6
    assert "rd_slew" in runtime.last_triggers
    assert bool(runtime.last_rd_slew[0])


def test_face_hold_keeps_frame_and_command_past_five_degrees():
    class _Tilted(_Flat):
        def query(self, p):
            hit = super().query(p)
            if not getattr(self, "_armed", False):
                self._armed = True
                return hit
            tilt = np.radians(30.0)
            cosine, sine = np.cos(tilt), np.sin(tilt)
            hit["R"][:] = np.array(
                [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]]
            )
            return hit

    runtime = MoldAwareRuntime(
        field=_Tilted(),
        R_off=np.eye(3),
        run_type="r4_world_fixb_tilt",
        limits=r4_safety_limits(),
        dry_run=False,
    )
    tcp = np.array([0.0, 0.0, 0.02])
    first = runtime.decode_chunk(
        _action(0.0), tcp, np.eye(3), np.array([1.0 / 15.0])
    )
    turned = _action(0.0)
    yaw = np.radians(30.0)
    cosine, sine = np.cos(yaw), np.sin(yaw)
    turned[:, 21:27] = [cosine, -sine, 0.0, sine, cosine, 0.0]
    second = runtime.decode_chunk(
        turned, tcp, np.eye(3), np.array([1.0 / 15.0])
    )
    assert np.allclose(second["R_field"][0], np.eye(3))
    assert np.allclose(second["R_d"][0], first["R_d"][0])
    assert bool(second["face_hold"][0])
    assert "face_hold" in runtime.last_triggers


def test_open_loop_query_lands_on_the_label_foot():
    """A carried foot must not move query 0. Each raw window re-queries the TCP."""
    zarr = pytest.importorskip("zarr")
    smooth = Path(
        "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
        "r4vt_smooth_frame/acp_replay_buffer.zarr"
    )
    meta = Path(
        "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb/shared/fixb_meta.json"
    )
    if not smooth.exists() or not FIELD.exists() or not meta.exists():
        pytest.skip("smooth r4vt zarr or mold field is not on this machine")
    from export_openloop import decode_raw_motion

    group = zarr.open(str(smooth), "r")["data"]["episode_0"]
    _tool_r, tcp = pose7_to_Rp(np.asarray(group["ts_pose_fb_0"], dtype=np.float64))
    foot = np.asarray(group["mold_x_s_0"], dtype=np.float64)
    robot_t = np.asarray(group["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
    rgb_t = np.asarray(group["rgb_time_stamps_0"], dtype=np.float64).reshape(-1)
    # Two windows whose label feet are the fresh TCP query, and far apart.
    chosen = []
    runtime = MoldAwareRuntime.from_assets(
        "r4_world_fixb_tilt",
        field_path=FIELD,
        meta_path=meta,
        dry_run=False,
        limits=r4_safety_limits(),
    )
    for rgb_id in range(0, len(rgb_t), 40):
        q = int(np.clip(np.searchsorted(robot_t, rgb_t[rgb_id]), 0, len(robot_t) - 1))
        runtime.release_command_foot()
        decoded = runtime.decode_chunk(
            _action(0.0), tcp[q], np.eye(3), np.array([1.0 / 15.0]), apply_safety=False
        )
        if np.linalg.norm(decoded["x_s"][0] - foot[q]) > 1e-3:
            continue
        if chosen and np.linalg.norm(foot[q] - foot[chosen[0]]) < 0.05:
            continue
        chosen.append(q)
        if len(chosen) == 2:
            break
    assert len(chosen) == 2

    runtime.decode_chunk(
        _action(0.0),
        tcp[chosen[0]] + np.array([0.25, 0.0, 0.0]),
        np.eye(3),
        np.array([1.0 / 15.0]),
        apply_safety=False,
    )
    assert np.linalg.norm(runtime._cmd_foot - foot[chosen[0]]) > 0.05
    preds = np.zeros((2, 1, 27), dtype=np.float64)
    preds[:, :, 21:27] = EYE6
    pos, _rot, _foot, _normal = decode_raw_motion(
        "r4vt",
        runtime,
        preds,
        tcp[chosen],
        np.broadcast_to(np.eye(3), (2, 3, 3)).copy(),
        np.zeros((2, 1)),
    )
    err = np.linalg.norm(pos[:, 0] - foot[chosen], axis=1)
    assert np.all(err < 1e-3), err


def _geodesic_deg(a, b) -> float:
    from scipy.spatial.transform import Rotation

    rel = np.asarray(a, dtype=np.float64).reshape(3, 3).T @ np.asarray(b).reshape(3, 3)
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(rel).as_rotvec())))


def test_first_orientation_slews_from_the_measured_tool():
    runtime = _runtime()
    tcp = np.array([0.0, 0.0, 0.0])
    action = _action(0.0)
    angle = np.radians(90.0)
    cosine, sine = np.cos(angle), np.sin(angle)
    action[:, 21:27] = [cosine, -sine, 0.0, sine, cosine, 0.0]
    decoded = runtime.decode_chunk(action, tcp, np.eye(3), np.array([1.0 / 15.0]))
    assert _geodesic_deg(np.eye(3), decoded["R_d"][0]) <= 3.0 + 1e-6
    assert bool(decoded["rd_slew"][0])


def test_start_31mm_off_the_mesh_stays_on_the_closest_point():
    """The foot is the TCP's closest point. A stationary TCP does not slide the foot."""
    runtime = _runtime()
    tcp = np.array([0.0, 0.0, 0.031])
    steps = 6
    decoded = runtime.decode_chunk(
        _action(0.0, steps=steps), tcp, np.eye(3), np.full(steps, 1.0 / 15.0), apply_safety=True
    )
    assert decoded["x_s"][0, 2] == pytest.approx(0.0)
    assert np.linalg.norm(decoded["x_s"][0] - tcp) == pytest.approx(0.031)
    assert np.allclose(decoded["x_s"], 0.0, atol=1e-6)
    assert not np.any(decoded["hop_hold"])


def test_tcp_past_the_flange_plane_is_not_a_table_trip():
    runtime = _runtime()
    runtime.set_table_plane(np.array([0.0, 1.0, 0.0]), -0.02)
    runtime.decode_chunk(
        _action(0.0), np.array([0.0, 0.05, 0.01]), np.eye(3), np.array([1.0 / 15.0])
    )
    assert "table_plane" not in runtime.last_triggers
    assert "mesh_far" not in runtime.last_triggers


def test_orientation_past_25_degrees_holds_the_previous():
    from mold_aware_runtime import ORI_LABEL_HOLD_DEG, hold_orientation_near_label

    angle = np.radians(40.0)
    cosine, sine = np.cos(angle), np.sin(angle)
    tipped = np.array([[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]])
    out, held = hold_orientation_near_label(
        tipped.reshape(1, 3, 3), np.eye(3).reshape(1, 3, 3), np.eye(3), ORI_LABEL_HOLD_DEG
    )
    assert bool(held[0])
    assert np.allclose(out[0], np.eye(3))
    near = hold_orientation_near_label(
        np.eye(3).reshape(1, 3, 3), np.eye(3).reshape(1, 3, 3), np.eye(3), ORI_LABEL_HOLD_DEG
    )[1]
    assert not bool(near[0])


def test_wrist_camera_through_the_table_holds():
    limits = r4_safety_limits()
    limits.table_y_m = 0.0
    runtime = _runtime(limits)
    # Tool +z points up, so the housing behind the tip hangs below the TCP.
    up = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    tcp = np.array([0.0, -0.05, 0.0])
    R_out, x_out = runtime._hold_if_camera_hits_table(
        up.reshape(1, 3, 3), tcp.reshape(1, 3), np.eye(3), tcp, None
    )
    assert "camera_table" in runtime.last_triggers
    assert np.allclose(R_out[0], np.eye(3))
    assert np.allclose(x_out[0], tcp)


def test_camera_uses_the_level_plane_not_fixture_y():
    runtime = _runtime()
    lowest = float(d405_housing_corners_in_tcp()[:, 1].max())
    # Lowest corner is 30 mm above fixture Y=0, and 30 mm through a plane at -60 mm.
    tcp = np.array([0.0, -0.030 - lowest, 0.0])
    runtime.set_table_plane(np.array([0.0, 1.0, 0.0]), -0.060)
    runtime._hold_if_camera_hits_table(
        np.eye(3).reshape(1, 3, 3), tcp.reshape(1, 3), np.eye(3), tcp, None
    )
    assert "camera_table" in runtime.last_triggers


def test_tcp_far_from_the_mesh_flags():
    runtime = _runtime()
    runtime.decode_chunk(
        _action(0.0), np.array([0.0, 0.0, 0.25]), np.eye(3), np.array([1.0 / 15.0])
    )
    assert "mesh_far" in runtime.last_triggers


def test_capped_foot_is_projected_back_onto_the_mesh():
    from mold_aware_runtime import _cap_onto_mesh

    tcp = np.array([0.0, 0.0, 0.031])
    foot = np.array([0.0, 0.0, 0.0])
    capped, _frame, limited = _cap_onto_mesh(_Flat(), foot, np.eye(3), tcp)
    assert limited
    assert capped[2] == pytest.approx(0.0)
    assert np.linalg.norm(capped - tcp) > FOOT_CAP_M


def test_new_face_at_chunk_start_stays_on_the_held_sheet():
    """The label gate refuses a flipped normal, so the foot does not follow it."""

    class _Flip(_Flat):
        def query(self, p):
            hit = super().query(p)
            if not getattr(self, "_armed", False):
                self._armed = True
                return hit
            hit["R"][:, :, 1] = np.array([0.0, -1.0, 0.0])
            hit["R"][:, :, 2] = np.array([0.0, 0.0, -1.0])
            return hit

    runtime = MoldAwareRuntime(
        field=_Flip(),
        R_off=np.eye(3),
        run_type="r4_world_fixb_tilt",
        limits=r4_safety_limits(),
        dry_run=False,
    )
    runtime.decode_chunk(_action(0.0), np.zeros(3), np.eye(3), np.array([1.0 / 15.0]))
    decoded = runtime.decode_chunk(
        _action(0.0), np.array([0.08, 0.0, 0.0]), np.eye(3), np.array([1.0 / 15.0])
    )
    assert np.allclose(decoded["x_s"][0], 0.0)
    assert decoded["R_field"][0, :, 2] @ np.array([0.0, 0.0, 1.0]) == pytest.approx(1.0)


def test_vt_ramps_in_and_a_later_25mm_step_holds():
    runtime = _runtime()
    tcp = np.array([0.0, 0.0, 0.0])
    dt = np.array([1.0 / 15.0])
    first = runtime.decode_chunk(_action(-0.05), tcp, np.eye(3), dt)
    allow = r4_safety_limits().max_step_m * float(dt[0]) / VT_RAMP_S
    moved = float(np.linalg.norm(first["x_d"][0] - tcp))
    assert moved == pytest.approx(allow, abs=1e-6)
    runtime._vt_elapsed_s = VT_RAMP_S
    second = runtime.decode_chunk(_action(-0.05), tcp, np.eye(3), dt)
    assert np.allclose(second["x_d"][0], first["x_d"][0], atol=1e-8)
    assert bool(second["clamp_step"][0])


E30_LOG = Path("/home/boeing/data/control_log/checkpoints_e30_rollout0_1794958462")
ORANGE = Path(
    "/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames/PyriteConfig/tasks/umift/"
    "layup_workcell_extrinsics_20261116_orange.yaml"
)


def test_runner_tracker_matches_the_stored_label_frame():
    """The runner's tracker on the label foot path reproduces the stored frame."""
    zarr = pytest.importorskip("zarr")
    side = FIELD.with_name("fixb_smooth_frames.npz")
    zarr_path = FIELD.parents[1] / "r4vt_smooth_tcp_track_clean" / "acp_replay_buffer.zarr"
    if not FIELD.is_file() or not side.is_file() or not zarr_path.is_dir():
        pytest.skip("smooth field or track-clean zarr is missing")
    from scipy.spatial.transform import Rotation
    from fixb_smooth_field import SmoothFrameField
    from mold_aware_runtime import rot6d_rows_to_R, track_feet

    field = SmoothFrameField(FIELD, side)
    group = zarr.open(str(zarr_path), "r")["data"]["episode_40"]
    feet = np.asarray(group["mold_x_s_0"], dtype=np.float64)
    stored = rot6d_rows_to_R(np.asarray(group["mold_R_field_rot6d_0"]))
    _x, got, _held = track_feet(field, feet)
    rel = np.einsum("nij,njk->nik", np.transpose(stored, (0, 2, 1)), got)
    ang = np.degrees(np.linalg.norm(Rotation.from_matrix(rel).as_rotvec(), axis=1))
    assert float(ang.max()) < 1.0


def test_e30_rollout_first_command_is_rate_limited():
    """The logged epoch-30 command was 117° and ~69 mm. The first step is now capped."""
    zarr = pytest.importorskip("zarr")
    if not E30_LOG.is_dir() or not FIELD.is_file() or not ORANGE.is_file():
        pytest.skip("e30 rollout log, mold field, or orange extrinsics is missing")
    from PyriteConfig.tasks.umift.workcell_frames import load_workcell_extrinsics
    from PyriteEnvSuites.env_runners.dataset_start_pose import pose7_change_frame
    from fixb_labels import so3_to_rot6d_rows

    log = zarr.open(str(E30_LOG), "r")
    fixture_from_base = load_workcell_extrinsics(str(ORANGE)).T_fixture_from_right_base
    tcp_pose = pose7_change_frame(
        np.asarray(log["horizon_0/obs_inference/ts_pose_fb_0"], dtype=np.float64),
        fixture_from_base,
    )
    nominal = pose7_change_frame(
        np.asarray(log["horizon_0/action/ts_nominal_targets_0"][0], dtype=np.float64),
        fixture_from_base,
    )
    virtual = pose7_change_frame(
        np.asarray(log["horizon_0/action/ts_virtual_targets_0"][0], dtype=np.float64),
        fixture_from_base,
    )
    tool, tcp = pose7_to_Rp(tcp_pose.reshape(1, 7))
    commanded, nominal_xyz = pose7_to_Rp(nominal.reshape(1, 7))
    _virtual_r, virtual_xyz = pose7_to_Rp(virtual.reshape(1, 7))
    tool, tcp = tool[0], tcp[0]
    commanded = commanded[0]
    nominal_xyz, virtual_xyz = nominal_xyz[0], virtual_xyz[0]
    field = FixBMoldField(FIELD)
    frame = np.asarray(field.query(tcp.reshape(1, 3))["R"][0], dtype=np.float64)
    delta_R = frame.T @ commanded
    delta_n = float(np.dot(virtual_xyz - nominal_xyz, frame[:, 2]))
    raw_deg = _geodesic_deg(tool, frame @ delta_R)
    raw_mm = float(np.linalg.norm(virtual_xyz - tcp) * 1e3)
    assert raw_deg > 90.0
    assert raw_mm > 50.0

    action = _action(delta_n)
    action[:, 21:27] = so3_to_rot6d_rows(delta_R.reshape(1, 3, 3))[0]
    runtime = MoldAwareRuntime(
        field=field,
        R_off=np.eye(3),
        run_type="r4_world_fixb_tilt",
        limits=r4_safety_limits(),
        dry_run=False,
    )
    decoded = runtime.decode_chunk(action, tcp, tool, np.array([1.0 / 15.0]))
    limited_deg = _geodesic_deg(tool, decoded["R_d"][0])
    limited_mm = float(np.linalg.norm(decoded["x_d"][0] - tcp) * 1e3)
    allow_mm = r4_safety_limits().max_step_m * (1.0 / 15.0) / VT_RAMP_S * 1e3
    print(
        f"e30 first command: orientation {raw_deg:.1f} -> {limited_deg:.2f} deg, "
        f"VT {raw_mm:.1f} -> {limited_mm:.2f} mm"
    )
    assert limited_deg <= 3.0 + 1e-6
    assert limited_mm <= allow_mm + 0.05
    assert bool(decoded["rd_slew"][0])


def test_bench_lift_is_fixture_up():
    lifted = shift_fixture_up(np.array([[1.0, 2.0, 3.0]]), 0.08)
    assert np.allclose(lifted, [[1.0, 1.92, 3.0]])
    pose = np.array([0.1, -0.2, 0.3, 1.0, 0.0, 0.0, 0.0])
    lifted_pose = shift_fixture_up(pose, 0.08)
    assert np.allclose(lifted_pose[:3], [0.1, -0.28, 0.3])
    assert np.allclose(lifted_pose[3:], pose[3:])


def test_runner_keeps_standard_bc_stiffness_path():
    """Stiffness still comes from VT − nominal in the runner. No second builder."""
    runner = (
        Path(__file__).resolve().parents[2]
        / "PyriteEnvSuites"
        / "env_runners"
        / "umift_env_runner.py"
    ).read_text()
    assert "compliance_direction_tool" in runner
    assert "default_stiffness = 5000" in runner
    assert "outputs_ts_targets[id] = ts_targets_virtual" in runner
    assert "UMIFT_ABORT_ON_MEASURED_MESH_DISTANCE" in runner
    assert "UMIFT_ENABLE_DELTA_N_FLOOR" in runner
    assert "UMIFT_BENCH_LIFT_M" in runner
    assert "shift_fixture_up" in runner

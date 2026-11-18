"""Open-loop export checks. Synthetic errors, r4vt label geometry, r1rel pose."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

SCRIPTS = Path(__file__).resolve().parents[1] / "surface_frames_scripts"
ROOT = SCRIPTS.parents[1]
for entry in (SCRIPTS, ROOT, ROOT / "PyriteML", ROOT / "PyriteUtility"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import openloop_lib as ol  # noqa: E402
from export_openloop import query_seed  # noqa: E402

R4VT = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "r4vt_world_fixb_tilt/acp_replay_buffer.zarr"
)
R1REL = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "r1rel_world/acp_replay_buffer.zarr"
)
FIELD = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz"
)


class FlatField:
    def query(self, p, with_curvature=False):
        p = np.atleast_2d(np.asarray(p, dtype=np.float64))
        n = len(p)
        R = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
        return {"x_s": p.copy(), "R": R, "sd": p[:, 2].copy(), "normal": R[:, :, 2].copy()}


def _driver():
    path = ROOT / "PyriteEnvSuites" / "scripts" / "offline_compare_driver.py"
    spec = importlib.util.spec_from_file_location("offline_compare_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_equal_prediction_is_zero_error():
    R = Rotation.from_euler("xyz", [12.0, -7.0, 30.0], degrees=True).as_matrix()
    label = np.zeros((8, 27), dtype=np.float64)
    label[:, 0] = np.linspace(-0.2, 0.2, 8)
    label[:, 1] = 0.05
    label[:, 2] = -0.01
    label[:, 18] = 40.0
    label[:, 21:27] = ol.rot6d_of_rotation(R)
    pred = label.copy()
    force = np.array([1, 9, 2, 12, 0, 8, 3, 20], dtype=np.float64)
    bad = np.array([0, 0, 1, 1, 0, 0, 0, 1], dtype=bool)
    long = np.array([0, 0, 0, 0, 1, 0, 0, 0], dtype=bool)
    valid = np.ones(8, dtype=bool)
    table = ol.episode_error_table(pred, label, force, bad, long, valid)
    for dim, groups in table["mae"].items():
        for group, value in groups.items():
            if group == "return_yaw":
                assert value is None, (dim, group, value)
            else:
                assert value == 0.0, (dim, group, value)
    assert table["n"]["near_identity"] == 8
    assert table["n"]["return_yaw"] == 0


def test_delta_r_modes_follow_the_label():
    eye = np.eye(3)
    yaw = np.diag([-1.0, -1.0, 1.0])
    label = np.zeros((2, 27))
    pred = np.zeros((2, 27))
    label[0, 21:27] = ol.rot6d_of_rotation(eye)
    label[1, 21:27] = ol.rot6d_of_rotation(yaw)
    pred[0, 21:27] = label[0, 21:27]
    pred[1, 21:27] = ol.rot6d_of_rotation(eye)
    valid = np.ones(2, dtype=bool)
    zeros = np.zeros(2, dtype=np.float64)
    table = ol.episode_error_table(pred, label, zeros, zeros.astype(bool), zeros.astype(bool), valid)
    assert table["n"]["near_identity"] == 1
    assert table["n"]["return_yaw"] == 1
    assert table["mae"]["delta_R_deg"]["near_identity"] == 0.0
    got = table["mae"]["delta_R_deg"]["return_yaw"]
    assert abs(got - 180.0) < 1e-4


def test_horizon_error_keeps_the_tail_out_of_the_exec_window():
    width = 8
    horizon = 6
    label = np.zeros((1, width, 27))
    pred = np.zeros((1, width, 27))
    eye = np.eye(3)
    yaw = np.diag([-1.0, -1.0, 1.0])
    for k in range(horizon):
        label[0, k, 21:27] = ol.rot6d_of_rotation(eye)
        pred[0, k, 21:27] = ol.rot6d_of_rotation(eye)
    for k in range(horizon, width):
        label[0, k, 21:27] = ol.rot6d_of_rotation(yaw)
        pred[0, k, 21:27] = ol.rot6d_of_rotation(eye)
        pred[0, k, 2] = 0.01
    pred_pos = np.zeros((1, width, 3))
    label_pos = np.zeros((1, width, 3))
    label_pos[0, horizon:, 0] = 0.01
    table = ol.chunk_horizon_errors(pred, label, pred_pos, label_pos, np.array([width]), horizon)
    assert table["horizon"] == horizon
    assert table["exec"]["position_mm"]["all"] == 0.0
    assert table["exec"]["delta_R_deg"]["near_identity"] == 0.0
    assert table["exec"]["delta_R_deg"]["return_yaw"] is None
    assert abs(table["full"]["position_mm"]["all"] - 2.5) < 1e-6
    assert abs(table["full"]["delta_R_deg"]["return_yaw"] - 180.0) < 1e-4
    assert abs(table["full"]["delta_n"]["return_yaw"] - 0.01) < 1e-9


def test_delta_r_is_geodesic_not_rot6d_difference():
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    pred = np.zeros((1, 27))
    label = np.zeros((1, 27))
    pred[:, 21:27] = ol.rot6d_of_rotation(R)
    label[:, 21:27] = ol.rot6d_of_rotation(np.eye(3))
    ang = ol.delta_r_deg(pred, label)
    assert abs(float(ang[0]) - 90.0) < 1e-6
    l2 = float(np.linalg.norm(pred[0, 21:27] - label[0, 21:27]))
    assert abs(float(ang[0]) - l2) > 1.0
    driver = _driver()
    got = ol.geodesic_deg(R.reshape(1, 3, 3), np.eye(3).reshape(1, 3, 3))
    ref = driver.geodesic_deg(R.reshape(1, 3, 3), np.eye(3).reshape(1, 3, 3))
    assert np.allclose(got, ref)


def test_query_seed_matches_offline_driver():
    driver = _driver()
    for rgb in (0, 1, 17, 400):
        assert query_seed("r4vt", rgb) == driver.query_seed("r4vt", rgb)


def test_synthetic_r4vt_decode_matches_vt_normal_and_tool_rotation():
    runtime = ol.make_r4_runtime(FlatField(), dry_run=False)
    R_tool = Rotation.from_euler("xyz", [10.0, -20.0, 15.0], degrees=True).as_matrix()
    x_s = np.array([0.1, 0.2, 0.0])
    n = np.array([0.0, 0.0, 1.0])
    p_vt = x_s + np.array([0.03, -0.04, -0.02])
    delta_n = float(np.dot(p_vt - x_s, n))
    action = ol.r4vt_action_row([[0.1, -0.2]], [delta_n], ol.rot6d_of_rotation(R_tool))
    dec = ol.decode_r4vt(runtime, action, x_s, R_tool, np.array([0.0]))
    lhs = ol.normal_offset(dec["x_d"], dec["x_s"], dec["n"])
    rhs = np.sum((p_vt - x_s) * n)
    assert abs(float(lhs[0]) - rhs) <= ol.TOL_POS_M
    assert float(ol.tangential_norm(dec["x_d"] - dec["x_s"], dec["n"])[0]) <= ol.TOL_POS_M
    assert float(ol.geodesic_deg(dec["R_d"], R_tool.reshape(1, 3, 3))[0]) <= ol.TOL_ROT_DEG
    offset = ol.tangential_vt_offset(p_vt.reshape(1, 3), x_s.reshape(1, 3), n.reshape(1, 3))
    expect = float(np.hypot(0.03, 0.04))
    assert abs(offset["median_m"] - expect) < 1e-12
    assert abs(offset["max_m"] - expect) < 1e-12


def test_r1rel_command_reconstructs_tcp():
    """The odom r1rel command is the corrected TCP, so the round-trip is ts_pose_fb."""
    import zarr
    from PyriteConfig.tasks.umift.umift_type_conversions import (
        action21_postprocess,
        action21_to_action_sample,
    )
    from PyriteUtility.spatial_math import spatial_utilities as su

    root = zarr.open(str(R1REL), mode="r")["data"]["episode_0"]
    fb = np.asarray(root["ts_pose_fb_0"], dtype=np.float64)
    command = np.asarray(root["ts_pose_command_0"], dtype=np.float64)
    if not np.array_equal(fb, command):
        raise AssertionError(
            "r1rel ts_pose_command_0 is not ts_pose_fb_0. "
            "The relative action reconstructs ts_pose_command_0, and this test "
            "must compare against that pose instead of loosening the TCP tolerance. "
            f"max |command-fb|={np.max(np.abs(command - fb))}"
        )
    assert ol.R1REL_LABEL_POSE == "ts_pose_fb_0"
    i0 = 200
    horizon = 8
    pose7 = fb[i0 : i0 + horizon]
    pose9 = su.SE3_to_pose9(su.pose7_to_SE3(pose7))
    action = np.zeros((horizon, 21), dtype=np.float64)
    action[:, :9] = pose9
    action[:, 9:18] = pose9
    action[:, 18] = 1.0
    base = [su.pose7_to_SE3(fb[i0])]
    rel = action21_to_action_sample(action, np.zeros((0, 21)), [0], base)["sparse"]
    mats, _vt, _stiff, _eoat = action21_postprocess(rel, base, [0])
    recon = su.SE3_to_pose7(mats[0].reshape(-1, 4, 4))
    pos = np.linalg.norm(recon[:, :3] - fb[i0 : i0 + horizon, :3], axis=1)
    from fixb_labels import pose7_to_Rp

    R_hat, _p = pose7_to_Rp(recon)
    R_fb, _ = pose7_to_Rp(fb[i0 : i0 + horizon])
    ang = ol.geodesic_deg(R_hat, R_fb)
    assert float(np.max(pos)) <= ol.TOL_POS_M
    assert float(np.max(ang)) <= ol.TOL_ROT_DEG


def test_r4vt_labels_match_normal_offset_and_tool_rotation():
    import zarr
    from fixb_field import FixBMoldField
    from fixb_labels import pose7_to_Rp

    r4 = zarr.open(str(R4VT), mode="r")["data"]["episode_0"]
    rel = zarr.open(str(R1REL), mode="r")["data"]["episode_0"]
    x_s = np.asarray(r4["mold_x_s_0"], dtype=np.float64).reshape(-1, 3)
    n = np.asarray(r4["mold_normal_0"], dtype=np.float64).reshape(-1, 3)
    v_par = np.asarray(r4["fixb_v_par_0"], dtype=np.float64)
    delta_n = np.asarray(r4["fixb_delta_n_0"], dtype=np.float64).reshape(-1)
    dR = np.asarray(r4["fixb_delta_R_rot6d_0"], dtype=np.float64).reshape(-1, 6)
    R_tool, _p = pose7_to_Rp(np.asarray(r4["ts_pose_fb_0"]))
    _Rv, p_vt = pose7_to_Rp(np.asarray(rel["ts_pose_virtual_target_0"]))
    m = min(len(x_s), len(p_vt), len(delta_n))
    offset = ol.tangential_vt_offset(p_vt[:m], x_s[:m], n[:m])
    print(
        f"episode 0 tangential VT offset median={offset['median_m']:.6f} m "
        f"max={offset['max_m']:.6f} m"
    )
    runtime = ol.make_r4_runtime(FixBMoldField(FIELD), dry_run=False)
    idx = np.linspace(0, m - 1, 12).astype(int)
    normal_err = []
    parallel = []
    rot_err = []
    for i in idx:
        action = ol.r4vt_action_row(v_par[i : i + 1], delta_n[i : i + 1], dR[i])
        # Anchor on the label foot so R(x_s) is the frame ΔR was written in.
        dec = ol.decode_r4vt(runtime, action, x_s[i], R_tool[i], np.array([0.0]))
        lhs = float(ol.normal_offset(dec["x_d"], dec["x_s"], dec["n"])[0])
        rhs = float(np.sum((p_vt[i] - x_s[i]) * n[i]))
        normal_err.append(abs(lhs - rhs))
        parallel.append(float(ol.tangential_norm(dec["x_d"] - dec["x_s"], dec["n"])[0]))
        rot_err.append(float(ol.geodesic_deg(dec["R_d"], R_tool[i : i + 1])[0]))
    assert max(normal_err) <= ol.TOL_POS_M, max(normal_err)
    assert max(parallel) <= ol.TOL_POS_M, max(parallel)
    assert max(rot_err) <= ol.TOL_ROT_DEG, max(rot_err)


def test_viewer_reads_r4vt_export():
    from view_openloop import episode_payload, load_export

    export = Path("/mnt/data/umidata/ft/open_loop_exports/r4vt_e66520fae9ae")
    loaded = load_export(export)
    payload = episode_payload([loaded], 0)
    assert payload["split"] == "train"
    assert payload["episode"] == 0
    run = payload["runs"][0]
    assert run["q_i"][0] == 0
    assert len(run["q_i"]) == len(run["q_t"]) == len(run["ang"])
    assert run["error"]["mae"]["delta_R_deg"]["clean"] is not None
    assert run["error"]["n"]["bad_span"] == 0
    val = episode_payload([loaded], 9)
    assert val["split"] == "val"
    assert val["runs"][0]["error"]["n"]["valid"] > 0


def test_span_toggle_changes_the_drawn_chunk():
    import numpy as np
    from view_openloop import cursor_payload, episode_payload, load_export

    loaded = load_export(Path("/mnt/data/umidata/ft/open_loop_exports/r4vt_e66520fae9ae"))
    run = episode_payload([loaded], 0)["runs"][0]
    assert len(run["pred_full"]) > len(run["pred_exec"])
    assert run["ckpt"]
    assert run["epoch"] == 185
    blob = loaded["episodes"][0]
    query = int(np.argmax(blob["n_valid"]))
    short = cursor_payload([loaded], 0, query, span="exec")["runs"][0]
    long = cursor_payload([loaded], 0, query, span="full")["runs"][0]
    assert len(long["future"]) > len(short["future"])
    assert len(short["future"]) == 6
    assert len(short["future_R"]) == len(short["future"])
    assert len(long["future_R"]) == len(long["future"])
    assert len(long["label_R"]) == len(long["label_xyz"])
    assert len(long["label_xyz"]) > len(short["future"])


def test_frames_overlay_is_off_unless_asked():
    from view_openloop import cursor_payload, episode_payload, load_export, load_frames_npz

    export = Path("/mnt/data/umidata/ft/open_loop_exports/r4vt_e66520fae9ae")
    frames_path = Path(
        "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr"
        "/r4vt_frame_continuity/ep0_ep9_frames.npz"
    )
    loaded = load_export(export)
    plain = episode_payload([loaded], 0)
    assert "frames" not in plain
    assert "frame" not in cursor_payload([loaded], 0, 0)
    frames = load_frames_npz(frames_path)
    over = episode_payload([loaded], 0, frames=frames)
    assert over["frames"]["dR_old"]
    assert over["frames"]["dR_new"]
    assert over["frames"]["jumps"]
    assert len(over["frames"]["t"]) == len(over["frames"]["dR_old"]) == len(over["frames"]["dR_new"])
    missing = episode_payload([loaded], 9, frames={0: frames[0]})
    assert "frames" not in missing
    cursor = cursor_payload([loaded], 0, 0, frames=frames)
    assert cursor["frame"]["x_old"] and cursor["frame"]["R_old"]
    assert cursor["frame"]["x_new"] and cursor["frame"]["R_new"]
    assert "switches" not in plain
    assert "events" not in frames


def test_frames_only_episode_and_switch_time():
    from view_openloop import cursor_payload, frames_only_payload

    n = 6
    eye = np.repeat(np.eye(3)[None], n, axis=0)
    robot_t = np.arange(n, dtype=np.float64) * 0.1 + 10.0
    frames = {
        4: {
            "x_old": np.zeros((n, 3)),
            "x_new": np.arange(n)[:, None] * np.array([0.01, 0.0, 0.0]),
            "R_old": eye,
            "R_new": eye,
            "old_jump": np.array([False, False, True, False, False, False]),
            "robot_time_s": robot_t,
            "t0": 10.0,
            "tcp_xyz": np.zeros((n, 3)),
            "tcp_R": eye,
            "split": "train",
        },
        "events": {
            "switches": [
                {"ep": 4, "i": 2, "time_s": 10.2, "t": 0.2, "frame": 25.0, "tool": 0.1, "x": [0.0, 0.0, 0.0]}
            ],
            "nearcap": [],
        },
    }
    payload = frames_only_payload(frames, 4)
    assert payload["policy"] is False
    assert payload["split"] == "train"
    assert payload["switches"][0]["t"] == 0.2
    assert payload["switches"][0]["frame"] == 25.0
    assert payload["frames"]["dR_old"]
    cursor = cursor_payload([], 4, payload["path_i"][0], frames=frames)
    assert cursor["frame"]["x_new"]


def test_region_bins_match_the_tilt_split():
    up = np.array([[0.0, -1.0, 0.0]])
    fillet = Rotation.from_rotvec([np.radians(40.0), 0.0, 0.0]).apply(up)
    wall = Rotation.from_rotvec([np.radians(70.0), 0.0, 0.0]).apply(up)
    codes = ol.region_code(np.vstack([up, fillet, wall]))
    assert list(codes) == [0, 1, 2]
    assert ol.REGION_NAMES == ("crown", "fillet", "wall")


def test_region_errors_follow_the_query_label_foot():
    pred = np.zeros((2, 4, 27))
    label = np.zeros((2, 4, 27))
    eye = ol.rot6d_of_rotation(np.eye(3))
    pred[:, :, 21:27] = eye
    label[:, :, 21:27] = eye
    pred[0, :, 2] = 0.01
    pred[1, :, 2] = -0.03
    yaw = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    pred[1, :, 21:27] = ol.rot6d_of_rotation(yaw)
    table = ol.region_action_errors(pred, label, np.array([4, 4]), np.array([0, 2]), horizon=6)
    assert table["exec"]["crown"]["delta_n_mm"] == 10.0
    assert table["exec"]["wall"]["delta_n_mm"] == 30.0
    assert table["exec"]["fillet"]["n"] == 0
    assert table["full"]["wall"]["delta_R_deg"] == 90.0
    assert table["exec"]["crown"]["n"] == 4


def test_label_virtual_target_is_the_foot_plus_delta_n():
    from view_openloop import checkpoint_label, label_virtual_target, PAGE

    foot = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    normal = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    got = label_virtual_target(foot, normal, np.array([0.02, -0.01]))
    assert np.allclose(got[0], [0.0, -0.02, 0.0])
    assert np.allclose(got[1], [1.0, 0.0, -0.01])
    marked = checkpoint_label({"role": "r1rel", "checkpoint_epoch": 195, "val_loss": 0.152}, "r1rel")
    assert "tool pose" in marked
    assert "tool pose" not in checkpoint_label({"role": "r4vt", "checkpoint_epoch": 55, "val_loss": 0.030}, "r4vt")
    assert 'id="geomFoot"' in PAGE
    assert 'id="onlySel"' in PAGE
    assert "dnSegments" in PAGE
    from view_openloop import foot_path_breaks
    gap = foot_path_breaks([0, 2, 4, 20], [False, False, False, False])
    assert list(gap) == [False, False, True, False]
    flagged = foot_path_breaks([0, 2, 4], [False, True, False])
    assert list(flagged) == [True, True, False]
    from view_openloop import table_keep
    order = table_keep(np.array([0.3, 0.0, 0.2, 0.1]), {"query_stride": 1})
    assert list(order) == [1]
    kept = table_keep(np.array([0.0, 0.1, 0.2, 0.3]), {"query_stride": 4})
    assert list(kept) == [0, 1, 2, 3]


def test_openloop_page_has_region_filter():
    from view_openloop import PAGE

    assert 'id="regFillet"' in PAGE
    assert 'id="regWall"' in PAGE
    assert "regionFilter" in PAGE


def test_saved_export_region_uses_the_label_foot_normal():
    zarr = pytest.importorskip("zarr")
    export = Path(
        "/mnt/data/umidata/ft/open_loop_exports/r4vt_smooth_e0030_43335222_release/episode_0.npz"
    )
    dataset = Path(
        "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
        "r4vt_smooth_frame/acp_replay_buffer.zarr"
    )
    if not export.is_file() or not dataset.is_dir():
        pytest.skip("released smooth export is not on this machine")
    from view_openloop import query_region_names

    blob = np.load(export)
    names = query_region_names(blob, 0, str(dataset))
    idx = ol.query_anchor_index(blob["query_rgb_idx"], blob["robot_time_s"], blob["rgb_time_s"])
    normal = np.asarray(zarr.open(str(dataset), "r")["data"]["episode_0"]["mold_normal_0"])
    codes = ol.region_code(normal[idx])
    assert names[0] == ol.REGION_NAMES[int(codes[0])]
    assert set(names) <= set(ol.REGION_NAMES)

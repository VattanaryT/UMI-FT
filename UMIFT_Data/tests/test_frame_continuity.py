"""Face-switch hysteresis and the motion-explained ΔR mask. No robot."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

SCRIPTS = Path(__file__).resolve().parents[1] / "surface_frames_scripts"
sys.path.insert(0, str(SCRIPTS))

from fixb_field import (  # noqa: E402
    FACE_SWITCH_MARGIN_M,
    FixBMoldField,
    accept_new_face,
    orthonormal_frames,
    slew_frame,
)
from fixb_labels import pose7_to_Rp, rot_angle_deg, smooth_positions  # noqa: E402
from fixb_r4_chunk_filter import episode_masks_from_zarr  # noqa: E402
from mold_aware_runtime import MoldAwareRuntime  # noqa: E402


def _pose(R):
    q = Rotation.from_matrix(np.asarray(R, dtype=np.float64)).as_quat()
    out = np.zeros((len(R), 7))
    out[:, 3] = q[:, 3]
    out[:, 4:] = q[:, :3]
    return out


def _pose_at(R, p):
    out = _pose(R)
    out[:, :3] = np.asarray(p, dtype=np.float64)
    return out


def _quad(origin, axis_u, axis_v):
    o = np.asarray(origin, dtype=np.float64)
    u = np.asarray(axis_u, dtype=np.float64)
    v = np.asarray(axis_v, dtype=np.float64)
    verts = np.stack([o, o + u, o + u + v, o + v])
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return verts, faces


def _field(tmp_path, sheets):
    verts, faces, normals, t1s = [], [], [], []
    base = 0
    for origin, axis_u, axis_v, normal, t1 in sheets:
        v, f = _quad(origin, axis_u, axis_v)
        n = np.asarray(normal, dtype=np.float64)
        n = n / np.linalg.norm(n)
        t = np.asarray(t1, dtype=np.float64)
        verts.append(v)
        faces.append(f + base)
        normals.append(np.repeat(n[None], len(v), axis=0))
        t1s.append(np.repeat(t[None], len(v), axis=0))
        base += len(v)
    path = tmp_path / "field.npz"
    frames = orthonormal_frames(np.concatenate(t1s), np.concatenate(normals))
    V = np.concatenate(verts)
    np.savez(
        path,
        vertices=V,
        faces=np.concatenate(faces),
        normals=np.concatenate(normals),
        frames=frames,
        curv_tensor=np.zeros((len(V), 3, 3)),
    )
    return FixBMoldField(path)


def _sheets(tmp_path, t1_sign=1.0):
    # Two parallel sheets, 30 mm apart. Global closest flips between them.
    u = (0.10, 0.0, 0.0)
    v = (0.0, 0.10, 0.0)
    return _field(
        tmp_path,
        [
            ((-0.05, -0.05, 0.0), u, v, (0, 0, 1), (t1_sign, 0, 0)),
            ((-0.05, -0.05, 0.03), u, v, (0, 0, 1), (t1_sign, 0, 0)),
        ],
    )


def _corner(tmp_path):
    # Crown z=0 (x<=0) and wall x=0 (normal +X).
    return _field(
        tmp_path,
        [
            ((-0.04, -0.02, 0.0), (0.04, 0, 0), (0, 0.04, 0), (0, 0, 1), (1, 0, 0)),
            ((0.0, -0.02, 0.0), (0, 0, 0.04), (0, 0.04, 0), (1, 0, 0), (0, 0, 1)),
        ],
    )


def test_distance_margin_is_5mm():
    assert FACE_SWITCH_MARGIN_M == 0.005
    switch, reason = accept_new_face(0.020, 0.010)
    assert switch and reason == "margin"
    switch, reason = accept_new_face(0.010, 0.006)
    assert not switch and reason == "hold"


def test_tool_rotation_toward_new_normal_switches_inside_the_margin():
    n_new = np.array([1.0, 0.0, 0.0])
    a_prev = np.array([0.0, 0.0, 1.0])
    tilt = np.radians(20.0)
    a_now = np.array([np.sin(tilt), 0.0, np.cos(tilt)])
    switch, reason = accept_new_face(0.010, 0.008, a_prev, a_now, n_new)
    assert switch and reason == "rotation"
    tilt = np.radians(2.0)
    a_now = np.array([np.sin(tilt), 0.0, np.cos(tilt)])
    switch, _ = accept_new_face(0.010, 0.008, a_prev, a_now, n_new)
    assert not switch


def test_track_holds_a_1mm_hop_and_switches_when_the_tool_moves_over(tmp_path):
    field = _sheets(tmp_path)
    # 1 mm closer, and the foot would jump 30 mm: hold.
    # The tool then travels onto the upper sheet, so the foot moves with it.
    p = np.array([[0.0, 0.0, 0.002], [0.0, 0.0, 0.0155], [0.0, 0.0, 0.050]])
    out = field.track_foot(p)
    assert out["x_s"][0, 2] < 0.005
    assert out["x_s"][1, 2] < 0.005
    assert out["x_s"][2, 2] > 0.02


def test_track_switches_crown_to_wall_when_the_tool_tilts(tmp_path):
    field = _corner(tmp_path)
    crown = np.array([-0.01, 0.0, 0.008])
    # Global closest is the wall, but only ~2 mm closer than the crown.
    ambiguous = np.array([-0.004, 0.0, 0.006])
    free = field.query(ambiguous[None])
    assert free["x_s"][0, 0] > -1e-4
    p = np.stack([crown, ambiguous])
    eye = np.broadcast_to(np.eye(3), (2, 3, 3)).copy()
    held = field.track_foot(p, R_tool=eye)
    assert held["x_s"][1, 2] < 1e-4
    tilt = Rotation.from_euler("y", 25.0, degrees=True).as_matrix()
    R_tool = np.stack([np.eye(3), tilt])
    switched = field.track_foot(p, R_tool=R_tool)
    assert switched["x_s"][1, 0] > -1e-4
    rel = switched["R"][0].T @ switched["R"][1]
    step = np.degrees(np.arccos(np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0)))
    assert step <= 25.0


def test_same_face_t1_flip_keeps_the_previous_tangent(tmp_path):
    field = _sheets(tmp_path, t1_sign=-1.0)
    p = np.array([[0.0, 0.0, 0.002]])
    R_seed = np.eye(3)
    out = field.track_foot(p, x_seed=np.array([0.0, 0.0, 0.0]), R_seed=R_seed)
    assert float(out["R"][0, :, 0] @ np.array([1.0, 0.0, 0.0])) > 0.9


def test_mask_keeps_motion_that_matches_the_tool_and_drops_a_bare_flip():
    R = Rotation.from_euler("y", np.array([0.0, 40.0])[:, None], degrees=True).as_matrix()
    eye = np.broadcast_to(np.eye(3), (2, 3, 3)).copy()
    x = np.zeros((2, 3))
    explained = {
        "mold_R_field_0": R.reshape(2, 9),
        "mold_x_s_0": x,
        "ts_pose_fb_0": _pose(R),
    }
    bare = {
        "mold_R_field_0": R.reshape(2, 9),
        "mold_x_s_0": x,
        "ts_pose_fb_0": _pose(eye),
    }
    assert not episode_masks_from_zarr(explained)["field_jump"][1]
    assert episode_masks_from_zarr(bare)["field_jump"][1]
    flip = eye.copy()
    flip[1] = np.diag([-1.0, -1.0, 1.0])
    signed = {
        "mold_R_field_0": flip.reshape(2, 9),
        "mold_x_s_0": x,
        "ts_pose_fb_0": _pose(eye),
    }
    assert episode_masks_from_zarr(signed)["field_jump"][1]


def test_slew_caps_one_step_at_25_degrees():
    target = Rotation.from_euler("y", 70.0, degrees=True).as_matrix()
    mid = slew_frame(np.eye(3), target)
    step = float(rot_angle_deg(np.eye(3)[None], mid[None])[0])
    assert abs(step - 25.0) < 1e-4


def test_live_windows_match_the_label_track_and_reset(tmp_path):
    """Overlapping live buffers, after reset, equal one track_foot from the start.

    The second sheet is 60° from the first, so the published frame is still
    mid-slew when the next buffer arrives. The decision face has to stay the
    unslewed sheet or the next step will not match the one-shot labels.
    """
    ang = np.radians(60.0)
    c, s = np.cos(ang), np.sin(ang)
    n_b = (s, 0.0, c)
    t_b = (c, 0.0, -s)
    origin = np.array([0.04, -0.05, 0.0])
    u = np.array([0.08 * c, 0.0, -0.08 * s])
    field = _field(
        tmp_path,
        [
            ((-0.06, -0.05, 0.0), (0.08, 0, 0), (0, 0.10, 0), (0, 0, 1), (1, 0, 0)),
            (tuple(origin), tuple(u), (0, 0.10, 0), n_b, t_b),
        ],
    )
    tool_b = origin + 0.5 * u + np.array([0.0, 0.05, 0.0]) + 0.01 * np.array(n_b)
    along = np.array(t_b)
    p = np.stack(
        [
            np.array([-0.02, 0.0, 0.008]),
            tool_b,
            tool_b + 0.004 * along,
            tool_b + 0.008 * along,
            tool_b + 0.012 * along,
        ]
    )
    R_tool = np.broadcast_to(np.eye(3), (len(p), 3, 3)).copy()
    times = np.arange(len(p), dtype=np.float64) * 0.005
    p_s = smooth_positions(times, p, 0.1)
    one = field.query(p_s, with_curvature=True, R_tool=R_tool)
    steps = rot_angle_deg(one["R"][:-1], one["R"][1:])
    assert float(steps.max()) <= 25.0 + 1e-4
    assert float(steps.max()) > 20.0
    rt = MoldAwareRuntime(
        field=field, R_off=np.eye(3), run_type="r4_world_fixb_tilt", dry_run=False
    )
    rt.reset()
    for start in range(0, len(p)):
        sl = slice(start, min(start + 2, len(p)))
        obs = {
            "ts_pose_fb_0": _pose_at(R_tool[sl], p[sl]),
            "robot_time_stamps_0": times[sl],
        }
        live = rt.fill_obs_raw(obs)
        assert np.allclose(live["x_s"], one["x_s"][sl], atol=1e-8)
        assert np.allclose(live["R"], one["R"][sl], atol=1e-6)
    rt.reset()
    obs = {
        "ts_pose_fb_0": _pose_at(R_tool[:1], p[:1]),
        "robot_time_stamps_0": times[:1],
    }
    live = rt.fill_obs_raw(obs)
    fresh = field.query(p[:1])
    assert np.allclose(live["x_s"], fresh["x_s"], atol=1e-8)
    runner = Path(__file__).resolve().parents[2] / "PyriteEnvSuites" / "env_runners" / "umift_env_runner.py"
    assert "mold_runtime.reset()" in runner.read_text()


def test_query_smooth_uses_only_past_samples_and_live_matches_it(tmp_path):
    t = np.arange(80, dtype=np.float64) * 0.005
    p = np.stack([0.02 * t, 0.005 * np.sin(40 * t), np.full(len(t), 0.004)], axis=1)
    base = smooth_positions(t, p, 0.1)
    bumped = p.copy()
    bumped[-1, 0] += 0.04
    assert np.allclose(base[:-1], smooth_positions(t, bumped, 0.1)[:-1])
    from scipy.signal import savgol_filter

    win = max(5, int(round(0.1 / 0.005)) | 1)
    assert not np.allclose(
        savgol_filter(p, win, 2, axis=0)[:-1],
        savgol_filter(bumped, win, 2, axis=0)[:-1],
    )
    field = _sheets(tmp_path)
    R_tool = np.broadcast_to(np.eye(3), (len(p), 3, 3)).copy()
    label = field.query(base, with_curvature=True, R_tool=R_tool)
    rt = MoldAwareRuntime(field=field, R_off=np.eye(3), run_type="r4_world_fixb_tilt", dry_run=False)
    rt.reset()
    for end in range(1, len(p) + 1):
        sl = slice(max(0, end - 3), end)
        live = rt.fill_obs_raw(
            {"ts_pose_fb_0": _pose_at(R_tool[sl], p[sl]), "robot_time_stamps_0": t[sl]}
        )
        assert np.array_equal(live["reason"], label["reason"][sl])
        assert np.allclose(live["R"], label["R"][sl], atol=1e-6)
        assert np.allclose(live["x_s"], label["x_s"][sl], atol=1e-8)


def test_query_with_tool_rotation_uses_the_same_tracker(tmp_path):
    field = _sheets(tmp_path)
    p = np.array([[0.0, 0.0, 0.002], [0.0, 0.0, 0.016]])
    R_tool = np.broadcast_to(np.eye(3), (2, 3, 3)).copy()
    out = field.query(p, R_tool=R_tool)
    direct = field.track_foot(p, R_tool=R_tool)
    assert np.allclose(out["x_s"], direct["x_s"])
    # pose7_to_Rp roundtrip stays available for the label path
    R_back, _ = pose7_to_Rp(_pose(R_tool))
    assert np.allclose(R_back, R_tool, atol=1e-6)

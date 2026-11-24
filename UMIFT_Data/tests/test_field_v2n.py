"""field_v2n: air δn round-trips the TCP. v1, v2, and ΔR match field_v2."""

import sys
from pathlib import Path

import numpy as np
import zarr

SCRIPTS = Path(__file__).resolve().parents[1] / "surface_frames_scripts"
sys.path.insert(0, str(SCRIPTS))

from field_v2n import K, SRC, blend, target_clearance  # noqa: E402
from fixb_field import FixBMoldField  # noqa: E402
from fixb_labels import next_control_times, pose7_to_Rp, so3_to_rot6d_rows  # noqa: E402
from r4vt_horizon import decode_horizon  # noqa: E402

OUT = SRC.parent.parent / "r4vt_field_v2n" / "acp_replay_buffer.zarr"
FIELD = SRC.parent.parent / "shared" / "fixb_field.npz"


def test_blend_is_linear_between_2_and_5n():
    force = np.array([-30.0 / K, -3.5 / K, -1.0 / K])
    fn = -force * K
    clearance = np.full(3, 0.02)
    got = blend(force, clearance, fn)
    assert got[0] == force[0]
    assert got[2] == clearance[2]
    weight = (3.5 - 2.0) / 3.0
    assert abs(got[1] - ((1.0 - weight) * 0.02 + weight * force[1])) < 1e-12


def test_air_offset_roundtrip_contact_keeps_tangent():
    frame = np.eye(3)
    foot = np.array([0.0, 0.04, 0.0])
    p0 = np.array([0.0, 0.0, -0.004])
    target = np.array([0.01, 0.0, -0.02])
    outward = -frame[:, 2]
    action = np.zeros((1, 10))
    action[0, 0:2] = [0.01, 0.0]
    action[0, 2] = float(np.dot(target - foot, outward))
    action[0, 3] = K
    action[0, 4:10] = so3_to_rot6d_rows(np.eye(3))[0]
    frames = frame.reshape(1, 3, 3)
    feet = foot.reshape(1, 3)
    air, _rot = decode_horizon(action, p0, frames, [1.0], air=[True], feet=feet)
    assert np.linalg.norm(air[0] - target) < 1e-9
    contact, _rot = decode_horizon(action, p0, frames, [1.0], air=[False], feet=feet)
    assert np.allclose(contact[0], p0 + np.array([0.01, 0.0, 0.0]))


def test_labels_match_field_v2_and_air_roundtrip():
    if not OUT.exists():
        raise AssertionError(f"missing {OUT}")
    old = zarr.open(str(SRC), "r")["data"]
    new = zarr.open(str(OUT), "r")["data"]
    field = FixBMoldField(str(FIELD))
    worst = 0.0
    for ep_id in range(50):
        key = f"episode_{ep_id}"
        src, dst = old[key], new[key]
        assert np.array_equal(np.asarray(src["fixb_v_par_0"]), np.asarray(dst["fixb_v_par_0"]))
        assert np.array_equal(np.asarray(src["fixb_delta_R_rot6d_0"]), np.asarray(dst["fixb_delta_R_rot6d_0"]))
        assert np.array_equal(
            np.asarray(src["ts_pose_command_0"])[:, [0, 1]],
            np.asarray(dst["ts_pose_command_0"])[:, [0, 1]],
        )
        stored = np.asarray(src["fixb_delta_n_0"], dtype=np.float64).reshape(-1)
        got = np.asarray(dst["fixb_delta_n_0"], dtype=np.float64).reshape(-1)
        fn = -stored * K
        assert np.array_equal(got[np.abs(fn) >= 5.0].astype(np.float32), stored[np.abs(fn) >= 5.0].astype(np.float32))
        if ep_id not in (0, 12, 43):
            continue
        robot_t = np.asarray(dst["robot_time_stamps_0"]).reshape(-1)
        rgb_t = np.asarray(dst["rgb_time_stamps_0"]).reshape(-1)
        _tool_r, position = pose7_to_Rp(np.asarray(dst["ts_pose_fb_0"], dtype=np.float64))
        frame = np.asarray(dst["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3)
        arrived, foot, clearance = target_clearance(position, frame, robot_t, rgb_t, field)
        air = np.abs(fn) <= 2.0
        assert np.max(np.abs(got[air] - clearance[air])) < 1e-6
        dt = next_control_times(robot_t, rgb_t) - robot_t
        velocity = np.asarray(dst["fixb_v_par_0"], dtype=np.float64)
        take = np.flatnonzero(air)[:: max(1, int(air.sum()) // 40)]
        for i in take:
            action = np.zeros((1, 10))
            action[0, 0:2] = velocity[i]
            action[0, 2] = got[i]
            action[0, 3] = K
            action[0, 4:10] = np.asarray(dst["fixb_delta_R_rot6d_0"][i])
            xyz, _rot = decode_horizon(
                action, position[i], frame[i:i + 1], dt[i:i + 1], air=[True], feet=foot[i:i + 1]
            )
            worst = max(worst, float(np.linalg.norm(xyz[0] - arrived[i])))
    assert worst < 1e-3, f"air round-trip {worst * 1e3:.3f} mm"

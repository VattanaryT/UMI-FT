import importlib.util
import pathlib

import numpy as np


SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts"
    / "convert_lerobot_teleop_to_umift_zarr.py"
)
SPEC = importlib.util.spec_from_file_location(
    "convert_lerobot_teleop_to_umift_zarr", SCRIPT_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

rot6d_to_quat_wxyz = MODULE.rot6d_to_quat_wxyz
right_state_to_pose7_wrench = MODULE.right_state_to_pose7_wrench


def test_rot6d_identity_to_quat():
    r6 = np.array([1.0, 0, 0, 0, 1, 0], dtype=np.float64)
    q = rot6d_to_quat_wxyz(r6)
    assert q.shape == (4,)
    assert np.allclose(np.abs(q), [1, 0, 0, 0], atol=1e-5) or np.allclose(
        q, [1, 0, 0, 0], atol=1e-5
    )


def test_right_state_unpack():
    state = np.zeros((2, 30), dtype=np.float32)
    state[0, 0:3] = [0.1, 0.2, 0.3]
    state[0, 3:9] = [1, 0, 0, 0, 1, 0]
    state[0, 9:15] = [1, 2, 3, 4, 5, 6]
    pose7, wrench = right_state_to_pose7_wrench(state)
    assert pose7.shape == (2, 7)
    assert wrench.shape == (2, 6)
    assert np.allclose(pose7[0, :3], [0.1, 0.2, 0.3])
    assert np.allclose(wrench[0], [1, 2, 3, 4, 5, 6])


def test_build_episode_arrays_geometry_only():
    T = 3
    H = W = 8
    timestamps = np.array([0.0, 0.1, 0.2], dtype=np.float64)
    state = np.zeros((T, 30), dtype=np.float32)
    state[:, 0:3] = np.arange(T).reshape(-1, 1) * 0.1
    state[:, 3:9] = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    state[:, 9:15] = 7.0
    rgb = np.zeros((T, H, W, 3), dtype=np.uint8)
    rgb[:, 2:5, 2:5] = 200

    ep = MODULE.build_episode_arrays_from_tables(
        timestamps=timestamps,
        observation_state=state,
        rgb_frames=rgb,
        image_size=224,
    )

    assert ep["rgb_0"].shape == (T, 224, 224, 3)
    assert ep["ultrawide_0"].shape == (T, 224, 224, 3)
    assert ep["ultrawide_0"].dtype == np.uint8
    assert np.array_equal(ep["ultrawide_0"], np.zeros_like(ep["ultrawide_0"]))
    assert not np.array_equal(ep["ultrawide_0"], ep["rgb_0"])
    assert ep["depth_0"].shape == (T, 224, 224, 3)
    assert ep["depth_0"].dtype == np.float16
    assert np.allclose(ep["depth_0"], 0.0)
    assert np.allclose(ep["ts_pose_command_0"], ep["ts_pose_fb_0"])
    assert np.allclose(ep["wrench_right_0"], 0.0)
    assert np.allclose(ep["wrench_left_0"], 7.0)
    assert np.allclose(ep["gripper_0"], 0.0)
    assert ep["map_to_uw_idx_0"].shape == (T, 1)
    assert ep["map_to_d_idx_0"].shape == (T, 1)
    assert np.array_equal(ep["map_to_uw_idx_0"].reshape(-1), np.arange(T))
    assert "ts_pose_virtual_target_0" not in ep
    assert "stiffness_0" not in ep
    for key in (
        "rgb_time_stamps_0",
        "robot_time_stamps_0",
        "wrench_time_stamps_left_0",
        "wrench_time_stamps_right_0",
    ):
        assert np.allclose(ep[key].reshape(-1), timestamps)


def test_parser_image_size_default():
    parser = MODULE.build_parser()
    args = parser.parse_args(["--input", "x", "--output", "y"])
    assert args.image_size == 224


def test_parser_start_pad_seconds_defaults_to_five():
    parser = MODULE.build_parser()
    args = parser.parse_args(["--input", "x", "--output", "y"])
    assert args.start_pad_seconds == 5.0
    args2 = parser.parse_args(
        ["--input", "x", "--output", "y", "--start-pad-seconds", "0"]
    )
    assert args2.start_pad_seconds == 0.0

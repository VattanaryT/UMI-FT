import numpy as np
from umift.processing.umift_image_prep import (
    resize_hwc_uint8,
    resize_hwc_float,
    make_uniform_time_grid,
    nearest_indices,
    resample_episode_to_hz,
)


def test_resize_hwc_uint8_to_224():
    frames = np.zeros((3, 48, 64, 3), dtype=np.uint8)
    frames[:, 10:20, 10:20] = 255
    out = resize_hwc_uint8(frames, (224, 224))
    assert out.shape == (3, 224, 224, 3)
    assert out.dtype == np.uint8


def test_resize_hwc_float_preserves_dtype_float16():
    depth = np.ones((2, 40, 50, 3), dtype=np.float16) * 0.25
    out = resize_hwc_float(depth, (224, 224))
    assert out.shape == (2, 224, 224, 3)
    assert out.dtype == np.float16


def test_make_uniform_time_grid_10hz():
    g = make_uniform_time_grid(0.0, 1.0, 10.0)
    assert len(g) == 11  # 0.0 .. 1.0 inclusive at 0.1
    assert np.isclose(g[1] - g[0], 0.1)


def test_nearest_indices_picks_closest():
    src = np.array([0.0, 0.1, 0.2])
    query = np.array([0.0, 0.09, 0.25])
    idx = nearest_indices(src, query)
    assert list(idx) == [0, 1, 2]


def test_resample_episode_to_hz_20_to_10():
    """Synthetic 20 Hz / 1s episode → 10 Hz: equal lengths, identity maps, ~0.1 s dt."""
    src_hz = 20.0
    duration_s = 1.0
    n_src = int(src_hz * duration_s) + 1  # 0.0 .. 1.0 inclusive
    t = np.linspace(0.0, duration_s, n_src)
    t_col = t.reshape(-1, 1)

    pose = np.zeros((n_src, 7), dtype=np.float64)
    pose[:, 3] = 1.0  # identity quat wxyz
    pose[:, 0] = t  # x drifts with time for a non-trivial resample

    ep = {
        "rgb_0": np.zeros((n_src, 8, 8, 3), dtype=np.uint8),
        "ultrawide_0": np.zeros((n_src, 8, 8, 3), dtype=np.uint8),
        "depth_0": np.zeros((n_src, 8, 8, 3), dtype=np.float16),
        "ts_pose_fb_0": pose,
        "gripper_0": np.linspace(0.0, 1.0, n_src).reshape(-1, 1),
        "wrench_left_0": np.ones((n_src, 6), dtype=np.float64),
        "wrench_right_0": np.zeros((n_src, 6), dtype=np.float64),
        "rgb_time_stamps_0": t_col.copy(),
        "ultrawide_time_stamps_0": t_col.copy(),
        "depth_time_stamps_0": t_col.copy(),
        "robot_time_stamps_0": t_col.copy(),
        "gripper_time_stamps_0": t_col.copy(),
        "wrench_time_stamps_left_0": t_col.copy(),
        "wrench_time_stamps_right_0": t_col.copy(),
        "map_to_uw_idx_0": np.arange(n_src, dtype=np.int64).reshape(-1, 1),
        "map_to_d_idx_0": np.arange(n_src, dtype=np.int64).reshape(-1, 1),
    }

    out = resample_episode_to_hz(ep, hz=10.0)
    n_out = len(out["rgb_0"])
    assert n_out == 11  # 0.0 .. 1.0 at 10 Hz

    stream_keys = [
        "rgb_0",
        "ultrawide_0",
        "depth_0",
        "ts_pose_fb_0",
        "gripper_0",
        "wrench_left_0",
        "wrench_right_0",
    ]
    for key in stream_keys:
        assert len(out[key]) == n_out, key

    for ts_key in (
        "rgb_time_stamps_0",
        "robot_time_stamps_0",
        "wrench_time_stamps_left_0",
    ):
        ts = np.asarray(out[ts_key]).reshape(-1)
        assert len(ts) == n_out
        assert np.isclose(ts[1] - ts[0], 0.1)
        assert np.isclose(ts[0], 0.0)
        assert np.isclose(ts[-1], 1.0)

    expected_id = np.arange(n_out, dtype=np.int64).reshape(-1, 1)
    assert np.array_equal(out["map_to_uw_idx_0"], expected_id)
    assert np.array_equal(out["map_to_d_idx_0"], expected_id)

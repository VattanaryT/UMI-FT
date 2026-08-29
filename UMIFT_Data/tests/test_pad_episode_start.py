import numpy as np

from umift.processing.pad_episode_start import pad_episode_start


def _uniform_episode(n: int = 20, hz: float = 10.0) -> dict:
    """Synthetic episode where every stream shares the same 10 Hz grid,
    mirroring the post `--resample-10hz` schema."""
    dt = 1.0 / hz
    t = (np.arange(n, dtype=np.float64) * dt).reshape(-1, 1)

    pose = np.zeros((n, 7), dtype=np.float64)
    pose[:, 3] = 1.0  # identity quat wxyz
    pose[:, 0] = np.arange(n, dtype=np.float64)  # x drifts so "real" rows differ from row 0

    wrench_left = np.stack([np.arange(n, dtype=np.float64)] * 6, axis=-1)
    wrench_right = np.zeros((n, 6), dtype=np.float64)
    wrench_concat = np.hstack((wrench_left, wrench_right))

    return {
        "rgb_0": np.arange(n, dtype=np.uint8).reshape(n, 1, 1, 1).repeat(3, axis=-1),
        "ultrawide_0": np.zeros((n, 1, 1, 3), dtype=np.uint8),
        "depth_0": np.zeros((n, 1, 1, 3), dtype=np.float16),
        "ts_pose_fb_0": pose,
        "gripper_0": np.arange(n, dtype=np.float64).reshape(-1, 1),
        "wrench_left_0": wrench_left,
        "wrench_right_0": wrench_right,
        "wrench_concat_0": wrench_concat,
        "rgb_time_stamps_0": t.copy(),
        "ultrawide_time_stamps_0": t.copy(),
        "depth_time_stamps_0": t.copy(),
        "robot_time_stamps_0": t.copy(),
        "gripper_time_stamps_0": t.copy(),
        "wrench_time_stamps_left_0": t.copy(),
        "wrench_time_stamps_right_0": t.copy(),
        "map_to_uw_idx_0": np.arange(n, dtype=np.int64).reshape(-1, 1),
        "map_to_d_idx_0": np.arange(n, dtype=np.int64).reshape(-1, 1),
    }


def test_zero_pad_duration_is_a_noop():
    ep = _uniform_episode()
    out = pad_episode_start(ep, pad_duration_s=0.0)
    assert out is not ep
    for key, value in ep.items():
        assert np.array_equal(out[key], value)


def test_uniform_grid_pads_every_stream_by_the_same_count():
    n = 20
    ep = _uniform_episode(n=n, hz=10.0)
    out = pad_episode_start(ep, pad_duration_s=2.0, id_list=[0])

    n_pad = 20  # 2.0s @ 10Hz
    for key in (
        "rgb_0",
        "ultrawide_0",
        "depth_0",
        "ts_pose_fb_0",
        "gripper_0",
        "wrench_left_0",
        "wrench_right_0",
        "wrench_concat_0",
    ):
        assert out[key].shape[0] == n + n_pad, key

    for ts_key in (
        "rgb_time_stamps_0",
        "robot_time_stamps_0",
        "wrench_time_stamps_left_0",
    ):
        assert out[ts_key].shape[0] == n + n_pad, ts_key


def test_padded_prefix_repeats_the_first_real_row_verbatim():
    ep = _uniform_episode(n=20, hz=10.0)
    out = pad_episode_start(ep, pad_duration_s=2.0, id_list=[0])
    n_pad = 20

    assert np.allclose(out["ts_pose_fb_0"][:n_pad], ep["ts_pose_fb_0"][0])
    assert np.allclose(out["wrench_left_0"][:n_pad], ep["wrench_left_0"][0])
    assert np.array_equal(
        out["rgb_0"][:n_pad], np.repeat(ep["rgb_0"][:1], n_pad, axis=0)
    )
    # real data is unchanged and simply shifted back by n_pad
    assert np.allclose(out["ts_pose_fb_0"][n_pad:], ep["ts_pose_fb_0"])
    assert np.allclose(out["wrench_left_0"][n_pad:], ep["wrench_left_0"])


def test_padded_timestamps_are_monotonic_and_start_at_zero():
    ep = _uniform_episode(n=20, hz=10.0)
    out = pad_episode_start(ep, pad_duration_s=2.0, id_list=[0])

    for ts_key in ("rgb_time_stamps_0", "robot_time_stamps_0", "wrench_time_stamps_left_0"):
        ts = np.asarray(out[ts_key]).reshape(-1)
        assert np.isclose(ts[0], 0.0)
        assert np.all(np.diff(ts) > 0)
        assert np.allclose(np.diff(ts), 0.1, atol=1e-9)


def test_index_maps_prefix_zero_and_shift_real_indices():
    ep = _uniform_episode(n=20, hz=10.0)
    out = pad_episode_start(ep, pad_duration_s=2.0, id_list=[0])
    n_pad = 20

    uw_map = out["map_to_uw_idx_0"].reshape(-1)
    d_map = out["map_to_d_idx_0"].reshape(-1)

    # padded region always points at index 0 of the (also padded) target stream
    assert np.all(uw_map[:n_pad] == 0)
    assert np.all(d_map[:n_pad] == 0)
    # real region indices are shifted by exactly n_pad (uw/d share rgb's grid)
    assert np.array_equal(uw_map[n_pad:], np.arange(20, dtype=np.int64) + n_pad)
    assert np.array_equal(d_map[n_pad:], np.arange(20, dtype=np.int64) + n_pad)

    # the shifted indices stay in-bounds for the padded target arrays
    assert uw_map.max() < out["ultrawide_0"].shape[0]
    assert d_map.max() < out["depth_0"].shape[0]


def test_heterogeneous_rates_pad_each_stream_by_its_own_dt():
    # rgb at 10 Hz, wrench at 100 Hz (pre-resample wild-converter shape)
    n_rgb, n_wrench = 10, 100
    rgb_t = (np.arange(n_rgb, dtype=np.float64) * 0.1).reshape(-1, 1)
    wrench_t = (np.arange(n_wrench, dtype=np.float64) * 0.01).reshape(-1, 1)

    ep = {
        "rgb_0": np.zeros((n_rgb, 1, 1, 3), dtype=np.uint8),
        "rgb_time_stamps_0": rgb_t,
        "wrench_left_0": np.arange(n_wrench, dtype=np.float64).reshape(-1, 1).repeat(6, axis=1),
        "wrench_time_stamps_left_0": wrench_t,
    }
    out = pad_episode_start(ep, pad_duration_s=1.0, id_list=[0])

    # 1.0s / 0.1s = 10 extra rgb frames; 1.0s / 0.01s = 100 extra wrench rows
    assert out["rgb_0"].shape[0] == n_rgb + 10
    assert out["wrench_left_0"].shape[0] == n_wrench + 100
    assert np.allclose(out["wrench_left_0"][:100], ep["wrench_left_0"][0])


def test_missing_optional_keys_are_skipped_without_error():
    # pre-VT episode: no ts_pose_virtual_target_0 / stiffness_0 / ts_pose_command_0
    ep = _uniform_episode(n=20, hz=10.0)
    assert "ts_pose_virtual_target_0" not in ep
    out = pad_episode_start(ep, pad_duration_s=2.0, id_list=[0])
    assert "ts_pose_virtual_target_0" not in out


def test_post_vt_keys_are_padded_when_present():
    ep = _uniform_episode(n=20, hz=10.0)
    ep["ts_pose_virtual_target_0"] = ep["ts_pose_fb_0"].copy()
    ep["ts_pose_command_0"] = ep["ts_pose_fb_0"].copy()
    ep["stiffness_0"] = np.arange(20, dtype=np.float64)
    ep["wrench_time_stamps_0"] = ep["wrench_time_stamps_left_0"].copy()

    out = pad_episode_start(ep, pad_duration_s=2.0, id_list=[0])
    n_pad = 20
    assert out["ts_pose_virtual_target_0"].shape[0] == 20 + n_pad
    assert out["ts_pose_command_0"].shape[0] == 20 + n_pad
    assert out["stiffness_0"].shape[0] == 20 + n_pad
    assert np.allclose(out["stiffness_0"][:n_pad], ep["stiffness_0"][0])

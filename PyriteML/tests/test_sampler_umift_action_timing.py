import numpy as np

from diffusion_policy.common.sampler_umift import (
    sparse_action_indices_from_rgb_timestamps,
)


def test_sparse_actions_follow_rgb_frame_grid_not_raw_action_stride():
    action_times = np.arange(0.0, 2.001, 0.005)
    rgb_times = np.arange(0.0, 2.001, 1.0 / 15.0)

    indices, target_times = sparse_action_indices_from_rgb_timestamps(
        action_times=action_times,
        rgb_times=rgb_times,
        rgb_id=3,
        action_horizon=4,
        rgb_frame_stride=2,
    )

    expected_target_times = rgb_times[[3, 5, 7, 9]]
    np.testing.assert_allclose(target_times, expected_target_times)
    np.testing.assert_allclose(
        action_times[indices], expected_target_times, atol=0.005
    )

    raw_query_index = np.searchsorted(action_times, rgb_times[3])
    old_raw_stride_indices = raw_query_index + np.array([0, 2, 4, 6])
    assert indices[1] - indices[0] > old_raw_stride_indices[1] - old_raw_stride_indices[0]


def test_sparse_actions_drop_targets_after_last_available_action():
    action_times = np.arange(0.0, 0.501, 0.005)
    rgb_times = np.arange(0.0, 1.001, 1.0 / 15.0)

    indices, target_times = sparse_action_indices_from_rgb_timestamps(
        action_times=action_times,
        rgb_times=rgb_times,
        rgb_id=5,
        action_horizon=6,
        rgb_frame_stride=2,
    )

    assert len(indices) == len(target_times)
    assert len(indices) < 6
    assert np.all(action_times[indices] >= target_times)

import numpy as np

from diffusion_policy.common.sampler_umift import (
    action_targets_have_timestamp_gap,
    filter_rgb_ids_without_action_gaps,
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


def test_action_targets_have_timestamp_gap_detects_large_step():
    # expected_dt = 0.133s; 3x => 0.4s threshold
    target_times = np.array([0.0, 0.133, 0.266, 1.266])  # last step = 1.0s
    assert action_targets_have_timestamp_gap(target_times, expected_dt=0.133, mult=3.0)
    assert not action_targets_have_timestamp_gap(
        np.array([0.0, 0.133, 0.266, 0.399]), expected_dt=0.133, mult=3.0
    )


def test_action_targets_have_timestamp_gap_short_horizon_is_safe():
    assert not action_targets_have_timestamp_gap(
        np.array([0.0]), expected_dt=0.133, mult=3.0
    )
    assert not action_targets_have_timestamp_gap(
        np.array([0.0, 0.133]), expected_dt=0.133, mult=3.0
    )


def test_filter_rgb_ids_drops_queries_whose_horizon_spans_rgb_hole():
    # 15 Hz RGB with a 1.0s hole after frame 10
    rgb_normal = np.arange(0.0, 10.1 / 15.0, 1.0 / 15.0)  # frames 0..10
    rgb_after = rgb_normal[-1] + 1.0 + np.arange(1, 21) / 15.0
    rgb_times = np.concatenate([rgb_normal, rgb_after])
    action_times = np.arange(rgb_times[0], rgb_times[-1] + 1e-3, 0.005)

    # horizon=4, stride=2 spans 6 RGB frames of nominal time
    candidate_ids = np.arange(0, len(rgb_times) - 8)
    kept, n_discarded = filter_rgb_ids_without_action_gaps(
        rgb_ids=candidate_ids,
        action_times=action_times,
        rgb_times=rgb_times,
        action_horizon=4,
        rgb_frame_stride=2,
        mult=3.0,
    )
    assert n_discarded > 0
    assert len(kept) + n_discarded == len(candidate_ids)

    # A query far before the hole should be kept
    assert 0 in kept
    # A query whose targets cross the hole should be discarded
    assert 8 not in kept


def test_filter_rgb_ids_disabled_when_mult_nonpositive():
    rgb_times = np.array([0.0, 0.067, 0.134, 1.134, 1.201, 1.268])
    action_times = np.arange(0.0, 2.0, 0.005)
    rgb_ids = np.array([0, 1])
    kept, n_discarded = filter_rgb_ids_without_action_gaps(
        rgb_ids=rgb_ids,
        action_times=action_times,
        rgb_times=rgb_times,
        action_horizon=3,
        rgb_frame_stride=1,
        mult=0.0,
    )
    np.testing.assert_array_equal(kept, rgb_ids)
    assert n_discarded == 0

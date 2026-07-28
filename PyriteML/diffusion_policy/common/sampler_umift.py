from typing import Callable, Optional

import numpy as np
from PyriteUtility.data_pipeline.indexing import (
    get_sample_ids,
    get_samples,
)


def sparse_action_indices_from_rgb_timestamps(
    action_times: np.ndarray,
    rgb_times: np.ndarray,
    rgb_id: int,
    action_horizon: int,
    rgb_frame_stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select future action labels on the RGB-frame time grid.

    ``rgb_frame_stride`` is a number of RGB frames, not an index stride into the
    higher-rate robot/action array. For target frames beyond the end of the RGB
    recording, use the episode's median RGB period; callers retain the existing
    action-padding behavior after the last available action label.
    """
    action_times = np.asarray(action_times, dtype=np.float64).reshape(-1)
    rgb_times = np.asarray(rgb_times, dtype=np.float64).reshape(-1)
    if len(action_times) == 0 or len(rgb_times) < 2:
        raise ValueError("Action and RGB timestamps must contain at least one and two entries.")
    if not 0 <= rgb_id < len(rgb_times):
        raise IndexError(f"rgb_id {rgb_id} is outside [0, {len(rgb_times) - 1}].")
    if action_horizon < 1 or rgb_frame_stride < 1:
        raise ValueError("action_horizon and rgb_frame_stride must be positive.")

    rgb_dt_s = float(np.median(np.diff(rgb_times)))
    if not np.isfinite(rgb_dt_s) or rgb_dt_s <= 0:
        raise ValueError("RGB timestamps must be strictly increasing.")

    frame_offsets = np.arange(action_horizon) * rgb_frame_stride
    target_rgb_ids = rgb_id + frame_offsets
    target_times = rgb_times[rgb_id] + frame_offsets * rgb_dt_s
    in_recording = target_rgb_ids < len(rgb_times)
    target_times[in_recording] = rgb_times[target_rgb_ids[in_recording]]

    action_indices = np.searchsorted(action_times, target_times)
    valid = action_indices < len(action_times)
    return action_indices[valid], target_times[valid]


def action_targets_have_timestamp_gap(
    target_times: np.ndarray,
    expected_dt: float,
    mult: float,
) -> bool:
    """Return True if any consecutive action-target dt exceeds mult * expected_dt."""
    if mult <= 0:
        return False
    target_times = np.asarray(target_times, dtype=np.float64).reshape(-1)
    if target_times.size < 2:
        return False
    if not np.isfinite(expected_dt) or expected_dt <= 0:
        raise ValueError("expected_dt must be a positive finite float.")
    return bool(np.any(np.diff(target_times) > mult * expected_dt))


def filter_rgb_ids_without_action_gaps(
    rgb_ids: np.ndarray,
    action_times: np.ndarray,
    rgb_times: np.ndarray,
    action_horizon: int,
    rgb_frame_stride: int,
    mult: float,
) -> tuple[np.ndarray, int]:
    """Keep rgb query ids whose sparse action target times have no large gap.

    Returns (kept_rgb_ids, n_discarded).
    """
    rgb_ids = np.asarray(rgb_ids, dtype=np.int64).reshape(-1)
    if mult <= 0 or rgb_ids.size == 0:
        return rgb_ids.copy(), 0

    rgb_times = np.asarray(rgb_times, dtype=np.float64).reshape(-1)
    action_times = np.asarray(action_times, dtype=np.float64).reshape(-1)
    expected_dt = float(np.median(np.diff(rgb_times))) * float(rgb_frame_stride)

    kept = []
    n_discarded = 0
    for rgb_id in rgb_ids:
        _, target_times = sparse_action_indices_from_rgb_timestamps(
            action_times=action_times,
            rgb_times=rgb_times,
            rgb_id=int(rgb_id),
            action_horizon=action_horizon,
            rgb_frame_stride=rgb_frame_stride,
        )
        if action_targets_have_timestamp_gap(target_times, expected_dt, mult):
            n_discarded += 1
        else:
            kept.append(int(rgb_id))
    return np.asarray(kept, dtype=np.int64), n_discarded


def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


class SequenceSamplerUmiFT:
    """Sample sequences of observations and actions from replay buffer.
    Query keys are determined by the shape_meta.
    Query frequency is based on rgb data, which is likely to be the most sparse data.
    Other data corresponding to the query ID is obtain based on timestamps.
    1. Given query id, find the corresponding low dim/rgb id.
    1. Construct sparse sample:
        Sample sparse obs horizon before idx,
        Sample sparse action horizon after idx.
    2. Construct dense sample:
        Find the indices of dense query points. For each dense query point:
            Sample dense obs horizon before idx,
            Sample dense action horizon after idx.
    """

    def __init__(
        self,
        shape_meta: dict,
        replay_buffer: dict,
        obs_to_obs_sample: Callable,
        action_to_action_sample: Callable,
        id_list: list,
        sparse_query_frequency_down_sample_steps: int = 1,
        episode_mask: Optional[np.ndarray] = None,
        action_padding: bool = False,
        weighted_sampling: int = 1,  # if > 1, duplicate the correction sample's ids weighted_sampling times
        correction_horizon: int = 1,  # the times of action horizon to be counted as "start of correction"
        action_gap_threshold_mult: float = 3.0,
    ):
        self.action_gap_threshold_mult = float(action_gap_threshold_mult)
        self.flag_has_dense = "dense" in shape_meta["sample"]["obs"]
        self.flag_has_gripper = (
            "robot0_gripper" in shape_meta["sample"]["obs"]["sparse"]
        )
        self.flag_has_rgb = "rgb_0" in shape_meta["sample"]["obs"]["sparse"]
        self.flag_has_ultrawide = "ultrawide_0" in shape_meta["sample"]["obs"]["sparse"]
        self.flag_has_depth = "depth_0" in shape_meta["sample"]["obs"]["sparse"]

        # assert that either rgb or ultrawide is present
        assert self.flag_has_rgb or self.flag_has_ultrawide
        print(
            f"[SequenceSamplerUmiFT] RGB: {self.flag_has_rgb}, Ultrawide: {self.flag_has_ultrawide}, Depth: {self.flag_has_depth}]"
        )

        episode_keys = replay_buffer["data"].keys()
        # Step one: Find the usable length of each episode
        episodes_length = replay_buffer["meta"]["episode_rgb0_len"][:]
        episodes_length_for_query = episodes_length.copy()
        episodes_start = episodes_length.copy()

        if self.flag_has_dense:
            buffer_time_ms = (
                shape_meta["sample"]["training_duration_per_sparse_query"]
                + shape_meta["sample"]["dense_action_duration_buffer_ms"]
            )
        else:
            buffer_time_ms = shape_meta["sample"]["training_duration_per_sparse_query"]
        episode_count = -1
        for episode in episode_keys:
            episode_count += 1

            ##
            ## Find feasible start and end times for this episode
            ##
            end_time = np.inf
            start_time = -np.inf
            for id in id_list:
                ## I. start time: find a time after all the obs horizons

                ## I.1.0 RGB
                # rgb_start_time = -1e9
                if self.flag_has_rgb:
                    rgb_key = f"rgb_{id}"
                elif self.flag_has_ultrawide:
                    rgb_key = f"ultrawide_{id}"
                else:
                    raise ValueError(
                        f"[SequenceSamplerUmiFT] No RGB or Ultrawide data found for id {id}"
                    )

                rgb_start_id = (
                    (
                        shape_meta["sample"]["obs"]["sparse"][rgb_key]["horizon"]
                        * shape_meta["sample"]["obs"]["sparse"][rgb_key][
                            "down_sample_steps"
                        ]
                    )
                    + 1
                )  

                rgb_start_time = np.squeeze(
                    replay_buffer["data"][episode]["obs"][f"rgb_time_stamps_{id}"][
                        rgb_start_id
                    ]
                )

                # HC NOTES: rgb time is used for indexing for vision and pose. Just use rgb time all the way. Use the mapping index to acquire actual uw or d data.

                ## I.1.1 Ultrawide
                # ultrawide_start_time = -1e9
                # if self.flag_has_ultrawide:
                #     ultrawide_start_id = (
                #         shape_meta["sample"]["obs"]["sparse"][f"ultrawide_{id}"][
                #             "horizon"
                #         ]
                #         * shape_meta["sample"]["obs"]["sparse"][f"ultrawide_{id}"][
                #             "down_sample_steps"
                #         ]
                #     ) + 1

                #     mapped_uw_idx = replay_buffer["data"][episode]["obs"][f"map_to_uw_idx_{id}"][ultrawide_start_id]
                #     ultrawide_start_time = np.squeeze(
                #         replay_buffer["data"][episode]["obs"][f"ultrawide_time_stamps_{id}"][
                #             mapped_uw_idx
                #         ]
                #     )

                ## I.1.2 Depth
                # depth_start_time = -1e9
                # if self.flag_has_depth:
                #     depth_start_id = (
                #         shape_meta["sample"]["obs"]["sparse"][f"depth_{id}"][
                #             "horizon"
                #         ]
                #         * shape_meta["sample"]["obs"]["sparse"][f"depth_{id}"][
                #             "down_sample_steps"
                #         ]
                #     ) + 1

                #     mapped_d_idx = replay_buffer["data"][episode]["obs"][f"map_to_d_idx_{id}"][depth_start_id]
                #     depth_start_time = np.squeeze(
                #         replay_buffer["data"][episode]["obs"][f"depth_time_stamps_{id}"][
                #             mapped_d_idx
                #         ]
                #     )

                ## I.2 robot
                robot_start_id = (
                    shape_meta["sample"]["obs"]["sparse"][f"robot{id}_eef_pos"][
                        "horizon"
                    ]
                    * shape_meta["sample"]["obs"]["sparse"][f"robot{id}_eef_pos"][
                        "down_sample_steps"
                    ]
                ) + 1

                if self.flag_has_dense:
                    dense_robot_start_id = (
                        shape_meta["sample"]["obs"]["dense"][f"robot{id}_eef_pos"][
                            "horizon"
                        ]
                        * shape_meta["sample"]["obs"]["dense"][f"robot{id}_eef_pos"][
                            "down_sample_steps"
                        ]
                    ) + 1
                    robot_start_id = max(robot_start_id, dense_robot_start_id)

                robot_start_time = np.squeeze(
                    replay_buffer["data"][episode]["obs"][f"robot_time_stamps_{id}"][
                        robot_start_id
                    ]
                )

                ## I.3 wrench (optional)
                # UMIFT: the time is aligned based on wrench_left. The time between left/right is a couple ms at most.
                # wrench_time_stamps_0 is an average of wrench_time_stamps_left and ~_right.
                if (
                    f"robot{id}_eef_wrench_left"
                    in shape_meta["sample"]["obs"]["sparse"]
                ):
                    wrench_start_id = (
                        shape_meta["sample"]["obs"]["sparse"][
                            f"robot{id}_eef_wrench_left"
                        ]["horizon"]
                        * shape_meta["sample"]["obs"]["sparse"][
                            f"robot{id}_eef_wrench_left"
                        ]["down_sample_steps"]
                    ) + 1

                    wrench_start_time = np.squeeze(
                        replay_buffer["data"][episode]["obs"][
                            f"wrench_time_stamps_{id}"
                        ][wrench_start_id]
                    )
                else:
                    wrench_start_time = -1e9

                ## I.4 gripper
                gripper_start_time = -1e9
                if self.flag_has_gripper:
                    gripper_start_id = (
                        shape_meta["sample"]["obs"]["sparse"][f"robot{id}_gripper"][
                            "horizon"
                        ]
                        * shape_meta["sample"]["obs"]["sparse"][f"robot{id}_gripper"][
                            "down_sample_steps"
                        ]
                    ) + 1
                    gripper_start_time = np.squeeze(
                        replay_buffer["data"][episode]["obs"][
                            f"gripper_time_stamps_{id}"
                        ][gripper_start_id]
                    )

                ## I.4 find max
                start_time = max(
                    start_time,
                    rgb_start_time,
                    # ultrawide_start_time,
                    # depth_start_time,
                    robot_start_time,
                    wrench_start_time,
                    gripper_start_time,
                )

                ## II. end time: the last time stamp before buffer_time_ms
                ## II.1 RGB

                # rgb_end_time = None
                # if self.flag_has_rgb:
                rgb_end_time = np.squeeze(
                    replay_buffer["data"][episode]["obs"][f"rgb_time_stamps_{id}"][-1]
                )
                # elif self.flag_has_ultrawide:
                #     rgb_end_time = np.squeeze(
                #         replay_buffer["data"][episode]["obs"][f"ultrawide_time_stamps_{id}"][
                #             -1
                #         ]
                #     )
                # assert rgb_end_time is not None, f"[SequenceSamplerUmiFT] No RGB or Ultrawide timestamp data found for episode {episode}"

                ## II.2 robot
                robot_end_time = np.squeeze(
                    replay_buffer["data"][episode]["obs"][f"robot_time_stamps_{id}"][-1]
                )

                ## II.2.1 wrench (optional)
                if (
                    f"robot{id}_eef_wrench_left"
                    in shape_meta["sample"]["obs"]["sparse"]
                ):
                    wrench_end_time = np.squeeze(
                        replay_buffer["data"][episode]["obs"][
                            f"wrench_time_stamps_{id}"
                        ][-1]
                    )
                else:
                    wrench_end_time = np.inf

                ## II.3 find min
                if not action_padding:
                    # if no action padding, truncate the indices to query so the last query point
                    #  still has access to the whole horizon of actions
                    #  This is enforced by limiting sparse action queries alone.
                    #  It is assumed that the dense action is not affected.
                    end_time = (
                        min(end_time, rgb_end_time, robot_end_time, wrench_end_time)
                        - buffer_time_ms
                    )
                else:
                    end_time = min(
                        end_time, rgb_end_time, robot_end_time, wrench_end_time
                    )
                assert end_time > 0
                if end_time < start_time:
                    print(
                        f"[SequenceSamplerUmiFT] Warning: end_time {end_time} is before start_time {start_time} for episode {episode}. Skipping this episode."
                    )
                    print(
                        f"rgb_end_time: {rgb_end_time}, robot_end_time: {robot_end_time}, wrench_end_time: {wrench_end_time}, buffer_time_ms: {buffer_time_ms}"
                    )
                    print(
                        f"rgb_start_time: {rgb_start_time}, robot_start_time: {robot_start_time}, start_time: {start_time}"
                    )
                    exit(0)

            last_rgb_idx = 1e9
            first_rgb_idx = -1

            # HC TODO: when going bimanual, this part needs to be modified. 
            for id in id_list:
                rgb_times = np.squeeze(
                    replay_buffer["data"][episode]["obs"][f"rgb_time_stamps_{id}"]
                )
                # find the last rgb_times index that is before low_dim_end_time
                rgb_id = np.searchsorted(rgb_times, end_time, side="right") - 1
                last_rgb_idx = min(last_rgb_idx, rgb_id)

                # find the first rgb_times index that is after low_dim_start_time
                rgb_id = np.searchsorted(rgb_times, start_time, side="left")
                first_rgb_idx = max(first_rgb_idx, rgb_id)

            episodes_length_for_query[episode_count] = last_rgb_idx - first_rgb_idx
            episodes_start[episode_count] = first_rgb_idx
        assert np.min(episodes_length_for_query) > 0

        # Step two: Computes indices from episodes_length_for_query. indices[i] = (epi_id, epi_len, id)
        #   epi_id: which episode the index i belongs to.
        #   id: the index within the episode.
        epi_id = []
        ids = []
        episode_count = -1
        gap_discard_total = 0
        gap_candidate_total = 0
        gap_discard_by_episode = {}
        action_horizon = shape_meta["sample"]["action"]["sparse"]["horizon"]
        action_rgb_frame_stride = shape_meta["sample"]["action"]["sparse"][
            "down_sample_steps"
        ]
        for key in episode_keys:
            episode_count += 1
            episode_index = int(key.split("_")[-1])
            if episode_mask is not None and not episode_mask[episode_count]:
                # skip episode
                continue

            has_mask = False
            if "mask_0" in replay_buffer["data"][key]["obs"]:
                masks = replay_buffer["data"][key]["obs"]["mask_0"]
                if np.any(masks):
                    has_mask = True

            action_times = np.squeeze(
                replay_buffer["data"][key]["action_time_stamps"]
            )
            rgb_times = np.squeeze(
                replay_buffer["data"][key]["obs"]["rgb_time_stamps_0"]
            )

            if not has_mask:
                # normal processing
                array_length = episodes_length_for_query[episode_count]
                ids_to_be_added = episodes_start[episode_count] + np.arange(
                    array_length
                )
                # Down sample the query indices to make the dataset smaller
                candidate_rgb_ids = ids_to_be_added[
                    ::sparse_query_frequency_down_sample_steps
                ]
                kept_rgb_ids, n_discarded = filter_rgb_ids_without_action_gaps(
                    rgb_ids=candidate_rgb_ids,
                    action_times=action_times,
                    rgb_times=rgb_times,
                    action_horizon=action_horizon,
                    rgb_frame_stride=action_rgb_frame_stride,
                    mult=self.action_gap_threshold_mult,
                )
                gap_discard_total += n_discarded
                gap_candidate_total += len(candidate_rgb_ids)
                if n_discarded > 0:
                    gap_discard_by_episode[episode_index] = (
                        n_discarded,
                        len(candidate_rgb_ids),
                    )

                epi_id.extend([episode_index] * len(kept_rgb_ids))
                ids.extend(kept_rgb_ids.tolist())
                if weighted_sampling > 1:
                    actions = replay_buffer["data"][key]["action"]
                    # detect correction occurance: if the first action is close to zero, and any of the other actions are not close to zero
                    reference_action = np.array(
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
                    )
                    action_down_sample_steps = action_rgb_frame_stride
                    sample_ids = np.searchsorted(
                        replay_buffer["data"][key]["action_time_stamps"],
                        replay_buffer["data"][key]["obs"]["rgb_time_stamps_0"][
                            kept_rgb_ids
                        ],
                    )

                    flag_not_in_correction = (
                        np.mean(
                            np.abs(actions[sample_ids, :9] - reference_action[None, :]),
                            axis=-1,
                        )
                        < 1e-3
                    )

                    action_id_range = np.arange(
                        1,
                        action_horizon * action_down_sample_steps,
                        action_down_sample_steps,
                    )
                    action_id_actual = np.clip(
                        sample_ids[:, None] + action_id_range, 0, len(actions) - 1
                    )
                    flag_before_correction = (
                        np.mean(
                            np.sum(
                                np.abs(
                                    actions[action_id_actual, :9]
                                    - reference_action[None, None, :]
                                ),
                                axis=1,
                            ),
                            axis=-1,
                        )
                        > 1e-2
                    )

                    is_correction_sample = (
                        flag_not_in_correction & flag_before_correction
                    )
                    correction_ids = kept_rgb_ids[is_correction_sample]

                    # duplicate the correction sample's ids weighted_sampling times
                    for _ in range(weighted_sampling):
                        ids.extend(correction_ids.tolist())
                        epi_id.extend([episode_index] * len(correction_ids))

                    # sample more (correction horizon) before and after correction
                    if correction_horizon > 0 and len(correction_ids) > 0:
                        addition_horizon = np.arange(
                            -correction_horizon,
                            correction_horizon + 1,
                            sparse_query_frequency_down_sample_steps,
                        )
                        additional_sample_ids = np.clip(
                            correction_ids[:, None] + addition_horizon,
                            ids_to_be_added[0],
                            ids_to_be_added[-1],
                        )
                        additional_sample_ids = np.unique(
                            additional_sample_ids.reshape(-1)
                        )

                        additional_sample_ids = additional_sample_ids[
                            ~np.isin(additional_sample_ids, correction_ids)
                        ]  # filter out the additional sample ids that are already in the list
                        additional_kept, add_discarded = (
                            filter_rgb_ids_without_action_gaps(
                                rgb_ids=additional_sample_ids,
                                action_times=action_times,
                                rgb_times=rgb_times,
                                action_horizon=action_horizon,
                                rgb_frame_stride=action_rgb_frame_stride,
                                mult=self.action_gap_threshold_mult,
                            )
                        )
                        gap_discard_total += add_discarded
                        gap_candidate_total += len(additional_sample_ids)
                        if add_discarded > 0:
                            prev = gap_discard_by_episode.get(episode_index, (0, 0))
                            gap_discard_by_episode[episode_index] = (
                                prev[0] + add_discarded,
                                prev[1] + len(additional_sample_ids),
                            )
                        for _ in range(weighted_sampling):
                            ids.extend(additional_kept.tolist())
                            epi_id.extend([episode_index] * len(additional_kept))
            else:
                # now we have an episode with masks.
                # we need to find the end id of each continuous segment of ones in the mask array.
                # The processing of mask assumes that mask_0 and mask_1 are the same.
                mask_start_ids = np.array(np.where(np.diff(masks) == -1)) + 1
                robot_time_at_mask_start = replay_buffer["data"][key]["obs"][
                    "robot_time_stamps_0"
                ][mask_start_ids]

                # find the corresponding rgb id
                rgb_id_at_mask_start = np.searchsorted(
                    np.squeeze(replay_buffer["data"][key]["obs"]["rgb_time_stamps_0"]),
                    robot_time_at_mask_start,
                )

                kept_rgb_ids, n_discarded = filter_rgb_ids_without_action_gaps(
                    rgb_ids=np.asarray(rgb_id_at_mask_start).reshape(-1),
                    action_times=action_times,
                    rgb_times=rgb_times,
                    action_horizon=action_horizon,
                    rgb_frame_stride=action_rgb_frame_stride,
                    mult=self.action_gap_threshold_mult,
                )
                gap_discard_total += n_discarded
                gap_candidate_total += len(np.asarray(rgb_id_at_mask_start).reshape(-1))
                if n_discarded > 0:
                    gap_discard_by_episode[episode_index] = (
                        n_discarded,
                        len(np.asarray(rgb_id_at_mask_start).reshape(-1)),
                    )

                epi_id.extend([episode_index] * len(kept_rgb_ids))
                ids.extend(kept_rgb_ids.tolist())

        indices = list(zip(epi_id, ids))

        if self.action_gap_threshold_mult > 0:
            pct = (
                100.0 * gap_discard_total / gap_candidate_total
                if gap_candidate_total > 0
                else 0.0
            )
            print(
                f"[SequenceSamplerUmiFT] action-gap filter "
                f"(mult={self.action_gap_threshold_mult}): "
                f"discarded {gap_discard_total}/{gap_candidate_total} queries "
                f"({pct:.1f}%)"
            )
            for ep_idx, (n_disc, n_cand) in sorted(gap_discard_by_episode.items()):
                print(f"  episode_{ep_idx}: {n_disc}/{n_cand}")

        self.shape_meta = shape_meta
        self.replay_buffer = replay_buffer
        self.action_padding = action_padding
        self.indices = indices
        self.obs_to_obs_sample = obs_to_obs_sample
        self.action_to_action_sample = action_to_action_sample
        self.id_list = id_list

        self.ignore_rgb_is_applied = (
            False  # speed up the interation when getting normalizer
        )

    def __len__(self):
        return len(self.indices)

    # @profile
    def sample_sequence(self, idx):
        """Sample a sequence of observations and actions at idx."""
        epi_id, rgb_id = self.indices[idx]
        episode = f"episode_{epi_id}"
        data_episode = self.replay_buffer["data"][episode]

        query_time = np.squeeze(data_episode["obs"]["rgb_time_stamps_0"][rgb_id])
        # query time is given for the rgb0 obs data.
        # To get others (rgb, low dim, action), we need to find their id
        sparse_obs_unprocessed = dict()
        sparse_action_unprocessed = []
        for key, attr in self.shape_meta["sample"]["obs"]["sparse"].items():
            input_arr = data_episode["obs"][key]
            this_horizon = attr["horizon"]
            this_downsample_steps = attr["down_sample_steps"]
            type = self.shape_meta["obs"][key]["type"]

            if "rgb" in key:
                id = int(key.split("_")[-1])
            elif "ultrawide" in key:
                id = int(key.split("_")[-1])
            elif "depth" in key:
                id = int(key.split("_")[-1])
            elif "policy" in key:
                id = int(key[12])
            else:
                id = int(key[5])  # robot0_xxxx

            # Assigning rgb time stamps for ultrawide and depth is not a bug. This allows consistent sampling of rgb, uw, and d.
            # Make sure the index found later is mapped to the correct index for ultrawide and depth.
            if "ultrawide" in key:
                time_stamp_key = f"rgb_time_stamps_{id}"
            elif "rgb" in key:
                time_stamp_key = f"rgb_time_stamps_{id}"
            elif "depth" in key:
                time_stamp_key = f"rgb_time_stamps_{id}"
            elif "wrench" in key:
                time_stamp_key = f"wrench_time_stamps_{id}"
            elif "gripper" in key:
                time_stamp_key = f"gripper_time_stamps_{id}"
            elif "policy" in key:
                time_stamp_key = f"policy_time_stamps_{id}"
            else:
                time_stamp_key = f"robot_time_stamps_{id}"

            # find the query id for the query time
            query_id = np.searchsorted(
                np.squeeze(data_episode["obs"][time_stamp_key]), query_time
            )
            found_time = data_episode["obs"][time_stamp_key][query_id]

            if abs(found_time - query_time) > 50.0:
                print("processing sparse key: ", key)
                print("sparse query_time: ", query_time)
                print(
                    "total time: ",
                    data_episode["obs"][time_stamp_key][-1],
                )
                print("query_id: ", query_id)
                print(
                    "total id: ",
                    len(data_episode["obs"][time_stamp_key]),
                )
                raise ValueError(
                    f"[sampler] {episode} Warning: closest data point at {found_time} is far from the query_time {query_time}"
                )

            # how many obs frames before the query time are valid
            num_valid = min(this_horizon, query_id // this_downsample_steps + 1)
            slice_start = query_id - (num_valid - 1) * this_downsample_steps
            assert slice_start >= 0

            # sample every this_downsample_steps frames from slice_start to query_id+1,
            # then fill the rest with the first frame if needed
            if type == "rgb":
                if self.ignore_rgb_is_applied:
                    continue

                rgb_indices = list(
                    range(slice_start, query_id + 1, this_downsample_steps)
                )

                if "rgb" in key:
                    output = input_arr[rgb_indices]
                elif "ultrawide" in key:
                    uw_indices = data_episode["obs"][f"map_to_uw_idx_{id}"][
                        rgb_indices
                    ].squeeze()
                    output = input_arr[uw_indices]
                elif "depth" in key:
                    d_indices = data_episode["obs"][f"map_to_d_idx_{id}"][
                        rgb_indices
                    ].squeeze()
                    output = input_arr[d_indices]

            elif type == "low_dim":
                output = input_arr[
                    slice_start : query_id + 1 : this_downsample_steps
                ].astype(np.float32)
            assert output.shape[0] == num_valid
            # solve padding
            if output.shape[0] < this_horizon:
                padding = np.repeat(output[:1], this_horizon - output.shape[0], axis=0)
                output = np.concatenate([padding, output], axis=0)
            sparse_obs_unprocessed[key] = output

        # sparse action
        action_id = np.searchsorted(
            np.squeeze(data_episode["action_time_stamps"]), query_time
        )
        found_time = data_episode["action_time_stamps"][action_id]
        if abs(found_time - query_time) > 5.0:
            raise ValueError(
                f"[sampler] {episode} Warning: action found_time {found_time} is not equal to query_time {query_time}"
            )
        if "sparse" in self.shape_meta["sample"]["action"]:
            input_arr = data_episode["action"]
            action_horizon = self.shape_meta["sample"]["action"]["sparse"]["horizon"]
            action_rgb_frame_stride = self.shape_meta["sample"]["action"]["sparse"][
                "down_sample_steps"
            ]
            action_times = np.squeeze(data_episode["action_time_stamps"])
            rgb_times = np.squeeze(data_episode["obs"]["rgb_time_stamps_0"])
            action_indices, target_times = sparse_action_indices_from_rgb_timestamps(
                action_times=action_times,
                rgb_times=rgb_times,
                rgb_id=rgb_id,
                action_horizon=action_horizon,
                rgb_frame_stride=action_rgb_frame_stride,
            )
            sparse_action_unprocessed = input_arr[action_indices].astype(np.float32)
            # if sparse_action_unprocessed.shape[0] != 16:
            #     import pdb; pdb.set_trace()

            # solve padding
            if not self.action_padding:
                if sparse_action_unprocessed.shape[0] != action_horizon:
                    print(sparse_action_unprocessed.shape[0], action_horizon)
                    print(
                        action_id,
                        action_indices[-1] if len(action_indices) else None,
                        action_horizon,
                        action_rgb_frame_stride,
                    )
                    # sparse_action_unprocessed = sparse_action_unprocessed[
                    #     :action_horizon
                    # ]
                assert sparse_action_unprocessed.shape[0] == action_horizon
            elif sparse_action_unprocessed.shape[0] < action_horizon:
                padding = np.repeat(
                    sparse_action_unprocessed[-1:],
                    action_horizon - sparse_action_unprocessed.shape[0],
                    axis=0,
                )
                sparse_action_unprocessed = np.concatenate(
                    [sparse_action_unprocessed, padding], axis=0
                )

        # dense obs and action
        # Given one query time, we sample H groups of dense obs and action,
        # spanning across sparse_total_steps action time steps.
        dense_obs_unprocessed = {}
        dense_action_unprocessed = []

        if self.flag_has_dense:
            # 1. Compute dense query time points
            total_duration_ms = self.shape_meta["sample"][
                "training_duration_per_sparse_query"
            ]
            delta_ms = self.shape_meta["sample"][
                "training_delta_time_between_dense_queries_ms"
            ]
            dense_query_times = np.arange(
                query_time,
                query_time + total_duration_ms,
                delta_ms,
            )

            # dense obs
            for key, attr in self.shape_meta["sample"]["obs"]["dense"].items():
                this_horizon = attr["horizon"]
                this_downsample_steps = attr["down_sample_steps"]
                type = self.shape_meta["obs"][key]["type"]
                id = int(key[5])  # robot0_xxxx
                if "wrench" in key:
                    time_stamp_key = f"wrench_time_stamps_{id}"
                elif "gripper" in key:
                    time_stamp_key = f"gripper_time_stamps_{id}"
                else:
                    time_stamp_key = f"robot_time_stamps_{id}"

                # 2. find the query ids for the query times
                dense_query_ids = np.searchsorted(
                    np.squeeze(data_episode["obs"][time_stamp_key]), dense_query_times
                )
                found_times = data_episode["obs"][time_stamp_key][dense_query_ids]
                if any(abs(found_times - dense_query_times) > 10.0):
                    print("processing dense key: ", key)
                    print("dense_query_times: ", dense_query_times)
                    print("found_times: ", found_times)
                    print(
                        "total time: ",
                        data_episode["obs"][time_stamp_key][-1],
                    )
                    print("dense_query_ids: ", dense_query_ids)
                    print(
                        "total id: ",
                        len(data_episode["obs"][time_stamp_key]),
                    )
                    raise ValueError(
                        f"[sampler] {episode} Warning: some dense sample points are far from the found_times"
                    )

                # 3. get samples from id
                dense_obs_unprocessed[key] = get_samples(
                    data_episode["obs"][key],
                    dense_query_ids,
                    this_horizon,
                    this_downsample_steps,
                    backwards=True,
                    closed=True,
                ).astype(np.float32)

            # dense action (H, T, D)
            # assuming no padding is needed
            # 2. find the query id for the query time
            dense_action_query_ids = np.searchsorted(
                np.squeeze(data_episode["action_time_stamps"]), dense_query_times
            )
            # 3. get samples from id
            dense_action_horizon = self.shape_meta["sample"]["action"]["dense"][
                "horizon"
            ]
            dense_action_down_sample_steps = self.shape_meta["sample"]["action"][
                "dense"
            ]["down_sample_steps"]
            dense_action_unprocessed_ids = get_sample_ids(
                dense_action_query_ids,
                dense_action_horizon,
                dense_action_down_sample_steps,
                backwards=False,
                closed=True,
            )
            dense_action_unprocessed = data_episode["action"][
                dense_action_unprocessed_ids
            ].astype(np.float32)  # [H, T, D]

            # experimental: get corresponding action wrench
            dense_action_unprocessed_times = data_episode["action_time_stamps"][
                dense_action_unprocessed_ids
            ]  # [H, T]
            dense_action_wrench_left_list = []
            dense_action_wrench_right_list = []
            for id in self.id_list:
                dense_action_wrench_id = np.searchsorted(
                    np.squeeze(data_episode["obs"][f"wrench_time_stamps_{id}"]),
                    dense_action_unprocessed_times,
                )
                dense_action_wrench_left = data_episode["obs"][
                    f"robot{id}_eef_wrench_left"
                ][dense_action_wrench_id].astype(np.float32)
                dense_action_wrench_right = data_episode["obs"][
                    f"robot{id}_eef_wrench_right"
                ][dense_action_wrench_id].astype(np.float32)
                dense_action_wrench_left_list.append(dense_action_wrench_left)
                dense_action_wrench_right_list.append(dense_action_wrench_right)
            dense_action_wrench_left = np.squeeze(
                np.stack(dense_action_wrench_left_list, axis=-1)
            )
            dense_action_wrench_right = np.squeeze(
                np.stack(dense_action_wrench_right_list, axis=-1)
            )

        #   convert to relative pose
        obs_sample, base_pose = self.obs_to_obs_sample(
            obs_sparse=sparse_obs_unprocessed,
            obs_dense=dense_obs_unprocessed,
            shape_meta=self.shape_meta,
            reshape_mode="reshape",
            id_list=self.id_list,
            ignore_rgb=self.ignore_rgb_is_applied,
        )
        action_sample = self.action_to_action_sample(
            action_sparse=sparse_action_unprocessed,
            action_dense=dense_action_unprocessed,
            id_list=self.id_list,
            base_pose=base_pose,
        )

        if self.flag_has_dense:
            action_sample["dense_wrench_left"] = dense_action_wrench_left
            action_sample["dense_wrench_right"] = dense_action_wrench_right

        # # debug plotting
        # sparse_obs_down_sample_steps = self.shape_meta['sample']['obs']['sparse']['robot0_eef_pos']['down_sample_steps']
        # sparse_obs_horizon = self.shape_meta['sample']['obs']['sparse']['robot0_eef_pos']['horizon']
        # slice_start = low_dim_id - (sparse_obs_horizon - 1) * sparse_obs_down_sample_steps
        # sparse_obs_time_steps = np.arange(slice_start, low_dim_id + 1, sparse_obs_down_sample_steps).astype(np.float32)
        # sparse_action_time_steps = np.arange(action_id, slice_end, action_down_sample_steps)
        # total_time_steps = np.arange(slice_start, slice_end, 1)
        # total_obs = dict()
        # for key, attr in self.shape_meta['sample']['obs']['dense'].items():
        #     total_obs[key] = data_episode['obs'][key][total_time_steps]
        # total_action = data_episode['action'][total_time_steps]
        # plot_sample(total_obs, total_action, total_time_steps,
        #             # sparse_obs_unprocessed, dense_obs_unprocessed, sparse_action_unprocessed, dense_action_unprocessed,
        #             obs_sample['sparse'], obs_sample['dense'], action_sample['sparse'], action_sample['dense'],
        #             sparse_obs_time_steps,
        #             sparse_action_time_steps,
        #             dense_obs_queries, dense_obs_horizon, dense_obs_down_sample_steps,
        #             dense_action_queries, dense_action_horizon, dense_action_down_sample_steps
        # )
        # print('debug: printing')

        return obs_sample, action_sample

    def ignore_rgb(self, apply=True):
        self.ignore_rgb_is_applied = apply

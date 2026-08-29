"""Prepend a frozen "hold at start state" prefix to a UMIFT episode dict.

Why: SequenceSamplerUmiFT only builds a trainable query once every modality
has a full backward observation history (see rgb_start_id in
sampler_umift.py), so it never samples inside the first
`horizon * down_sample_steps` steps of an episode. At inference time,
however, every real rollout *starts* exactly there: t=0, with nothing but
the reset state to fill the observation history buffer. Padding the front
of each episode with repeated first-frame data (at data-prep time, before
any zarr write) makes that situation part of the trainable index range
instead of always being skipped.
"""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

# data key -> the "_time_stamps" family that indexes it (both templated on
# the robot id). Keys absent from a given episode dict are skipped, so this
# table is safely shared across pre-VT and post-VT schemas.
_DATA_KEY_TO_TIMESTAMP_KEY = {
    "rgb_{id}": "rgb_time_stamps_{id}",
    "ultrawide_{id}": "ultrawide_time_stamps_{id}",
    "depth_{id}": "depth_time_stamps_{id}",
    "ts_pose_fb_{id}": "robot_time_stamps_{id}",
    "ts_pose_virtual_target_{id}": "robot_time_stamps_{id}",
    "ts_pose_command_{id}": "robot_time_stamps_{id}",
    "stiffness_{id}": "robot_time_stamps_{id}",
    "gripper_{id}": "gripper_time_stamps_{id}",
    "wrench_left_{id}": "wrench_time_stamps_left_{id}",
    "wrench_left_coinft_{id}": "wrench_time_stamps_left_{id}",
    "wrench_right_{id}": "wrench_time_stamps_right_{id}",
    "wrench_right_coinft_{id}": "wrench_time_stamps_right_{id}",
    "wrench_concat_{id}": "wrench_time_stamps_left_{id}",
    "wrench_concat_coinft_{id}": "wrench_time_stamps_left_{id}",
}

# index-map key -> (timestamp family it is queried by, timestamp family it
# points into).
_INDEX_MAP_KEYS = {
    "map_to_uw_idx_{id}": ("rgb_time_stamps_{id}", "ultrawide_time_stamps_{id}"),
    "map_to_d_idx_{id}": ("rgb_time_stamps_{id}", "depth_time_stamps_{id}"),
}

# every timestamp family we know how to extend. "wrench_time_stamps_{id}"
# (no left/right) only exists after VT postprocessing (averaged left/right).
_ALL_TIMESTAMP_KEYS = [
    "rgb_time_stamps_{id}",
    "ultrawide_time_stamps_{id}",
    "depth_time_stamps_{id}",
    "robot_time_stamps_{id}",
    "gripper_time_stamps_{id}",
    "wrench_time_stamps_left_{id}",
    "wrench_time_stamps_right_{id}",
    "wrench_time_stamps_{id}",
]


def _median_dt(ts: np.ndarray) -> float:
    ts = np.asarray(ts, dtype=np.float64).reshape(-1)
    if ts.size < 2:
        return 0.1  # arbitrary fallback for a degenerate single-sample stream
    dt = float(np.median(np.diff(ts)))
    return dt if dt > 0 else 0.1


def pad_episode_start(
    ep_data: Dict[str, np.ndarray],
    pad_duration_s: float,
    id_list: Iterable[int] = (0,),
) -> Dict[str, np.ndarray]:
    """Return a copy of `ep_data` with `pad_duration_s` of repeated start state prepended.

    Every recognized per-timestep data array is extended by repeating its
    own first row; every recognized "_time_stamps_" array is extended
    backwards at its own native sampling rate. All timestamp arrays are then
    re-zeroed so the episode still starts at t=0, matching the existing
    converter convention (see `normalize_timestamps`).

    Keys not present in `ep_data`, or not in the known UMIFT key tables, are
    left untouched -- this keeps the function safe to call on both pre-VT
    (converter-time) and post-VT episode dicts.
    """
    if pad_duration_s <= 0:
        return dict(ep_data)

    out = dict(ep_data)

    for id_ in id_list:
        n_pad_by_ts_key: Dict[str, int] = {}

        for ts_template in _ALL_TIMESTAMP_KEYS:
            ts_key = ts_template.format(id=id_)
            if ts_key not in ep_data:
                continue
            raw = np.asarray(ep_data[ts_key], dtype=np.float64)
            ts = raw.reshape(-1)
            dt = _median_dt(ts)
            n_pad = max(1, int(round(pad_duration_s / dt)))
            n_pad_by_ts_key[ts_key] = n_pad
            prefix = ts[0] - dt * np.arange(n_pad, 0, -1)
            new_ts = np.concatenate([prefix, ts])
            out[ts_key] = new_ts.reshape(-1, 1) if raw.ndim == 2 else new_ts

        for data_template, ts_template in _DATA_KEY_TO_TIMESTAMP_KEY.items():
            key = data_template.format(id=id_)
            ts_key = ts_template.format(id=id_)
            if key not in ep_data or ts_key not in n_pad_by_ts_key:
                continue
            arr = np.asarray(ep_data[key])
            n_pad = n_pad_by_ts_key[ts_key]
            prefix = np.repeat(arr[:1], n_pad, axis=0)
            out[key] = np.concatenate([prefix, arr], axis=0)

        for map_template, (own_template, target_template) in _INDEX_MAP_KEYS.items():
            key = map_template.format(id=id_)
            own_ts_key = own_template.format(id=id_)
            target_ts_key = target_template.format(id=id_)
            if key not in ep_data or own_ts_key not in n_pad_by_ts_key:
                continue
            n_pad_own = n_pad_by_ts_key[own_ts_key]
            n_pad_target = n_pad_by_ts_key.get(target_ts_key, 0)
            arr = np.asarray(ep_data[key])
            prefix = np.zeros((n_pad_own,) + arr.shape[1:], dtype=arr.dtype)
            out[key] = np.concatenate([prefix, arr + n_pad_target], axis=0)

        # map_to_scan_idx_{id}: doesn't fit the _INDEX_MAP_KEYS shape above -- it has no
        # real "target timestamp family" to shift against. It indexes into
        # scan_heatmap_{id} by SCAN COUNT (N), not by time, and padding never adds new
        # scans, so index VALUES must NOT be shifted (no "+ n_pad_target" equivalent).
        # Handled as its own small loop: extend the array's LENGTH by n_pad_own (rgb's
        # own pad count, since map_to_scan_idx is queried on the rgb timeline just like
        # map_to_d_idx/map_to_uw_idx), by prepending n_pad_own COPIES OF THE ARRAY'S OWN
        # FIRST VALUE -- not zeros, not shifted. Zeros would silently be wrong if the
        # real episode's first value is e.g. 0 (a scan legitimately completed before the
        # recording's nominal start): repeating the real first value is the only choice
        # that means "whatever scan (or -1) was valid at the real episode's first
        # timestep, keep pointing at that through the padded prefix" -- the same
        # "repeat own first row" pattern used for ordinary data keys above, not the
        # "prepend zeros" pattern used for the other index maps.
        scan_idx_key = f"map_to_scan_idx_{id_}"
        rgb_ts_key = f"rgb_time_stamps_{id_}"
        if scan_idx_key in ep_data and rgb_ts_key in n_pad_by_ts_key:
            n_pad_own = n_pad_by_ts_key[rgb_ts_key]
            arr = np.asarray(ep_data[scan_idx_key])
            assert arr.shape[0] == np.asarray(ep_data[rgb_ts_key]).reshape(-1).shape[0], (
                f"{scan_idx_key} length must match {rgb_ts_key} length before padding "
                f"({arr.shape[0]} != {np.asarray(ep_data[rgb_ts_key]).reshape(-1).shape[0]})"
            )
            prefix = np.repeat(arr[:1], n_pad_own, axis=0)
            out[scan_idx_key] = np.concatenate([prefix, arr], axis=0)

        # re-zero so the episode still starts at t=0, matching converter convention
        ts_keys_present = [
            ts_template.format(id=id_)
            for ts_template in _ALL_TIMESTAMP_KEYS
            if ts_template.format(id=id_) in out
        ]
        if ts_keys_present:
            offset = min(float(np.asarray(out[k]).reshape(-1)[0]) for k in ts_keys_present)
            for k in ts_keys_present:
                out[k] = out[k] - offset

            # scan_gap_intervals_{id} lives in the same pre-pad time convention as
            # rgb_time_stamps_{id} (see convert_unified_t265_wild_to_umift_zarr.py's
            # ts_norm rebasing), but it isn't itself a "_time_stamps_" family so the
            # generic re-zero loop above skips it. Apply the identical scalar shift here
            # so it stays directly comparable to the now-re-zeroed timestamps (this is
            # what SequenceSamplerUmiFT's scan-gap window filter compares against).
            gap_key = f"scan_gap_intervals_{id_}"
            if gap_key in out:
                gap_arr = np.asarray(out[gap_key])
                out[gap_key] = (gap_arr.astype(np.float64) - offset).astype(gap_arr.dtype)

    return out

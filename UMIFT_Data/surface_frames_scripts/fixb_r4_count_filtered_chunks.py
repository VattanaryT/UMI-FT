#!/usr/bin/env python3
"""Count r4 train/val chunks before/after off-edge ∪ field-jump filter (no image load).

Mirrors SequenceSamplerUmiFT index construction for action21 mold-aware tasks
(sparse_query_frequency_down_sample_steps, action horizon/stride, val split).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import zarr

from diffusion_policy.common.sampler_umift import (
    get_val_mask,
    sparse_action_indices_from_rgb_timestamps,
)

from fixb_r4_chunk_filter import (
    FIELD_JUMP_DEG,
    OFF_EDGE_M,
    build_episode_mask_table,
    filter_sampler_indices,
)

ROOT = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_scanner_v2_fixb")
ZARR = ROOT / "r4_world_fixb_tilt" / "acp_replay_buffer.zarr"
OUT = ROOT / "r4_world_fixb_tilt" / "checks" / "r4_chunk_filter_stats.json"

# Match umift_mold_aware + r4
SPARSE_QUERY_DS = 2
ACTION_HORIZON = 24
ACTION_RGB_STRIDE = 1
VAL_RATIO = 0.1
SEED = 42
ACTION_PADDING = True


def build_indices(data, episode_mask: np.ndarray):
    """Return list of (episode_number, rgb_id) like SequenceSamplerUmiFT."""
    keys = sorted(
        (k for k in data.keys() if k.startswith("episode_")),
        key=lambda s: int(s.split("_")[-1]),
    )
    indices = []
    for count, key in enumerate(keys):
        if not episode_mask[count]:
            continue
        ep = data[key]
        epi_num = int(key.split("_")[-1])
        rgb_times = np.squeeze(np.asarray(ep["rgb_time_stamps_0"], dtype=np.float64))
        robot_times = np.squeeze(np.asarray(ep["robot_time_stamps_0"], dtype=np.float64))
        # Approximate end/start like sampler (robot span vs rgb)
        start_time = float(robot_times[0])
        end_time = float(robot_times[-1])
        first = int(np.searchsorted(rgb_times, start_time, side="left"))
        last = int(np.searchsorted(rgb_times, end_time, side="right") - 1)
        if last <= first:
            continue
        length = last - first
        candidate = first + np.arange(0, length, SPARSE_QUERY_DS)
        # Keep only those that can form a full action horizon under padding
        action_times = robot_times  # Fix B: action stamps == robot
        for rgb_id in candidate:
            try:
                aidx, _ = sparse_action_indices_from_rgb_timestamps(
                    action_times=action_times,
                    rgb_times=rgb_times,
                    rgb_id=int(rgb_id),
                    action_horizon=ACTION_HORIZON,
                    rgb_frame_stride=ACTION_RGB_STRIDE,
                )
            except (ValueError, IndexError):
                continue
            if len(aidx) == 0:
                continue
            if (not ACTION_PADDING) and len(aidx) < ACTION_HORIZON:
                continue
            indices.append((epi_num, int(rgb_id)))
    return indices


def main() -> int:
    data = zarr.open(str(ZARR), "r")["data"]
    keys = sorted(
        (k for k in data.keys() if k.startswith("episode_")),
        key=lambda s: int(s.split("_")[-1]),
    )
    n_ep = len(keys)
    val_mask = get_val_mask(n_episodes=n_ep, val_ratio=VAL_RATIO, seed=SEED)
    train_mask = ~val_mask

    # Fake replay_buffer structure expected by filter (obs timestamps + action stamps)
    rb_data = {}
    for key in keys:
        ep = data[key]
        rb_data[key] = {
            "action_time_stamps": np.asarray(ep["robot_time_stamps_0"]),
            "obs": {"rgb_time_stamps_0": np.asarray(ep["rgb_time_stamps_0"])},
        }
    rb = {"data": rb_data}
    shape_meta = {
        "sample": {
            "action": {
                "sparse": {"horizon": ACTION_HORIZON, "down_sample_steps": ACTION_RGB_STRIDE}
            }
        }
    }
    masks = build_episode_mask_table(ZARR, off_edge_m=OFF_EDGE_M, jump_deg=FIELD_JUMP_DEG)

    payload = {}
    for split, mask in (("train", train_mask), ("val", val_mask)):
        before = build_indices(data, mask)
        # filter_sampler_indices expects replay_buffer with same episode keys
        kept, stats = filter_sampler_indices(
            before,
            replay_buffer=rb,
            raw_masks=masks,
            shape_meta=shape_meta,
            action_padding=ACTION_PADDING,
        )
        stats["split"] = split
        stats["off_edge_m"] = OFF_EDGE_M
        stats["field_jump_deg"] = FIELD_JUMP_DEG
        payload[split] = stats
        print(
            f"[{split}] {stats['chunks_before']} -> {stats['chunks_after']} "
            f"(removed {stats['frac_removed']:.1%}; "
            f"off_only={stats['removed_off_edge_only']}, "
            f"jump_only={stats['removed_field_jump_only']}, both={stats['removed_both']})"
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

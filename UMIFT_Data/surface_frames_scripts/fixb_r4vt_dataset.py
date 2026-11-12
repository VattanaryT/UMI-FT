"""r4vt training set: same chunks as r4, then drop excluded episodes and samples.

A query is dropped when any action target in its chunk hits an excluded robot
sample. Exclusions are episodes 35 and 44, bad_span_0, and the long-dt mask.
Existing r4 datasets are unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import zarr

from diffusion_policy.common.sampler_umift import sparse_action_indices_from_rgb_timestamps
from fixb_dataset import UmiFTDatasetOctR4


def _robot_masks(dataset_path: str, long_dt_mask_path: str, n_action: dict[int, int]):
    root = zarr.open(dataset_path, "r")["data"]
    blob = np.load(long_dt_mask_path)
    long_idx: dict[int, set[int]] = {}
    for ep, idx in zip(blob["flagged_episode"], blob["flagged_index"]):
        long_idx.setdefault(int(ep), set()).add(int(idx))
    span = {}
    long = {}
    for key in root.keys():
        if not key.startswith("episode_"):
            continue
        ep = int(key.split("_")[-1])
        if ep < 0 or ep > 49:
            raise RuntimeError(f"{key} is outside October episodes 0-49")
        if ep not in n_action:
            continue
        n = int(n_action[ep])
        robot_n = int(np.asarray(root[key]["robot_time_stamps_0"]).reshape(-1).shape[0])
        if n > robot_n:
            raise RuntimeError(f"{key} action length {n} exceeds robot length {robot_n}")
        bad = np.zeros(robot_n, dtype=bool)
        if "bad_span_0" in root[key]:
            bad |= np.asarray(root[key]["bad_span_0"]).reshape(-1)[:robot_n] > 0
        span[ep] = bad[:n]
        dt = np.zeros(n, dtype=bool)
        for i in long_idx.get(ep, ()):
            if 0 <= i < n:
                dt[i] = True
        long[ep] = dt
    return span, long


def drop_excluded_queries(
    sampler,
    dataset_path: str,
    exclude_episodes,
    long_dt_mask_path: str,
    tag: str = "r4vt-exclude",
) -> dict:
    shape = sampler.shape_meta
    horizon = int(shape["sample"]["action"]["sparse"]["horizon"])
    stride = int(shape["sample"]["action"]["sparse"]["down_sample_steps"])
    n_action = {}
    for epi_id, _rgb_id in sampler.indices:
        epi_id = int(epi_id)
        if epi_id not in n_action:
            ep = sampler.replay_buffer["data"][f"episode_{epi_id}"]
            n_action[epi_id] = int(np.asarray(ep["action_time_stamps"]).reshape(-1).shape[0])
    exclude = {int(ep) for ep in exclude_episodes}
    for ep in exclude:
        n_action.setdefault(ep, 0)
    # Lengths for episodes that never entered this split still need a mask entry
    # only if a query references them. Queries do not.
    span, long = _robot_masks(dataset_path, long_dt_mask_path, {k: v for k, v in n_action.items() if v})
    kept = []
    n_ep = n_span = n_dt = n_both = 0
    for epi_id, rgb_id in sampler.indices:
        epi_id = int(epi_id)
        rgb_id = int(rgb_id)
        if epi_id in exclude:
            n_ep += 1
            continue
        ep = sampler.replay_buffer["data"][f"episode_{epi_id}"]
        action_idx, _times = sparse_action_indices_from_rgb_timestamps(
            action_times=np.asarray(ep["action_time_stamps"]),
            rgb_times=np.asarray(ep["obs"]["rgb_time_stamps_0"]),
            rgb_id=rgb_id,
            action_horizon=horizon,
            rgb_frame_stride=stride,
        )
        hit_span = bool(np.any(span[epi_id][action_idx])) if action_idx.size else False
        hit_dt = bool(np.any(long[epi_id][action_idx])) if action_idx.size else False
        if hit_span and hit_dt:
            n_both += 1
            continue
        if hit_span:
            n_span += 1
            continue
        if hit_dt:
            n_dt += 1
            continue
        kept.append((epi_id, rgb_id))
    before = len(sampler.indices)
    sampler.indices = kept
    stats = {
        "before": before,
        "dropped_episode": n_ep,
        "dropped_bad_span": n_span,
        "dropped_long_dt": n_dt,
        "dropped_span_and_long_dt": n_both,
        "kept": len(kept),
    }
    print(
        f"[{tag}] "
        + " ".join(f"{k}={v}" for k, v in stats.items()),
        flush=True,
    )
    return stats


class UmiFTDatasetR4VT(UmiFTDatasetOctR4):
    """October r4vt. Episodes 35 and 44, bad spans, and long waits are not sampled."""

    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        action_normalizer_path: str,
        action_normalizer_sha256: str,
        exclude_episodes,
        long_dt_mask_path: str,
        exclusion_stats_path: str | None = None,
        **kwargs,
    ):
        self._exclude = [int(ep) for ep in exclude_episodes]
        self._long_dt_mask_path = str(long_dt_mask_path)
        self._exclusion_stats_path = exclusion_stats_path
        self._dataset_path = dataset_path
        super().__init__(
            shape_meta=shape_meta,
            dataset_path=dataset_path,
            action_normalizer_path=action_normalizer_path,
            action_normalizer_sha256=action_normalizer_sha256,
            **kwargs,
        )
        keys = list(self.replay_buffer["data"].keys())
        for i, key in enumerate(keys):
            if int(key.split("_")[-1]) in set(self._exclude):
                self.val_mask[i] = False
        self._exclusion_stats = {"train": self._apply("train")}

    def _apply(self, split: str) -> dict:
        stats = drop_excluded_queries(
            self.sampler,
            self._dataset_path,
            self._exclude,
            self._long_dt_mask_path,
        )
        stats["split"] = split
        if self._exclusion_stats_path:
            path = Path(self._exclusion_stats_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {}
            if path.exists():
                try:
                    payload = json.loads(path.read_text())
                except json.JSONDecodeError:
                    payload = {}
            payload[split] = stats
            path.write_text(json.dumps(payload, indent=2))
        return stats

    def get_validation_dataset(self):
        val_set = super().get_validation_dataset()
        val_set._exclude = self._exclude
        val_set._long_dt_mask_path = self._long_dt_mask_path
        val_set._dataset_path = self._dataset_path
        val_set._exclusion_stats_path = self._exclusion_stats_path
        stats = val_set._apply("val")
        self._exclusion_stats["val"] = stats
        return val_set

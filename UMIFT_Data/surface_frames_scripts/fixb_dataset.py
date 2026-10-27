"""Dataset helpers for Fix B runs (r3 curv obs; r4 ΔR loss mask).

Put this directory on PYTHONPATH. r1/r2 keep the default UmiFTDataset.
r4 uses UmiFTDatasetR4: **same sampler indices as r2**, with a per-step loss
weight that zeros only the ΔR dims after a field-frame jump inside the chunk.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import zarr

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.sampler_umift import (
    SequenceSamplerUmiFT,
    sparse_action_indices_from_rgb_timestamps,
)
from diffusion_policy.dataset.umift_dataset import UmiFTDataset
from PyriteConfig.tasks.umift.umift_type_conversions import (
    action21_to_action_sample,
    obs_to_obs_sample,
)
from threadpoolctl import threadpool_limits

from fixb_r4_chunk_filter import FIELD_JUMP_DEG, build_episode_mask_table


class UmiFTDatasetFixB(UmiFTDataset):
    """Run 3: add robot0_mold_curv from mold_curvature_0."""

    def raw_episodes_conversion(self, replay_buffer_raw, shape_meta):
        replay_buffer = super().raw_episodes_conversion(replay_buffer_raw, shape_meta)
        if "robot0_mold_curv" not in shape_meta.get("obs", {}):
            return replay_buffer
        for ep in replay_buffer_raw["data"].keys():
            replay_buffer["data"][ep]["obs"]["robot0_mold_curv"] = np.asarray(
                replay_buffer_raw["data"][ep]["mold_curvature_0"], dtype=np.float32
            )
        return replay_buffer


class UmiFTDatasetR4(UmiFTDataset):
    """r4: identical chunk sampling to r2; mask ΔR loss after field-frame jumps."""

    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        mask_delta_R_after_field_jumps: bool = True,
        field_jump_deg: float = FIELD_JUMP_DEG,
        filter_stats_path: str | None = None,
        **kwargs,
    ):
        self._r4_mask = bool(mask_delta_R_after_field_jumps)
        self._r4_jump_deg = float(field_jump_deg)
        self._r4_filter_stats_path = filter_stats_path
        super().__init__(shape_meta=shape_meta, dataset_path=dataset_path, **kwargs)
        self._r4_field_jump: Dict[str, np.ndarray] = {}
        self._r4_mask_stats: Dict[str, float] = {}
        if self._r4_mask:
            table = build_episode_mask_table(
                dataset_path, jump_deg=self._r4_jump_deg
            )
            self._r4_field_jump = {k: v["field_jump"] for k, v in table.items()}
            self._r4_mask_stats = self._log_delta_R_mask_fraction(train=True)
            if self._r4_filter_stats_path:
                path = Path(self._r4_filter_stats_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = {}
                if path.exists():
                    try:
                        payload = json.loads(path.read_text())
                    except json.JSONDecodeError:
                        payload = {}
                payload["train"] = self._r4_mask_stats
                path.write_text(json.dumps(payload, indent=2))

    def _action_indices_for_query(self, epi_id: int, rgb_id: int) -> np.ndarray:
        key = f"episode_{int(epi_id)}"
        ep = self.replay_buffer["data"][key]
        action_horizon = int(self.shape_meta["sample"]["action"]["sparse"]["horizon"])
        rgb_frame_stride = int(
            self.shape_meta["sample"]["action"]["sparse"]["down_sample_steps"]
        )
        action_times = np.squeeze(np.asarray(ep["action_time_stamps"], dtype=np.float64))
        rgb_times = np.squeeze(np.asarray(ep["obs"]["rgb_time_stamps_0"], dtype=np.float64))
        action_idx, _ = sparse_action_indices_from_rgb_timestamps(
            action_times=action_times,
            rgb_times=rgb_times,
            rgb_id=int(rgb_id),
            action_horizon=action_horizon,
            rgb_frame_stride=rgb_frame_stride,
        )
        return np.asarray(action_idx, dtype=np.int64)

    def _delta_R_weight_for_query(self, epi_id: int, rgb_id: int, H: int, D: int) -> np.ndarray:
        """(H,D) weights: 1 everywhere; 0 on dims 21:27 after a field jump since chunk start."""
        w = np.ones((H, D), dtype=np.float32)
        if D < 27 or not self._r4_mask:
            return w
        key = f"episode_{int(epi_id)}"
        jump = self._r4_field_jump.get(key)
        if jump is None:
            return w
        try:
            action_idx = self._action_indices_for_query(epi_id, rgb_id)
        except (ValueError, IndexError):
            return w
        if len(action_idx) == 0:
            return w
        a0 = int(action_idx[0])
        for t in range(H):
            if t >= len(action_idx):
                a_t = int(action_idx[-1])
            else:
                a_t = int(action_idx[t])
            if t == 0:
                continue
            # Jump strictly after chunk start up to step t (inclusive).
            lo = a0 + 1
            hi = a_t + 1
            if lo < hi and np.any(jump[lo:hi]):
                w[t, 21:27] = 0.0
        return w

    def _log_delta_R_mask_fraction(self, train: bool) -> dict:
        """Scan sampler indices (no images) and report fraction of ΔR steps masked."""
        n_steps = 0
        n_masked = 0
        H = int(self.shape_meta["sample"]["action"]["sparse"]["horizon"])
        D = int(self.shape_meta["action"]["shape"][0])
        for epi_id, rgb_id in self.sampler.indices:
            w = self._delta_R_weight_for_query(epi_id, rgb_id, H, D)
            n_steps += H
            n_masked += int(np.sum(w[:, 21:27] == 0.0) // 6)  # count masked timesteps
        frac = float(n_masked / max(n_steps, 1))
        split = "train" if train else "val"
        stats = {
            "split": split,
            "n_chunks": len(self.sampler.indices),
            "n_action_steps": n_steps,
            "n_delta_R_steps_masked": n_masked,
            "frac_delta_R_steps_masked": frac,
            "field_jump_deg": self._r4_jump_deg,
            "note": (
                "Same chunks as r2 (no chunk drop). ΔR loss weight=0 from the first "
                "field-frame jump after chunk start through the rest of the horizon."
            ),
        }
        print(
            f"[UmiFTDatasetR4] {split} ΔR loss mask: "
            f"{n_masked}/{n_steps} steps masked ({frac:.1%}) "
            f"over {len(self.sampler.indices)} chunks "
            f"(field_jump>{self._r4_jump_deg:.0f}°)"
        )
        return stats

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.threadpool_limits_is_applied:
            threadpool_limits(1)
            self.threadpool_limits_is_applied = True
        obs_dict, action_array = self.sampler.sample_sequence(idx)
        torch_data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": dict_apply(action_array, torch.from_numpy),
        }
        if self._r4_mask:
            epi_id, rgb_id = self.sampler.indices[idx]
            act = torch_data["action"]["sparse"]
            H, D = int(act.shape[0]), int(act.shape[1])
            w = self._delta_R_weight_for_query(epi_id, rgb_id, H, D)
            torch_data["action_loss_weight"] = {
                "sparse": torch.from_numpy(w),
            }
        return torch_data

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSamplerUmiFT(
            shape_meta=self.shape_meta,
            replay_buffer=self.replay_buffer,
            sparse_query_frequency_down_sample_steps=self.sparse_query_frequency_down_sample_steps,
            episode_mask=self.val_mask,
            action_padding=self.action_padding,
            obs_to_obs_sample=obs_to_obs_sample,
            action_to_action_sample=action21_to_action_sample,
            id_list=self.id_list,
            action_gap_threshold_mult=self.action_gap_threshold_mult,
            scan_pre_first_fill=self.scan_pre_first_fill,
            max_abs_pos_m=self.max_abs_pos_m,
        )
        val_set.val_mask = ~self.val_mask
        if self._r4_mask:
            val_set._r4_field_jump = self._r4_field_jump
            val_set._r4_jump_deg = self._r4_jump_deg
            val_set._r4_filter_stats_path = self._r4_filter_stats_path
            val_set._r4_mask = True
            stats = val_set._log_delta_R_mask_fraction(train=False)
            if self._r4_filter_stats_path:
                path = Path(self._r4_filter_stats_path)
                payload = {}
                if path.exists():
                    try:
                        payload = json.loads(path.read_text())
                    except json.JSONDecodeError:
                        payload = {}
                payload["val"] = stats
                path.write_text(json.dumps(payload, indent=2))
        return val_set

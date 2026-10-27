"""Dataset helpers for Fix B runs (r3 curv obs; r4 jump/off-edge filter).

Put this directory on PYTHONPATH. r1/r2 keep the default UmiFTDataset.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from diffusion_policy.common.sampler_umift import SequenceSamplerUmiFT
from diffusion_policy.dataset.umift_dataset import UmiFTDataset
from PyriteConfig.tasks.umift.umift_type_conversions import (
    action21_to_action_sample,
    obs_to_obs_sample,
)

from fixb_r4_chunk_filter import (
    FIELD_JUMP_DEG,
    OFF_EDGE_M,
    build_episode_mask_table,
    filter_sampler_indices,
)


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
    """r4 only: same as UmiFTDataset, plus off-edge ∪ field-jump chunk filter."""

    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        drop_off_edge_and_field_jumps: bool = True,
        off_edge_m: float = OFF_EDGE_M,
        field_jump_deg: float = FIELD_JUMP_DEG,
        filter_stats_path: str | None = None,
        **kwargs,
    ):
        self._r4_drop = bool(drop_off_edge_and_field_jumps)
        self._r4_off_edge_m = float(off_edge_m)
        self._r4_jump_deg = float(field_jump_deg)
        self._r4_filter_stats_path = filter_stats_path
        super().__init__(shape_meta=shape_meta, dataset_path=dataset_path, **kwargs)
        self._r4_raw_masks = None
        if self._r4_drop:
            self._r4_raw_masks = build_episode_mask_table(
                dataset_path,
                off_edge_m=self._r4_off_edge_m,
                jump_deg=self._r4_jump_deg,
            )
            self._apply_r4_chunk_filter(self.sampler, train=True)

    def _apply_r4_chunk_filter(self, sampler: SequenceSamplerUmiFT, train: bool) -> dict:
        before = list(sampler.indices)
        kept, stats = filter_sampler_indices(
            before,
            replay_buffer=self.replay_buffer,
            raw_masks=self._r4_raw_masks,
            shape_meta=self.shape_meta,
            action_padding=self.action_padding,
        )
        # patch OFF_EDGE / jump into stats from instance
        stats["off_edge_m"] = self._r4_off_edge_m
        stats["field_jump_deg"] = self._r4_jump_deg
        stats["split"] = "train" if train else "val"
        sampler.indices = kept
        msg = (
            f"[UmiFTDatasetR4] {stats['split']} chunk filter "
            f"(off-edge>{self._r4_off_edge_m*1e3:.0f}mm ∪ field-jump>{self._r4_jump_deg:.0f}°): "
            f"{stats['chunks_before']} -> {stats['chunks_after']} "
            f"(removed {stats['chunks_removed']}, {stats['frac_removed']:.1%}; "
            f"off_only={stats['removed_off_edge_only']}, "
            f"jump_only={stats['removed_field_jump_only']}, "
            f"both={stats['removed_both']})"
        )
        print(msg)
        if self._r4_filter_stats_path:
            path = Path(self._r4_filter_stats_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {}
            if path.exists():
                try:
                    payload = json.loads(path.read_text())
                except json.JSONDecodeError:
                    payload = {}
            payload[stats["split"]] = stats
            path.write_text(json.dumps(payload, indent=2))
        self._r4_last_filter_stats = stats
        return stats

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
        if self._r4_drop and self._r4_raw_masks is not None:
            val_set._r4_raw_masks = self._r4_raw_masks
            val_set._r4_off_edge_m = self._r4_off_edge_m
            val_set._r4_jump_deg = self._r4_jump_deg
            val_set._r4_filter_stats_path = self._r4_filter_stats_path
            val_set._apply_r4_chunk_filter(val_set.sampler, train=False)
        return val_set

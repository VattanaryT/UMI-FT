"""r1rel training set on the corrected October labels.

Same relative-pose dataset as r1rel. Queries are dropped when any action
target hits bad_span_0 or the long-dt mask. Episodes 35 and 44 stay.
The old pose-jump flag file is not a drop and not a loss mask.
"""

from __future__ import annotations

import json
from pathlib import Path

from fixb_dataset import UmiFTDatasetOctNorm
from fixb_r4vt_dataset import drop_excluded_queries


class UmiFTDatasetR1RelFiltered(UmiFTDatasetOctNorm):
    """October r1rel. Bad spans and long waits are not sampled."""

    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        action_normalizer_path: str,
        action_normalizer_sha256: str,
        action_normalizer_role: str,
        long_dt_mask_path: str,
        exclude_episodes=(),
        exclusion_stats_path: str | None = None,
        pose_jump_flag_path: str | None = None,
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
            action_normalizer_role=action_normalizer_role,
            pose_jump_flag_path=pose_jump_flag_path,
            **kwargs,
        )
        keys = list(self.replay_buffer["data"].keys())
        excluded = set(self._exclude)
        for i, key in enumerate(keys):
            if int(key.split("_")[-1]) in excluded:
                self.val_mask[i] = False
        self._exclusion_stats = {"train": self._apply("train")}

    def _apply(self, split: str) -> dict:
        stats = drop_excluded_queries(
            self.sampler,
            self._dataset_path,
            self._exclude,
            self._long_dt_mask_path,
            tag="r1rel-exclude",
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

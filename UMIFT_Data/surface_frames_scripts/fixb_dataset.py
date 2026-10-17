"""Dataset for Fix B run 3.

UmiFTDataset already packs signed distance, normal and R_field rot6d into
robot0_mold_geom. Run 3 adds local curvature as robot0_mold_curv, read from
mold_curvature_0. Nothing else changes. Put this directory on PYTHONPATH and
set dataset._target_ to fixb_dataset.UmiFTDatasetFixB (see
config/task/umift_fixb_r3.yaml).
"""

from __future__ import annotations

import numpy as np

from diffusion_policy.dataset.umift_dataset import UmiFTDataset


class UmiFTDatasetFixB(UmiFTDataset):
    def raw_episodes_conversion(self, replay_buffer_raw, shape_meta):
        replay_buffer = super().raw_episodes_conversion(replay_buffer_raw, shape_meta)
        if "robot0_mold_curv" not in shape_meta.get("obs", {}):
            return replay_buffer
        for ep in replay_buffer_raw["data"].keys():
            replay_buffer["data"][ep]["obs"]["robot0_mold_curv"] = np.asarray(
                replay_buffer_raw["data"][ep]["mold_curvature_0"], dtype=np.float32
            )
        return replay_buffer

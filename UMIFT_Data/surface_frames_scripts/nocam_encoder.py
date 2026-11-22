"""Wrist-RGB-off encoder and policy for the fixed-k nocam run.

The dataset and the sampler still load ``rgb_0``. That key is the query clock,
so deleting it from ``sample.obs.sparse`` would change which rows are trained.
These two classes drop ``obs["sparse"]["rgb_0"]`` after the batch is built and
build the encoder with no wrist-RGB branch. F/T, TCP pose, and mold features
stay. Depth is removed by the existing ``task.drop_depth_inputs`` flag.
"""

from __future__ import annotations

import copy

from omegaconf import OmegaConf

from diffusion_policy.model.vision.timm_obs_encoder_with_force import (
    TimmObsEncoderWithForce,
)
from diffusion_policy.policy.diffusion_unet_timm_mod1_policy import (
    DiffusionUnetTimmMod1Policy,
)


def _is_wrist_rgb(key: str) -> bool:
    return key.startswith("rgb_") and key[4:].isdigit()


def _copy_without_wrist_rgb(shape_meta):
    if OmegaConf.is_config(shape_meta):
        copied = OmegaConf.to_container(shape_meta, resolve=True)
    else:
        copied = copy.deepcopy(dict(shape_meta))
    obs = copied.get("obs", {})
    for key in list(obs):
        if _is_wrist_rgb(key):
            del obs[key]
    return copied


def strip_wrist_rgb(tree):
    """Return a shallow copy of an obs or batch dict with wrist RGB removed."""
    if not isinstance(tree, dict):
        return tree
    if "sparse" in tree and isinstance(tree["sparse"], dict):
        sparse = {key: value for key, value in tree["sparse"].items() if not _is_wrist_rgb(key)}
        out = dict(tree)
        out["sparse"] = sparse
        return out
    return {key: value for key, value in tree.items() if not _is_wrist_rgb(key)}


class NoWristRgbEncoder(TimmObsEncoderWithForce):
    def __init__(self, shape_meta, **kwargs):
        super().__init__(shape_meta=_copy_without_wrist_rgb(shape_meta), **kwargs)
        if self.rgb_keys:
            raise RuntimeError(f"nocam encoder still has rgb keys: {self.rgb_keys}")

    def forward(self, obs_dict):
        return super().forward(strip_wrist_rgb(obs_dict))


class NoWristRgbPolicy(DiffusionUnetTimmMod1Policy):
    def compute_loss(self, batch, args):
        batch = dict(batch)
        batch["obs"] = strip_wrist_rgb(batch["obs"])
        return super().compute_loss(batch, args)

    def predict_action(self, obs, debug_action=None):
        return super().predict_action(strip_wrist_rgb(obs), debug_action)

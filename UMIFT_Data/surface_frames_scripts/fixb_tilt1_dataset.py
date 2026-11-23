"""5-d 1-tilt action on the causal-foot zarr. Does not write that zarr.

The parent loader only accepts the 10-d causal-foot action. This class loads
that, then replaces it with v1, v2, δn, stiffness, a. ψ stays the field heading.
"""

from __future__ import annotations

import copy
import hashlib
import pickle
from pathlib import Path

import numpy as np

from diffusion_policy.dataset.umift_dataset import UmiFTDataset
from diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer
from fixb_r4vt_dataset import UmiFTDatasetR4VT
from heading_tilt import long_axis
from tilt1 import action_tilt1_from_group, action_tilt1_to_action_sample


def _as_dict(shape_meta):
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(shape_meta):
            shape_meta = OmegaConf.to_container(shape_meta, resolve=True)
    except ImportError:
        pass
    return copy.deepcopy(dict(shape_meta))


def _pin_tilt1(sparse_normalizer, path: str, sha256: str):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if not sha256 or digest != sha256:
        raise RuntimeError(f"action normalizer sha256 {digest} != config {sha256} ({path})")
    with path.open("rb") as handle:
        saved = pickle.load(handle)
    src = saved.params_dict["action"]
    scale = src["scale"].detach().float().cpu().reshape(-1)
    offset = src["offset"].detach().float().cpu().reshape(-1)
    if scale.numel() != 5 or offset.numel() != 5:
        raise RuntimeError(f"tilt1 file action dim {scale.numel()}, expected 5")
    online = sparse_normalizer.params_dict["action"]["scale"].detach().float().cpu().reshape(-1)
    if online.numel() != 5:
        raise RuntimeError(f"tilt1 online action dim {online.numel()}, expected 5")
    stats = {}
    for key in ("min", "max", "mean", "std"):
        values = src["input_stats"][key].detach().float().cpu().reshape(-1)
        if values.numel() != 5:
            raise RuntimeError(f"tilt1 input_stats[{key}] dim {values.numel()}")
        stats[key] = values.numpy()
    sparse_normalizer["action"] = SingleFieldLinearNormalizer.create_manual(
        scale=scale.numpy(),
        offset=offset.numpy(),
        input_stats_dict=stats,
    )


class UmiFTDatasetTilt1(UmiFTDatasetR4VT):
    """Causal-foot rows. The action is v1, v2, δn, stiffness, a."""

    def __init__(self, shape_meta: dict, reduced_r4: bool = False, **kwargs):
        del reduced_r4
        kwargs.pop("mask_delta_R_after_field_jumps", None)
        lied = _as_dict(shape_meta)
        lied["action"]["shape"] = [10]
        lied["reduced_r4"] = True
        lied.pop("heading_tilt", None)
        super().__init__(
            shape_meta=lied,
            reduced_r4=True,
            mask_delta_R_after_field_jumps=False,
            **kwargs,
        )
        self._install_tilt1()

    def _install_tilt1(self) -> None:
        import zarr

        field = Path(self._dataset_path).resolve().parents[1] / "shared" / "fixb_field.npz"
        if not field.is_file():
            raise RuntimeError(f"missing field {field}")
        axis = long_axis(np.load(field)["vertices"])
        disk = zarr.open(self._dataset_path, "r")["data"]
        for key, episode in self.replay_buffer["data"].items():
            group = disk[key]
            new = action_tilt1_from_group(group, axis)
            old = np.asarray(episode["action"])
            if old.shape[-1] != 10 or len(old) != len(new):
                raise RuntimeError(f"{key} action {old.shape} vs tilt1 {new.shape}")
            if not np.allclose(old[:, :4], new[:, :4], atol=1e-5):
                raise RuntimeError(f"{key} v1/v2/δn/stiffness drifted from the causal-foot action")
            episode["action"] = new
        self.shape_meta["action"]["shape"] = [5]
        self.shape_meta["reduced_r4"] = False
        # Parent range-fits only a known action type. heading8's slice keeps a
        # shorter vector, then _pin_tilt1 replaces it. The policy dim is 5.
        self.action_type = "r4_heading8"
        self.action_to_action_sample = action_tilt1_to_action_sample
        self.sampler.action_to_action_sample = action_tilt1_to_action_sample

    def get_normalizer(self, **kwargs):
        sparse, dense = UmiFTDataset.get_normalizer(self, **kwargs)
        _pin_tilt1(sparse, self._an_path, self._an_sha)
        return sparse, dense

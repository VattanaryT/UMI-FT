"""8-d heading-tilt action: v1, v2, δn, stiffness, sin ψ, cos ψ, a, b.

Same episode and long-dt exclusions as r4vt. The field-jump ΔR mask stays off:
ψ is not measured from t1, so a t1 jump is not a label hole.
"""

from __future__ import annotations

import copy

from diffusion_policy.dataset.umift_dataset import UmiFTDataset
from fixb_dataset import apply_shared_action_normalizer
from fixb_r4vt_dataset import UmiFTDatasetR4VT


def _mark_heading(shape_meta):
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(shape_meta):
            shape_meta = OmegaConf.to_container(shape_meta, resolve=True)
    except ImportError:
        pass
    copied = copy.deepcopy(dict(shape_meta))
    copied["heading_tilt"] = True
    copied["reduced_r4"] = False
    return copied


class UmiFTDatasetHeadingTilt(UmiFTDatasetR4VT):
    """October causal-foot rows with the heading-tilt action."""

    def __init__(self, shape_meta: dict, reduced_r4: bool = False, **kwargs):
        del reduced_r4
        super().__init__(shape_meta=_mark_heading(shape_meta), reduced_r4=False, **kwargs)

    def get_normalizer(self, **kwargs):
        sparse, dense = UmiFTDataset.get_normalizer(self, **kwargs)
        apply_shared_action_normalizer(sparse, self._an_path, self._an_sha, "r4_heading8")
        return sparse, dense

"""Liftos dataset with a higher loss on the climb velocity and the lift.

Same zarr and the same normalizer. v2 is the tangent that carries the foot
from the crown onto the far flange. δn is the clearance. Both stay at weight 1
in the parent. This class multiplies those two columns after the field-jump
ΔR mask, so a masked rotation stays masked.
"""

from __future__ import annotations

import torch

from field_v2n_v3r_liftos_dataset import UmiFTDatasetFieldV2NV3RLiftOS

V2_DIM = 1
DN_DIM = 2


class UmiFTDatasetLiftOSH24(UmiFTDatasetFieldV2NV3RLiftOS):
    def __init__(self, *args, v2_loss_weight: float = 4.0, dn_loss_weight: float = 4.0, **kwargs):
        self._v2_loss_weight = float(v2_loss_weight)
        self._dn_loss_weight = float(dn_loss_weight)
        if self._v2_loss_weight < 1.0 or self._dn_loss_weight < 1.0:
            raise ValueError("climb weights must be >= 1")
        super().__init__(*args, **kwargs)
        print(
            f"[climb-h24] loss weight v2 x{self._v2_loss_weight:g} "
            f"δn x{self._dn_loss_weight:g}",
            flush=True,
        )

    def __getitem__(self, idx: int):
        data = super().__getitem__(idx)
        action = data["action"]["sparse"]
        weights = data.get("action_loss_weight", {}).get("sparse")
        if weights is None:
            weights = torch.ones_like(action)
        else:
            weights = weights.clone()
        weights[:, V2_DIM] *= self._v2_loss_weight
        weights[:, DN_DIM] *= self._dn_loss_weight
        data["action_loss_weight"] = {"sparse": weights}
        return data


def _check() -> None:
    action = torch.zeros(4, 10)
    weights = torch.ones(4, 10)
    weights[:, 4:] = 0.0
    weights[:, V2_DIM] *= 4.0
    weights[:, DN_DIM] *= 4.0
    if float(weights[0, V2_DIM]) != 4.0 or float(weights[0, DN_DIM]) != 4.0:
        raise RuntimeError("climb weights did not land on v2 and δn")
    if float(weights[:, 4:].sum()) != 0.0:
        raise RuntimeError("ΔR mask was overwritten")
    del action
    print("[climb-h24] weight check ok")


if __name__ == "__main__":
    _check()

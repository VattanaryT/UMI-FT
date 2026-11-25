"""sv4b liftos dataset with a higher loss on δn and a larger lift fraction.

v2 stays at weight 1. δn is multiplied after the field-jump ΔR mask, so a
masked rotation stays masked. Horizon stays 64.

Lift chunks are the kept training queries whose action horizon contains an
isolated release in the stringer_v4 end file. The October liftos end file
uses episode ids from a different zarr. Those ids are counted and are not
applied onto stringer_v4 rows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from field_v2_dataset import UmiFTDatasetFieldV2
from field_v2n_v3r_liftos_dataset import (
    _ends_by_episode,
    _is_lift,
    action_liftos_to_action_sample,
    expand_indices,
)

DN_DIM = 2
V2_DIM = 1


class UmiFTDatasetSv4bLift(UmiFTDatasetFieldV2):
    def __init__(
        self,
        *args,
        lift_index_path: str,
        lift_batch_fraction: float = 0.30,
        dn_loss_weight: float = 4.0,
        extra_lift_index_path: str | None = None,
        **kwargs,
    ):
        self._lift_index_path = str(lift_index_path)
        self._extra_lift_index_path = (
            None if extra_lift_index_path in (None, "null", "") else str(extra_lift_index_path)
        )
        self._lift_fraction = float(lift_batch_fraction)
        self._dn_loss_weight = float(dn_loss_weight)
        if self._dn_loss_weight < 1.0:
            raise ValueError("δn weight must be >= 1")
        super().__init__(*args, **kwargs)
        self.action_to_action_sample = action_liftos_to_action_sample
        self.sampler.action_to_action_sample = action_liftos_to_action_sample
        ends = _ends_by_episode(self._lift_index_path)
        flags = _is_lift(self.sampler, ends)
        expanded, plan = expand_indices(
            list(self.sampler.indices), flags, self._lift_fraction
        )
        self.sampler.indices = expanded
        self._lift_plan = plan
        extra_n = 0
        if self._extra_lift_index_path:
            extra = _ends_by_episode(self._extra_lift_index_path)
            extra_n = int(sum(len(v) for v in extra.values()))
        print(
            f"[sv4b-lift] loss weight v2 x1 δn x{self._dn_loss_weight:g} "
            f"horizon column {DN_DIM}",
            flush=True,
        )
        print(
            f"[sv4b-lift] lift {plan['lift']}/{plan['lift'] + plan['other']} "
            f"slots {plan['slots']} epoch_len {plan['epoch']} "
            f"fraction {plan['fraction']:.4f}",
            flush=True,
        )
        print(
            f"[sv4b-lift] extra ends {extra_n} from {self._extra_lift_index_path} "
            f"not applied (October episode ids are a different zarr)",
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
        if float(weights[0, V2_DIM]) != 0.0:
            weights[:, V2_DIM] *= 1.0
        weights[:, DN_DIM] *= self._dn_loss_weight
        data["action_loss_weight"] = {"sparse": weights}
        return data

    def get_validation_dataset(self):
        val = super().get_validation_dataset()
        val.action_to_action_sample = action_liftos_to_action_sample
        val.sampler.action_to_action_sample = action_liftos_to_action_sample
        val._dn_loss_weight = self._dn_loss_weight
        val.__class__ = UmiFTDatasetSv4bLift
        return val


def _check() -> None:
    weights = torch.ones(4, 10)
    weights[:, 4:] = 0.0
    weights[:, DN_DIM] *= 4.0
    if float(weights[0, V2_DIM]) != 1.0 or float(weights[0, DN_DIM]) != 4.0:
        raise RuntimeError("δn weight did not leave v2 at 1")
    if float(weights[:, 4:].sum()) != 0.0:
        raise RuntimeError("ΔR mask was overwritten")
    if not Path(__file__).name == "sv4b_lift_dataset.py":
        raise RuntimeError("renamed")
    print("[sv4b-lift] weight check ok")


if __name__ == "__main__":
    _check()

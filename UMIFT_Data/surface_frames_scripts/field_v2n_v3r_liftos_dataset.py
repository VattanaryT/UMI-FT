"""field_v2n observations and the original δn. TCP-relative ΔR. Lift chunks repeated.

The zarr stores absolute tool rot6d and the field_v2n δn. At sample time the
rotation becomes R_tcp(t).T @ R_tcp(t+k). v1, v2, δn, and stiffness pass through.
pose_frames.action must be relative. Observations stay absolute.

Lift chunks are the kept training queries whose action horizon contains an
isolated 1 s |F·n| >= 5 N release. Repeating them makes those chunks the
requested fraction of each training epoch. Validation is not repeated.
"""

from __future__ import annotations

import numpy as np

from diffusion_policy.common.sampler_umift import sparse_action_indices_from_rgb_timestamps
from field_v2_dataset import UmiFTDatasetFieldV2
from rot6d import rot6d_rows_to_R, so3_to_rot6d_rows

TARGET_FRACTION = 0.15


def action_liftos_to_action_sample(action_sparse, action_dense, id_list, base_pose):
    del id_list
    if len(action_dense) > 0:
        raise NotImplementedError
    action = np.array(action_sparse, copy=True)
    if action.ndim != 2 or action.shape[1] != 10:
        raise ValueError(f"liftos action {getattr(action, 'shape', None)}")
    R0 = np.asarray(base_pose[0], dtype=np.float64)[:3, :3]
    future = rot6d_rows_to_R(np.asarray(action[:, 4:10], dtype=np.float64))
    delta = np.einsum("ji,njk->nik", R0, future)
    action[:, 4:10] = so3_to_rot6d_rows(delta).astype(action.dtype, copy=False)
    return {"sparse": action, "dense": {}}


def repeat_plan(n_lift: int, n_other: int, target: float = TARGET_FRACTION):
    """How many copies of the lift list make `target` of the epoch."""
    if n_lift <= 0:
        raise RuntimeError("no lift chunks to oversample")
    slots = int(round(target / (1.0 - target) * n_other))
    slots = max(slots, n_lift)
    base, extra = divmod(slots, n_lift)
    return {
        "slots": slots,
        "base": base,
        "extra": extra,
        "weight": slots / n_lift,
        "fraction": slots / (slots + n_other),
        "epoch": slots + n_other,
    }


def expand_indices(indices, is_lift, target: float = TARGET_FRACTION):
    lift = [item for item, flag in zip(indices, is_lift) if flag]
    other = [item for item, flag in zip(indices, is_lift) if not flag]
    plan = repeat_plan(len(lift), len(other), target)
    out = list(other)
    for i, item in enumerate(lift):
        out.extend([item] * (plan["base"] + (1 if i < plan["extra"] else 0)))
    plan["lift"] = len(lift)
    plan["other"] = len(other)
    return out, plan


def _ends_by_episode(path: str) -> dict[int, np.ndarray]:
    blob = np.load(path)
    out: dict[int, list[int]] = {}
    for epi, idx in zip(blob["episode"], blob["index"]):
        out.setdefault(int(epi), []).append(int(idx))
    return {epi: np.asarray(idxs, dtype=np.int64) for epi, idxs in out.items()}


def _is_lift(sampler, ends) -> np.ndarray:
    shape = sampler.shape_meta
    horizon = int(shape["sample"]["action"]["sparse"]["horizon"])
    stride = int(shape["sample"]["action"]["sparse"]["down_sample_steps"])
    flags = np.zeros(len(sampler.indices), dtype=bool)
    for i, (epi, rgb) in enumerate(sampler.indices):
        epi = int(epi)
        hits = ends.get(epi)
        if hits is None or hits.size == 0:
            continue
        ep = sampler.replay_buffer["data"][f"episode_{epi}"]
        idx, _ = sparse_action_indices_from_rgb_timestamps(
            action_times=np.asarray(ep["action_time_stamps"]).reshape(-1),
            rgb_times=np.asarray(ep["obs"]["rgb_time_stamps_0"]).reshape(-1),
            rgb_id=int(rgb),
            action_horizon=horizon,
            rgb_frame_stride=stride,
        )
        if idx.size and np.intersect1d(idx, hits).size:
            flags[i] = True
    return flags


class UmiFTDatasetFieldV2NV3RLiftOS(UmiFTDatasetFieldV2):
    def __init__(self, *args, lift_index_path: str, lift_batch_fraction: float = TARGET_FRACTION, **kwargs):
        self._lift_index_path = str(lift_index_path)
        self._lift_fraction = float(lift_batch_fraction)
        super().__init__(*args, **kwargs)
        self.action_to_action_sample = action_liftos_to_action_sample
        self.sampler.action_to_action_sample = action_liftos_to_action_sample
        flags = _is_lift(self.sampler, _ends_by_episode(self._lift_index_path))
        expanded, plan = expand_indices(list(self.sampler.indices), flags, self._lift_fraction)
        self.sampler.indices = expanded
        self._lift_plan = plan
        print(
            f"[liftos] lift {plan['lift']}/{plan['lift'] + plan['other']} "
            f"weight {plan['weight']:.4f} copies {plan['base']}"
            f"+{plan['extra']} at {plan['base'] + 1} "
            f"epoch_fraction {plan['fraction']:.4f} epoch_len {plan['epoch']}",
            flush=True,
        )

    def get_validation_dataset(self):
        val = super().get_validation_dataset()
        val.action_to_action_sample = action_liftos_to_action_sample
        val.sampler.action_to_action_sample = action_liftos_to_action_sample
        return val


def _check() -> None:
    plan = repeat_plan(348, 10779, 0.15)
    if abs(plan["fraction"] - 0.15) > 5e-4:
        raise RuntimeError(f"fraction {plan['fraction']}")
    yaw = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    action = np.zeros((2, 10), dtype=np.float32)
    action[:, 2] = np.array([0.01, -0.002], dtype=np.float32)
    action[:, 4:10] = so3_to_rot6d_rows(np.stack([np.eye(3), yaw])).astype(np.float32)
    base = np.eye(4)
    got = action_liftos_to_action_sample(action, np.zeros((0,)), [0], [base])["sparse"]
    if not np.array_equal(got[:, :4], action[:, :4]):
        raise RuntimeError("δn column moved")
    if not np.allclose(rot6d_rows_to_R(got[:, 4:10])[1], yaw, atol=1e-5):
        raise RuntimeError("query delta did not decode")
    print(f"[liftos] check weight {plan['weight']:.4f} fraction {plan['fraction']:.4f}")


if __name__ == "__main__":
    _check()

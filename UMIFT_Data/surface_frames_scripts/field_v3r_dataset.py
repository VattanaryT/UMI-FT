"""Field v2 inputs and position. Rotation is relative to the query TCP.

ΔR = R_tcp(t).T @ R_tcp(t+k). The zarr stores absolute tool rot6d. This conversion
runs at sample time, which is when t is known. v1, v2, δn, and stiffness pass through.
"""

from __future__ import annotations

import numpy as np

from field_v2_dataset import UmiFTDatasetFieldV2
from rot6d import rot6d_rows_to_R, so3_to_rot6d_rows


def action_v3r_to_action_sample(action_sparse, action_dense, id_list, base_pose):
    """Leave position columns alone. Replace absolute rot6d with R(t).T @ R(t+k)."""
    del id_list
    if len(action_dense) > 0:
        raise NotImplementedError
    action = np.array(action_sparse, copy=True)
    if action.ndim != 2 or action.shape[1] != 10:
        raise ValueError(f"field_v3r action {getattr(action, 'shape', None)}")
    R0 = np.asarray(base_pose[0], dtype=np.float64)[:3, :3]
    future = rot6d_rows_to_R(np.asarray(action[:, 4:10], dtype=np.float64))
    delta = np.einsum("ji,njk->nik", R0, future)
    action[:, 4:10] = so3_to_rot6d_rows(delta).astype(action.dtype, copy=False)
    return {"sparse": action, "dense": {}}


class UmiFTDatasetFieldV3R(UmiFTDatasetFieldV2):
    """Same obs and position labels as field v2. Rotation matches r1rel."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_to_action_sample = action_v3r_to_action_sample
        self.sampler.action_to_action_sample = action_v3r_to_action_sample


def _check() -> None:
    R0 = np.eye(3)
    yaw = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    absolute = so3_to_rot6d_rows(yaw[None])
    action = np.zeros((1, 10), dtype=np.float32)
    action[0, 4:10] = absolute
    base = np.eye(4)
    base[:3, :3] = R0
    got = action_v3r_to_action_sample(action, np.zeros((0,)), [0], [base])["sparse"]
    if not np.allclose(got[0, :4], 0.0):
        raise RuntimeError("position columns moved")
    back = R0 @ rot6d_rows_to_R(got[:, 4:10])[0]
    if not np.allclose(back, yaw, atol=1e-5):
        raise RuntimeError("query-relative rotation did not decode")


if __name__ == "__main__":
    _check()
    print("field_v3r_dataset ok")

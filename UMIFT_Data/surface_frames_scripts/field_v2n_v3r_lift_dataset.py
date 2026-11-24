"""field_v2n observations and position. TCP-relative ΔR. δn is the lift gate.

The zarr stores absolute tool rot6d. At sample time that becomes
R_tcp(t).T @ R_tcp(t+k). v1, v2, δn, and stiffness pass through.
pose_frames.action must be relative so base_pose carries the query TCP rotation.
Observations stay absolute.
"""

from __future__ import annotations

import numpy as np

from field_v2_dataset import UmiFTDatasetFieldV2
from rot6d import rot6d_rows_to_R, so3_to_rot6d_rows


def action_lift_to_action_sample(action_sparse, action_dense, id_list, base_pose):
    del id_list
    if len(action_dense) > 0:
        raise NotImplementedError
    action = np.array(action_sparse, copy=True)
    if action.ndim != 2 or action.shape[1] != 10:
        raise ValueError(f"lift action {getattr(action, 'shape', None)}")
    R0 = np.asarray(base_pose[0], dtype=np.float64)[:3, :3]
    future = rot6d_rows_to_R(np.asarray(action[:, 4:10], dtype=np.float64))
    delta = np.einsum("ji,njk->nik", R0, future)
    action[:, 4:10] = so3_to_rot6d_rows(delta).astype(action.dtype, copy=False)
    return {"sparse": action, "dense": {}}


class UmiFTDatasetFieldV2NV3RLift(UmiFTDatasetFieldV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_to_action_sample = action_lift_to_action_sample
        self.sampler.action_to_action_sample = action_lift_to_action_sample

    def get_validation_dataset(self):
        val = super().get_validation_dataset()
        val.action_to_action_sample = action_lift_to_action_sample
        val.sampler.action_to_action_sample = action_lift_to_action_sample
        return val


def _check() -> None:
    yaw = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    action = np.zeros((2, 10), dtype=np.float32)
    action[:, 2] = np.array([0.01, -0.002], dtype=np.float32)
    action[:, 4:10] = so3_to_rot6d_rows(np.stack([np.eye(3), yaw])).astype(np.float32)
    base = np.eye(4)
    got = action_lift_to_action_sample(action, np.zeros((0,)), [0], [base])["sparse"]
    if not np.array_equal(got[:, :4], action[:, :4]):
        raise RuntimeError("position columns moved")
    if not np.allclose(rot6d_rows_to_R(got[:, 4:10])[1], yaw, atol=1e-5):
        raise RuntimeError("query delta did not decode")


if __name__ == "__main__":
    _check()
    print("field_v2n_v3r_lift_dataset ok")

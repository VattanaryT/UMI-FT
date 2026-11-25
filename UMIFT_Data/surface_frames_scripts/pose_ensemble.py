"""Pose-space ensemble for r4vt. Off unless UMIFT_POSE_ENSEMBLE is set.

The chunk is integrated to SE3 first. Overlapping targets are then averaged
the way standard_bc does: positions by a weighted mean, rotations by the
sign-aligned quaternion sum in PoseChunkEnsembler. Decay is 0.005.
"""

from __future__ import annotations

import copy
import os

import numpy as np

DECAY_M = 0.005


def enabled(environ=None) -> bool:
    source = os.environ if environ is None else environ
    return str(source.get("UMIFT_POSE_ENSEMBLE", "")).strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _copy_carry(value):
    """Copy carry state. The train-frame tracker holds an Open3D scene, which
    deepcopy cannot pickle. That object is shared; the numpy carry still copies.
    """
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def snapshot_carry(runtime):
    """Copy the runtime state a full-chunk decode would otherwise leave behind."""
    return {
        name: _copy_carry(value)
        for name, value in vars(runtime).items()
        if name.startswith("_")
    }


def restore_carry(runtime, snap) -> None:
    for name, value in snap.items():
        setattr(runtime, name, _copy_carry(value))


def pose7_base_to_fixture(pose7, T_fixture_from_base) -> np.ndarray:
    from fixb_labels import Rp_to_pose7, pose7_to_Rp

    rotation, position = pose7_to_Rp(np.asarray(pose7, dtype=np.float64).reshape(-1, 7))
    transform = np.asarray(T_fixture_from_base, dtype=np.float64)
    rotation_fix = np.einsum("ij,njk->nik", transform[:3, :3], rotation)
    position_fix = (transform[:3, :3] @ position.T).T + transform[:3, 3]
    return Rp_to_pose7(rotation_fix, position_fix)


def apply_blended_carry(runtime, vt_pose7, nominal_pose7, T_fixture_from_base) -> None:
    """Next chunk slews from the pose that was actually scheduled."""
    from fixb_labels import pose7_to_Rp

    virtual = pose7_base_to_fixture(vt_pose7, T_fixture_from_base).reshape(-1, 7)
    nominal = pose7_base_to_fixture(nominal_pose7, T_fixture_from_base).reshape(-1, 7)
    rotation, position = pose7_to_Rp(virtual[-1])
    _, foot = pose7_to_Rp(nominal[-1])
    runtime._cmd_vt = np.asarray(position, dtype=np.float64).reshape(3).copy()
    runtime._cmd_foot = np.asarray(foot, dtype=np.float64).reshape(3).copy()
    runtime.note_commanded_rd(rotation[0])

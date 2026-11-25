"""Shift the live tool frame along tool z. Off unless UMIFT_TCP_OFFSET_MM is set.

The liftos labels use the wand TCP. A longer pendant TCP sits further along
tool +z than that blade. A negative offset moves the query and the decode
onto the blade. Commands sent to the robot are shifted back, so the pendant
TCP is what the controller moves.
"""

from __future__ import annotations

import os

import numpy as np


def millimeters(environ=None) -> float:
    source = os.environ if environ is None else environ
    raw = str(source.get("UMIFT_TCP_OFFSET_MM", "")).strip()
    if not raw:
        return 0.0
    return float(raw)


def shift_pose7(pose7, offset_mm: float) -> np.ndarray:
    """Move each pose by ``offset_mm`` along its own tool +z. Rotation stays."""
    poses = np.asarray(pose7, dtype=np.float64)
    single = poses.ndim == 1
    if offset_mm == 0.0:
        return poses.reshape(1, 7).copy() if single else poses.copy()
    from fixb_labels import Rp_to_pose7, pose7_to_Rp

    rotation, position = pose7_to_Rp(poses.reshape(-1, 7))
    position = position + rotation @ np.array([0.0, 0.0, float(offset_mm) / 1000.0])
    shifted = Rp_to_pose7(rotation, position)
    return shifted.reshape(7) if single else shifted

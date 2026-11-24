"""Rot6d pack and unpack. Numpy only, so training does not import open3d.

Same math as fixb_labels.so3_to_rot6d_rows and mold_aware_runtime.rot6d_rows_to_R.
"""

from __future__ import annotations

import numpy as np


def so3_to_rot6d_rows(R: np.ndarray) -> np.ndarray:
    """UMI-FT spatial_utilities.SO3_to_rot6d: first two rows."""
    return np.asarray(R).reshape(-1, 3, 3)[:, :2, :].reshape(-1, 6).copy()


def rot6d_rows_to_R(r6: np.ndarray) -> np.ndarray:
    """Gram-Schmidt. Columns of the result are the rotation's axes."""
    a1 = r6[..., 0:3]
    a2 = r6[..., 3:6]
    b1 = a1 / np.clip(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-9, None)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.clip(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-9, None)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2)

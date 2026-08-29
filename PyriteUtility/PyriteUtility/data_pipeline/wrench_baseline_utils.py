"""Utilities for wrench baseline removal and time-based smoothing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Tuple

import numpy as np

BaselineMode = Literal["first_n", "session_tare"]


def load_session_wrench_offset(offset_path: str | Path) -> np.ndarray:
    """Load a 6D wrench offset from JSON (list or {'wrench_offset': [...]})."""
    path = Path(offset_path)
    with path.open("r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "wrench_offset" in data:
        offset = np.asarray(data["wrench_offset"], dtype=np.float64)
    else:
        offset = np.asarray(data, dtype=np.float64)
    if offset.shape != (6,):
        raise ValueError(f"session offset must be 6D, got shape {offset.shape}")
    return offset


def save_session_wrench_offset(
    offset: np.ndarray,
    output_path: str | Path,
    metadata: dict | None = None,
) -> None:
    """Save a 6D wrench offset for session_tare baseline mode."""
    offset = np.asarray(offset, dtype=np.float64).reshape(6)
    payload = {
        "wrench_offset": offset.tolist(),
        "description": "Free-space wrench offset subtracted during postprocessing",
    }
    if metadata:
        payload.update(metadata)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def compute_first_n_sample_count(
    timestamps: np.ndarray,
    duration_s: float,
    min_samples: int = 1,
) -> int:
    """Convert a target duration to a sample count using median dt."""
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if ts.size < 2:
        return max(min_samples, ts.size)
    dts = np.diff(ts)
    dts = dts[dts > 0]
    if dts.size == 0:
        return max(min_samples, ts.size)
    median_dt = float(np.median(dts))
    n = int(np.ceil(duration_s / median_dt))
    return max(min_samples, min(n, ts.size))


def subtract_wrench_baseline(
    wrench: np.ndarray,
    timestamps: np.ndarray,
    baseline_mode: BaselineMode = "first_n",
    baseline_duration_s: float = 1.0,
    session_offset_path: str | Path | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove wrench DC bias.

    Returns (corrected_wrench, baseline_vector).
    """
    wrench = np.asarray(wrench, dtype=np.float64)
    if baseline_mode == "session_tare":
        if session_offset_path is None:
            raise ValueError("session_tare requires session_offset_path")
        baseline = load_session_wrench_offset(session_offset_path)
    elif baseline_mode == "first_n":
        n = compute_first_n_sample_count(timestamps, baseline_duration_s)
        baseline = np.mean(wrench[:n], axis=0)
    else:
        raise ValueError(f"Unknown baseline_mode: {baseline_mode}")
    return wrench - baseline, baseline


def time_based_moving_average(
    wrench: np.ndarray,
    timestamps: np.ndarray,
    window_duration_s: float = 1.0,
    min_periods: int = 1,
) -> np.ndarray:
    """
    Edge-preserving moving average using reflection padding at boundaries.
    """
    wrench = np.asarray(wrench, dtype=np.float64)
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    n = wrench.shape[0]
    if n == 0:
        return wrench.copy()

    window_samples = compute_first_n_sample_count(ts, window_duration_s, min_samples=1)
    half = max(1, window_samples // 2)

    padded = np.pad(wrench, ((half, half), (0, 0)), mode="reflect")
    kernel = np.ones(window_samples, dtype=np.float64) / window_samples
    smoothed = np.empty_like(wrench)
    for axis in range(6):
        conv = np.convolve(padded[:, axis], kernel, mode="valid")
        smoothed[:, axis] = conv[:n]

    if min_periods > 1 and n < window_samples:
        # Fall back to expanding mean for very short episodes
        for i in range(n):
            start = max(0, i - half)
            end = min(n, i + half + 1)
            smoothed[i] = np.mean(wrench[start:end], axis=0)

    return smoothed


def _quat_wxyz_rotate_vectors(quat_wxyz: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    """Rotate vectors by pose7 quaternions (wxyz). quat (N,4), vecs (N,3) -> (N,3)."""
    w = quat_wxyz[:, 0]
    x = quat_wxyz[:, 1]
    y = quat_wxyz[:, 2]
    z = quat_wxyz[:, 3]
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    r00 = ww + xx - yy - zz
    r01 = 2 * (xy - wz)
    r02 = 2 * (xz + wy)
    r10 = 2 * (xy + wz)
    r11 = ww - xx + yy - zz
    r12 = 2 * (yz - wx)
    r20 = 2 * (xz - wy)
    r21 = 2 * (yz + wx)
    r22 = ww - xx - yy + zz
    return np.stack(
        [
            r00 * vecs[:, 0] + r01 * vecs[:, 1] + r02 * vecs[:, 2],
            r10 * vecs[:, 0] + r11 * vecs[:, 1] + r12 * vecs[:, 2],
            r20 * vecs[:, 0] + r21 * vecs[:, 1] + r22 * vecs[:, 2],
        ],
        axis=-1,
    )


def check_virtual_target_push_direction(
    ts_pose_fb: np.ndarray,
    ts_pose_virtual_target: np.ndarray,
    wrench_tcp: np.ndarray,
) -> dict:
    """
    Check that virtual targets lie on the far side of the command along dominant force.
    Returns summary metrics for validation logging.

    Force is rotated from tool/TCP frame into the world frame using ``ts_pose_fb``
    so it is comparable to the world-frame pose delta (VT - fb).
    """
    pose_fb = np.asarray(ts_pose_fb, dtype=np.float64)
    pose_vt = np.asarray(ts_pose_virtual_target, dtype=np.float64)
    wrench = np.asarray(wrench_tcp, dtype=np.float64)

    delta = pose_vt[:, :3] - pose_fb[:, :3]
    # wrench_tcp is tool-frame; delta is world-frame
    force = _quat_wxyz_rotate_vectors(pose_fb[:, 3:7], wrench[:, :3])
    force_norm = np.linalg.norm(force, axis=1)
    delta_norm = np.linalg.norm(delta, axis=1)
    valid = (force_norm > 2.0) & (delta_norm > 1e-9)
    if not np.any(valid):
        return {"push_alignment": 0.0, "valid_samples": 0.0}

    cos = np.sum(delta[valid] * force[valid], axis=1) / (
        delta_norm[valid] * force_norm[valid]
    )
    # virtual target should oppose measured wrench (f=-wrench in estimator)
    return {
        "push_alignment": float(np.mean(-cos)),
        "valid_samples": float(np.sum(valid)),
    }

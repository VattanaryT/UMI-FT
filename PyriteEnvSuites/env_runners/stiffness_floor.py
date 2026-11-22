"""Press-axis stiffness floor applied when a chunk is scheduled.

Off unless ``UMIFT_STIFFNESS_FLOOR`` is a finite stiffness in N/m. The floor
raises the compliance-axis gain. It does not change the other two axes.
"""

from __future__ import annotations

import os
from typing import Mapping, Optional, Tuple

import numpy as np

_ENV = "UMIFT_STIFFNESS_FLOOR"


def read_press_stiffness_floor(
    environ: Optional[Mapping[str, str]] = None,
) -> Optional[float]:
    """Return the floor in N/m, or None when the flag is unset."""
    source = os.environ if environ is None else environ
    text = str(source.get(_ENV, "")).strip()
    if text == "":
        return None
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"{_ENV}={text!r} is not a stiffness in N/m") from exc
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{_ENV}={text!r} must be a finite stiffness >= 0")
    return float(value)


def apply_press_stiffness_floor(
    stiffness, floor: Optional[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(raw, floored)`` along the press axis. ``floor is None`` copies raw."""
    raw = np.asarray(stiffness, dtype=np.float64).reshape(-1).copy()
    if floor is None:
        return raw, raw.copy()
    return raw, np.maximum(raw, float(floor))


def write_stiffness_trace(action_log, robot_id: int, raw, floored) -> None:
    """Store the scheduled press-axis stiffness, raw and after the floor."""
    action_log.create_dataset(
        f"stiffness_raw_{robot_id}",
        data=np.asarray(raw, dtype=np.float64).reshape(-1),
    )
    action_log.create_dataset(
        f"stiffness_floored_{robot_id}",
        data=np.asarray(floored, dtype=np.float64).reshape(-1),
    )

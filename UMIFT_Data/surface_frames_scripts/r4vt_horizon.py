"""r4vt horizon decode. The closed-loop sim runner imports this.

Position is the integral of tangential v. dt is the label builder's control
step (the next RGB sample), not the robot-sample step. δn is the force target
and is not a position. Rotation at every step is R_field @ ΔR, which is how
the label builder stored it.
"""

from __future__ import annotations

import numpy as np

from fixb_labels import next_control_times
from mold_aware_runtime import rot6d_rows_to_R


def control_dts(robot_t, rgb_t, index) -> np.ndarray:
    """Seconds from each action sample to the next control sample."""
    robot_t = np.asarray(robot_t, dtype=np.float64).reshape(-1)
    dt = next_control_times(robot_t, np.asarray(rgb_t, dtype=np.float64)) - robot_t
    return dt[np.asarray(index, dtype=np.int64)]


def integrate_tangent(v, frames, dts, p0) -> np.ndarray:
    """Tool position. xyz[0] is p0. Later steps add B(v dt). δn is ignored."""
    v = np.asarray(v, dtype=np.float64).reshape(-1, 2)
    frames = np.asarray(frames, dtype=np.float64).reshape(len(v), 3, 3)
    dts = np.asarray(dts, dtype=np.float64).reshape(-1)
    if len(dts) != len(v):
        raise ValueError(f"dt length {len(dts)} != horizon {len(v)}")
    p = np.asarray(p0, dtype=np.float64).reshape(3).copy()
    xyz = np.empty((len(v), 3), dtype=np.float64)
    for k in range(len(v)):
        xyz[k] = p
        p = p + frames[k, :, :2] @ (v[k] * float(dts[k]))
    return xyz


def compose_field_delta(frames, delta_rot6d) -> np.ndarray:
    """R_tool = R_field @ ΔR at every horizon step."""
    frames = np.asarray(frames, dtype=np.float64).reshape(-1, 3, 3)
    delta = rot6d_rows_to_R(np.asarray(delta_rot6d, dtype=np.float64))
    if len(delta) != len(frames):
        raise ValueError(f"ΔR {len(delta)} != frames {len(frames)}")
    return np.einsum("nij,njk->nik", frames, delta)


def decode_horizon(action, p0, frames, dts):
    """xyz (H, 3), R_tool (H, 3, 3) for a 10-d r4vt action."""
    action = np.asarray(action, dtype=np.float64)
    if action.ndim != 2 or action.shape[1] < 10:
        raise ValueError(f"r4vt action {action.shape}")
    frames = np.asarray(frames, dtype=np.float64).reshape(len(action), 3, 3)
    xyz = integrate_tangent(action[:, 0:2], frames, dts, p0)
    return xyz, compose_field_delta(frames, action[:, 4:10])

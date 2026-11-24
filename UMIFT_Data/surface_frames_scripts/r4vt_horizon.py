"""r4vt horizon decode. The closed-loop sim runner imports this.

Position is the integral of tangential v. dt is the label builder's control
step (the next RGB sample), not the robot-sample step. In contact, δn is the
force target and is not a position. Out of contact, pass air=True and the foot:
the arrival TCP's offset along the outward frame normal is δn. Rotation at
every step is R_field @ ΔR, which is how the label builder stored it.
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


def set_clearance(point, foot, frame, clearance) -> np.ndarray:
    """Move point along the outward normal. Frame z is inward. Positive is away."""
    outward = -np.asarray(frame, dtype=np.float64).reshape(3, 3)[:, 2]
    point = np.asarray(point, dtype=np.float64).reshape(3)
    foot = np.asarray(foot, dtype=np.float64).reshape(3)
    gap = float(np.dot(point - foot, outward))
    return point + outward * (float(clearance) - gap)


def decode_horizon(action, p0, frames, dts, air=None, feet=None):
    """xyz (H, 3), R_tool (H, 3, 3) for a 10-d r4vt action.

    air is None: xyz[k] is the TCP at the start of step k, and δn is ignored.
    air is set: xyz[k] is the TCP after step k. Air steps sit at clearance δn.
    Contact steps keep the tangent integral and leave the normal alone.
    """
    action = np.asarray(action, dtype=np.float64)
    if action.ndim != 2 or action.shape[1] < 10:
        raise ValueError(f"r4vt action {action.shape}")
    frames = np.asarray(frames, dtype=np.float64).reshape(len(action), 3, 3)
    rotation = compose_field_delta(frames, action[:, 4:10])
    if air is None:
        return integrate_tangent(action[:, 0:2], frames, dts, p0), rotation
    air = np.asarray(air, dtype=bool).reshape(-1)
    if feet is None or len(air) != len(action):
        raise ValueError(f"air decode needs one foot per step ({len(air)} vs {len(action)})")
    feet = np.asarray(feet, dtype=np.float64).reshape(len(action), 3)
    v = action[:, 0:2]
    dts = np.asarray(dts, dtype=np.float64).reshape(-1)
    p = np.asarray(p0, dtype=np.float64).reshape(3).copy()
    xyz = np.empty((len(v), 3), dtype=np.float64)
    for k in range(len(v)):
        p = p + frames[k, :, :2] @ (v[k] * float(dts[k]))
        if air[k]:
            p = set_clearance(p, feet[k], frames[k], action[k, 2])
        xyz[k] = p
    return xyz, rotation

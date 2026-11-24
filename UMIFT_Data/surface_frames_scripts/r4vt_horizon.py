"""r4vt horizon decode. The closed-loop sim runner imports this.

Position is the integral of tangential v. dt is the label builder's control
step (the next RGB sample), not the robot-sample step. With no extra
arguments, δn is not a position and rotation is R_field @ ΔR, the
surface_frame decode. air/feet selects the field_v2n clearance path.
R_tcp selects field_v3r: R_tcp @ ΔR.
"""

from __future__ import annotations

import numpy as np

from fixb_labels import interp_rows, next_control_times
from mold_aware_runtime import rot6d_rows_to_R


def control_dts(robot_t, rgb_t, index) -> np.ndarray:
    """Seconds from each action sample to the next control sample."""
    robot_t = np.asarray(robot_t, dtype=np.float64).reshape(-1)
    dt = next_control_times(robot_t, np.asarray(rgb_t, dtype=np.float64)) - robot_t
    return dt[np.asarray(index, dtype=np.int64)]


def arrival_tcp(tool_p, robot_t, rgb_t, index) -> np.ndarray:
    """Demo TCP where each action lands. Action k covers [t_k, t_k+]."""
    robot_t = np.asarray(robot_t, dtype=np.float64).reshape(-1)
    index = np.asarray(index, dtype=np.int64)
    t_next = next_control_times(robot_t, np.asarray(rgb_t, dtype=np.float64))[index]
    return interp_rows(t_next, robot_t, np.asarray(tool_p, dtype=np.float64))


def arrival_frame(frames, robot_t, rgb_t, index) -> np.ndarray:
    """Field frame at the robot sample each action lands on."""
    robot_t = np.asarray(robot_t, dtype=np.float64).reshape(-1)
    index = np.asarray(index, dtype=np.int64)
    t_next = next_control_times(robot_t, np.asarray(rgb_t, dtype=np.float64))[index]
    landed = np.clip(np.searchsorted(robot_t, t_next), 0, len(robot_t) - 1)
    return np.asarray(frames, dtype=np.float64)[landed]


def integrate_tangent(v, frames, dts, p0) -> np.ndarray:
    """xyz[k] is the pose after action k. δn is ignored."""
    v = np.asarray(v, dtype=np.float64).reshape(-1, 2)
    frames = np.asarray(frames, dtype=np.float64).reshape(len(v), 3, 3)
    dts = np.asarray(dts, dtype=np.float64).reshape(-1)
    if len(dts) != len(v):
        raise ValueError(f"dt length {len(dts)} != horizon {len(v)}")
    p = np.asarray(p0, dtype=np.float64).reshape(3).copy()
    xyz = np.empty((len(v), 3), dtype=np.float64)
    for k in range(len(v)):
        p = p + frames[k, :, :2] @ (v[k] * float(dts[k]))
        xyz[k] = p
    return xyz


def _integrate_query(v, frames, dts, p0) -> np.ndarray:
    """field_v3r position. xyz[0] is p0. Later steps add B(v dt)."""
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


def compose_tcp_delta(R_tcp, delta_rot6d) -> np.ndarray:
    """R_next = R_tcp @ ΔR. ΔR is relative to the query TCP, as in r1rel."""
    R0 = np.asarray(R_tcp, dtype=np.float64).reshape(3, 3)
    delta = rot6d_rows_to_R(np.asarray(delta_rot6d, dtype=np.float64))
    return np.einsum("ij,njk->nik", R0, delta)


def set_clearance(point, foot, frame, clearance) -> np.ndarray:
    """Move point along the outward normal. Frame z is inward. Positive is away."""
    outward = -np.asarray(frame, dtype=np.float64).reshape(3, 3)[:, 2]
    point = np.asarray(point, dtype=np.float64).reshape(3)
    foot = np.asarray(foot, dtype=np.float64).reshape(3)
    gap = float(np.dot(point - foot, outward))
    return point + outward * (float(clearance) - gap)


def decode_horizon(action, p0, frames, dts, R_tcp=None, air=None, feet=None):
    """xyz (H, 3), R_tool (H, 3, 3) for a 10-d r4vt action.

    No extra arguments: surface_frame. xyz[k] is the TCP after step k, and
    rotation is R_field @ ΔR. R_tcp is the fifth argument, matching field_v3r:
    rotation is R_tcp @ ΔR and xyz[0] is the query position. air/feet, by
    keyword, select the field_v2n clearance path.
    """
    action = np.asarray(action, dtype=np.float64)
    if action.ndim != 2 or action.shape[1] < 10:
        raise ValueError(f"r4vt action {action.shape}")
    frames = np.asarray(frames, dtype=np.float64).reshape(len(action), 3, 3)
    if air is not None:
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
        if R_tcp is None:
            rotation = compose_field_delta(frames, action[:, 4:10])
        else:
            rotation = compose_tcp_delta(R_tcp, action[:, 4:10])
        return xyz, rotation
    if R_tcp is None:
        xyz = integrate_tangent(action[:, 0:2], frames, dts, p0)
        return xyz, compose_field_delta(frames, action[:, 4:10])
    xyz = _integrate_query(action[:, 0:2], frames, dts, p0)
    return xyz, compose_tcp_delta(R_tcp, action[:, 4:10])

"""TCP velocity on a foot that cannot hop to a distant face.

The search at each sample starts from the previous foot. A global closest point
is accepted only when it lands nearby and its outward normal still agrees.
Otherwise the foot walks along the current tangent toward the tool.
"""

from __future__ import annotations

import numpy as np

from fixb_labels import interp_rows, next_control_times

# One robot sample is ~5 ms. A real foot step is about 1 mm. 6 mm accepts that
# and a fast sample, and is too short to reach the other sheet of the fillet.
GATE_M = 0.006
MAX_FOOT_STEP_M = 0.006
NORMAL_COS = float(np.cos(np.deg2rad(40.0)))
WALK_STEP_M = 0.002
WALK_MAX_STEPS = 8


def project_tcp_velocity(
    R: np.ndarray, p: np.ndarray, robot_t: np.ndarray, rgb_t: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project the measured TCP step onto the frame's t1, t2.

    The step is the control interval used by the existing labels (the next RGB
    sample), not the foot's own displacement. Returns v (N, 2) and dt (N,).
    """
    p = np.asarray(p, dtype=np.float64).reshape(-1, 3)
    R = np.asarray(R, dtype=np.float64).reshape(-1, 3, 3)
    t_next = next_control_times(robot_t, rgb_t)
    dt = t_next - robot_t
    p_next = interp_rows(t_next, robot_t, p)
    vel = np.zeros_like(p)
    ok = dt > 1e-6
    vel[ok] = (p_next[ok] - p[ok]) / dt[ok, None]
    v = np.einsum("nia,ni->na", R[..., :2], vel)
    return v, dt


def _outward(R: np.ndarray) -> np.ndarray:
    return -np.asarray(R, dtype=np.float64).reshape(3, 3)[:, 2]


def accept_local(x_free: np.ndarray, R_free: np.ndarray, x_prev: np.ndarray, R_prev: np.ndarray) -> bool:
    """True when the global closest point is still the same neighborhood."""
    if float(np.linalg.norm(x_free - x_prev)) > GATE_M:
        return False
    return float(np.dot(_outward(R_free), _outward(R_prev))) >= NORMAL_COS


def _walk(query, p: np.ndarray, x: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Step along the tangent. Stop rather than jump to another sheet."""
    n = _outward(R)
    delta = np.asarray(p, dtype=np.float64).reshape(3) - x
    tang = delta - np.dot(delta, n) * n
    origin = np.array(x, copy=True)
    full = float(np.linalg.norm(tang))
    dist = min(full, MAX_FOOT_STEP_M)
    if dist < 1e-5:
        return x, R
    steps = min(WALK_MAX_STEPS, int(np.ceil(dist / WALK_STEP_M)))
    step = tang / full * (dist / steps)
    for _ in range(steps):
        guess = x + step
        xh, Rh = query(guess.reshape(1, 3))
        xh = np.asarray(xh, dtype=np.float64).reshape(3)
        Rh = np.asarray(Rh, dtype=np.float64).reshape(3, 3)
        if float(np.linalg.norm(xh - origin)) > MAX_FOOT_STEP_M + 1e-6:
            break
        if float(np.linalg.norm(xh - x)) > WALK_STEP_M + 0.002:
            break
        if float(np.dot(_outward(Rh), n)) < np.cos(np.deg2rad(25.0)):
            break
        if float(np.linalg.norm(p - xh)) > float(np.linalg.norm(p - x)) + 1e-4:
            break
        x, R, n = xh, Rh, _outward(Rh)
    return x, R


def step_foot(query, p, x_prev, R_prev, x_free, R_free):
    """One sample. ``x_free`` is the global closest point of this TCP."""
    x_free = np.asarray(x_free, dtype=np.float64).reshape(3)
    R_free = np.asarray(R_free, dtype=np.float64).reshape(3, 3)
    x_prev = np.asarray(x_prev, dtype=np.float64).reshape(3)
    R_prev = np.asarray(R_prev, dtype=np.float64).reshape(3, 3)
    if accept_local(x_free, R_free, x_prev, R_prev):
        return x_free, R_free
    return _walk(query, np.asarray(p, dtype=np.float64).reshape(3), x_prev, R_prev)


def track_local_feet(query, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Feet for a TCP series. ``query(points) -> (x_s, R)`` uses the smooth field.

    The first sample is the global closest point. Later samples stay near the
    previous foot. Returns x (N, 3), R (N, 3, 3), and how many samples refused
    the global closest point.
    """
    p = np.asarray(p, dtype=np.float64).reshape(-1, 3)
    x_free, R_free = query(p)
    x_free = np.asarray(x_free, dtype=np.float64).reshape(-1, 3)
    R_free = np.asarray(R_free, dtype=np.float64).reshape(-1, 3, 3)
    n = len(p)
    x = np.empty((n, 3), dtype=np.float64)
    R = np.empty((n, 3, 3), dtype=np.float64)
    if n == 0:
        return x, R, 0
    x[0], R[0] = x_free[0], R_free[0]
    held = 0
    for i in range(1, n):
        if not accept_local(x_free[i], R_free[i], x[i - 1], R[i - 1]):
            held += 1
        x[i], R[i] = step_foot(query, p[i], x[i - 1], R[i - 1], x_free[i], R_free[i])
    return x, R, held

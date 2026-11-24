"""Air/contact gate chosen by sweep_lift_gate.py.

0.5 s wrench smoothing. Contact latches on above 6 N and off below 3 N.
In contact, δn is -(F·n)/3000. In the air it is the TCP clearance.
"""

from __future__ import annotations

import numpy as np

ON_N = 6.0
OFF_N = 3.0
WINDOW_S = 0.5
K = 3000.0


def hysteresis(magnitude: np.ndarray, on_n: float = ON_N, off_n: float = OFF_N) -> np.ndarray:
    """True while contact is latched. Starts in the air."""
    mag = np.asarray(magnitude, dtype=np.float64).reshape(-1)
    contact = np.zeros(len(mag), dtype=bool)
    state = False
    for i, value in enumerate(mag):
        if state:
            if value < off_n:
                state = False
        elif value > on_n:
            state = True
        contact[i] = state
    return contact


def delta_n(force_n: np.ndarray, clearance_m: np.ndarray, contact: np.ndarray) -> np.ndarray:
    depth = np.asarray(clearance_m, dtype=np.float64).reshape(-1).copy()
    depth[contact] = -np.asarray(force_n, dtype=np.float64).reshape(-1)[contact] / K
    return depth


def crossings(contact: np.ndarray) -> np.ndarray:
    """Indices of the last contact sample before the latch opens."""
    contact = np.asarray(contact, dtype=bool).reshape(-1)
    return np.flatnonzero(contact[:-1] & ~contact[1:])


def isolated(contact: np.ndarray, time_s: np.ndarray, hold_s: float = 0.5) -> np.ndarray:
    """Releases that are not followed by contact again within hold_s."""
    time_s = np.asarray(time_s, dtype=np.float64).reshape(-1)
    ends = crossings(contact)
    keep = []
    for end in ends:
        later = np.flatnonzero(contact[end + 1:])
        if later.size == 0 or float(time_s[end + 1 + later[0]] - time_s[end]) > hold_s:
            keep.append(int(end))
    return np.asarray(keep, dtype=np.int64)


def _check() -> None:
    mag = np.array([0.0, 4.5, 6.0, 4.0, 3.5, 2.0, 6.0, 2.5])
    got = hysteresis(mag, 5.0, 3.0)
    expect = np.array([False, False, True, True, True, False, True, False])
    if not np.array_equal(got, expect):
        raise RuntimeError(f"hysteresis {got} != {expect}")
    depth = delta_n(np.array([6.0, 4.0]), np.array([0.02, 0.02]), got[:2])
    if abs(depth[0] - 0.02) > 1e-12 or abs(depth[1] - 0.02) > 1e-12:
        raise RuntimeError("air δn was not clearance")
    latched = delta_n(np.array([6.0]), np.array([0.02]), np.array([True]))
    if abs(latched[0] + 6.0 / K) > 1e-12:
        raise RuntimeError("contact δn was not -F/K")


if __name__ == "__main__":
    _check()
    print("lift_gate ok")

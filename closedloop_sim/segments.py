"""Contact, approach, lift-off, and air-travel labels.

Contact is |F| > 5 N. Air-travel is |F| < 2 N and clearance > 10 mm.
Lift-off is the band after contact until air-travel. Approach is the band
after air-travel (or from the start) until contact.
"""

from __future__ import annotations

import numpy as np

CONTACT_N = 5.0
AIR_N = 2.0
AIR_CLEARANCE_M = 0.010
LIFT_WINDOW_S = 1.0
NAMES = ("contact", "approach", "lift-off", "air-travel")


def label_segments(force_n, clearance_m) -> np.ndarray:
    force = np.asarray(force_n, dtype=np.float64).reshape(-1)
    clearance = np.asarray(clearance_m, dtype=np.float64).reshape(-1)
    labels = np.empty(force.shape[0], dtype=object)
    phase = "approach"
    for i, (f, c) in enumerate(zip(force, clearance)):
        if f > CONTACT_N:
            phase = "contact"
        elif f < AIR_N and c > AIR_CLEARANCE_M:
            phase = "air-travel"
        elif phase == "contact":
            phase = "lift-off"
        elif phase == "air-travel":
            phase = "approach"
        labels[i] = phase
    return labels


def run_span(demo: dict, mode: str) -> tuple[float, float, np.ndarray]:
    """Full episode, or from the first contact sample to the first lift-off."""
    if mode == "full":
        return 0.0, float(demo["duration_s"]), np.asarray(demo["pose0"], dtype=np.float64)
    if mode != "contact":
        raise ValueError(mode)
    segment = np.asarray(demo["segment"])
    contact = np.flatnonzero(segment == "contact")
    if contact.size == 0:
        raise RuntimeError("demo has no |F|>5 N sample")
    i0 = int(contact[0])
    lifts = np.flatnonzero(segment == "lift-off")
    later = lifts[lifts > i0]
    t_end = float(demo["t_s"][int(later[0])]) if later.size else float(demo["duration_s"])
    return float(demo["t_s"][i0]), t_end, np.asarray(demo["pose7"][i0], dtype=np.float64)


def _starts(t, labels, name):
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    on = np.asarray(labels) == name
    if on.size == 0:
        return np.zeros(0)
    prev = np.concatenate([[False], on[:-1]])
    return t[on & ~prev]


def _ends(t, labels, name):
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    on = np.asarray(labels) == name
    if on.size == 0:
        return np.zeros(0)
    nxt = np.concatenate([on[1:], [False]])
    return t[on & ~nxt]


def events(demo_t, demo_labels, demo_clearance, demo_xyz, sim_t, sim_clearance, sim_xyz) -> dict:
    """Lift-off success, lift-height error, and the air-travel end-point error."""
    lifts = _starts(demo_t, demo_labels, "lift-off")
    demo_t = np.asarray(demo_t, dtype=np.float64).reshape(-1)
    sim_t = np.asarray(sim_t, dtype=np.float64).reshape(-1)
    demo_c = np.asarray(demo_clearance, dtype=np.float64).reshape(-1)
    sim_c = np.asarray(sim_clearance, dtype=np.float64).reshape(-1)
    demo_xyz = np.asarray(demo_xyz, dtype=np.float64).reshape(-1, 3)
    sim_xyz = np.asarray(sim_xyz, dtype=np.float64).reshape(-1, 3)
    covered = float(sim_t[-1]) if sim_t.size else -1.0
    lift_rows = []
    for t0 in lifts:
        if sim_t.size == 0 or t0 + LIFT_WINDOW_S > covered + 1e-6 or t0 < float(sim_t[0]) - 1e-6:
            lift_rows.append(
                {"t_s": float(t0), "success": None, "lift_height_error_mm": None, "in_run": False}
            )
            continue
        sim_w = (sim_t >= t0) & (sim_t <= t0 + LIFT_WINDOW_S)
        demo_w = (demo_t >= t0) & (demo_t <= t0 + LIFT_WINDOW_S)
        got = sim_c[sim_w]
        ref = demo_c[demo_w]
        success = bool(got.size and np.max(got) > AIR_CLEARANCE_M)
        err = None
        if got.size and ref.size:
            err = float((np.max(got) - np.max(ref)) * 1e3)
        lift_rows.append(
            {
                "t_s": float(t0),
                "success": success,
                "lift_height_error_mm": err,
                "in_run": True,
            }
        )
    ends = []
    for t1 in _ends(demo_t, demo_labels, "air-travel"):
        if sim_t.size == 0 or t1 < float(sim_t[0]) or t1 > covered:
            continue
        i = int(np.argmin(np.abs(sim_t - t1)))
        j = int(np.argmin(np.abs(demo_t - t1)))
        ends.append(
            {
                "t_s": float(t1),
                "reposition_error_mm": float(np.linalg.norm(sim_xyz[i] - demo_xyz[j]) * 1e3),
            }
        )
    scored = [row for row in lift_rows if row["in_run"]]
    return {
        "lift_off": lift_rows,
        "lift_off_scored": len(scored),
        "lift_off_success_rate": (
            float(np.mean([row["success"] for row in scored])) if scored else None
        ),
        "air_travel_end": ends,
    }


def _check() -> None:
    force = np.array([0.0, 0.0, 6.0, 3.0, 0.0, 0.0, 1.0])
    clearance = np.array([0.02, 0.005, 0.0, 0.004, 0.02, 0.02, 0.004])
    labels = label_segments(force, clearance)
    assert labels.tolist() == [
        "air-travel",
        "approach",
        "contact",
        "lift-off",
        "air-travel",
        "air-travel",
        "approach",
    ]
    ev = events(
        [0, 1, 2, 3, 4, 5, 6],
        labels,
        clearance,
        np.zeros((7, 3)),
        [0, 1, 2, 3, 4, 5, 6],
        [0.0, 0.0, 0.0, 0.012, 0.02, 0.02, 0.0],
        np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0.01, 0, 0], [0, 0, 0]]),
    )
    assert ev["lift_off"][0]["in_run"] is True
    assert ev["lift_off"][0]["success"] is True
    assert ev["lift_off"][0]["lift_height_error_mm"] == 0.0
    assert abs(ev["air_travel_end"][-1]["reposition_error_mm"] - 10.0) < 1e-6


if __name__ == "__main__":
    _check()

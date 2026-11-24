"""Deploy-time lift-off override. Default off. The runner installs it.

Tracks |F·n| with at most a 20 ms box. After contact (|F·n| > on_n), a stretch
below off_n for hold_s with a non-pressing command enters RETRACT: δn ramps
to clearance_m over ramp_s. It drops when the policy's own δn exceeds
release_dn_m, or clearance falls approach_drop_m from its peak (a new approach).
"""

from __future__ import annotations

from collections import deque

import numpy as np


class LiftRetract:
    def __init__(
        self,
        on_n: float = 5.0,
        off_n: float = 3.0,
        hold_s: float = 0.050,
        smooth_s: float = 0.0,
        press_dn_m: float = -0.001,
        release_dn_m: float = 0.005,
        clearance_m: float = 0.010,
        ramp_s: float = 0.20,
        approach_drop_m: float = 0.001,
        dt: float = 0.002,
    ):
        self.on_n = float(on_n)
        self.off_n = float(off_n)
        self.hold_s = float(hold_s)
        self.press_dn_m = float(press_dn_m)
        self.release_dn_m = float(release_dn_m)
        self.clearance_m = float(clearance_m)
        self.ramp_s = float(ramp_s)
        self.approach_drop_m = float(approach_drop_m)
        self.dt = float(dt)
        # ponytail: raw sample. smooth_s is capped at 20 ms; raise it only to kill a noisy FT.
        window = 0.0 if smooth_s <= 0.0 else min(float(smooth_s), 0.020)
        self._fn: deque[float] = deque(maxlen=max(1, int(round(window / self.dt))))
        self._fn_now = 0.0
        self.latched = False
        self.low_s = 0.0
        self.retracting = False
        self._ramp_t = 0.0
        self._ramp_from = 0.0
        self._peak = 0.0
        self._saw_contact = False
        self.retracts: list[dict] = []

    def observe(self, fn_abs: float) -> None:
        self._fn.append(float(abs(fn_abs)))
        self._fn_now = float(np.mean(self._fn))

    def command(self, policy_dn: float, away: bool, clearance: float, t_s: float) -> float | None:
        fn = self._fn_now
        if fn > self.on_n:
            self.latched = True
            self.low_s = 0.0
        elif self.latched and fn < self.off_n:
            self.low_s += self.dt
        else:
            self.low_s = 0.0
        nonpress = float(policy_dn) > self.press_dn_m or bool(away)
        if (
            not self.retracting
            and self.latched
            and self.low_s + 1e-12 >= self.hold_s
            and nonpress
        ):
            self.retracting = True
            self.latched = False
            self.low_s = 0.0
            self._ramp_t = 0.0
            self._ramp_from = float(clearance)
            self._peak = float(clearance)
            self._saw_contact = False
            self.retracts.append(
                {"t_s": float(t_s), "spurious": False, "reached_10mm": False, "peak_mm": None}
            )
        if not self.retracting:
            return None
        if fn > self.on_n:
            self._saw_contact = True
        self._peak = max(self._peak, float(clearance))
        self._ramp_t += self.dt
        alpha = min(1.0, self._ramp_t / self.ramp_s)
        cmd = self._ramp_from + alpha * (self.clearance_m - self._ramp_from)
        approach = self._peak - float(clearance) > self.approach_drop_m
        if float(policy_dn) > self.release_dn_m or approach:
            self._close()
            return None
        return float(cmd)

    def finish(self) -> None:
        if self.retracting:
            self._close()

    def _close(self) -> None:
        row = self.retracts[-1]
        row["peak_mm"] = round(self._peak * 1e3, 2)
        row["reached_10mm"] = self._peak + 1e-6 >= self.clearance_m
        row["spurious"] = bool(self._saw_contact and not row["reached_10mm"])
        self.retracting = False


def _check() -> None:
    gate = LiftRetract(dt=0.002)
    gate.observe(6.0)
    assert gate.command(-0.002, False, 0.0, 0.0) is None
    for i in range(24):
        gate.observe(2.0)
        assert gate.command(0.0, False, 0.0, 0.002 * (i + 1)) is None
    gate.observe(2.0)
    cmd = gate.command(0.0, False, 0.0, 0.050)
    assert cmd is not None and abs(cmd - 0.0001) < 1e-6, cmd
    held = LiftRetract(dt=0.002)
    held.observe(6.0)
    held.command(0.0, False, 0.0, 0.0)
    for _ in range(25):
        held.observe(2.0)
        got = held.command(-0.002, False, 0.0, 0.0)
        assert got is None
    assert held.retracting is False
    away = LiftRetract(dt=0.002)
    away.observe(6.0)
    away.command(-0.002, False, 0.0, 0.0)
    entered = None
    for i in range(25):
        away.observe(2.0)
        entered = away.command(-0.002, True, 0.0, 0.05)
    assert entered is not None and away.retracting
    released = LiftRetract(dt=0.002)
    released.observe(6.0)
    released.command(0.0, False, 0.0, 0.0)
    for _ in range(25):
        released.observe(1.0)
        released.command(0.0, False, 0.0, 0.0)
    assert released.retracting
    assert released.command(0.006, False, 0.004, 0.1) is None
    assert released.retracting is False


if __name__ == "__main__":
    _check()
    print("lift_retract ok")

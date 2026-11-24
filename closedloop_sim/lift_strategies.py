"""Deploy-time lift strategies. Every flag defaults off.

force: |F·n| contact on > on_n, off < off_n for hold_s, and the policy is not
pressing (δn > press_dn_m or normal speed away) → ramp δn to retract_m.
stroke: same retract when the foot is within end_m of either stringer end, or
the tangential speed along the mold has reversed for hold_s.
air: while |F·n| < air_n, δn is at least air_floor_m.

RETRACT hands δn back after a 0.2 s ramp when the hold has lasted t_hold_s
and the policy's tangential command has moved travel_m from the lift point or
reversed, or at timeout_s. Contact detection re-arms after that hand-back.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class LiftStrategy:
    def __init__(
        self,
        force: bool = False,
        stroke: bool = False,
        air: bool = False,
        on_n: float = 5.0,
        off_n: float = 3.0,
        hold_s: float = 0.050,
        press_dn_m: float = -0.001,
        release_dn_m: float = 0.005,
        retract_m: float = 0.010,
        ramp_s: float = 0.20,
        air_n: float = 2.0,
        air_floor_m: float = 0.003,
        end_m: float = 0.020,
        approach_drop_m: float = 0.001,
        release_on_press: bool = False,
        t_hold_s: float = 0.5,
        travel_m: float = 0.02,
        timeout_s: float = 2.0,
        dt: float = 0.002,
    ):
        self.force = bool(force)
        self.stroke = bool(stroke)
        self.air = bool(air)
        self.on_n = float(on_n)
        self.off_n = float(off_n)
        self.hold_s = float(hold_s)
        self.press_dn_m = float(press_dn_m)
        self.release_dn_m = float(release_dn_m)
        self.retract_m = float(retract_m)
        self.ramp_s = float(ramp_s)
        self.air_n = float(air_n)
        self.air_floor_m = float(air_floor_m)
        self.end_m = float(end_m)
        self.approach_drop_m = float(approach_drop_m)
        self.release_on_press = bool(release_on_press)
        self.t_hold_s = float(t_hold_s)
        self.travel_m = float(travel_m)
        self.timeout_s = float(timeout_s)
        self.dt = float(dt)
        self.v_tan = None
        self._fn: deque[float] = deque(maxlen=1)
        self._along: deque[float] = deque()
        self.fn = 0.0
        self.latched = False
        self.low_s = 0.0
        self.rev_s = 0.0
        self._rev_sign = 0
        self.retracting = False
        self.reason = None
        self._ramp_t = 0.0
        self._ramp_from = 0.0
        self._peak = 0.0
        self._saw_contact = False
        self.retracts: list[dict] = []

    def observe(self, fn_abs: float) -> None:
        self._fn.append(float(abs(fn_abs)))
        self.fn = float(self._fn[-1])

    def step(self, t, policy_dn, away, along, s_end, clearance) -> float | None:
        self._along.append(float(along))
        if self.retracting:
            return self._hold(t, float(policy_dn), float(along), float(clearance))
        fn = self.fn
        if fn > self.on_n:
            self.latched = True
            self.low_s = 0.0
        elif self.latched and fn < self.off_n:
            self.low_s += self.dt
        else:
            self.low_s = 0.0
        nonpress = float(policy_dn) > self.press_dn_m or bool(away)
        want = None
        if self.force and self.latched and self.low_s + 1e-12 >= self.hold_s and nonpress:
            want = "force"
        if self.stroke and (float(s_end) <= self.end_m or self._reversed()):
            want = "stroke"
        if want is not None:
            self._begin(t, clearance, want)
            return self._hold(t, float(policy_dn), float(along), float(clearance))
        if self.air and fn < self.air_n and float(policy_dn) < self.air_floor_m:
            return float(self.air_floor_m)
        return None

    def finish(self) -> None:
        if self.retracting:
            self._close(None)

    def _hold(self, t, policy_dn, along, clearance) -> float | None:
        self._held += self.dt
        if self.fn > self.on_n:
            self._saw_contact = True
        self._peak = max(self._peak, clearance)
        self._accumulate(along)
        if self._phase == "in":
            self._ramp_t += self.dt
            alpha = min(1.0, self._ramp_t / self.ramp_s)
            cmd = self._ramp_from + alpha * (self.retract_m - self._ramp_from)
            if self._release_due():
                self._phase = "out"
                self._ramp_t = 0.0
                self._ramp_from = float(cmd)
                self._held_at_release = self._held
            return float(cmd)
        self._ramp_t += self.dt
        alpha = min(1.0, self._ramp_t / self.ramp_s)
        cmd = self._ramp_from + alpha * (float(policy_dn) - self._ramp_from)
        if alpha >= 1.0:
            self._close(t)
            return None
        return float(cmd)

    def _begin(self, t, clearance, reason) -> None:
        self.retracting = True
        self.reason = reason
        self.latched = False
        self.low_s = 0.0
        self._phase = "in"
        self._held = 0.0
        self._held_at_release = 0.0
        self._ramp_t = 0.0
        self._ramp_from = float(clearance)
        self._peak = float(clearance)
        self._saw_contact = False
        self._disp = np.zeros(3)
        self._v_at = None
        self._v_now = np.zeros(3)
        self._release_why = None
        self.retracts.append(
            {"t_s": float(t), "reason": reason, "spurious": False, "reached_10mm": False, "peak_mm": None}
        )

    def _close(self, t) -> None:
        row = self.retracts[-1]
        row["peak_mm"] = round(self._peak * 1e3, 2)
        row["reached_10mm"] = self._peak + 1e-6 >= self.retract_m
        row["spurious"] = bool(self._saw_contact and not row["reached_10mm"])
        row["release"] = self._release_why or "finish"
        row["held_s"] = round(self._held_at_release or self._held, 3)
        row["owned_s"] = round(self._held, 3)
        row["travel_mm"] = round(float(np.linalg.norm(self._disp)) * 1e3, 1)
        if t is not None:
            row["t_release_s"] = float(t)
        self.retracting = False
        self.latched = self.fn > self.on_n
        self.low_s = 0.0

    def _accumulate(self, along) -> None:
        vel = self.v_tan if self.v_tan is not None else np.array([float(along), 0.0, 0.0])
        vel = np.asarray(vel, dtype=np.float64).reshape(-1)
        if vel.size == 1:
            vel = np.array([float(vel[0]), 0.0, 0.0])
        elif vel.size == 2:
            vel = np.array([float(vel[0]), float(vel[1]), 0.0])
        self._v_now = vel
        if self._v_at is None:
            self._v_at = vel.copy()
        self._disp = self._disp + vel * self.dt

    def _cmd_reversed(self) -> bool:
        if self._v_at is None:
            return False
        a = float(np.linalg.norm(self._v_at))
        b = float(np.linalg.norm(self._v_now))
        if a < 0.005 or b < 0.005:
            return False
        return float(np.dot(self._v_at, self._v_now)) < 0.0

    def _release_due(self) -> bool:
        if self._held + 1e-12 >= self.timeout_s:
            self._release_why = "timeout"
            return True
        if self._held + 1e-12 < self.t_hold_s:
            return False
        if float(np.linalg.norm(self._disp)) + 1e-12 >= self.travel_m:
            self._release_why = "travel"
            return True
        if self._cmd_reversed():
            self._release_why = "reverse"
            return True
        return False

    def _reversed(self) -> bool:
        n = max(1, int(round(self.hold_s / self.dt)))
        hist = max(1, int(round(0.20 / self.dt)))
        if len(self._along) < n + hist:
            return False
        vals = np.fromiter(self._along, dtype=np.float64)
        recent = float(np.mean(vals[-n:]))
        older = float(np.mean(vals[-(n + hist):-n]))
        if abs(recent) < 0.005 or abs(older) < 0.005 or recent * older >= 0.0:
            self.rev_s = 0.0
            return False
        self.rev_s += self.dt
        return self.rev_s + 1e-12 >= self.hold_s


def _check() -> None:
    gate = LiftStrategy(force=True, dt=0.002)
    gate.observe(6.0)
    assert gate.step(0.0, -0.002, False, 0.0, 1.0, 0.0) is None
    cmd = None
    for i in range(25):
        gate.observe(2.0)
        cmd = gate.step(0.002 * (i + 1), 0.0, False, 0.0, 1.0, 0.0)
    assert cmd is not None and gate.retracting and gate.retracts[-1]["reason"] == "force"
    air = LiftStrategy(air=True, dt=0.002)
    air.observe(1.0)
    assert abs(air.step(0.0, 0.0, False, 0.0, 1.0, 0.02) - 0.003) < 1e-9
    air.observe(4.0)
    assert air.step(0.0, 0.0, False, 0.0, 1.0, 0.0) is None
    end = LiftStrategy(stroke=True, dt=0.002)
    end.observe(0.0)
    assert end.step(0.0, -0.004, False, 0.0, 0.01, 0.0) is not None
    assert end.retracts[-1]["reason"] == "stroke"
    rev = LiftStrategy(stroke=True, dt=0.002, end_m=0.0)
    rev.observe(0.0)
    fired = False
    for i in range(200):
        along = 0.02 if i < 120 else -0.02
        got = rev.step(i * 0.002, -0.004, False, along, 1.0, 0.0)
        fired = fired or got is not None
    if not fired:
        raise RuntimeError("reversal did not retract")
    # Held past the old δn>+5 mm exit. Releases on travel after T_hold, else on the 2 s timeout.
    hold = LiftStrategy(force=True, dt=0.01, t_hold_s=0.5, travel_m=0.04, timeout_s=2.0, ramp_s=0.2)
    hold.observe(6.0)
    hold.step(0.0, 0.02, False, 0.0, 1.0, 0.0)
    for i in range(5):
        hold.observe(2.0)
        hold.step(0.01 * (i + 1), 0.02, False, 0.0, 1.0, 0.0)
    assert hold.retracting
    for i in range(40):
        hold.observe(0.0)
        cmd = hold.step(0.1 + 0.01 * i, 0.02, False, 0.2, 1.0, 0.01)
    assert hold._phase == "in" and cmd is not None
    for i in range(20):
        hold.observe(0.0)
        cmd = hold.step(0.5 + 0.01 * i, 0.02, False, 0.2, 1.0, 0.01)
    assert hold._phase == "out" and hold._release_why == "travel"
    for i in range(30):
        hold.observe(0.0)
        cmd = hold.step(0.7 + 0.01 * i, -0.003, False, 0.0, 1.0, 0.01)
    assert cmd is None and not hold.retracting
    hold.observe(6.0)
    assert hold.step(1.2, -0.002, False, 0.0, 1.0, 0.0) is None
    again = None
    for i in range(6):
        hold.observe(1.0)
        again = hold.step(1.3 + 0.01 * i, 0.0, False, 0.0, 1.0, 0.0)
    assert again is not None and len(hold.retracts) == 2
    stuck = LiftStrategy(force=True, dt=0.01, t_hold_s=0.5, travel_m=0.04, timeout_s=2.0, ramp_s=0.2)
    stuck.observe(6.0)
    stuck.step(0.0, 0.02, False, 0.0, 1.0, 0.0)
    for i in range(5):
        stuck.observe(2.0)
        stuck.step(0.01 * (i + 1), 0.02, False, 0.0, 1.0, 0.0)
    for i in range(230):
        stuck.observe(0.0)
        stuck.step(0.1 + 0.01 * i, -0.004, False, 0.0, 1.0, 0.01)
    assert stuck.retracts[-1].get("release") == "timeout" and not stuck.retracting


if __name__ == "__main__":
    _check()
    print("lift_strategies ok")

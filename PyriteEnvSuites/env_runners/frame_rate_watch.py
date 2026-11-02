"""Warn when a live camera falls behind the rate the policy was trained on."""

from __future__ import annotations

import time
from typing import Optional


class FrameRateWatch:
    """Track frame arrival times and warn when a gap or the average rate is low.

    Training and collection RGB is 15 Hz (one D405 frame = 1/15 s). A gap longer
    than 1.5 frame periods counts as a drop. Warnings are throttled to one per
    second so a stall does not flood the terminal.
    """

    def __init__(
        self,
        name: str,
        expected_hz: float = 15.0,
        warn_ratio: float = 0.8,
    ) -> None:
        if expected_hz <= 0:
            raise ValueError(f"expected_hz must be > 0, got {expected_hz}")
        self.name = name
        self.expected_hz = float(expected_hz)
        self.warn_ratio = float(warn_ratio)
        self._t: list[float] = []
        self.n_gap_warns = 0
        self._last_warn_t = -1e9

    def tick(self, t: Optional[float] = None) -> Optional[str]:
        now = time.monotonic() if t is None else float(t)
        msg = None
        if self._t:
            dt = now - self._t[-1]
            period = 1.0 / self.expected_hz
            if dt > 1.5 * period and now - self._last_warn_t >= 1.0:
                missed = max(1, int(round(dt * self.expected_hz)) - 1)
                self.n_gap_warns += 1
                self._last_warn_t = now
                msg = (
                    f"[cam] {self.name} frame gap {dt * 1000:.0f} ms "
                    f"(expected {period * 1000:.0f} ms at {self.expected_hz:.0f} Hz, "
                    f"~{missed} frame(s) dropped)"
                )
        self._t.append(now)
        return msg

    def summary(self) -> dict:
        n = len(self._t)
        if n < 2:
            duration_s = 0.0
            hz = 0.0
        else:
            duration_s = float(self._t[-1] - self._t[0])
            hz = (n - 1) / duration_s if duration_s > 0 else 0.0
        expected_n = self.expected_hz * duration_s if duration_s > 0 else 0.0
        # n arrivals over duration_s cover about expected_n intervals, so the
        # frame count to compare is n, against expected_hz * duration.
        short = max(0, int(round(expected_n)) - (n - 1 if n else 0))
        ok = n >= 2 and hz >= self.warn_ratio * self.expected_hz
        return {
            "name": self.name,
            "frames": n,
            "duration_s": round(duration_s, 3),
            "hz": round(hz, 2),
            "expected_hz": self.expected_hz,
            "frames_short": short,
            "gap_warns": self.n_gap_warns,
            "ok": bool(ok),
        }

    def summary_line(self) -> str:
        s = self.summary()
        flag = "ok" if s["ok"] else "LOW"
        return (
            f"[cam] {s['name']} {flag}: {s['hz']:.1f} Hz over {s['duration_s']:.1f}s "
            f"({s['frames']} frames, expected {s['expected_hz']:.0f} Hz, "
            f"~{s['frames_short']} short, gap_warns={s['gap_warns']})"
        )

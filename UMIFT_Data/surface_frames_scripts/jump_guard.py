"""Hold an r4vt commanded target when one step jumps past normal motion.

The demo TCP at the runner's RGB step (decode dt 1/15 s), episodes 0 and 9
pooled, is the tool motion: position mm p50 6.668, p99 39.173, p99.9 86.335,
max 133.813; rotation deg p50 1.081, p99 6.474, p99.9 10.054, max 36.367.
1.5 times that rotation p99.9 is 15.1 deg. The commanded heading R_d still
moves 20 to 30 deg on ordinary steps, so that limit holds normal motion.

The limit below is 1.5 times the p99.9 of the commanded target itself
(action-blended x_d and R_d, H=6), on steps whose rotation is under 60 deg.
The steps at or above 60 deg are the frame-flip discontinuities.
Position mm: p50 17.190, p99 119.188, p99.9 165.801, max 185.209.
Rotation deg: p50 1.306, p99 37.425, p99.9 50.640, max 58.563.
"""

from __future__ import annotations

import numpy as np

DEMO_P999_MM = 165.801
DEMO_P999_DEG = 50.640
MAX_STEP_MM = 1.5 * DEMO_P999_MM
MAX_STEP_DEG = 1.5 * DEMO_P999_DEG
MAX_CONSECUTIVE_HOLDS = 3


def _geodesic_deg(ra: np.ndarray, rb: np.ndarray) -> float:
    rel = ra.T @ rb
    cos = float(np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


class TargetJumpGuard:
    """Replace a step that exceeds the limit with the previous accepted target.

    More than ``max_holds`` excesses in a row aborts. The aborting step is not
    returned. The first step of an episode is accepted, because there is no
    previous target.
    """

    def __init__(
        self,
        max_step_mm: float = MAX_STEP_MM,
        max_step_deg: float = MAX_STEP_DEG,
        max_holds: int = MAX_CONSECUTIVE_HOLDS,
    ):
        self.max_step_mm = float(max_step_mm)
        self.max_step_deg = float(max_step_deg)
        self.max_holds = int(max_holds)
        self.reset()

    def reset(self) -> None:
        self.prev_x = None
        self.prev_R = None
        self.prev_field = None
        self.run = 0
        self.holds: list[dict] = []
        self.aborted = False

    def apply(
        self,
        x_d: np.ndarray,
        R_d: np.ndarray,
        robot_index: np.ndarray | None = None,
        R_field: np.ndarray | None = None,
    ):
        x = np.array(x_d, dtype=np.float64, copy=True)
        R = np.array(R_d, dtype=np.float64, copy=True)
        field = None if R_field is None else np.array(R_field, dtype=np.float64, copy=True)
        keep = x.shape[0]
        if self.aborted:
            empty = x[:0], R[:0]
            return (*empty, None if field is None else field[:0], True)
        for i in range(x.shape[0]):
            if self.prev_x is None:
                self._accept(x[i], R[i], None if field is None else field[i])
                continue
            pos_mm = float(np.linalg.norm(x[i] - self.prev_x) * 1e3)
            rot_deg = _geodesic_deg(self.prev_R, R[i])
            if pos_mm <= self.max_step_mm and rot_deg <= self.max_step_deg:
                self._accept(x[i], R[i], None if field is None else field[i])
                continue
            self.run += 1
            row = {
                "robot_index": None if robot_index is None else int(robot_index[i]),
                "step": int(i),
                "pos_mm": pos_mm,
                "rot_deg": rot_deg,
                "run": int(self.run),
                "abort": self.run > self.max_holds,
            }
            self.holds.append(row)
            if row["abort"]:
                self.aborted = True
                keep = i
                break
            x[i] = self.prev_x
            R[i] = self.prev_R
            if field is not None:
                field[i] = self.prev_field
        out_field = None if field is None else field[:keep]
        return x[:keep], R[:keep], out_field, self.aborted

    def _accept(self, x, R, field) -> None:
        self.prev_x = np.array(x, dtype=np.float64, copy=True)
        self.prev_R = np.array(R, dtype=np.float64, copy=True)
        self.prev_field = None if field is None else np.array(field, dtype=np.float64, copy=True)
        self.run = 0

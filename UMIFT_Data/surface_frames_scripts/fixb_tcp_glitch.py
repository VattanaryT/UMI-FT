"""Find T265 pose glitches and interpolate the short ones.

A glitch is a TCP step faster than 0.5 m/s, or a fast step that reverses
within 200 ms. Spans of 200 ms or less are filled by a straight line and a
slerp between the poses on either side. Longer spans are left in the pose and
returned so the caller can drop training windows that overlap them.

The raw pose is never written here.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from fixb_labels import Rp_to_pose7, pose7_to_Rp

SPEED_MAX_M_S = 0.5
REVERSE_WINDOW_S = 0.200
SHORT_MAX_S = 0.200


def glitch_mask(position: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    """True on samples that arrive on a fast step, or lie inside a fast reversal."""
    position = np.asarray(position, dtype=np.float64).reshape(-1, 3)
    time_s = np.asarray(time_s, dtype=np.float64).reshape(-1)
    n = len(position)
    mask = np.zeros(n, dtype=bool)
    if n < 2:
        return mask
    dt = np.diff(time_s)
    delta = np.diff(position, axis=0)
    step = np.linalg.norm(delta, axis=1)
    speed = np.zeros(n, dtype=np.float64)
    ok = dt > 1e-6
    speed[1:][ok] = step[ok] / dt[ok]
    fast_step = np.flatnonzero(speed[1:] > SPEED_MAX_M_S)
    mask[fast_step + 1] = True
    unit = delta / np.maximum(step[:, None], 1e-12)
    for i in fast_step:
        for j in fast_step:
            if j <= i:
                continue
            if time_s[j] > time_s[i] + REVERSE_WINDOW_S:
                break
            if float(np.dot(unit[i], unit[j])) < 0.0:
                mask[i + 1 : j + 2] = True
    return mask


def glitch_spans(mask: np.ndarray, time_s: np.ndarray) -> list[dict]:
    """Inclusive spans. kind is 'short' when the span can be interpolated."""
    time_s = np.asarray(time_s, dtype=np.float64).reshape(-1)
    spans = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        duration = float(time_s[j] - time_s[i])
        bracketed = i > 0 and j + 1 < n
        kind = "short" if bracketed and duration <= SHORT_MAX_S else "long"
        spans.append({"i0": int(i), "i1": int(j), "duration_s": duration, "kind": kind})
        i = j + 1
    return spans


def can_interpolate(position: np.ndarray, time_s: np.ndarray, span: dict) -> bool:
    """A short span can be filled only when the poses on either side still agree.

    A one-way tracking jump leaves the later pose somewhere else. The straight
    line between those poses is the jump itself, so it is not filled in.
    """
    if span["kind"] != "short":
        return False
    position = np.asarray(position, dtype=np.float64).reshape(-1, 3)
    time_s = np.asarray(time_s, dtype=np.float64).reshape(-1)
    a = span["i0"] - 1
    b = span["i1"] + 1
    if a < 0 or b >= len(time_s):
        return False
    dt = float(time_s[b] - time_s[a])
    if dt <= 1e-6:
        return False
    return float(np.linalg.norm(position[b] - position[a]) / dt) <= SPEED_MAX_M_S


def clean_pose7(pose7: np.ndarray, time_s: np.ndarray, spans: list[dict]) -> np.ndarray:
    """Copy of pose7 with short glitches replaced. Long spans are unchanged."""
    pose = np.array(pose7, dtype=np.float64, copy=True)
    time_s = np.asarray(time_s, dtype=np.float64).reshape(-1)
    R, p = pose7_to_Rp(pose)
    for span in spans:
        if span["kind"] != "short":
            continue
        a = span["i0"] - 1
        b = span["i1"] + 1
        if b <= a or time_s[b] <= time_s[a]:
            continue
        slerp = Slerp([time_s[a], time_s[b]], Rotation.from_matrix(np.stack([R[a], R[b]])))
        for k in range(span["i0"], span["i1"] + 1):
            alpha = (time_s[k] - time_s[a]) / (time_s[b] - time_s[a])
            p[k] = (1.0 - alpha) * p[a] + alpha * p[b]
            R[k] = slerp(time_s[k]).as_matrix()
    pose[:, :] = Rp_to_pose7(R, p)
    return pose

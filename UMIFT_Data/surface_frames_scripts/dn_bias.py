"""Add a press bias to scheduled δn. Off unless UMIFT_DN_BIAS_MM is set.

The bias is applied only in contact: |F·n| at least 2 N, or the tool is
within 3 mm of the surface. In the air the scheduled δn is unchanged.
Negative N pushes the virtual target further along the inward normal.
"""

from __future__ import annotations

import os

import numpy as np

CONTACT_FORCE_N = 2.0
CONTACT_SD_M = 0.003


def millimeters(environ=None) -> float:
    source = os.environ if environ is None else environ
    raw = str(source.get("UMIFT_DN_BIAS_MM", "")).strip()
    if not raw:
        return 0.0
    return float(raw)


def in_contact(force_n: float, signed_distance_m: float) -> bool:
    return abs(float(force_n)) >= CONTACT_FORCE_N or float(signed_distance_m) < CONTACT_SD_M


def apply_bias(decoded: dict, bias_mm: float) -> None:
    """Shift δn and the virtual target. The foot (nominal) stays put."""
    bias_m = float(bias_mm) / 1000.0
    if bias_m == 0.0:
        return
    normal = np.asarray(decoded["n_out"], dtype=np.float64).reshape(-1, 3)
    normal = normal / np.clip(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9, None)
    decoded["delta_n"] = np.asarray(decoded["delta_n"], dtype=np.float64).reshape(-1) + bias_m
    decoded["x_d"] = np.asarray(decoded["x_d"], dtype=np.float64).reshape(-1, 3) + bias_m * normal
    if decoded.get("sd") is not None:
        decoded["sd"] = np.asarray(decoded["sd"], dtype=np.float64).reshape(-1) + bias_m

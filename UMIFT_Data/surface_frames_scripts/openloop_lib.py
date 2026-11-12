"""Shared open-loop decode, ΔR angle, and episode error table.

No policy and no GUI. The export script and the viewer both import this.
"""

from __future__ import annotations

import numpy as np

from fixb_labels import pose7_to_Rp, so3_to_rot6d_rows
from mold_aware_runtime import MoldAwareRuntime, SafetyLimits, rot6d_rows_to_R

# Same copy-check tolerances as PyriteEnvSuites/scripts/offline_compare_driver.py.
TOL_POS_M = 1e-5
TOL_ROT_DEG = 1e-3
TOL_STIFFNESS = 1e-3
CONTACT_N = 8.0

# r1rel labels. Confirmed from the converters, not from a tolerance change.
# raw_to_action21 builds the absolute action from ts_pose_command_0 and
# ts_pose_virtual_target_0. action21_to_action_sample then writes both poses
# relative to the query TCP, which is the newest obs pose (SE3_WT[-1] in
# sparse_obs_to_obs_sample). action21_postprocess left-multiplies that same
# TCP back on, so the command half reconstructs ts_pose_command_0 and the
# virtual-target half reconstructs ts_pose_virtual_target_0.
# fixb_relabel_odom_corr writes pack["fb"] (the corrected TCP) into both
# ts_pose_fb_0 and ts_pose_command_0 on the r1rel zarr. The reconstructed
# command is therefore ts_pose_fb. It is not the virtual target.
R1REL_LABEL_POSE = "ts_pose_fb_0"


def geodesic_deg(ra: np.ndarray, rb: np.ndarray) -> np.ndarray:
    """arccos((tr(Ra^T Rb) - 1) / 2) in degrees. Argument clipped to [-1, 1]."""
    rel = np.einsum("...ji,...jk->...ik", ra, rb)
    tr = np.trace(rel, axis1=-2, axis2=-1)
    cos = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def integration_dt(times: np.ndarray) -> np.ndarray:
    """dt used by integrate_fixb. The last column is unused. times (H,) or (Q, H)."""
    times = np.asarray(times, dtype=np.float64)
    single = times.ndim == 1
    if single:
        times = times.reshape(1, -1)
    if times.shape[1] == 1:
        dt = np.zeros_like(times)
    else:
        dt = np.diff(times, axis=1)
        dt = np.concatenate([dt, dt[:, -1:]], axis=1)
    return dt[0] if single else dt


def make_r4_runtime(field, dry_run: bool = False) -> MoldAwareRuntime:
    """Runtime decode with clamps available but not applied when dry_run is false.

    Call decode_chunk(..., apply_safety=False). dry_run must stay false: a true
    value replaces δn with the hover height even when apply_safety is false.
    """
    return MoldAwareRuntime(
        field=field,
        R_off=np.eye(3),
        run_type="r4_world_fixb_tilt",
        dry_run=bool(dry_run),
        limits=SafetyLimits(),
    )


def decode_r4vt(runtime: MoldAwareRuntime, action: np.ndarray, p0, R0, times) -> dict:
    """One chunk through MoldAwareRuntime.decode_chunk, safety clamps off.

    action (H, 27), times (H,). Returns x_d, R_d, x_s, n, delta_n.
    x_d = x_s + δn n and R_d = R(x_s) @ Gram-Schmidt(ΔR rot6d).
    """
    action = np.asarray(action, dtype=np.float64)
    if action.ndim != 2 or action.shape[1] < 27:
        raise ValueError(f"r4vt action must be (H, 27), got {action.shape}")
    dt = integration_dt(times)
    dec = runtime.decode_chunk(
        action,
        np.asarray(p0, dtype=np.float64).reshape(3),
        np.asarray(R0, dtype=np.float64).reshape(3, 3),
        dt,
        apply_safety=False,
    )
    n = dec["R_field"][:, :, 2]
    return {
        "x_d": dec["x_d"],
        "R_d": dec["R_d"],
        "x_s": dec["x_s"],
        "n": n,
        "delta_n": np.asarray(dec["delta_n"], dtype=np.float64),
        "R_field": dec["R_field"],
    }


def normal_offset(x_d, x_s, n) -> np.ndarray:
    return np.sum((x_d - x_s) * n, axis=-1)


def tangential_norm(delta, n) -> np.ndarray:
    n = np.asarray(n, dtype=np.float64)
    n = n / np.clip(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12, None)
    along = np.sum(delta * n, axis=-1, keepdims=True)
    return np.linalg.norm(delta - along * n, axis=-1)


def tangential_vt_offset(p_vt, x_s, n) -> dict:
    """Size of (p_vt - x_s) rejected by δn = (p_vt - x_s) · n. Meters."""
    mag = tangential_norm(np.asarray(p_vt) - np.asarray(x_s), n)
    mag = mag[np.isfinite(mag)]
    if mag.size == 0:
        return {"n": 0, "median_m": None, "max_m": None}
    return {
        "n": int(mag.size),
        "median_m": float(np.median(mag)),
        "max_m": float(np.max(mag)),
    }


def delta_r_deg(pred_action, label_action) -> np.ndarray:
    """Geodesic degrees between Gram-Schmidt rotations of the ΔR rot6d slots."""
    rp = rot6d_rows_to_R(np.asarray(pred_action, dtype=np.float64)[..., 21:27])
    rl = rot6d_rows_to_R(np.asarray(label_action, dtype=np.float64)[..., 21:27])
    return geodesic_deg(rp, rl)


def relative_command_rot_deg(pred_action, label_action) -> np.ndarray:
    """Geodesic degrees between the relative command rotations (pose9 rot6d)."""
    import PyriteUtility.spatial_math.spatial_utilities as su

    pred_action = np.asarray(pred_action, dtype=np.float64)
    label_action = np.asarray(label_action, dtype=np.float64)
    rp = su.pose9_to_SE3(pred_action[..., :9])[..., :3, :3]
    rl = su.pose9_to_SE3(label_action[..., :9])[..., :3, :3]
    return geodesic_deg(rp, rl)


def _mae(err, mask) -> float | None:
    err = np.asarray(err, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if err.shape != mask.shape:
        raise ValueError(f"error {err.shape} vs mask {mask.shape}")
    take = mask & np.isfinite(err)
    if not np.any(take):
        return None
    return float(np.mean(np.abs(err[take])))


def episode_error_table(pred, label, force_n, bad_span, long_dt, valid) -> dict:
    """Mean absolute error. Contact is ||F|| > 8 N. Clean drops bad spans and long-dt.

    pred and label are flattened over queries and steps.
    r4vt (width 27): v1, v2, δn, geodesic ΔR, stiffness.
    r1rel (width 21): relative command x, y, z, geodesic rotation, stiffness.
    """
    pred = np.asarray(pred, dtype=np.float64)
    label = np.asarray(label, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    force_n = np.asarray(force_n, dtype=np.float64).reshape(-1)
    bad = np.asarray(bad_span, dtype=bool).reshape(-1)
    long = np.asarray(long_dt, dtype=bool).reshape(-1)
    contact = valid & (force_n > CONTACT_N)
    free = valid & (force_n <= CONTACT_N)
    clean = valid & ~bad & ~long
    bad_m = valid & bad
    groups = {
        "contact": contact,
        "free": free,
        "clean": clean,
        "bad_span": bad_m,
    }
    if pred.shape[1] >= 27:
        per_dim = {
            "v1": pred[:, 0] - label[:, 0],
            "v2": pred[:, 1] - label[:, 1],
            "delta_n": pred[:, 2] - label[:, 2],
            "delta_R_deg": delta_r_deg(pred, label),
            "stiffness": pred[:, 18] - label[:, 18],
        }
    elif pred.shape[1] >= 21:
        per_dim = {
            "x": pred[:, 0] - label[:, 0],
            "y": pred[:, 1] - label[:, 1],
            "z": pred[:, 2] - label[:, 2],
            "rot_deg": relative_command_rot_deg(pred, label),
            "stiffness": pred[:, 18] - label[:, 18],
        }
    else:
        raise ValueError(f"action width {pred.shape[1]} is neither r4vt nor r1rel")
    table = {}
    for name, err in per_dim.items():
        table[name] = {group: _mae(err, mask) for group, mask in groups.items()}
    counts = {group: int(np.sum(mask)) for group, mask in groups.items()}
    counts["valid"] = int(np.sum(valid))
    counts["long_dt"] = int(np.sum(valid & long))
    return {"mae": table, "n": counts}


def r4vt_action_row(v_par, delta_n, delta_r_rot6d, stiffness=0.0) -> np.ndarray:
    """One (H, 27) label row: v1, v2, δn, identity rot6d, unused VT, stiffness, ΔR."""
    v_par = np.asarray(v_par, dtype=np.float64).reshape(-1, 2)
    h = len(v_par)
    action = np.zeros((h, 27), dtype=np.float64)
    action[:, 0:2] = v_par
    action[:, 2] = np.asarray(delta_n, dtype=np.float64).reshape(-1)
    action[:, 3:9] = np.array([1, 0, 0, 0, 1, 0], dtype=np.float64)
    action[:, 18] = stiffness
    action[:, 21:27] = np.asarray(delta_r_rot6d, dtype=np.float64).reshape(h, 6)
    return action


def rot6d_of_rotation(R: np.ndarray) -> np.ndarray:
    return so3_to_rot6d_rows(np.asarray(R, dtype=np.float64).reshape(1, 3, 3))[0]

"""Temporally consistent foot for Fix B labels and the decoder.

Independent closest-point queries hop when the tool leaves a fold or the
apron edge. This file does not change fixb_labels.py. It re-projects only
those hops from the previous foot plus the tool step, then rebuilds
(v_par, delta_n, R_field) on that tracked foot.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from scipy.signal import savgol_filter

from fixb_field import FixBMoldField
from fixb_labels import (
    interp_rows,
    next_control_times,
    pose7_to_Rp,
    rot_angle_deg,
    so3_to_rot6d_rows,
    smooth_positions,
    tangent_steps,
)

HOP_EXCESS_M = 0.015


def tracked_project(
    field: FixBMoldField,
    p: np.ndarray,
    robot_t: np.ndarray | None = None,
    rgb_t: np.ndarray | None = None,
    excess_m: float = HOP_EXCESS_M,
    R_tool: np.ndarray | None = None,
    x_seed: np.ndarray | None = None,
    R_seed: np.ndarray | None = None,
) -> np.ndarray:
    """Foot positions from FixBMoldField.track_foot.

    robot_t, rgb_t, and excess_m stay so older callers still import. A field
    without track_foot keeps the old one-step hop repair.
    """
    p = np.asarray(p, dtype=np.float64).reshape(-1, 3)
    if hasattr(field, "track_foot"):
        return field.track_foot(p, R_tool=R_tool, x_seed=x_seed, R_seed=R_seed)["x_s"]
    x = field.project(p)
    if len(p) < 2:
        return x
    if robot_t is None or rgb_t is None:
        excess = np.linalg.norm(np.diff(x, axis=0), axis=1) - np.linalg.norm(np.diff(p, axis=0), axis=1)
        hops = np.flatnonzero(excess > excess_m) + 1
        for k in hops:
            x[k] = field.project((x[k - 1] + (p[k] - p[k - 1]))[None])[0]
        return x
    ir = np.clip(np.searchsorted(robot_t, rgb_t), 0, len(p) - 1)
    j1 = np.minimum(np.arange(len(ir)) + 1, len(ir) - 1)
    excess = np.linalg.norm(x[ir[j1]] - x[ir], axis=1) - np.linalg.norm(p[ir[j1]] - p[ir], axis=1)
    for k in np.flatnonzero(excess > excess_m):
        a, b = int(ir[k]), int(ir[j1[k]])
        for i in range(a + 1, b + 1):
            x[i] = field.project((x[i - 1] + (p[i] - p[i - 1]))[None])[0]
    return x


def centered_smooth_positions(robot_t: np.ndarray, p: np.ndarray, window_s: float) -> np.ndarray:
    """Centered Savitzky-Golay. About half the window is future samples.

    Labels and the live query no longer use this. The overlay of
    r4vt_frame_continuity still does, because that zarr was built with it.
    """
    t = np.asarray(robot_t, dtype=np.float64).reshape(-1)
    p = np.asarray(p, dtype=np.float64)
    if len(p) < 2:
        return np.array(p, dtype=np.float64, copy=True)
    dt = float(np.median(np.diff(t)))
    win = max(5, int(round(float(window_s) / dt)) | 1)
    if len(p) <= win:
        return np.array(p, dtype=np.float64, copy=True)
    return savgol_filter(p, win, 2, axis=0)


def query_smoothed_frames(
    field: FixBMoldField,
    robot_t: np.ndarray,
    position: np.ndarray,
    R_tool: np.ndarray | None = None,
    smooth_window_s: float = 0.1,
    *,
    align: str = "causal",
    which: str = "tracked",
) -> Dict[str, np.ndarray]:
    """The only smoothed foot query. align is 'causal' or 'center'. which is tracked, free, or both."""
    if align == "causal":
        p_s = smooth_positions(robot_t, position, smooth_window_s)
    elif align == "center":
        p_s = centered_smooth_positions(robot_t, position, smooth_window_s)
    else:
        raise ValueError(align)
    if which not in ("tracked", "free", "both"):
        raise ValueError(which)
    out: Dict[str, np.ndarray] = {"p_s": p_s}
    if which in ("tracked", "both"):
        if not hasattr(field, "track_foot"):
            raise ValueError("tracked query needs FixBMoldField.track_foot")
        out["tracked"] = field.query(p_s, with_curvature=True, R_tool=R_tool)
    if which in ("free", "both"):
        out["free"] = field.query(p_s)
    return out


def rising_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    prev = np.concatenate([[False], mask[:-1]])
    return mask & ~prev


def geodesic_jump(R: np.ndarray, deg: float = 30.0) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(-1, 3, 3)
    out = np.zeros(len(R), dtype=bool)
    if len(R) > 1:
        out[1:] = rot_angle_deg(R[:-1], R[1:]) > deg
    return out


def unexplained_jump(R: np.ndarray, R_tool: np.ndarray, deg: float = 30.0) -> np.ndarray:
    """Frame step the tool rotation does not explain."""
    R = np.asarray(R, dtype=np.float64).reshape(-1, 3, 3)
    R_tool = np.asarray(R_tool, dtype=np.float64).reshape(-1, 3, 3)
    out = np.zeros(len(R), dtype=bool)
    if len(R) > 1 and len(R) == len(R_tool):
        field_step = np.einsum("nji,njk->nik", R[:-1], R[1:])
        tool_step = np.einsum("nji,njk->nik", R_tool[:-1], R_tool[1:])
        out[1:] = rot_angle_deg(field_step, tool_step) > deg
    return out


def crown_wall_edges(reason: np.ndarray, free_R: np.ndarray) -> np.ndarray:
    """Rising edges where the closer face is accepted and the free normal jumps more than 45 deg.

    The returned mask has one entry per step. The sample that arrives on the new face is index + 1.
    """
    free_R = np.asarray(free_R, dtype=np.float64)
    if len(free_R) < 2:
        return np.zeros(0, dtype=bool)
    free_step = np.degrees(
        np.arccos(np.clip(np.sum(free_R[:-1, :, 2] * free_R[1:, :, 2], axis=1), -1.0, 1.0))
    )
    return rising_mask((np.asarray(reason)[1:] == 1) & (free_step > 45.0))


def near_cap_steps(R: np.ndarray, R_tool: np.ndarray):
    """Steps whose published frame moves more than 24.5 deg while the tool moves less than 1 deg."""
    R = np.asarray(R, dtype=np.float64)
    R_tool = np.asarray(R_tool, dtype=np.float64)
    if len(R) < 2:
        empty = np.zeros(0, dtype=np.float64)
        return np.zeros(0, dtype=bool), empty, empty
    out_step = rot_angle_deg(R[:-1], R[1:])
    tool_step = rot_angle_deg(R_tool[:-1], R_tool[1:])
    return (out_step > 24.5) & (tool_step < 1.0), out_step, tool_step


def episode_arrays_tracked(
    field: FixBMoldField,
    ep,
    *,
    smooth_window_s: float = 0.1,
    v_par_mode: str = "foot",
    delta_n_source: str = "fb",
) -> Dict[str, np.ndarray]:
    robot_t = np.asarray(ep["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
    rgb_t = np.asarray(ep["rgb_time_stamps_0"], dtype=np.float64).reshape(-1)
    R_fb, p_fb = pose7_to_Rp(ep["ts_pose_fb_0"])
    p_src = p_fb if delta_n_source == "fb" else pose7_to_Rp(ep["ts_pose_virtual_target_0"])[1]
    if hasattr(field, "track_foot"):
        pack = query_smoothed_frames(
            field, robot_t, p_src, R_fb, smooth_window_s, align="causal", which="tracked"
        )
        p_s = pack["p_s"]
        obs = pack["tracked"]
        x_s = np.asarray(obs["x_s"], dtype=np.float64)
        R = np.asarray(obs["R"], dtype=np.float64)
        sd = np.asarray(obs["sd"], dtype=np.float64).reshape(-1)
    else:
        p_s = smooth_positions(robot_t, p_src, smooth_window_s)
        x_s = tracked_project(field, p_s, robot_t, rgb_t)
        obs = field.query(x_s, with_curvature=True)
        R = obs["R"]
        sd = np.sum((p_s - x_s) * R[:, :, 2], axis=1)
    n = R[:, :, 2]
    t_next = next_control_times(robot_t, rgb_t)
    dt = t_next - robot_t
    x_next = interp_rows(t_next, robot_t, x_s)
    p_next = interp_rows(t_next, robot_t, p_s)
    if v_par_mode == "tool":
        v_par = np.einsum("nia,ni->na", R[..., :2], p_next - p_s)
    elif v_par_mode == "foot":
        v_par = tangent_steps(field, x_s, R, x_next)
    else:
        raise ValueError(v_par_mode)
    ok = dt > 1e-6
    v_par[ok] /= dt[ok, None]
    v_par[~ok] = 0.0
    v_par = np.clip(v_par, -1.0, 1.0)
    out = {
        "mold_x_s_0": x_s.astype(np.float32),
        "mold_R_field_0": R.reshape(-1, 9).astype(np.float32),
        "mold_R_field_rot6d_0": so3_to_rot6d_rows(R).astype(np.float32),
        "mold_normal_0": n.astype(np.float32),
        "mold_signed_distance_0": sd.reshape(-1, 1).astype(np.float32),
        "mold_curvature_0": obs["curv"].astype(np.float32),
        "fixb_v_par_0": v_par.astype(np.float32),
        "fixb_delta_n_0": sd.reshape(-1, 1).astype(np.float32),
        "fixb_dt_next_0": dt.reshape(-1, 1).astype(np.float32),
    }
    if "reason" in obs:
        out["face_reason_0"] = np.asarray(obs["reason"], dtype=np.int8)
    return out

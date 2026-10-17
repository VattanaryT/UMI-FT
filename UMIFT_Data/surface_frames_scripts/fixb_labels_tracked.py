"""Temporally consistent foot for Fix B labels and the decoder.

Independent closest-point queries hop when the tool leaves a fold or the
apron edge. This file does not change fixb_labels.py. It re-projects only
those hops from the previous foot plus the tool step, then rebuilds
(v_par, delta_n, R_field) on that tracked foot.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from fixb_field import FixBMoldField
from fixb_labels import (
    interp_rows,
    next_control_times,
    pose7_to_Rp,
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
) -> np.ndarray:
    """Batch closest points, then repair RGB-grid hops along the robot samples between them.

    Independent closest points hop by centimeters over one RGB step (~67 ms) while
    each 5 ms robot step stays under a naive threshold. Detect hops on the RGB
    grid, then walk the robot samples in that interval from the previous foot.
    """
    p = np.asarray(p, dtype=np.float64).reshape(-1, 3)
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
    _, p_fb = pose7_to_Rp(ep["ts_pose_fb_0"])
    p_src = p_fb if delta_n_source == "fb" else pose7_to_Rp(ep["ts_pose_virtual_target_0"])[1]
    p_s = smooth_positions(robot_t, p_src, smooth_window_s)
    x_s = tracked_project(field, p_s, robot_t, rgb_t)
    obs = field.query(x_s, with_curvature=True)
    R = obs["R"]
    n = R[:, :, 2]
    sd = np.sum((p_s - x_s) * n, axis=1)
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
    return {
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

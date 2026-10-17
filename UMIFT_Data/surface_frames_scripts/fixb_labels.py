"""Per-episode mold observations and Fix B labels (v_par, delta_n).

Labels live on the robot-rate grid but are built for the training action grid:
the sampler takes 24 action steps at consecutive RGB-frame times (index =
searchsorted(robot_t, rgb_t[j])). For a robot sample at t_k, the "next control
time" t_k+ is the robot sample of the next RGB frame, and

  v_par(t_k) = B(x_s(t_k))^T (c(t_k+) - c(t_k)) / (t_k+ - t_k),  B = [t1 t2]
  delta_n(t_k) = (p~(t_k) - x_s(t_k)) . n(x_s(t_k))

with p~ the lightly smoothed tool position and c either the tool position
("tool": projected tool velocity) or its foot point ("foot": surface-point
velocity; identical in contact, exact for the x_s integration off-surface).
In "foot" mode the tangent step is refined so that the decode step
x_s+ = foot(x_s + B v dt) lands on the next demo foot point (the tangent-plane
chord alone loses ~3 mm per 15 mm step on the 20 mm fillets).
Labels are commands over [t_k, t_k+]; decoding needs only the measured pose at
the chunk start.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from fixb_field import FixBMoldField


def pose7_to_Rp(pose7: np.ndarray):
    p = np.asarray(pose7, dtype=np.float64).reshape(-1, 7)
    R = Rotation.from_quat(p[:, [4, 5, 6, 3]]).as_matrix()
    return R, p[:, :3].copy()


def Rp_to_pose7(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    q = Rotation.from_matrix(R).as_quat()
    return np.concatenate([p, q[:, 3:4], q[:, :3]], axis=1)


def so3_to_rot6d_rows(R: np.ndarray) -> np.ndarray:
    """UMI-FT spatial_utilities.SO3_to_rot6d: first two rows."""
    return np.asarray(R).reshape(-1, 3, 3)[:, :2, :].reshape(-1, 6).copy()


def next_control_times(robot_t: np.ndarray, rgb_t: np.ndarray) -> np.ndarray:
    """Robot time of the action step that follows each robot sample."""
    rgb_dt = float(np.median(np.diff(rgb_t)))
    j_next = np.searchsorted(rgb_t, robot_t, side="right")
    t_rgb_next = np.where(j_next < len(rgb_t), rgb_t[np.minimum(j_next, len(rgb_t) - 1)], robot_t + rgb_dt)
    i_next = np.clip(np.searchsorted(robot_t, t_rgb_next), 0, len(robot_t) - 1)
    return robot_t[i_next]


def smooth_positions(robot_t: np.ndarray, p: np.ndarray, window_s: float) -> np.ndarray:
    dt = float(np.median(np.diff(robot_t)))
    win = max(5, int(round(window_s / dt)) | 1)
    if len(p) <= win:
        return p.copy()
    return savgol_filter(p, win, 2, axis=0)


def interp_rows(t_query: np.ndarray, t: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(t_query, t, x[:, d]) for d in range(x.shape[1])], axis=1)


def tangent_steps(
    field: FixBMoldField, x_s: np.ndarray, R: np.ndarray, x_next: np.ndarray, iters: int = 8
) -> np.ndarray:
    """2D steps s with foot(x_s + B s) ~= x_next.

    Keeps the best residual. If that step is much longer than the straight
    tangent chord, the fixed point has wandered around a fold; use the chord.
    """
    B = R[..., :2]
    chord = np.einsum("nia,ni->na", B, x_next - x_s)
    s = chord.copy()
    best_s, best_r = s.copy(), np.full(len(s), np.inf)
    for _ in range(iters):
        land = field.project(x_s + np.einsum("nia,na->ni", B, s))
        resid = x_next - land
        r = np.linalg.norm(resid, axis=1)
        better = r < best_r
        best_s[better], best_r[better] = s[better], r[better]
        s = s + np.einsum("nia,ni->na", B, resid)
    wandered = np.linalg.norm(best_s, axis=1) > np.maximum(1.5 * np.linalg.norm(chord, axis=1), 0.02)
    best_s[wandered] = chord[wandered]
    return best_s


def episode_arrays(
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

    obs = field.query(p_fb, with_curvature=True)
    R = obs["R"]

    p_src = p_fb if delta_n_source == "fb" else pose7_to_Rp(ep["ts_pose_virtual_target_0"])[1]
    p_s = smooth_positions(robot_t, p_src, smooth_window_s)
    lab = field.query(p_s)
    t_next = next_control_times(robot_t, rgb_t)
    dt = t_next - robot_t
    p_next = interp_rows(t_next, robot_t, p_s)
    if v_par_mode == "tool":
        v_par = np.einsum("nia,ni->na", lab["R"][..., :2], p_next - p_s)
    elif v_par_mode == "foot":
        v_par = tangent_steps(field, lab["x_s"], lab["R"], field.project(p_next))
    else:
        raise ValueError(v_par_mode)
    ok = dt > 1e-6
    v_par[ok] /= dt[ok, None]
    v_par[~ok] = 0.0
    # A handful of closest-point hops are ~30 m/s. The range normalizer uses
    # min/max, so those hops would flatten the real ~0.05 m/s motion.
    v_par = np.clip(v_par, -1.0, 1.0)

    return {
        "mold_x_s_0": obs["x_s"].astype(np.float32),
        "mold_R_field_0": R.reshape(-1, 9).astype(np.float32),
        "mold_R_field_rot6d_0": so3_to_rot6d_rows(R).astype(np.float32),
        "mold_normal_0": R[..., 2].astype(np.float32),
        "mold_signed_distance_0": obs["sd"].reshape(-1, 1).astype(np.float32),
        "mold_curvature_0": obs["curv"].astype(np.float32),
        "fixb_v_par_0": v_par.astype(np.float32),
        "fixb_delta_n_0": lab["sd"].reshape(-1, 1).astype(np.float32),
        "fixb_dt_next_0": dt.reshape(-1, 1).astype(np.float32),
    }


def polar_rotation(M: np.ndarray) -> np.ndarray:
    """Closest rotation to M; with M = sum R_field^T R_tool this is the chordal
    L2 mean heading offset (R_tool ~ R(x_s) R_off)."""
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


RZ_PI = np.diag([-1.0, -1.0, 1.0])


def fit_heading_offset(L: np.ndarray, iters: int = 10):
    """Fixed heading R_off from L = R_field^T R_tool samples, modulo a 180 deg
    turn about the surface normal (demos run the stringer in both directions).

    Returns R_off, per-sample flip flags and residual angles (deg).
    """
    R_off = polar_rotation(L.sum(axis=0))
    for _ in range(iters):
        keep = np.einsum("ij,nij->n", R_off, L)
        flip = np.einsum("ij,nij->n", RZ_PI @ R_off, L) > keep
        Lf = np.where(flip[:, None, None], np.einsum("ij,njk->nik", RZ_PI, L), L)
        R_off = polar_rotation(Lf.sum(axis=0))
    cos = np.clip((np.einsum("ij,nij->n", R_off, Lf) - 1.0) / 2.0, -1.0, 1.0)
    return R_off, flip, np.degrees(np.arccos(cos))


def heading_branch(R_field: np.ndarray, R_tool: np.ndarray, R_off: np.ndarray) -> np.ndarray:
    """R_off or RZ_PI @ R_off, whichever is closer to the measured tool (per row)."""
    L = np.einsum("nji,njk->nik", R_field, R_tool)
    flip = np.einsum("ij,nij->n", RZ_PI @ R_off, L) > np.einsum("ij,nij->n", R_off, L)
    return np.where(flip[:, None, None], (RZ_PI @ R_off)[None], R_off[None])


def fixb_pose7_command(v_par: np.ndarray, delta_n: np.ndarray) -> np.ndarray:
    """Pack (v_par, delta_n) into the ts_pose_command_0 slot (identity rotation)."""
    n = len(v_par)
    out = np.zeros((n, 7), dtype=np.float64)
    out[:, 0:2] = v_par
    out[:, 2] = np.asarray(delta_n).reshape(-1)
    out[:, 3] = 1.0
    return out


def identity_pose7(n: int) -> np.ndarray:
    out = np.zeros((n, 7), dtype=np.float64)
    out[:, 3] = 1.0
    return out

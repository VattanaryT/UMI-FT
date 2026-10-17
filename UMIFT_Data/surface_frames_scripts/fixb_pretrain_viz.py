#!/usr/bin/env python3
"""Pre-training checks for the Fix B datasets (r1, r2, r3). Reads only.

All three runs share timestamps, so every section uses the same episodes and
the trainer's chunk starts: query stride 2 on the RGB grid, 24 RGB-step action
chunks, action-gap filter (mult 3), obs windows of 5 robot samples 16 apart.
Physical units are mm, mm/s, deg. Normalized values use the range normalizer
each dataset fits on its train chunks (identity for rot6d dims), as in
UmiFTDataset.get_normalizer.

  1  flip timeline: R(x_s) frame-to-frame angle, normal flips, |d sd| > 50 mm,
     and how many train chunks see them in obs or labels
  2  label round trip: integrate (v_par, delta_n) from the measured pose,
     error vs the demo path, split by closest point within 30 mm over the whole chunk
  3  normalized label / obs histograms, v_par clip fraction, sd at force peaks
  4  twist about the normal, 180 deg branch, chunks whose branch could flip
  5  per-episode label std, across-episode spread by task phase, chunk PCA
  6  6_scrubber.html: offline 3D scrubber over the chunk starts (tool and R(x_s)
     axes, nearest-point line with distance / angles, 24-step label chunk,
     r1 / r2 / r3 toggles)

Usage (system python, needs open3d and plotly):
  python fixb_pretrain_viz.py
  python fixb_pretrain_viz.py --sections 6 --scrub_episodes 9 12 20
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import zarr
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from scipy.signal import find_peaks

from fixb_data_check import VAL_EPISODES, frame_step_deg, pct
from fixb_field import FixBMoldField, integrate_fixb
from fixb_labels import RZ_PI, polar_rotation, pose7_to_Rp, smooth_positions, so3_to_rot6d_rows

TRAIN_WT = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames")
sys.path[:0] = [str(TRAIN_WT), str(TRAIN_WT / "PyriteUtility"), str(TRAIN_WT / "PyriteML")]
from diffusion_policy.common.sampler_umift import filter_rgb_ids_without_action_gaps  # noqa: E402

RUNS = {"r1": "r1_world_world", "r2": "r2_world_fixb", "r3": "r3_surface_fixb"}
COLORS = {"r1": "C0", "r2": "C1", "r3": "C2"}

# trainer sampling (umift_mold_aware.yaml + sampler_umift.py)
H = 24
QUERY_STRIDE = 2
GAP_MULT = 3.0
OBS_H, OBS_DS = 5, 16
RGB_START_ID = 5 * 4 + 1
LOWDIM_START_ID = 5 * 16 + 1
WRENCH_START_ID = 32 * 1 + 1

SD_JUMP_M = 0.050
NEAR_M = 0.030
OFF_EDGE_M = 0.005
FOOT_HOP_M = 0.020
EXCURSION_MAX = 40
LABEL_HOP_EXCESS_M = 0.015
V_CLIP = 1.0
TRAVEL_MIN_MPS = 0.02
BRANCH_MARGIN_DEG = 15.0
FORCE_PEAK_N, FORCE_PEAK_DIST = 15.0, 5
SMOOTH_S = 0.1
N_PHASE = 10

TAIL_DIMS = ["vt_x", "vt_y", "vt_z"] + [f"vt_r6_{i}" for i in range(6)] + ["stiffness", "gripper", "grasp_force"]
DIMS = {
    "r1": ["cmd_x", "cmd_y", "cmd_z"] + [f"cmd_r6_{i}" for i in range(6)] + TAIL_DIMS,
    "r2": ["v_t1", "v_t2", "delta_n"] + [f"cmd_r6_{i}" for i in range(6)] + TAIL_DIMS,
}
DIMS["r3"] = DIMS["r2"]
RANGE_SLICES = (slice(0, 3), slice(9, 12), slice(18, 19), slice(19, 20), slice(20, 21))
MOTION_DIMS = {"r1": list(range(9)), "r2": [0, 1, 2], "r3": [0, 1, 2]}


def unit(dim: str):
    """(display unit, factor from SI)."""
    if dim.startswith("v_t"):
        return "mm/s", 1e3
    if dim in ("delta_n",) or dim[-2:] in ("_x", "_y", "_z"):
        return "mm", 1e3
    if dim == "stiffness":
        return "N/m", 1.0
    if dim == "grasp_force":
        return "N", 1.0
    if dim == "gripper":
        return "m", 1.0
    return "-", 1.0


def arr(x, dtype=np.float64):
    return np.asarray(x, dtype=dtype)


def range_normalizer(x: np.ndarray):
    lo, hi = x.min(0), x.max(0)
    rng = hi - lo
    ignore = rng < 1e-7
    rng = np.where(ignore, 2.0, rng)
    scale = 2.0 / rng
    offset = np.where(ignore, -lo, -1.0 - scale * lo)
    return scale, offset


def action_normalizer(X: np.ndarray):
    scale, offset = np.ones(X.shape[1]), np.zeros(X.shape[1])
    for sl in RANGE_SLICES:
        scale[sl], offset[sl] = range_normalizer(X[:, sl])
    return scale, offset


def pose9(pose7) -> np.ndarray:
    R, p = pose7_to_Rp(pose7)
    return np.concatenate([p, so3_to_rot6d_rows(R)], axis=1)


def action21(g, rt: np.ndarray, wt: np.ndarray) -> np.ndarray:
    """raw_to_action21 on the robot grid."""
    wl, wr = arr(g["wrench_left_0"]), arr(g["wrench_right_0"])
    force = (-wl[:, 0:1] + wr[:, 0:1]) / 2
    wi = np.clip(np.searchsorted(wt, rt, side="left"), 0, len(wt) - 1)
    return np.concatenate(
        [
            pose9(g["ts_pose_command_0"]),
            pose9(g["ts_pose_virtual_target_0"]),
            arr(g["stiffness_0"]).reshape(-1, 1),
            arr(g["gripper_0"]).reshape(-1, 1),
            force[wi],
        ],
        axis=1,
    )


def chunk_grid(rt, rgb, wt):
    """Train query rgb ids q, action robot ids A (Q,H), obs robot ids O (Q,5)."""
    dt_rgb = float(np.median(np.diff(rgb)))
    start = max(rgb[RGB_START_ID], rt[LOWDIM_START_ID], wt[WRENCH_START_ID])
    end = min(rgb[-1], rt[-1], wt[-1])
    first = int(np.searchsorted(rgb, start, side="left"))
    last = int(np.searchsorted(rgb, end, side="right") - 1)
    cand = first + np.arange(last - first)[::QUERY_STRIDE]
    q, _ = filter_rgb_ids_without_action_gaps(cand, rt, rgb, H, 1, GAP_MULT)
    tid = q[:, None] + np.arange(H)
    tt = np.where(tid < len(rgb), rgb[np.minimum(tid, len(rgb) - 1)], rgb[q][:, None] + np.arange(H) * dt_rgb)
    A = np.minimum(np.searchsorted(rt, tt), len(rt) - 1)
    O = A[:, :1] - OBS_DS * np.arange(OBS_H - 1, -1, -1)
    return q, A, O, len(cand)


class ScanOnlySD:
    """Signed distance to the scanned faces only (no flange apron)."""

    def __init__(self, field: FixBMoldField):
        faces = field.faces if field.is_apron is None else field.faces[~field.is_apron[field.faces].any(axis=1)]
        mesh = o3d.t.geometry.TriangleMesh()
        mesh.vertex.positions = o3d.core.Tensor(field.vertices.astype(np.float32))
        mesh.triangle.indices = o3d.core.Tensor(faces.astype(np.int32))
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(mesh)
        self.field = field

    def __call__(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64).reshape(-1, 3)
        if len(p) == 0:
            return np.zeros(0)
        x = self.scene.compute_closest_points(o3d.core.Tensor(p.astype(np.float32)))["points"].numpy().astype(np.float64)
        n = self.field.query(x)["R"][:, :, 2]
        return np.sum((p - x) * n, axis=1)


def geodesic_deg(Ra, Rb):
    tr = np.einsum("...ji,...ji->...", Ra, Rb)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def process_episode(name, data, field, scan_sd, R_off, checks):
    eid = int(name.split("_")[-1])
    g = {k: data[k][name] for k in data}
    rt = arr(g["r1"]["robot_time_stamps_0"]).reshape(-1)
    rgb = arr(g["r1"]["rgb_time_stamps_0"]).reshape(-1)
    wt = arr(g["r1"]["wrench_time_stamps_0"]).reshape(-1)
    for k in ("r2", "r3"):
        checks["timestamps_equal"] &= bool(
            np.array_equal(rt, arr(g[k]["robot_time_stamps_0"]).reshape(-1))
            and np.array_equal(rgb, arr(g[k]["rgb_time_stamps_0"]).reshape(-1))
        )
    q, A, O, n_cand = chunk_grid(rt, rgb, wt)
    ir = np.clip(np.searchsorted(rt, rgb), 0, len(rt) - 1)
    steps = np.minimum(q[:, None] + np.arange(H), len(rgb) - 1)

    R_fb, p_fb = pose7_to_Rp(g["r1"]["ts_pose_fb_0"])
    m = g["r2"]
    Rf = arr(m["mold_R_field_0"]).reshape(-1, 3, 3)
    n = Rf[:, :, 2]
    sd = arr(m["mold_signed_distance_0"]).reshape(-1)
    x_s = arr(m["mold_x_s_0"])
    dist = np.linalg.norm(p_fb - x_s, axis=1)
    in_plane = np.sqrt(np.maximum(dist**2 - sd**2, 0.0))
    v = arr(m["fixb_v_par_0"])
    dn = arr(m["fixb_delta_n_0"]).reshape(-1)
    for key in ("fixb_v_par_0", "fixb_delta_n_0", "ts_pose_command_0", "ts_pose_virtual_target_0", "stiffness_0"):
        d = float(np.max(np.abs(arr(g["r2"][key]) - arr(g["r3"][key]))))
        checks["labels_r2_vs_r3_max_abs"] = max(checks["labels_r2_vs_r3_max_abs"], d)
    for key in ("mold_signed_distance_0", "mold_normal_0", "mold_R_field_rot6d_0", "mold_curvature_0"):
        for k in ("r1", "r3"):
            d = float(np.max(np.abs(arr(g[k][key]) - arr(m[key]))))
            checks["mold_obs_max_abs_vs_r2"] = max(checks["mold_obs_max_abs_vs_r2"], d)
    checks["fb_r1_vs_r2_max_abs"] = max(
        checks["fb_r1_vs_r2_max_abs"], float(np.max(np.abs(arr(g["r1"]["ts_pose_fb_0"]) - arr(g["r2"]["ts_pose_fb_0"]))))
    )

    # 1 flips, sd jumps, foot jumps (robot rate, obs) and foot hops (rgb grid, labels)
    flips = np.flatnonzero(np.sum(n[1:] * n[:-1], axis=1) < 0)
    jumps = np.flatnonzero(np.abs(np.diff(sd)) > SD_JUMP_M)
    fstep = frame_step_deg(Rf)
    foot_step = np.linalg.norm(np.diff(x_s, axis=0), axis=1)
    big = np.flatnonzero(foot_step > FOOT_HOP_M)
    excursion = np.zeros(len(sd), bool)
    exc_len, paired = [], set()
    for a, b in zip(big[:-1], big[1:]):
        if a in paired:
            continue
        if b - a <= EXCURSION_MAX and np.linalg.norm(x_s[b + 1] - x_s[a]) < FOOT_HOP_M:
            excursion[a + 1 : b + 1] = True
            exc_len.append(b - a)
            paired.update((a, b))
    persistent = np.asarray([i for i in big if i not in paired], dtype=int)
    flip_big = foot_step[flips] > FOOT_HOP_M if len(flips) else np.zeros(0, bool)

    p_s = smooth_positions(rt, p_fb, SMOOTH_S)
    c = field.project(p_s[ir])
    j1 = np.minimum(np.arange(len(ir)) + 1, len(ir) - 1)
    j2 = np.minimum(np.arange(len(ir)) + 2, len(ir) - 1)
    excess1 = np.linalg.norm(c[j1] - c, axis=1) - np.linalg.norm(p_s[ir[j1]] - p_s[ir], axis=1)
    excess2 = np.linalg.norm(c[j2] - c, axis=1) - np.linalg.norm(p_s[ir[j2]] - p_s[ir], axis=1)
    label_hop = excess1 > LABEL_HOP_EXCESS_M
    hop_back = label_hop & (excess2 < LABEL_HOP_EXCESS_M)
    near_rgb = dist[ir] < NEAR_M
    clip_rgb = (np.abs(v[ir]) >= V_CLIP - 1e-6).any(axis=1)

    def in_span(events, lo, hi):
        return (events[None, :] >= lo[:, None]) & (events[None, :] < hi[:, None])

    fl_in = in_span(flips, A[:, 0], A[:, -1])
    jm_in = in_span(jumps, A[:, 0], A[:, -1])
    obs_flip = in_span(flips, O[:, 0], O[:, -1]).any(axis=1)
    obs_jump = in_span(big, O[:, 0], O[:, -1]).any(axis=1)
    obs_exc = excursion[O].any(axis=1)

    # 2 round trip
    dt = np.diff(rt[A], axis=1)
    dt = np.concatenate([dt, dt[:, -1:]], axis=1)
    dec = integrate_fixb(field, p_fb[A[:, 0]], v[A], dn[A], dt)
    err = np.linalg.norm(dec["x_d"] - p_fb[A], axis=-1) * 1e3
    near = (dist[A] < NEAR_M).all(axis=1)
    off_edge = in_plane[A[:, 0]] > OFF_EDGE_M
    checks["r1_cmd_vs_fb_max_abs"] = max(
        checks["r1_cmd_vs_fb_max_abs"], float(np.max(np.abs(arr(g["r1"]["ts_pose_command_0"]) - arr(g["r1"]["ts_pose_fb_0"]))))
    )
    vt_p = pose7_to_Rp(g["r1"]["ts_pose_virtual_target_0"])[1]
    r1_vt_off = np.linalg.norm(vt_p[A] - p_fb[A], axis=-1) * 1e3

    # 3 actions / obs on the chunk grid
    act = {k: action21(g[k], rt, wt) for k in data}
    chunks = {k: act[k][A].astype(np.float32) for k in data}
    geom = np.concatenate([sd[:, None], n, arr(m["mold_R_field_rot6d_0"])], axis=1)[O].astype(np.float32)
    eef = {k: arr(g[k]["ts_pose_fb_0"])[:, :3][O].astype(np.float32) for k in data}
    curv_o = arr(m["mold_curvature_0"])[O].astype(np.float32)
    F = np.linalg.norm(arr(g["r1"]["wrench_concat_0"])[:, 0:3], axis=1)
    pk, _ = find_peaks(F, height=FORCE_PEAK_N, distance=FORCE_PEAK_DIST)
    pk_idx = np.clip(np.searchsorted(rt, wt[pk]), 0, len(rt) - 1)

    # 4 twist about the normal (rgb grid)
    def twist_tilt(idx):
        M = np.einsum("nji,njk->nik", Rf[idx], R_fb[idx]) @ R_off.T
        psi_ = np.degrees(np.arctan2(M[:, 1, 0] - M[:, 0, 1], M[:, 0, 0] + M[:, 1, 1]))
        return psi_, np.degrees(np.arccos(np.clip(M[:, 2, 2], -1.0, 1.0))), (M[:, 0, 0] + M[:, 1, 1]) < 0

    psi, tilt, flipped = twist_tilt(ir)
    psi_pk, tilt_pk, _ = twist_tilt(pk_idx)
    v1 = v[ir, 0]
    travel = np.where(v1 > TRAVEL_MIN_MPS, 1, np.where(v1 < -TRAVEL_MIN_MPS, -1, 0))
    contact = near_rgb
    b_steps = flipped[steps]
    branch_change = (b_steps != b_steps[:, :1]).any(axis=1)
    near_boundary = np.abs(np.abs(psi[q]) - 90.0) < BRANCH_MARGIN_DEG
    B0 = np.where(flipped[q][:, None, None], (RZ_PI @ R_off)[None], R_off[None])
    head_err = geodesic_deg(np.einsum("qkij,qjl->qkil", Rf[A], B0), R_fb[A])
    Rd_end = np.einsum("qij,qjl->qil", dec["R"][:, -1], B0)

    # 5 per-sample labels for variation / phase
    uniq = np.unique(A)
    phase = (rt[uniq] - rt[uniq[0]]) / max(rt[uniq[-1]] - rt[uniq[0]], 1e-9)
    R_cmd = pose7_to_Rp(g["r1"]["ts_pose_command_0"])[0][uniq]
    cmd_rot_rms = float(np.sqrt(np.mean(geodesic_deg(R_cmd, polar_rotation(R_cmd.sum(0))[None]) ** 2)))

    print(
        f"[viz] {name}: chunks {len(q)}/{n_cand} flips {len(flips)} jumps {len(jumps)} "
        f"k23 p50 {np.median(err[:, -1]):.2f} mm near {near.mean():.2f}"
    )
    return {
        "id": eid,
        "split": "val" if eid in VAL_EPISODES else "train",
        "rt": rt, "rgb": rgb, "ir": ir, "q": q, "A": A, "O": O, "n_cand": n_cand,
        "p_fb": p_fb, "sd": sd, "x_s": x_s, "dist": dist,
        "flips": flips, "flip_big": flip_big, "jumps": jumps, "fstep": fstep,
        "big": big, "persistent": persistent, "exc_len": np.asarray(exc_len, dtype=int), "excursion": excursion,
        "label_hop": label_hop, "hop_back": hop_back, "near_rgb": near_rgb, "clip_rgb": clip_rgb,
        "n_flip_span": fl_in.sum(1), "chunks_per_flip": fl_in.sum(0), "n_jump_span": jm_in.sum(1),
        "obs_flip": obs_flip, "obs_jump": obs_jump, "obs_exc": obs_exc,
        "label_hit": label_hop[steps].any(axis=1),
        "label_hit_near": (label_hop & near_rgb)[steps].any(axis=1),
        "label_hit_persistent": (label_hop & ~hop_back)[steps].any(axis=1),
        "clip_hit": clip_rgb[steps].any(axis=1),
        "err": err, "near": near, "off_edge": off_edge, "r1_vt_off": r1_vt_off, "x_d": dec["x_d"].astype(np.float32),
        "chunks": chunks, "geom": geom, "eef": eef, "curv": curv_o,
        "sd_peak": sd[pk_idx], "sd_peak_scan": scan_sd(p_fb[pk_idx]), "dn_peak": dn[pk_idx], "n_peaks": len(pk),
        "psi_peak": psi_pk, "tilt_peak": tilt_pk,
        "psi": psi, "tilt": tilt, "flipped": flipped, "travel": travel, "contact": contact,
        "branch_change": branch_change, "near_boundary": near_boundary, "head_err": head_err,
        "samples": {k: act[k][uniq] for k in data}, "phase": phase, "cmd_rot_rms_deg": cmd_rot_rms,
        "rgb_view": {
            "R_fb": R_fb[ir], "Rf": Rf[ir], "curv": arr(m["mold_curvature_0"])[ir], "vt": vt_p[ir],
            "loc": arr(g["r3"]["ts_pose_fb_0"])[ir, :3], "v": v[ir], "dn": dn[ir],
            "flip_rgb": np.unique(np.clip(np.searchsorted(ir, flips + 1), 0, len(ir) - 1)),
        },
        "Rd_end": Rd_end,
    }


# ---------------------------------------------------------------- plotting


def mold_axes(field: FixBMoldField):
    V = field.vertices[~field.is_apron] if field.is_apron is not None else field.vertices
    c = V.mean(0)
    _, _, Wt = np.linalg.svd(V - c, full_matrices=False)
    if Wt[2, 1] > 0:  # fixture +Y is down; make the third axis point up
        Wt[2] *= -1
    return c, Wt, V[:: max(1, len(V) // 6000)]


def proj(P, c, Wt):
    return (np.asarray(P) - c) @ Wt.T * 1e3


def paged(eps, draw, prefix: Path, title: str, per_page=12):
    for p0 in range(0, len(eps), per_page):
        page = eps[p0 : p0 + per_page]
        fig, axs = plt.subplots(4, 3, figsize=(18, 12), squeeze=False)
        for ax, e in zip(axs.flat, page):
            draw(ax, e)
        for ax in axs.flat[len(page) :]:
            ax.axis("off")
        handles, labels = axs.flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right", fontsize=8, ncol=len(handles))
        fig.suptitle(title, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(f"{prefix}_p{p0 // per_page + 1}.png", dpi=85)
        plt.close(fig)


def ep_label(e):
    return f"ep {e['id']}" + (" (val)" if e["split"] == "val" else "")


def write_csv(path: Path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- sections


def section_flips(eps, field, out):
    tr = [e for e in eps if e["split"] == "train"]
    cat = lambda key, es=tr: np.concatenate([e[key] for e in es])
    n_chunks = sum(len(e["q"]) for e in tr)
    frac = lambda x: round(float(np.mean(x)), 4)
    obs_flip, obs_jump, obs_exc = cat("obs_flip"), cat("obs_jump"), cat("obs_exc")
    lab, lab_near, lab_pers, clip_hit = cat("label_hit"), cat("label_hit_near"), cat("label_hit_persistent"), cat("clip_hit")
    flip_span, jump_span = cat("n_flip_span"), cat("n_jump_span")
    cpf = cat("chunks_per_flip")
    exc_len = cat("exc_len")
    lh, back, near_rgb = cat("label_hop"), cat("hop_back"), cat("near_rgb")
    n_big = int(sum(len(e["big"]) for e in tr))
    n_pers = int(sum(len(e["persistent"]) for e in tr))
    flip_big = cat("flip_big")
    res = {
        "train_normal_flips": int(sum(len(e["flips"]) for e in tr)),
        "train_sd_jumps_over_50mm": int(sum(len(e["jumps"]) for e in tr)),
        "all_normal_flips": int(sum(len(e["flips"]) for e in eps)),
        "flips_with_foot_step_over_20mm": int(flip_big.sum()),
        "flips_inside_any_train_chunk_span": int((cpf > 0).sum()),
        "train_chunks_per_flip_mean": round(float(cpf.mean()), 2) if len(cpf) else 0.0,
        "train_chunks": int(n_chunks),
        "train_chunk_candidates_before_gap_filter": int(sum(e["n_cand"] for e in tr)),
        "frac_chunks": {
            "flip_in_action_span": frac(flip_span > 0),
            "sd_jump_in_action_span": frac(jump_span > 0),
            "flip_in_obs_window": frac(obs_flip),
            "foot_jump_in_obs_window": frac(obs_jump),
            "obs_sample_inside_excursion": frac(obs_exc),
            "label_step_foot_hop": frac(lab),
            "label_step_foot_hop_near_surface": frac(lab_near),
            "label_step_persistent_hop": frac(lab_pers),
            "clipped_v_step": frac(clip_hit),
            "flip_in_obs_or_near_surface_label_hop": frac(obs_flip | lab_near),
        },
        "obs_foot_jumps_robot_rate": {
            "count": n_big,
            "in_out_and_back_excursions": n_big - n_pers,
            "persistent_switches": n_pers,
            "excursion_length_samples_p50_p90_max": [*pct(exc_len, (50, 90)), int(exc_len.max()) if len(exc_len) else 0],
        },
        "label_foot_hops_rgb_grid": {
            "rgb_steps": int(len(lh)),
            "hop_steps": int(lh.sum()),
            "hop_steps_near_surface": int((lh & near_rgb).sum()),
            "hop_steps_out_and_back": int(back.sum()),
            "frac_steps": round(float(lh.mean()), 4),
            "frac_near_surface_steps": round(float((lh & near_rgb).sum() / max(near_rgb.sum(), 1)), 4),
        },
        "definitions": {
            "flip": "n(x_s) . n(x_s) of the next robot sample < 0 (obs, from measured fb)",
            "foot jump": f"|x_s step| > {FOOT_HOP_M * 1e3:.0f} mm in one robot step (5 ms)",
            "excursion": f"a foot jump that returns within {FOOT_HOP_M * 1e3:.0f} mm of the start within {EXCURSION_MAX} samples",
            "label foot hop": f"foot chord of the smoothed pose exceeds the tool chord by > {LABEL_HOP_EXCESS_M * 1e3:.0f} mm "
                              "over one RGB step (what v_par sees)",
            "near surface": f"closest surface point within {NEAR_M * 1e3:.0f} mm (|p - x_s|, not |sd|)",
            "action span": "robot samples from chunk step 0 to step 23",
            "obs window": f"{OBS_H} robot samples {OBS_DS} apart ending at the query",
        },
    }
    fc = res["frac_chunks"]
    res["decision"] = {
        "obs": (
            f"mask: {fc['flip_in_obs_window']:.1%} of chunks have a normal flip inside the obs window; "
            "drop those chunks (dataset-subclass index filter, no training code change)"
            if fc["flip_in_obs_window"] <= 0.05
            else f"repair: {fc['flip_in_obs_window']:.1%} of chunks have a flip in the obs window, too many to drop"
        ),
        "labels": (
            f"mask near-surface hops: {fc['label_step_foot_hop_near_surface']:.1%} of chunks; keep free-space hops "
            "(the decoder follows the same closest point)"
            if fc["label_step_foot_hop_near_surface"] <= 0.05
            else f"repair: {fc['label_step_foot_hop']:.1%} of chunks cross a foot hop "
                 f"({fc['label_step_foot_hop_near_surface']:.1%} near the surface) but only "
                 f"{res['label_foot_hops_rgb_grid']['frac_steps']:.1%} of steps; masking whole chunks is too costly. "
                 f"Most hops are persistent closest-point switches ({res['label_foot_hops_rgb_grid']['hop_steps_out_and_back']} "
                 f"of {res['label_foot_hops_rgb_grid']['hop_steps']} return the next frame), so interpolation does not fix them; "
                 "the fix is a temporally consistent foot used by both the labels and the decoder"
        ),
    }

    per_ep = [
        {
            "episode": e["id"], "split": e["split"], "chunks": len(e["q"]),
            "flips": len(e["flips"]), "sd_jumps_50mm": len(e["jumps"]),
            "foot_jumps": len(e["big"]), "persistent_switches": len(e["persistent"]),
            "label_hop_steps": int(e["label_hop"].sum()),
            "label_hop_steps_near": int((e["label_hop"] & e["near_rgb"]).sum()),
            "chunks_flip_in_span": int((e["n_flip_span"] > 0).sum()),
            "chunks_flip_in_obs": int(e["obs_flip"].sum()),
            "chunks_label_hop": int(e["label_hit"].sum()),
            "chunks_label_hop_near": int(e["label_hit_near"].sum()),
            "chunks_clip": int(e["clip_hit"].sum()),
        }
        for e in eps
    ]
    write_csv(out / "1_chunk_impact_per_episode.csv", per_ep)

    # overview raster + locations
    c, Wt, Vbg = mold_axes(field)
    dur = max(e["rt"][-1] - e["rt"][0] for e in eps)
    nb = int(np.ceil(dur)) + 1
    img = np.full((len(eps), nb), np.nan)
    for row, e in enumerate(eps):
        b = ((e["rt"][1:] - e["rt"][0]) // 1.0).astype(int)
        rowv = np.full(nb, -np.inf)
        np.maximum.at(rowv, b, e["fstep"])
        img[row] = np.where(np.isfinite(rowv), rowv, np.nan)
    fig = plt.figure(figsize=(22, 13))
    gs = fig.add_gridspec(2, 3, width_ratios=[2.4, 1, 1], height_ratios=[1, 1])
    ax = fig.add_subplot(gs[:, 0])
    im = ax.imshow(np.maximum(img, 1e-2), aspect="auto", cmap="viridis", norm=LogNorm(1e-2, 180), interpolation="nearest",
                   extent=(0, nb, len(eps) - 0.5, -0.5))
    for row, e in enumerate(eps):
        t0 = e["rt"][0]
        ax.plot(e["rt"][e["flips"] + 1] - t0, np.full(len(e["flips"]), row), "x", color="red", ms=5, mew=1.2)
        ax.plot(e["rt"][e["jumps"] + 1] - t0, np.full(len(e["jumps"]), row), "o", mfc="none", mec="white", ms=7, mew=1.2)
        ax.text(nb + 1, row, f"{len(e['flips'])}/{len(e['jumps'])}", va="center", fontsize=6)
    ax.set_yticks(range(len(eps)))
    ax.set_yticklabels([ep_label(e) for e in eps], fontsize=6)
    ax.set_xlabel("time in episode [s] (1 s bins; right column: flips / sd jumps)")
    ax.set_title("max R(x_s) frame-to-frame angle per 1 s [deg]; red x = normal flip, white o = |Δsd| > 50 mm")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.06)

    P_flip = np.concatenate([e["p_fb"][e["flips"] + 1] for e in eps]) if res["all_normal_flips"] else np.zeros((0, 3))
    X_flip = np.concatenate([e["x_s"][e["flips"] + 1] for e in eps]) if res["all_normal_flips"] else np.zeros((0, 3))
    P_jump = np.concatenate([e["p_fb"][e["jumps"] + 1] for e in eps])
    for k, (a, b, lab) in enumerate(((0, 1, "plan: along stringer vs across [mm]"), (0, 2, "elevation: along stringer vs up [mm]"))):
        axm = fig.add_subplot(gs[k, 1])
        vb = proj(Vbg, c, Wt)
        axm.scatter(vb[:, a], vb[:, b], s=0.3, c="0.8")
        for P, col, lbl in ((P_flip, "red", "tool at flip"), (X_flip, "orange", "hopped foot"), (P_jump, "k", "tool at sd jump")):
            pp = proj(P, c, Wt)
            axm.scatter(pp[:, a], pp[:, b], s=6, c=col, label=lbl)
        axm.set_title(lab, fontsize=9)
        axm.set_aspect("equal", adjustable="datalim")
        axm.legend(fontsize=7)
    axh = fig.add_subplot(gs[0, 2])
    lab_sd = np.concatenate([e["dist"][e["ir"][e["label_hop"]]] for e in tr]) * 1e3
    axh.hist(np.minimum(lab_sd, 300), bins=60, color="C1")
    axh.axvline(NEAR_M * 1e3, color="k", ls="--", lw=0.8)
    axh.set_xlabel("distance to closest surface point at label foot hops [mm] (300 = 300+); dashed = 30 mm")
    axh.set_title(
        f"label foot hops (train): {int(lh.sum())} of {len(lh)} RGB steps, "
        f"{int((lh & near_rgb).sum())} near the surface, {int(back.sum())} out-and-back",
        fontsize=8,
    )
    axt = fig.add_subplot(gs[1, 2])
    axt.axis("off")
    oj = res["obs_foot_jumps_robot_rate"]
    lines = [
        f"train flips {res['train_normal_flips']} ({res['flips_with_foot_step_over_20mm']} with foot step > 20 mm), "
        f"sd jumps {res['train_sd_jumps_over_50mm']}",
        f"flips inside >=1 train chunk: {res['flips_inside_any_train_chunk_span']} "
        f"(mean {res['train_chunks_per_flip_mean']} chunks each)",
        f"obs foot jumps {oj['count']}: {oj['persistent_switches']} persistent, "
        f"{oj['in_out_and_back_excursions']} in excursions",
        f"train chunks {n_chunks}; fraction with",
    ] + [f"  {k:<40s} {v:6.2%}" for k, v in fc.items()]
    lines += ["", "decision:"]
    for key in ("obs", "labels"):
        words, cur = res["decision"][key].split(), f"  {key}:"
        for w in words:
            if len(cur) + len(w) > 78:
                lines.append(cur)
                cur = "     "
            cur += " " + w
        lines.append(cur)
    axt.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "1_flip_timeline_overview.png", dpi=90)
    plt.close(fig)

    def draw(ax, e):
        t = e["rt"] - e["rt"][0]
        k = 10
        nbk = len(e["fstep"]) // k
        fs = e["fstep"][: nbk * k].reshape(nbk, k).max(1)
        ax.semilogy(t[1 : nbk * k + 1 : k], np.maximum(fs, 1e-3), lw=0.5, color="0.35", label="R(x_s) step (max / 50 ms)")
        lh = e["label_hop"]
        ln = lh & e["near_rgb"]
        ax.plot(e["rgb"][lh & ~ln] - e["rt"][0], np.full((lh & ~ln).sum(), 8.0), "|", color="0.6", ms=5,
                label="label foot hop, free space")
        ax.plot(e["rgb"][ln] - e["rt"][0], np.full(ln.sum(), 12.0), "|", color="C1", ms=6, label="label foot hop, within 30 mm")
        ax.plot(t[e["flips"] + 1], np.full(len(e["flips"]), 150.0), "v", color="red", ms=5, label="normal flip")
        ax.plot(t[e["jumps"] + 1], np.full(len(e["jumps"]), 40.0), "o", mfc="none", mec="k", ms=5, label="|Δsd| > 50 mm")
        ax.set_ylim(1e-3, 400)
        ax.set_title(
            f"{ep_label(e)}: flips {len(e['flips'])}, jumps {len(e['jumps'])}; chunks with flip in obs "
            f"{e['obs_flip'].sum()}, near-surface label hop {e['label_hit_near'].sum()} of {len(e['q'])}",
            fontsize=8,
        )
        ax.tick_params(labelsize=7)

    paged(eps, draw, out / "1_flip_timeline_episodes", "R(x_s) frame-to-frame angle [deg] vs time [s], robot rate")
    return res


def section_round_trip(eps, field, out):
    tr = [e for e in eps if e["split"] == "train"]
    va = [e for e in eps if e["split"] == "val"]

    def stats(es):
        err = np.concatenate([e["err"] for e in es])
        near = np.concatenate([e["near"] for e in es])
        vt_off = np.concatenate([e["r1_vt_off"] for e in es])
        edge = np.concatenate([e["off_edge"] for e in es])
        o = {"n_chunks": int(len(err)), "frac_near": round(float(near.mean()), 4),
             "frac_start_off_mesh_edge": round(float(edge.mean()), 4)}
        for lab, msk in (("near_30mm", near), ("not_near", ~near), ("not_near_on_mesh", ~near & ~edge),
                         ("start_off_mesh_edge", edge), ("all", np.ones_like(near))):
            e_ = err[msk]
            o[lab] = {
                "n_chunks": int(msk.sum()),
                "k0_mm_p50_p90": pct(e_[:, 0], (50, 90)),
                "k23_mm_p50_p90_p99": pct(e_[:, -1]),
                "full_horizon_mean_mm": round(float(e_.mean()), 3) if len(e_) else None,
                "frac_chunks_k23_under_5mm": round(float((e_[:, -1] < 5).mean()), 4) if len(e_) else None,
                "r1_virtual_target_vs_fb_mm_p50_p90": pct(vt_off[msk], (50, 90)),
            }
        return o, err, near

    res_tr, err, near = stats(tr)
    res = {"train": res_tr, "val": stats(va)[0],
           "off_mesh_edge": f"closest point on the mesh boundary with > {OFF_EDGE_M * 1e3:.0f} mm in-plane offset at the "
                            "chunk start; x_s + delta_n n cannot represent that offset, so k0 already misses by it",
           "note": "r2 and r3 labels are identical, so their round trip is the same. r1 actions are absolute "
                   "poses and r1 ts_pose_command_0 equals the measured fb exactly, so r1 has no round-trip error; "
                   "its virtual target sits off the demo path (offset listed)."}
    rows = []
    for e in eps:
        for lab, msk in (("near", e["near"]), ("not_near", ~e["near"])):
            rows.append({"episode": e["id"], "split": e["split"], "bucket": lab, "chunks": int(msk.sum()),
                         "k23_p50_mm": round(float(np.median(e["err"][msk, -1])), 3) if msk.any() else "",
                         "k23_p90_mm": round(float(np.percentile(e["err"][msk, -1], 90)), 3) if msk.any() else "",
                         "full_mean_mm": round(float(e["err"][msk].mean()), 3) if msk.any() else ""})
    write_csv(out / "2_round_trip_per_episode.csv", rows)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    k = np.arange(H)
    for msk, col, lab in ((near, "C2", "near (closest point < 30 mm, all steps)"), (~near, "C3", "not near")):
        e_ = err[msk]
        axs[0].plot(k, np.median(e_, 0), color=col, label=f"{lab} p50 (n={msk.sum()})")
        axs[0].plot(k, np.percentile(e_, 90, 0), color=col, ls="--", label=f"{lab} p90")
    axs[0].set_yscale("log")
    axs[0].set_xlabel("chunk step k")
    axs[0].set_ylabel("|x_d - demo fb| [mm]")
    axs[0].set_title("r2/r3 integrated labels vs demo, train chunks")
    axs[0].legend(fontsize=7)
    for msk, col, lab in ((near, "C2", "near"), (~near, "C3", "not near")):
        x = np.sort(err[msk, -1])
        axs[1].plot(x, np.linspace(0, 1, len(x)), color=col, label=lab)
    axs[1].set_xscale("log")
    axs[1].set_xlim(1e-2, 1e3)
    axs[1].set_xlabel("error at k=23 [mm]")
    axs[1].set_ylabel("CDF")
    axs[1].axvline(5, color="0.6", ls=":")
    axs[1].legend(fontsize=8)
    ep_ids = [e["id"] for e in eps]
    med_near = [np.median(e["err"][e["near"], -1]) if e["near"].any() else np.nan for e in eps]
    med_far = [np.median(e["err"][~e["near"], -1]) if (~e["near"]).any() else np.nan for e in eps]
    xx = np.arange(len(eps))
    axs[2].bar(xx - 0.2, med_near, 0.4, color="C2", label="near")
    axs[2].bar(xx + 0.2, med_far, 0.4, color="C3", label="not near")
    axs[2].set_yscale("log")
    axs[2].set_xticks(xx)
    axs[2].set_xticklabels([f"{i}*" if i in VAL_EPISODES else str(i) for i in ep_ids], fontsize=6, rotation=90)
    axs[2].set_ylabel("k=23 median [mm]")
    axs[2].set_title("per episode (* = val)")
    axs[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "2_round_trip_horizon.png", dpi=100)
    plt.close(fig)

    # demo paths for 4 train episodes spread over the error ranking
    c, Wt, Vbg = mold_axes(field)
    rank = sorted(tr, key=lambda e: np.median(e["err"][:, -1]))
    pick = [rank[int(f * (len(rank) - 1))] for f in (0.1, 0.4, 0.7, 0.95)]
    norm = LogNorm(0.5, 100)
    fig, axs = plt.subplots(len(pick), 4, figsize=(22, 4.6 * len(pick)), squeeze=False)
    vb = proj(Vbg, c, Wt)
    for r, e in enumerate(pick):
        demo = proj(e["p_fb"], c, Wt)
        for col, (msk_name, view) in enumerate(((m, v) for m in ("near", "not near") for v in ((0, 1), (0, 2)))):
            ax = axs[r, col]
            a, b = view
            ax.scatter(vb[:, a], vb[:, b], s=0.2, c="0.88")
            ax.plot(demo[:, a], demo[:, b], color="0.4", lw=0.5)
            msk = e["near"] if msk_name == "near" else ~e["near"]
            sel = np.flatnonzero(msk)[::4]
            if len(sel):
                xd = proj(e["x_d"][sel].reshape(-1, 3), c, Wt).reshape(len(sel), H, 3)
                segs = np.stack([xd[:, :-1][..., [a, b]], xd[:, 1:][..., [a, b]]], axis=2).reshape(-1, 2, 2)
                cols = e["err"][sel][:, 1:].reshape(-1)
                lc = LineCollection(segs, cmap="turbo", norm=norm, linewidths=1.0)
                lc.set_array(np.maximum(cols, 0.5))
                ax.add_collection(lc)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_title(
                f"{ep_label(e)} {msk_name} ({msk.sum()} chunks, every 4th) "
                f"{'plan' if b == 1 else 'elevation'}; k23 p50 "
                f"{np.median(e['err'][msk, -1]) if msk.any() else float('nan'):.1f} mm",
                fontsize=8,
            )
            ax.tick_params(labelsize=7)
    sm = plt.cm.ScalarMappable(norm=norm, cmap="turbo")
    fig.colorbar(sm, ax=axs, fraction=0.015, label="integrated label error [mm]")
    fig.suptitle("Fix B labels integrated from the measured pose (colored) over the demo path (gray); axes in mm along mold PCA axes")
    fig.savefig(out / "2_round_trip_paths.png", dpi=85)
    plt.close(fig)
    return res


def section_normalization(eps, out):
    tr = [e for e in eps if e["split"] == "train"]
    norms = {}
    X = {}
    for k in RUNS:
        X[k] = np.concatenate([e["chunks"][k].reshape(-1, 21) for e in tr]).astype(np.float64)
        norms[k] = action_normalizer(X[k])
    G = np.concatenate([e["geom"].reshape(-1, 10) for e in tr]).astype(np.float64)
    g_scale, g_off = range_normalizer(G)
    obs_norm = {"robot0_mold_geom": (g_scale, g_off)}
    for k in RUNS:
        E = np.concatenate([e["eef"][k].reshape(-1, 3) for e in tr]).astype(np.float64)
        obs_norm[f"{k}:robot0_eef_pos"] = range_normalizer(E)
    C = np.concatenate([e["curv"].reshape(-1, 4) for e in tr]).astype(np.float64)
    obs_norm["r3:robot0_mold_curv"] = range_normalizer(C)

    v = X["r2"][:, 0:2]
    clipped = np.abs(v) >= V_CLIP - 1e-6
    sd_peak_tr = np.concatenate([e["sd_peak"] for e in tr])
    sd_peak_all = np.concatenate([e["sd_peak"] for e in eps])
    sd_scan_tr = np.concatenate([e["sd_peak_scan"] for e in tr])
    sd_scan_all = np.concatenate([e["sd_peak_scan"] for e in eps])
    sd_all = G[:, 0]
    res = {
        "action_normalizer": {
            k: {d: {"scale_per_SI": round(float(norms[k][0][i]), 6), "offset": round(float(norms[k][1][i]), 6),
                    "raw_std_display": round(float(X[k][:, i].std() * unit(d)[1]), 4), "unit": unit(d)[0],
                    "normalized_std": round(float((X[k][:, i] * norms[k][0][i] + norms[k][1][i]).std()), 4),
                    "raw_min_max_display": [round(float(X[k][:, i].min() * unit(d)[1]), 3), round(float(X[k][:, i].max() * unit(d)[1]), 3)]}
                for i, d in enumerate(DIMS[k])}
            for k in RUNS
        },
        "obs_normalizer": {
            key: {"scale_per_SI": np.round(s, 5).tolist(), "offset": np.round(o, 5).tolist()} for key, (s, o) in obs_norm.items()
        },
        "v_par_clip": {
            "clip_mps": V_CLIP,
            "frac_steps_v_t1_clipped": round(float(clipped[:, 0].mean()), 5),
            "frac_steps_v_t2_clipped": round(float(clipped[:, 1].mean()), 5),
            "frac_steps_either_clipped": round(float(clipped.any(1).mean()), 5),
            "speed_mm_s_p50_p90_p99_unclipped": pct(np.linalg.norm(v[~clipped.any(1)], axis=1) * 1e3),
        },
        "sd_at_force_peaks": {
            "definition": f"find_peaks(|wrench_concat_0[:, 0:3]|, height={FORCE_PEAK_N} N, distance={FORCE_PEAK_DIST})",
            "train_peaks": int(len(sd_peak_tr)),
            "all_peaks": int(len(sd_peak_all)),
            "note": "field sd (scan + flange apron) is what the obs and delta_n use; scan-only sd is the distance "
                    "to the scanned faces alone (the earlier ~33 mm figure)",
            "train_sd_mm_p5_p25_p50_p75_p95": pct(sd_peak_tr * 1e3, (5, 25, 50, 75, 95)),
            "all_sd_mm_p5_p25_p50_p75_p95": pct(sd_peak_all * 1e3, (5, 25, 50, 75, 95)),
            "train_scan_only_sd_mm_p5_p25_p50_p75_p95": pct(sd_scan_tr * 1e3, (5, 25, 50, 75, 95)),
            "all_scan_only_sd_mm_p5_p25_p50_p75_p95": pct(sd_scan_all * 1e3, (5, 25, 50, 75, 95)),
            "train_frac_inside_surface": round(float((sd_peak_tr < 0).mean()), 4),
            "train_frac_abs_lt_5mm": round(float((np.abs(sd_peak_tr) < 0.005).mean()), 4),
            "train_normalized_p50": round(float(np.median(sd_peak_tr * g_scale[0] + g_off[0])), 4),
            "per_episode_median_mm": {e["id"]: round(float(np.median(e["sd_peak"]) * 1e3), 1) for e in eps if len(e["sd_peak"])},
        },
        "sd_obs_all_train_mm_p5_p50_p95": pct(sd_all * 1e3, (5, 50, 95)),
    }

    cols = [
        ("pos 0", [0]), ("pos 1", [1]), ("pos 2", [2]), ("cmd rot6d (identity)", list(range(3, 9))),
        ("vt pos", [9, 10, 11]), ("stiffness", [18]), ("grasp force", [20]),
    ]
    fig, axs = plt.subplots(3, len(cols) + 1, figsize=(30, 11))
    bins = np.linspace(-1.05, 1.05, 85)
    for r, k in enumerate(RUNS):
        Xn = X[k] * norms[k][0] + norms[k][1]
        for ci, (lab, dims) in enumerate(cols):
            ax = axs[r, ci]
            for d in dims:
                ax.hist(Xn[:, d], bins=bins, histtype="step", log=True, label=DIMS[k][d])
            d0 = dims[0]
            name = DIMS[k][d0]
            u, f = unit(name)
            raw_std = X[k][:, d0].std() * f
            if np.ptp(X[k][:, dims]) < 1e-7:
                info = "constant"
            elif len(dims) == 1:
                info = f"raw std {raw_std:.3g} {u}\nnorm std {Xn[:, d0].std():.3f}, 1 norm = {f / norms[k][0][d0]:.3g} {u}"
            else:
                info = f"norm std {', '.join(f'{Xn[:, d].std():.2f}' for d in dims)}"
            if name in ("v_t1", "v_t2"):
                info += f"\nclipped ±1 m/s: {clipped[:, d0].mean():.3%}"
            ax.set_title(f"{k} {name if len(dims) == 1 else lab}\n{info}", fontsize=8)
            if len(dims) > 1:
                ax.legend(fontsize=6)
            ax.tick_params(labelsize=7)
        ax = axs[r, -1]
        sn = sd_all * g_scale[0] + g_off[0]
        ax.hist(sn, bins=bins, histtype="step", log=True, density=True, label="all obs")
        ax.hist(sd_peak_tr * g_scale[0] + g_off[0], bins=bins, histtype="step", log=True, density=True, label="force peaks")
        ax.set_title(f"{k} obs mold sd (normalized, density)\n1 norm = {1e3 / g_scale[0]:.0f} mm; norm std {sn.std():.3f}", fontsize=8)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=7)
    fig.suptitle("train chunk steps after each dataset's normalizer (x axis normalized, log count)")
    fig.tight_layout()
    fig.savefig(out / "3_normalized_histograms.png", dpi=85)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    bins_mm = np.linspace(-100, 250, 141)
    axs[0].hist(sd_all * 1e3, bins=bins_mm, density=True, histtype="step", color="0.5", label="all train obs samples")
    axs[0].hist(sd_peak_tr * 1e3, bins=bins_mm, density=True, histtype="step", color="C3", label=f"train force peaks (n={len(sd_peak_tr)})")
    axs[0].hist(sd_scan_tr * 1e3, bins=bins_mm, density=True, histtype="step", color="C4", ls="--",
                label="train force peaks, scan only (no apron)")
    med, med_scan = np.median(sd_peak_tr) * 1e3, np.median(sd_scan_tr) * 1e3
    axs[0].axvline(0, color="k", lw=0.8)
    axs[0].axvline(med, color="C3", ls="--", label=f"median {med:.1f} mm (field)")
    axs[0].axvline(med_scan, color="C4", ls=":", label=f"median {med_scan:.1f} mm (scan only)")
    axs[0].set_xlabel("signed distance of measured tool to scan [mm] (+ = free space)")
    axs[0].legend(fontsize=8)
    sec = axs[0].secondary_xaxis("top", functions=(lambda x: x * 1e-3 * g_scale[0] + g_off[0], lambda y: (y - g_off[0]) / g_scale[0] * 1e3))
    sec.set_xlabel("normalized (robot0_mold_geom[0])", fontsize=8)
    ids = [e["id"] for e in eps if len(e["sd_peak"])]
    meds = [np.median(e["sd_peak"]) * 1e3 for e in eps if len(e["sd_peak"])]
    axs[1].bar(range(len(ids)), meds, color=["C1" if i in VAL_EPISODES else "C0" for i in ids])
    axs[1].set_xticks(range(len(ids)))
    axs[1].set_xticklabels(ids, fontsize=6, rotation=90)
    axs[1].set_ylabel("median sd at force peaks [mm]")
    axs[1].set_title("per episode (orange = val)")
    dn_n = X["r2"][:, 2] * norms["r2"][0][2] + norms["r2"][1][2]
    dn_pk = np.concatenate([e["dn_peak"] for e in tr]) * norms["r2"][0][2] + norms["r2"][1][2]
    axs[2].hist(dn_n, bins=bins, density=True, histtype="step", label="delta_n, train chunk steps")
    axs[2].hist(dn_pk, bins=bins, density=True, histtype="step", label="delta_n at force peaks")
    axs[2].set_xlabel(f"r2/r3 delta_n normalized (1 norm = {1e3 / norms['r2'][0][2]:.0f} mm)")
    axs[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "3_sd_at_force_peaks.png", dpi=100)
    plt.close(fig)
    return res, norms, X


def section_twist(eps, out):
    tr = [e for e in eps if e["split"] == "train"]
    psi = np.concatenate([e["psi"] for e in tr])
    flipped = np.concatenate([e["flipped"] for e in tr])
    travel = np.concatenate([e["travel"] for e in tr])
    contact = np.concatenate([e["contact"] for e in tr])
    bc = np.concatenate([e["branch_change"] for e in tr])
    nb = np.concatenate([e["near_boundary"] for e in tr])
    he = np.concatenate([e["head_err"] for e in tr])
    tilt = np.concatenate([e["tilt"] for e in tr])
    psi_pk = np.concatenate([e["psi_peak"] for e in tr])
    tilt_pk = np.concatenate([e["tilt_peak"] for e in tr])
    resid = np.minimum(np.abs(psi), 180 - np.abs(psi))
    resid_pk = np.minimum(np.abs(psi_pk), 180 - np.abs(psi_pk))
    near_ch = np.concatenate([e["near"] for e in tr])
    moving = contact & (travel != 0)
    agree = (flipped[moving] == (travel[moving] < 0)).mean() if moving.any() else float("nan")
    res = {
        "frac_rgb_samples_flipped_branch": round(float(flipped.mean()), 4),
        "frac_flipped_in_contact": round(float(flipped[contact].mean()), 4),
        "twist_residual_deg_p50_p90": {"near_surface": pct(resid[contact], (50, 90)), "free_space": pct(resid[~contact], (50, 90)),
                                       "force_peaks": pct(resid_pk, (50, 90))},
        "tilt_tool_z_from_minus_n_deg_p50_p90": {"near_surface": pct(tilt[contact], (50, 90)),
                                                 "free_space": pct(tilt[~contact], (50, 90)),
                                                 "force_peaks": pct(tilt_pk, (50, 90))},
        "flipped_vs_reverse_travel_agreement_in_contact": round(float(agree), 4),
        "train_chunks": int(len(bc)),
        "frac_chunks_branch_changes_within_chunk": round(float(bc.mean()), 4),
        "frac_chunks_start_within_15deg_of_boundary": round(float(nb.mean()), 4),
        "frac_chunks_could_flip": round(float((bc | nb).mean()), 4),
        "frac_chunks_could_flip_near_chunks": round(float((bc | nb)[near_ch].mean()), 4),
        "frac_chunks_could_flip_not_near": round(float((bc | nb)[~near_ch].mean()), 4),
        "frac_samples_psi_within_15deg_of_90": {
            "near_surface": round(float((np.abs(np.abs(psi[contact]) - 90) < BRANCH_MARGIN_DEG).mean()), 4),
            "free_space": round(float((np.abs(np.abs(psi[~contact]) - 90) < BRANCH_MARGIN_DEG).mean()), 4),
        },
        "decoded_heading_err_deg_k0_p50_p90": pct(he[:, 0], (50, 90)),
        "decoded_heading_err_deg_k23_p50_p90": pct(he[:, -1], (50, 90)),
        "decoded_heading_err_deg_k23_p50_p90_near_chunks": pct(he[near_ch, -1], (50, 90)),
        "definitions": {
            "psi": "rotation about n of R_field^T R_tool R_off^T (0 = R_off branch, ±180 = reversed)",
            "tilt": "angle between the z axis of R_field^T R_tool R_off^T and n (tool z vs inward normal, after R_off)",
            "flipped": "heading_branch picks diag(-1,-1,1) R_off (|psi| > 90)",
            "could flip": f"branch differs at any chunk step, or |psi| at the chunk start within {BRANCH_MARGIN_DEG:.0f} deg of 90",
            "heading err": "angle between R(x_s demo) B0 (B0 from the chunk start) and the demo tool",
            "reverse travel": f"v_t1 < -{TRAVEL_MIN_MPS * 1e3:.0f} mm/s with the closest point within 30 mm",
        },
    }
    rows = [
        {"episode": e["id"], "split": e["split"], "frac_flipped": round(float(e["flipped"].mean()), 3),
         "frac_chunks_branch_change": round(float(e["branch_change"].mean()), 3),
         "frac_chunks_near_boundary": round(float(e["near_boundary"].mean()), 3),
         "twist_resid_p50_deg": round(float(np.median(np.minimum(np.abs(e["psi"]), 180 - np.abs(e["psi"])))), 1)}
        for e in eps
    ]
    write_csv(out / "4_twist_per_episode.csv", rows)

    def draw(ax, e):
        t = e["rgb"] - e["rt"][0]
        f = e["flipped"]
        ax.scatter(t[~f], e["psi"][~f], s=1, c="C0", label="R_off branch")
        ax.scatter(t[f], e["psi"][f], s=1, c="C3", label="180° (reverse) branch")
        for y in (-90, 90):
            ax.axhline(y, color="0.6", lw=0.6, ls="--")
        cf = e["branch_change"] | e["near_boundary"]
        tq = t[e["q"][cf]]
        ax.plot(tq, np.full(len(tq), 195), "|", color="k", ms=6, label="chunk start that could flip")
        rev = (e["travel"] < 0) & e["contact"]
        ax.plot(t[rev], np.full(rev.sum(), -195), "|", color="purple", ms=3, label="reverse travel (v_t1<0, contact)")
        ax.set_ylim(-210, 210)
        ax.set_yticks([-180, -90, 0, 90, 180])
        ax.set_title(f"{ep_label(e)}: flipped {f.mean():.0%}, could-flip chunks {cf.mean():.0%}", fontsize=8)
        ax.tick_params(labelsize=7)

    paged(eps, draw, out / "4_twist_episodes", "twist about the normal psi [deg] vs time [s] (RGB grid)")

    fig, axs = plt.subplots(1, 4, figsize=(22, 4.8))
    axs[0].hist(psi[contact], bins=72, range=(-180, 180), histtype="step", density=True, label="within 30 mm")
    axs[0].hist(psi[~contact], bins=72, range=(-180, 180), histtype="step", density=True, label="free space")
    axs[0].set_xlabel("psi [deg]")
    axs[0].legend(fontsize=8)
    axs[0].set_title("twist about n relative to R_off")
    axs[1].hist(resid[contact], bins=45, range=(0, 90), histtype="step", density=True, label="twist, within 30 mm")
    axs[1].hist(resid_pk, bins=45, range=(0, 90), histtype="step", density=True, label="twist, force peaks")
    axs[1].hist(tilt[contact], bins=45, range=(0, 90), histtype="step", density=True, label="tilt, within 30 mm")
    axs[1].hist(tilt_pk, bins=45, range=(0, 90), histtype="step", density=True, label="tilt, force peaks")
    axs[1].set_xlabel("deg")
    axs[1].set_title("twist residual after branch vs tool-z tilt from -n")
    axs[1].legend(fontsize=7)
    k = np.arange(H)
    for msk, col, lab in ((near_ch, "C2", "near chunks"), (~near_ch, "C3", "not near")):
        axs[2].plot(k, np.median(he[msk], 0), color=col, label=f"{lab} p50")
        axs[2].plot(k, np.percentile(he[msk], 90, 0), color=col, ls="--", label=f"{lab} p90")
    axs[2].set_xlabel("chunk step k")
    axs[2].set_ylabel("deg")
    axs[2].set_title("decoded heading R(x_s) B0 vs demo tool")
    axs[2].legend(fontsize=8)
    xx = np.arange(len(eps))
    axs[3].bar(xx - 0.2, [e["flipped"].mean() for e in eps], 0.4, label="frac flipped samples")
    axs[3].bar(xx + 0.2, [(e["branch_change"] | e["near_boundary"]).mean() for e in eps], 0.4, label="frac could-flip chunks")
    axs[3].set_xticks(xx)
    axs[3].set_xticklabels([f"{e['id']}*" if e["split"] == "val" else str(e["id"]) for e in eps], fontsize=6, rotation=90)
    axs[3].legend(fontsize=8)
    axs[3].set_title("per episode (* = val)")
    fig.tight_layout()
    fig.savefig(out / "4_twist_summary.png", dpi=100)
    plt.close(fig)
    return res


def section_variation(eps, norms, X, out):
    tr = [e for e in eps if e["split"] == "train"]
    long_rows, summary_rows, phase_rows = [], [], []
    nonconst = {k: [i for i in range(21) if np.ptp(X[k][:, i]) >= 1e-7] for k in RUNS}
    for k in RUNS:
        scale, offset = norms[k]
        for e in eps:
            S = e["samples"][k]
            for i in nonconst[k]:
                d = DIMS[k][i]
                u, f = unit(d)
                s = S[:, i].std()
                long_rows.append({"episode": e["id"], "split": e["split"], "run": k, "dim": d, "unit": u,
                                  "std_phys": round(float(s * f), 5), "std_norm": round(float(s * scale[i]), 5)})
            if k == "r1":
                long_rows.append({"episode": e["id"], "split": e["split"], "run": k, "dim": "cmd_rot_angle_rms",
                                  "unit": "deg", "std_phys": round(e["cmd_rot_rms_deg"], 3), "std_norm": ""})
        pooled = np.concatenate([e["samples"][k] for e in tr])
        for i in nonconst[k]:
            d = DIMS[k][i]
            u, f = unit(d)
            per = np.array([e["samples"][k][:, i].std() for e in tr])
            means = np.full((len(tr), N_PHASE), np.nan)
            stds = np.full((len(tr), N_PHASE), np.nan)
            for j, e in enumerate(tr):
                b = np.minimum((e["phase"] * N_PHASE).astype(int), N_PHASE - 1)
                for p in range(N_PHASE):
                    x = e["samples"][k][b == p, i]
                    if len(x):
                        means[j, p], stds[j, p] = x.mean(), x.std()
            between, within = np.nanstd(means, 0), np.nanmean(stds, 0)
            for p in range(N_PHASE):
                phase_rows.append({"run": k, "dim": d, "unit": u, "phase_bin": f"{p / N_PHASE:.1f}-{(p + 1) / N_PHASE:.1f}",
                                   "between_episode_std_phys": round(float(between[p] * f), 5),
                                   "within_episode_std_phys": round(float(within[p] * f), 5),
                                   "between_episode_std_norm": round(float(between[p] * scale[i]), 5),
                                   "within_episode_std_norm": round(float(within[p] * scale[i]), 5)})
            summary_rows.append({
                "run": k, "dim": d, "unit": u,
                "norm_scale_per_unit": round(float(scale[i] / f), 6),
                "pooled_std_phys": round(float(pooled[:, i].std() * f), 5),
                "pooled_std_norm": round(float(pooled[:, i].std() * scale[i]), 5),
                "per_episode_std_phys_p10": round(float(np.percentile(per, 10) * f), 5),
                "per_episode_std_phys_p50": round(float(np.median(per) * f), 5),
                "per_episode_std_phys_p90": round(float(np.percentile(per, 90) * f), 5),
                "per_episode_std_norm_p50": round(float(np.median(per) * scale[i]), 5),
                "phase_between_episode_std_norm_median": round(float(np.median(between) * scale[i]), 5),
                "phase_within_episode_std_norm_median": round(float(np.median(within) * scale[i]), 5),
            })
    write_csv(out / "5_variation_per_episode.csv", long_rows)
    write_csv(out / "5_variation_summary.csv", summary_rows)
    write_csv(out / "5_phase_spread.csv", phase_rows)

    # PCA on train chunks, normalized as the model sees them
    pca = {}
    curves = {}
    for k in RUNS:
        scale, offset = norms[k]
        C = np.concatenate([e["chunks"][k] for e in tr]).astype(np.float64) * scale + offset  # (Q,H,21)
        for set_name, dims in (("motion", MOTION_DIMS[k]), ("full", nonconst[k])):
            for shape_only in (False, True):
                Y = C[:, :, dims]
                if shape_only:
                    Y = Y - Y.mean(axis=1, keepdims=True)
                Y = Y.reshape(len(Y), -1)
                s = np.linalg.svd(Y - Y.mean(0), compute_uv=False)
                ev = s**2 / np.sum(s**2)
                cum = np.cumsum(ev)
                key = f"{set_name}{'_shape' if shape_only else ''}"
                curves[(k, key)] = cum
                pca.setdefault(k, {})[key] = {
                    "dims": [DIMS[k][i] for i in dims],
                    "n_components_90_95_99": [int(np.searchsorted(cum, x) + 1) for x in (0.90, 0.95, 0.99)],
                    "first5_explained": np.round(ev[:5], 4).tolist(),
                }

    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    for k in RUNS:
        for key, ls in (("motion", "-"), ("motion_shape", ":")):
            cum = curves[(k, key)]
            axs[0].plot(np.arange(1, 31), cum[:30], ls=ls, color=COLORS[k], lw=2.5 if k == "r2" else 1.2,
                        label=f"{k} {key}")
        for key, ls in (("full", "-"), ("full_shape", ":")):
            cum = curves[(k, key)]
            axs[1].plot(np.arange(1, 31), cum[:30], ls=ls, color=COLORS[k], lw=2.5 if k == "r2" else 1.2,
                        label=f"{k} {key}")
    for ax, t in zip(axs[:2], ("motion dims (r1 cmd pose9; r2/r3 v_t1, v_t2, delta_n)", "all non-constant action dims")):
        ax.set_xlabel("components")
        ax.set_ylabel("cumulative explained variance")
        ax.set_title(f"PCA of 24-step train chunks, {t}", fontsize=9)
        ax.axhline(0.95, color="0.6", lw=0.6)
        ax.set_ylim(0, 1.01)
        ax.legend(fontsize=7)
    sm = {r["run"] + ":" + r["dim"]: r for r in summary_rows}
    labels, per_ep = [], []
    for k in RUNS:
        for i in MOTION_DIMS[k] + [18, 20]:
            d = DIMS[k][i]
            if i not in nonconst[k]:
                continue
            labels.append(f"{k}:{d}")
            per_ep.append([e["samples"][k][:, i].std() * norms[k][0][i] for e in tr])
    axs[2].boxplot(per_ep, vert=False, tick_labels=labels)
    axs[2].tick_params(labelsize=7)
    axs[2].set_xlabel("per-episode std, normalized units (train)")
    axs[2].set_title("per-episode label std (dataset normalization)")
    fig.tight_layout()
    fig.savefig(out / "5_pca_and_per_episode_std.png", dpi=100)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(24, 8))
    for ax, k in zip(axs, RUNS):
        dims = [DIMS[k][i] for i in nonconst[k]]
        M = np.array([[r["between_episode_std_norm"] for r in phase_rows if r["run"] == k and r["dim"] == d] for d in dims])
        W = np.array([[r["within_episode_std_norm"] for r in phase_rows if r["run"] == k and r["dim"] == d] for d in dims])
        im = ax.imshow(M, aspect="auto", cmap="magma", vmin=0, vmax=max(0.05, float(np.nanmax(M))))
        for (y, x), val in np.ndenumerate(M):
            ax.text(x, y, f"{val:.2f}\n({W[y, x]:.2f})", ha="center", va="center", fontsize=5, color="w" if val < 0.6 * M.max() else "k")
        ax.set_yticks(range(len(dims)))
        ax.set_yticklabels(dims, fontsize=7)
        ax.set_xticks(range(N_PHASE))
        ax.set_xticklabels([f"{p / N_PHASE:.1f}" for p in range(N_PHASE)], fontsize=7)
        ax.set_xlabel("task phase (normalized time over the episode's chunk range)")
        ax.set_title(f"{k}: across-episode std of phase means (within-episode std), normalized", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.03)
    fig.suptitle("single mold in the data (stringer_scan_20261008_170729): across-episode spread stands in for across-mold spread")
    fig.tight_layout()
    fig.savefig(out / "5_phase_spread.png", dpi=95)
    plt.close(fig)
    return {"pca": pca, "summary": summary_rows}


SCRUB_FLAGS = {
    "near": "closest point within 30 mm over the whole chunk",
    "off_edge": "in-plane offset > 5 mm at k=0 (tool beside the mesh edge)",
    "could_flip": "heading branch changes in the chunk or |psi| within 15 deg of 90 at k=0",
    "label_hop": "label foot hop in the action span",
    "label_hop_near": "near-surface label foot hop in the action span",
    "clip": "|v_par| clipped at 1 m/s in the action span",
    "flip_obs": "normal flip in the obs window",
    "flip_span": "normal flip in the action span",
    "jump_obs": "foot jump > 20 mm / 5 ms in the obs window",
}


def rnd(x, nd):
    return np.round(np.asarray(x, dtype=np.float64).reshape(-1), nd).tolist()


def display_mesh(field: FixBMoldField, apron: bool, n_tri: int):
    if field.is_apron is None:
        faces = field.faces[:0] if apron else field.faces
    else:
        faces = field.faces[field.is_apron[field.faces].any(axis=1) == apron]
    if len(faces) == 0:
        return None
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(field.vertices), o3d.utility.Vector3iVector(faces))
    mesh.remove_unreferenced_vertices()
    if len(faces) > n_tri:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=n_tri)
    V, F = np.asarray(mesh.vertices) * 1e3, np.asarray(mesh.triangles)
    return {"x": rnd(V[:, 0], 1), "y": rnd(V[:, 1], 1), "z": rnd(V[:, 2], 1),
            "i": F[:, 0].tolist(), "j": F[:, 1].tolist(), "k": F[:, 2].tolist(), "n_tri": int(len(F))}


def scrub_episode(e):
    rv, ir = e["rgb_view"], e["ir"]
    bits = [e["near"], e["off_edge"], e["branch_change"] | e["near_boundary"], e["label_hit"], e["label_hit_near"],
            e["clip_hit"], e["obs_flip"], e["n_flip_span"] > 0, e["obs_jump"]]
    flags = sum(b.astype(int) << i for i, b in enumerate(bits))
    return {
        "split": e["split"],
        "t": rnd(e["rgb"] - e["rgb"][0], 3),
        "P": rnd(e["p_fb"][ir] * 1e3, 1), "Rt": rnd(rv["R_fb"], 3),
        "X": rnd(e["x_s"][ir] * 1e3, 1), "Rf": rnd(rv["Rf"], 3),
        "sd": rnd(e["sd"][ir] * 1e3, 1), "dist": rnd(e["dist"][ir] * 1e3, 1),
        "psi": rnd(e["psi"], 1), "tilt": rnd(e["tilt"], 1), "flip": e["flipped"].astype(int).tolist(),
        "v": rnd(rv["v"] * 1e3, 1), "dn": rnd(rv["dn"] * 1e3, 1), "vt": rnd(rv["vt"] * 1e3, 1),
        "curv": rnd(rv["curv"][:, :2], 3), "loc": rnd(rv["loc"] * 1e3, 1),
        "flips": rv["flip_rgb"].tolist(), "hops": np.flatnonzero(e["label_hop"]).tolist(),
        "q": e["q"].tolist(), "xd": rnd(e["x_d"] * 1e3, 1), "err": rnd(e["err"], 1),
        "Rd": rnd(e["Rd_end"], 3), "he": rnd(e["head_err"][:, [0, -1]], 1), "flags": flags.tolist(),
    }


def section_scrubber(eps, field, out, ids=None):
    from plotly.offline import get_plotlyjs

    tr = [e for e in eps if e["split"] == "train"]
    if not ids:
        rank = sorted(tr, key=lambda e: np.median(e["err"][:, -1]))
        ids = [e["id"] for e in eps if e["split"] == "val"]
        ids += [rank[int(f * (len(rank) - 1))]["id"] for f in (0.1, 0.4, 0.7, 0.95)]
        ids.append(max(tr, key=lambda e: len(e["flips"]))["id"])
    by_id = {e["id"]: e for e in eps}
    ids = [i for i in dict.fromkeys(ids) if i in by_id]
    payload = {
        "H": H, "chunk_stride": QUERY_STRIDE, "near_mm": NEAR_M * 1e3, "off_edge_mm": OFF_EDGE_M * 1e3,
        "branch_margin_deg": BRANCH_MARGIN_DEG, "flags": list(SCRUB_FLAGS), "flag_text": SCRUB_FLAGS,
        "mesh": {"scan": display_mesh(field, False, 30000), "apron": display_mesh(field, True, 4000)},
        "order": [str(i) for i in ids],
        "episodes": {str(i): scrub_episode(by_id[i]) for i in ids},
    }
    html = SCRUB_HTML.replace("/*DATA*/", json.dumps(payload, separators=(",", ":"))).replace("/*PLOTLY*/", get_plotlyjs())
    path = out / "6_scrubber_fixb_only.html"
    path.write_text(html)
    print(f"[viz] scrubber {path} ({path.stat().st_size / 1e6:.1f} MB, episodes {ids})")
    return {"file": path.name, "episodes": ids, "size_mb": round(path.stat().st_size / 1e6, 1)}


SCRUB_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Fix B pretrain scrubber</title>
<script>/*PLOTLY*/</script>
<style>
body{margin:0;font:13px system-ui,sans-serif;color:#222;background:#fff}
#bar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;padding:6px 10px;border-bottom:1px solid #ddd}
#bar label{white-space:nowrap}
#slider{flex:1;min-width:280px}
#wrap{display:grid;grid-template-columns:minmax(0,1fr) 360px;grid-template-rows:60vh 34vh;gap:4px;padding:4px}
#scene{grid-column:1;grid-row:1;min-width:0}
#side{grid-column:2;grid-row:1/3;overflow:auto;border-left:1px solid #eee;padding:0 10px}
#bottom{grid-column:1;grid-row:2;display:flex;gap:6px;min-width:0}
.cell{flex:1;display:flex;flex-direction:column;min-width:0}
#cell_tl{flex:1.6}
.cap{font-size:11px;line-height:1.35;padding:2px 4px;color:#444}
.plot{flex:1;min-height:0}
table{border-collapse:collapse;width:100%}
td{padding:2px 4px;border-bottom:1px solid #f2f2f2;vertical-align:top}
td:first-child{color:#555;white-space:nowrap}
.on{color:#b00020;font-weight:600}
.off{color:#999}
h3{margin:10px 0 4px;font-size:13px}
.note{color:#666;font-size:11px;line-height:1.4}
.sw{display:inline-block;width:10px;height:10px;margin-right:3px;vertical-align:middle}
</style></head><body>
<div id="bar">
  <label>episode <select id="ep"></select></label>
  <button id="prev">&lt;</button>
  <input type="range" id="slider" min="0" max="0" value="0">
  <button id="next">&gt;</button>
  <button id="play">play</button>
  <label>speed <select id="speed"><option value="0.5">0.5x</option><option value="1" selected>1x</option><option value="2">2x</option><option value="4">4x</option></select></label>
  <label>jump to <select id="flag"></select></label>
  <button id="fprev">prev</button><button id="fnext">next</button>
  <label><input type="checkbox" id="follow" checked> follow tool, box</label>
  <select id="box"><option value="75">&plusmn;75 mm</option><option value="150" selected>&plusmn;150 mm</option><option value="300">&plusmn;300 mm</option><option value="600">&plusmn;600 mm</option></select>
  <label><span class="sw" style="background:#1f77b4"></span><input type="checkbox" id="show_r1" checked> r1 world/world</label>
  <label><span class="sw" style="background:#ff7f0e"></span><input type="checkbox" id="show_r2" checked> r2 world/fixb</label>
  <label><span class="sw" style="background:#2ca02c"></span><input type="checkbox" id="show_r3" checked> r3 surface/fixb</label>
  <label><input type="checkbox" id="legend" checked> legend</label>
</div>
<div id="wrap">
  <div id="scene"></div>
  <div id="side"></div>
  <div id="bottom">
    <div class="cell" id="cell_r1"><div class="cap"><b>r1 label chunk</b>: command position minus k=0 [mm] vs step k:
      <span style="color:#d62728">x</span>, <span style="color:#2ca02c">y</span>, <span style="color:#1f77b4">z</span></div>
      <div class="plot" id="lab_r1"></div></div>
    <div class="cell" id="cell_fixb"><div class="cap"><b>r2/r3 label chunk</b> vs step k:
      <span style="color:#d62728">v_t1</span>, <span style="color:#2ca02c">v_t2</span> [mm/s, left];
      <span style="color:#1f77b4">delta_n</span> [mm, right]; r3 dotted (same values)</div>
      <div class="plot" id="lab_fixb"></div></div>
    <div class="cell" id="cell_tl"><div class="cap"><b>episode timeline</b>: |p - x_s| (black), signed distance (gray) [mm];
      twist psi [deg] (<span style="color:#1f77b4">R_off</span> / <span style="color:#d62728">180 deg</span> branch);
      <span style="color:#d62728">&times;</span> normal flip, <span style="color:#9467bd">&#9650;</span> label foot hop;
      shaded = current chunk; click to jump</div>
      <div class="plot" id="timeline"></div></div>
  </div>
</div>
<script>
const D = /*DATA*/;
const H = D.H;
const RC = {r1: "#1f77b4", r2: "#ff7f0e", r3: "#2ca02c"};
const AXC = ["#d62728", "#2ca02c", "#1f77b4"];
const ERR_CS = [[0, "#fff5eb"], [0.1, "#fdd0a2"], [0.3, "#fd8d3c"], [0.6, "#d94801"], [1, "#7f2704"]];
const $ = id => document.getElementById(id);
const scene = $("scene"), labR1 = $("lab_r1"), labFixb = $("lab_fixb"), tl = $("timeline"), slider = $("slider");
const CFG = {displaylogo: false, responsive: true};
let E = null, ci = 0, timer = null, tlBound = false;
const show = {r1: true, r2: true, r3: true};

const p3 = (a, i) => [a[3 * i], a[3 * i + 1], a[3 * i + 2]];
const c3 = (R, i, c) => [R[9 * i + c], R[9 * i + 3 + c], R[9 * i + 6 + c]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const deg = r => r * 180 / Math.PI;
const seg = (p, d, s) => ({x: [p[0], p[0] + s * d[0]], y: [p[1], p[1] + s * d[1]], z: [p[2], p[2] + s * d[2]]});
const pts = l => ({x: l.map(p => p[0]), y: l.map(p => p[1]), z: l.map(p => p[2])});
const f1 = x => (x === null || Number.isNaN(x)) ? "nan" : x.toFixed(1);

function line3(name, color, width, extra) {
  return Object.assign({type: "scatter3d", mode: "lines", x: [], y: [], z: [], name, hoverinfo: "skip",
                        line: {color, width}}, extra || {});
}
function meshTrace(m, name, color, opacity) {
  return {type: "mesh3d", x: m.x, y: m.y, z: m.z, i: m.i, j: m.j, k: m.k, name, color, opacity,
          showlegend: true, hoverinfo: "skip", flatshading: true,
          lighting: {ambient: 0.55, diffuse: 0.75, specular: 0.05, roughness: 0.9}};
}

const DYN = [
  ["pts", null], ["near", null], ["ntxt", null], ["tx", null], ["ty", null], ["tz", null], ["ft1", null], ["ft2", null], ["fn", null],
  ["r1", "r1"], ["r1vt", "r1"], ["r1x", "r1"], ["r1y", "r1"], ["r1z", "r1"],
  ["r2", "r2"], ["r3", "r3"], ["dx", "fixb"], ["dy", "fixb"], ["dz", "fixb"],
];
function dynTraces() {
  const dash = {dash: "dash"};
  return [
    {type: "scatter3d", mode: "markers", x: [], y: [], z: [], name: "tool p (black), nearest point x_s (magenta)",
     marker: {size: 5, color: ["#000000", "#d62ad6"]}, hoverinfo: "skip"},
    line3("nearest-point line p to x_s", "#000000", 6),
    {type: "scatter3d", mode: "text", x: [], y: [], z: [], text: [], name: "distance / angle", showlegend: false,
     textposition: "middle right", textfont: {size: 11, color: "#000000"}, hoverinfo: "skip"},
    line3("tool x", AXC[0], 8, {legendgroup: "tool", showlegend: false}),
    line3("tool y", AXC[1], 8, {legendgroup: "tool", showlegend: false}),
    line3("tool axes x/y/z (solid, measured)", AXC[2], 8, {legendgroup: "tool"}),
    line3("t1", AXC[0], 4, {line: Object.assign({color: AXC[0], width: 4}, dash), legendgroup: "field", showlegend: false}),
    line3("t2", AXC[1], 4, {line: Object.assign({color: AXC[1], width: 4}, dash), legendgroup: "field", showlegend: false}),
    line3("R(x_s) t1/t2/n (dashed, at x_s)", AXC[2], 4, {line: Object.assign({color: AXC[2], width: 4}, dash), legendgroup: "field"}),
    {type: "scatter3d", mode: "lines+markers", x: [], y: [], z: [], name: "r1 label: 24 command positions (= future fb)",
     line: {color: RC.r1, width: 5}, marker: {size: 3, color: RC.r1}, hoverinfo: "skip"},
    {type: "scatter3d", mode: "lines+markers", x: [], y: [], z: [], name: "r1 label: 24 virtual-target positions",
     line: {color: RC.r1, width: 2, dash: "dot"}, marker: {size: 2, color: RC.r1}, opacity: 0.6, hoverinfo: "skip"},
    line3("r1 x", AXC[0], 3, {legendgroup: "r1ax", showlegend: false}),
    line3("r1 y", AXC[1], 3, {legendgroup: "r1ax", showlegend: false}),
    line3("r1 command axes at k=23 (thin)", AXC[2], 3, {legendgroup: "r1ax"}),
    {type: "scatter3d", mode: "lines+markers", x: [], y: [], z: [], name: "r2 label decoded from p(k=0) (marker = error vs demo)",
     line: {color: RC.r2, width: 5}, hoverinfo: "skip",
     marker: {size: 4, color: [], cmin: 0, cmax: 30, colorscale: ERR_CS,
              colorbar: {title: {text: "r2/r3 decode<br>error [mm]"}, len: 0.35, y: 0.18, thickness: 10}}},
    {type: "scatter3d", mode: "lines+markers", x: [], y: [], z: [], name: "r3 label decoded (labels identical to r2)",
     line: {color: RC.r3, width: 2, dash: "dot"}, marker: {size: 7, symbol: "circle-open", color: RC.r3}, hoverinfo: "skip"},
    line3("dec x", AXC[0], 3, {legendgroup: "decax", showlegend: false, line: {color: AXC[0], width: 3, dash: "dot"}}),
    line3("dec y", AXC[1], 3, {legendgroup: "decax", showlegend: false, line: {color: AXC[1], width: 3, dash: "dot"}}),
    line3("r2/r3 decoded heading R(x_s) B0 at k=23 (dotted)", AXC[2], 3, {legendgroup: "decax", line: {color: AXC[2], width: 3, dash: "dot"}}),
  ];
}
const MC = {pts: ["#000000", "#d62ad6"], r1: RC.r1, r1vt: RC.r1, r3: RC.r3};
let NSTATIC = 0, PATH_IDX = 0, MESHES = [], clipKey = "", clipC = null, lastVis = "";

function staticTraces() {
  const t = [];
  MESHES = [];
  if (D.mesh.scan) { MESHES.push([D.mesh.scan, t.length]); t.push(meshTrace(D.mesh.scan, "scan mesh", "#b8b8b8", 0.6)); }
  if (D.mesh.apron) { MESHES.push([D.mesh.apron, t.length]); t.push(meshTrace(D.mesh.apron, "flange apron", "#e3dccb", 0.35)); }
  PATH_IDX = t.length;
  t.push({type: "scatter3d", mode: "lines", x: [], y: [], z: [], name: "demo path (measured tool, RGB grid)",
          line: {color: "#8c8c8c", width: 2}, hoverinfo: "skip"});
  t.push({type: "scatter3d", mode: "markers", x: [], y: [], z: [], name: "normal flips",
          marker: {color: "#d62728", size: 3, symbol: "x"}, hoverinfo: "skip"});
  NSTATIC = t.length;
  clipKey = "";
  lastVis = "";
  return t;
}

function clipStatic(p) {
  const follow = $("follow").checked, b = +$("box").value;
  const key = `${$("ep").value}|${follow}|${b}`;
  if (key === clipKey && (!follow || Math.max(...sub(p, clipC).map(Math.abs)) < b / 3)) return;
  clipKey = key; clipC = p;
  const B = 1.6 * b;
  const ins = (x, y, z) => !follow || (Math.abs(x - p[0]) <= B && Math.abs(y - p[1]) <= B && Math.abs(z - p[2]) <= B);
  const upd = {i: [], j: [], k: []};
  for (const [m] of MESHES) {
    if (!follow) { upd.i.push(m.i); upd.j.push(m.j); upd.k.push(m.k); continue; }
    const inV = m.x.map((x, v) => ins(x, m.y[v], m.z[v]));
    const i = [], j = [], k = [];
    for (let f = 0; f < m.i.length; f++) {
      if (inV[m.i[f]] || inV[m.j[f]] || inV[m.k[f]]) { i.push(m.i[f]); j.push(m.j[f]); k.push(m.k[f]); }
    }
    upd.i.push(i); upd.j.push(j); upd.k.push(k);
  }
  if (MESHES.length) Plotly.restyle(scene, upd, MESHES.map(([, ti]) => ti));
  const path = [], fl = [];
  for (let i = 0; i < E.t.length; i++) {
    const q = p3(E.P, i);
    path.push(ins(...q) ? q : [null, null, null]);
  }
  E.flips.forEach(i => { const q = p3(E.P, i); if (ins(...q)) fl.push(q); });
  const a = pts(path), f = pts(fl);
  Plotly.restyle(scene, {x: [a.x, f.x], y: [a.y, f.y], z: [a.z, f.z]}, [PATH_IDX, PATH_IDX + 1]);
}

function axisLen() { return $("follow").checked ? +$("box").value / 5 : 60; }

function sceneLayout() {
  return {
    margin: {l: 0, r: 0, t: 0, b: 0}, uirevision: "keep", showlegend: $("legend").checked,
    legend: {x: 0, y: 1, font: {size: 10}, bgcolor: "rgba(255,255,255,0.7)"},
    scene: {
      aspectmode: "data",
      xaxis: {title: {text: "x [mm]"}}, yaxis: {title: {text: "y [mm] (fixture, +y down)"}}, zaxis: {title: {text: "z [mm]"}},
      camera: {up: {x: 0, y: -1, z: 0}, eye: {x: 1.0, y: -1.2, z: 1.3}},
    },
  };
}

function sceneRanges(p) {
  if (!$("follow").checked) {
    return {"scene.xaxis.autorange": true, "scene.yaxis.autorange": true, "scene.zaxis.autorange": true, "scene.aspectmode": "data"};
  }
  const b = +$("box").value;
  return {"scene.xaxis.range": [p[0] - b, p[0] + b], "scene.yaxis.range": [p[1] - b, p[1] + b],
          "scene.zaxis.range": [p[2] - b, p[2] + b], "scene.aspectmode": "cube"};
}

function nearInfo(q) {
  const p = p3(E.P, q), xs = p3(E.X, q), n = c3(E.Rf, q, 2), d = sub(p, xs);
  const dist = Math.hypot(d[0], d[1], d[2]), sd = E.sd[q];
  const lineAng = dist > 1e-6 ? deg(Math.acos(Math.min(1, Math.abs(dot(d, n)) / dist))) : 0;
  const tzn = deg(Math.acos(Math.max(-1, Math.min(1, dot(c3(E.Rt, q, 2), n)))));
  return {p, xs, n, dist, sd, lineAng, tzn, inpl: Math.sqrt(Math.max(dist * dist - sd * sd, 0))};
}

function chunkGeometry() {
  const q = E.q[ci], N = E.t.length, L = axisLen();
  const ni = nearInfo(q), p = ni.p, xs = ni.xs, g = {};
  g.pts = pts([p, xs]);
  g.near = pts([p, xs]);
  g.ntxt = Object.assign(pts([[(p[0] + xs[0]) / 2, (p[1] + xs[1]) / 2, (p[2] + xs[2]) / 2]]), {
    text: [`${f1(ni.dist)} mm (sd ${f1(ni.sd)})<br>line vs n ${f1(ni.lineAng)} deg, tilt ${f1(E.tilt[q])} deg`]});
  ["tx", "ty", "tz"].forEach((k, c) => g[k] = seg(p, c3(E.Rt, q, c), L));
  ["ft1", "ft2", "fn"].forEach((k, c) => g[k] = seg(xs, c3(E.Rf, q, c), L));
  const r1 = [], vt = [], dec = [];
  for (let k = 0; k < H; k++) {
    const i = Math.min(q + k, N - 1);
    r1.push(p3(E.P, i)); vt.push(p3(E.vt, i)); dec.push(p3(E.xd, ci * H + k));
  }
  g.r1 = pts(r1); g.r1vt = pts(vt); g.r2 = pts(dec); g.r3 = pts(dec);
  const ie = Math.min(q + H - 1, N - 1);
  ["r1x", "r1y", "r1z"].forEach((k, c) => g[k] = seg(r1[H - 1], c3(E.Rt, ie, c), L));
  ["dx", "dy", "dz"].forEach((k, c) => g[k] = seg(dec[H - 1], c3(E.Rd, ci, c), L));
  return {g, p};
}

function visible(group) {
  if (group === null) return true;
  if (group === "fixb") return show.r2 || show.r3;
  return show[group];
}

function drawScene() {
  const {g, p} = chunkGeometry();
  clipStatic(p);
  const idx = DYN.map((_, j) => NSTATIC + j);
  const data = {x: DYN.map(([k]) => g[k].x), y: DYN.map(([k]) => g[k].y), z: DYN.map(([k]) => g[k].z),
                text: DYN.map(([k]) => g[k].text || []),
                "marker.color": DYN.map(([k]) => k === "r2" ? E.err.slice(ci * H, ci * H + H) : (MC[k] || "#000000"))};
  const vis = DYN.map(([, grp]) => visible(grp)), vkey = vis.join();
  if (vkey !== lastVis) { data.visible = vis; lastVis = vkey; }
  Plotly.update(scene, data, sceneRanges(p), idx);
}

function drawLabels() {
  const q = E.q[ci], N = E.t.length, ks = [...Array(H).keys()], p0 = p3(E.P, q);
  const r1 = [[], [], []], v1 = [], v2 = [], dn = [];
  for (const k of ks) {
    const i = Math.min(q + k, N - 1), p = p3(E.P, i);
    for (let c = 0; c < 3; c++) r1[c].push(p[c] - p0[c]);
    v1.push(E.v[2 * i]); v2.push(E.v[2 * i + 1]); dn.push(E.dn[i]);
  }
  const base = {margin: {l: 44, r: 44, t: 6, b: 34}, font: {size: 10}, showlegend: false,
                xaxis: {title: {text: "chunk step k (RGB frames)"}, range: [-0.5, H - 0.5]}};
  if (show.r1) {
    Plotly.react(labR1, ["x", "y", "z"].map((a, c) => ({x: ks, y: r1[c], name: `cmd ${a}`, mode: "lines+markers",
                                                        line: {color: AXC[c]}, marker: {size: 3}})),
                 Object.assign({}, base, {yaxis: {title: {text: "mm"}}}), CFG);
  }
  const runs = ["r2", "r3"].filter(r => show[r]);
  if (runs.length) {
    const t = [];
    runs.forEach(r => {
      const d = r === "r3" ? "dot" : "solid";
      t.push({x: ks, y: v1, name: `${r} v_t1`, mode: "lines+markers", marker: {size: 3}, line: {color: AXC[0], dash: d}});
      t.push({x: ks, y: v2, name: `${r} v_t2`, mode: "lines+markers", marker: {size: 3}, line: {color: AXC[1], dash: d}});
      t.push({x: ks, y: dn, name: `${r} delta_n`, mode: "lines+markers", marker: {size: 3}, yaxis: "y2", line: {color: AXC[2], dash: d}});
    });
    Plotly.react(labFixb, t, Object.assign({}, base, {
      yaxis: {title: {text: "mm/s"}}, yaxis2: {title: {text: "mm"}, overlaying: "y", side: "right", showgrid: false}}), CFG);
  }
}

function layoutCells() {
  const vis = {cell_r1: show.r1, cell_fixb: show.r2 || show.r3};
  let changed = false;
  for (const [id, on] of Object.entries(vis)) {
    const want = on ? "" : "none";
    if ($(id).style.display !== want) { $(id).style.display = want; changed = true; }
  }
  return changed;
}

function buildTimeline() {
  const fl = E.flips, hp = E.hops;
  const t = [
    {x: E.t, y: E.dist, name: "|p - x_s| [mm]", mode: "lines", line: {color: "#000", width: 1}},
    {x: E.t, y: E.sd, name: "signed distance [mm]", mode: "lines", line: {color: "#999", width: 1}},
    {x: fl.map(i => E.t[i]), y: fl.map(i => E.dist[i]), name: "normal flip", mode: "markers",
     marker: {color: "#d62728", symbol: "x", size: 6}},
    {x: hp.map(i => E.t[i]), y: hp.map(i => E.dist[i]), name: "label foot hop", mode: "markers",
     marker: {color: "#9467bd", symbol: "triangle-up", size: 5}},
    {x: E.t, y: E.psi, name: "twist psi [deg] (red = 180 deg branch)", mode: "markers", yaxis: "y2",
     marker: {size: 2, color: E.flip.map(b => b ? "#d62728" : "#1f77b4")}},
  ];
  const layout = {
    margin: {l: 48, r: 10, t: 6, b: 34}, font: {size: 10}, showlegend: false,
    xaxis: {title: {text: "time [s]"}, anchor: "y2"},
    yaxis: {title: {text: "mm"}, domain: [0.42, 1], range: [-60, 200]},
    yaxis2: {title: {text: "psi [deg]"}, domain: [0, 0.34], range: [-185, 185], tickvals: [-180, -90, 0, 90, 180]},
    shapes: [
      {type: "rect", xref: "x", yref: "paper", x0: 0, x1: 0, y0: 0, y1: 1, fillcolor: "#ff7f0e", opacity: 0.18, line: {width: 0}},
      {type: "line", xref: "x", yref: "paper", x0: 0, x1: 0, y0: 0, y1: 1, line: {color: "#ff7f0e", width: 2}},
    ],
  };
  Plotly.react(tl, t, layout, CFG);
  if (!tlBound) {
    tl.on("plotly_click", ev => {
      const tc = ev.points[0].x;
      let best = 0;
      E.q.forEach((q, j) => { if (Math.abs(E.t[q] - tc) < Math.abs(E.t[E.q[best]] - tc)) best = j; });
      setChunk(best);
    });
    tlBound = true;
  }
}

function drawCursor() {
  const q = E.q[ci], t0 = E.t[q], t1 = E.t[Math.min(q + H - 1, E.t.length - 1)];
  Plotly.relayout(tl, {"shapes[0].x0": t0, "shapes[0].x1": t1, "shapes[1].x0": t0, "shapes[1].x1": t0});
}

function readout() {
  const q = E.q[ci], f = E.flags[ci];
  const {p, dist, sd, inpl, lineAng, tzn} = nearInfo(q);
  const loc = p3(E.loc, q), vtOff = Math.hypot(...sub(p3(E.vt, q), p));
  const e0 = E.err[ci * H], e23 = E.err[ci * H + H - 1];
  const row = (k, v, hot) => `<tr><td>${k}</td><td class="${hot ? "on" : ""}">${v}</td></tr>`;
  let h = `<h3>episode ${$("ep").value} (${E.split}), t = ${E.t[q].toFixed(2)} s</h3>`;
  h += `<div class="note">chunk ${ci + 1} / ${E.q.length} (trainer query grid, stride ${D.chunk_stride} RGB frames), RGB frame ${q}</div>`;
  h += "<h3>nearest point</h3><table>";
  h += row("distance |p - x_s|", `${f1(dist)} mm`, dist >= D.near_mm);
  h += row("signed distance", `${f1(sd)} mm`);
  h += row("in-plane offset", `${f1(inpl)} mm`, inpl > D.off_edge_mm);
  h += row("line vs normal n", `${f1(lineAng)} deg`, lineAng > 10);
  h += row("tool z vs n", `${f1(tzn)} deg`);
  h += row("tilt (after R_off)", `${f1(E.tilt[q])} deg`);
  h += row("twist psi about n", `${f1(E.psi[q])} deg`, Math.abs(Math.abs(E.psi[q]) - 90) < D.branch_margin_deg);
  h += row("heading branch B0", E.flip[q] ? "diag(-1,-1,1) R_off (180 deg)" : "R_off");
  h += "</table><h3>chunk flags</h3><table>";
  D.flags.forEach((k, b) => { h += `<tr><td title="${D.flag_text[k]}">${k}</td><td class="${(f >> b) & 1 ? "on" : "off"}">${(f >> b) & 1 ? "yes" : "no"}</td></tr>`; });
  h += "</table>";
  if (show.r1) {
    h += `<h3><span class="sw" style="background:${RC.r1}"></span>r1 label (absolute)</h3><table>`;
    h += row("command", "= measured fb (decode error 0)");
    h += row("virtual target offset k=0", `${f1(vtOff)} mm`);
    h += "</table>";
  }
  if (show.r2 || show.r3) {
    const runs = ["r2", "r3"].filter(r => show[r]).join(" / ");
    h += `<h3><span class="sw" style="background:${RC.r2}"></span>${runs} label (Fix B)</h3><table>`;
    h += row("v_t1, v_t2 at k=0", `${f1(E.v[2 * q])}, ${f1(E.v[2 * q + 1])} mm/s`);
    h += row("delta_n at k=0", `${f1(E.dn[q])} mm`);
    h += row("decode error k=0 / k=23", `${f1(e0)} / ${f1(e23)} mm`, e23 > 20);
    h += row("heading error k=0 / k=23", `${f1(E.he[2 * ci])} / ${f1(E.he[2 * ci + 1])} deg`);
    h += "</table>";
  }
  if (show.r3) {
    h += `<h3><span class="sw" style="background:${RC.r3}"></span>r3 obs (surface frame)</h3><table>`;
    h += row("R^T (p - x_s)", `${f1(loc[0])}, ${f1(loc[1])}, ${f1(loc[2])} mm`);
    h += row("curvature k1, k2", `${E.curv[2 * q].toFixed(2)}, ${E.curv[2 * q + 1].toFixed(2)} 1/m`);
    h += "</table>";
  }
  h += `<h3>notes</h3><div class="note">
    Fixture frame, mm. Solid axes: measured tool; dashed: field frame R(x_s) = [t1 t2 n] at the nearest point;
    red/green/blue = x/y/z (t1/t2/n).<br>
    r1 actions are absolute command poses (identical to the measured fb) plus a virtual target.
    r2/r3 actions (v_t1, v_t2, delta_n) are integrated from the measured pose at k=0 on the field
    (x_s+ = foot(x_s + R[:, :2] v dt), x_d = x_s + delta_n n) with heading R(x_s) B0, B0 picked at k=0.
    r2 and r3 labels are identical; r3 differs only in its obs (local pose and curvature).<br>
    Flags (hover the names): ${D.flags.map(k => `${k}: ${D.flag_text[k]}`).join("; ")}.</div>`;
  $("side").innerHTML = h;
}

function setChunk(j) {
  ci = Math.max(0, Math.min(E.q.length - 1, j));
  slider.value = ci;
  const resized = layoutCells();
  drawScene(); drawLabels(); drawCursor(); readout();
  if (resized) [labR1, labFixb, tl].forEach(d => { if (d.offsetParent) Plotly.Plots.resize(d); });
  history.replaceState(null, "", `#ep=${$("ep").value}&chunk=${ci}&box=${$("box").value}&follow=${+$("follow").checked}`);
}

function setEpisode(id, chunk) {
  E = D.episodes[id];
  $("ep").value = id;
  slider.max = E.q.length - 1;
  Plotly.react(scene, staticTraces().concat(dynTraces()), sceneLayout(), CFG);
  buildTimeline();
  setChunk(chunk || 0);
}

function stop() { if (timer) { clearInterval(timer); timer = null; $("play").textContent = "play"; } }
function play() {
  if (timer) return stop();
  $("play").textContent = "pause";
  const dtChunk = 1000 * (E.t[E.q[Math.min(1, E.q.length - 1)]] - E.t[E.q[0]] || 0.133);
  timer = setInterval(() => { if (ci >= E.q.length - 1) return stop(); setChunk(ci + 1); }, dtChunk / +$("speed").value);
}
function jumpFlag(dir) {
  const b = D.flags.indexOf($("flag").value);
  for (let j = ci + dir; j >= 0 && j < E.q.length; j += dir) if ((E.flags[j] >> b) & 1) return setChunk(j);
}

D.order.forEach(id => {
  const o = document.createElement("option");
  o.value = id; o.textContent = `ep ${id}` + (D.episodes[id].split === "val" ? " (val)" : "");
  $("ep").appendChild(o);
});
D.flags.forEach(k => {
  const o = document.createElement("option");
  o.value = k; o.textContent = k; o.title = D.flag_text[k];
  $("flag").appendChild(o);
});
$("ep").onchange = () => { stop(); setEpisode($("ep").value); };
slider.oninput = () => setChunk(+slider.value);
$("prev").onclick = () => setChunk(ci - 1);
$("next").onclick = () => setChunk(ci + 1);
$("play").onclick = play;
$("speed").onchange = () => { if (timer) { stop(); play(); } };
$("fprev").onclick = () => jumpFlag(-1);
$("fnext").onclick = () => jumpFlag(1);
$("follow").onchange = () => setChunk(ci);
$("box").onchange = () => setChunk(ci);
["r1", "r2", "r3"].forEach(r => $(`show_${r}`).onchange = () => { show[r] = $(`show_${r}`).checked; setChunk(ci); });
$("legend").onchange = () => Plotly.relayout(scene, {showlegend: $("legend").checked});
document.addEventListener("keydown", ev => {
  if (ev.target.tagName === "SELECT" || ev.target.tagName === "INPUT" && ev.target.type !== "range") return;
  if (ev.key === "ArrowRight") { ev.preventDefault(); setChunk(ci + 1); }
  if (ev.key === "ArrowLeft") { ev.preventDefault(); setChunk(ci - 1); }
  if (ev.key === " ") { ev.preventDefault(); play(); }
});
const hash = Object.fromEntries(location.hash.slice(1).split("&").filter(Boolean).map(s => s.split("=")));
if ([...$("box").options].some(o => o.value === hash.box)) $("box").value = hash.box;
if (hash.follow === "0") $("follow").checked = false;
setEpisode(D.episodes[hash.ep] ? hash.ep : D.order[0], +hash.chunk || 0);
</script></body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_data_v2_full_fixb"))
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "eval" / "pretrain_viz")
    ap.add_argument("--episodes", type=int, nargs="*", default=None, help="subset (debug)")
    ap.add_argument("--sections", type=int, nargs="*", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--scrub_episodes", type=int, nargs="*", default=None,
                    help="episodes in 6_scrubber.html (default: val + 4 train over the round-trip ranking + most flips)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    field = FixBMoldField(args.root / "shared" / "fixb_field.npz")
    npz = np.load(args.root / "shared" / "fixb_field.npz")
    field.is_apron = npz["is_apron"].astype(bool) if "is_apron" in npz.files else None
    meta = json.loads((args.root / "shared" / "fixb_meta.json").read_text())
    R_off = np.asarray(meta["R_off"], dtype=np.float64)
    data = {k: zarr.open(str(args.root / v / "acp_replay_buffer.zarr"), "r")["data"] for k, v in RUNS.items()}
    names = sorted(data["r1"].keys(), key=lambda s: int(s.split("_")[-1]))
    if args.episodes:
        names = [n for n in names if int(n.split("_")[-1]) in set(args.episodes)]
    mold_ids = sorted({str(data[k][n].attrs.get("mold_id")) for k in data for n in names})

    checks = {"timestamps_equal": True, "labels_r2_vs_r3_max_abs": 0.0, "mold_obs_max_abs_vs_r2": 0.0,
              "fb_r1_vs_r2_max_abs": 0.0, "r1_cmd_vs_fb_max_abs": 0.0}
    scan_sd = ScanOnlySD(field)
    eps = [process_episode(n, data, field, scan_sd, R_off, checks) for n in names]
    if not any(e["split"] == "train" for e in eps):
        raise RuntimeError("no train episodes selected")

    summary_path = args.out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() and set(args.sections) != set(range(1, 7)) else {}
    summary |= {
        "root": str(args.root),
        "label_version": meta["label_version"],
        "mold_ids": mold_ids,
        "episodes": {"train": [e["id"] for e in eps if e["split"] == "train"], "val": [e["id"] for e in eps if e["split"] == "val"]},
        "same_inputs_across_runs": checks,
        "chunk_grid": f"query stride {QUERY_STRIDE} RGB frames, H={H} RGB steps, action-gap mult {GAP_MULT}, "
                      f"obs {OBS_H} samples x {OBS_DS} robot steps; same chunks for r1/r2/r3",
    }
    sec = set(args.sections)
    if 1 in sec:
        print("[viz] 1 flips")
        summary["1_flips"] = section_flips(eps, field, args.out)
    if 2 in sec:
        print("[viz] 2 round trip")
        summary["2_round_trip"] = section_round_trip(eps, field, args.out)
    if sec & {3, 5}:
        print("[viz] 3 normalization")
        res3, norms, X = section_normalization(eps, args.out)
        if 3 in sec:
            summary["3_normalization"] = res3
    if 4 in sec:
        print("[viz] 4 twist")
        summary["4_twist"] = section_twist(eps, args.out)
    if 5 in sec:
        print("[viz] 5 variation")
        var = section_variation(eps, norms, X, args.out)
        summary["5_variation"] = {"pca": var["pca"], "across_mold": "not computable: one mold id in all episodes; "
                                  "5_phase_spread.* reports across-episode spread at matching normalized-time phase"}
    if 6 in sec:
        print("[viz] 6 scrubber")
        summary["6_scrubber"] = section_scrubber(eps, field, args.out, args.scrub_episodes)
    summary_path.write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: summary[k] for k in ("same_inputs_across_runs", "1_flips", "2_round_trip") if k in summary},
                     indent=1, default=float))
    print(f"[viz] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

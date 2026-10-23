#!/usr/bin/env python3
"""Post-training open-loop eval for scanner_v2 Fix B runs (r1/r2/r3).

Reads mold_aware_open_loop_export.py outputs. Anchored decode only:
  r1  absolute fixture pose from the command slot
  r2/r3  integrate (v∥, δn) from the measured pose at each chunk start

Skips chunks that are off-edge (in-plane foot offset > 5 mm) or have a normal
flip inside the observation window. Scores first step and full horizon.

Usage:
  python scanner_fixb_posttrain_eval.py --ckpt epoch0035
  python scanner_fixb_posttrain_eval.py --ckpt epoch0020
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.transform import Rotation
import zarr

# open3d (via fixb_field) is only needed for decode/metrics, not training curves.
FixBMoldField = integrate_fixb = heading_branch = pose7_to_Rp = None  # type: ignore


def _ensure_fixb():
    global FixBMoldField, integrate_fixb, heading_branch, pose7_to_Rp
    if FixBMoldField is None:
        from fixb_field import FixBMoldField as _F, integrate_fixb as _I
        from fixb_labels import heading_branch as _H, pose7_to_Rp as _P

        FixBMoldField, integrate_fixb, heading_branch, pose7_to_Rp = _F, _I, _H, _P

TOOL_Z = 2
OFF_EDGE_M = 0.005
OBS_H, OBS_DS = 5, 16
RUNS = ("r1_world_world", "r2_world_fixb", "r3_surface_fixb")
RUN_COLORS = {
    "r1_world_world": "#1f77b4",
    "r2_world_fixb": "#ff7f0e",
    "r3_surface_fixb": "#2ca02c",
}
STEP_METRICS = (
    "sd_err_mm",
    "sd_bias_mm",
    "tangential_mm",
    "normal_mm",
    "normal_angle_deg",
    "pos_mm",
    "rot_deg",
)


def rot6d_rows_to_R(r6: np.ndarray) -> np.ndarray:
    a1 = r6[..., 0:3]
    a2 = r6[..., 3:6]
    b1 = a1 / np.clip(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-9, None)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.clip(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-9, None)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2)


def rot_angle_deg(Ra, Rb) -> np.ndarray:
    rel = np.einsum("...ji,...jk->...ik", Ra, Rb)
    rv = Rotation.from_matrix(rel.reshape(-1, 3, 3)).as_rotvec()
    return np.degrees(np.linalg.norm(rv, axis=-1)).reshape(rel.shape[:-2])


def sparc(speed, fs, padlevel=4, fc=10.0, amp_th=0.05) -> float:
    n = len(speed)
    nfft = int(2 ** (np.ceil(np.log2(max(n, 2))) + padlevel))
    f = np.arange(0, fs, fs / nfft)
    mf = np.abs(np.fft.fft(speed, nfft))
    mf = mf / max(mf.max(), 1e-12)
    sel = f <= min(fc, fs / 2.0)
    f_sel, m_sel = f[sel], mf[sel]
    keep = np.nonzero(m_sel >= amp_th)[0]
    if len(keep) < 2:
        return 0.0
    f_sel, m_sel = f_sel[keep[0] : keep[-1] + 1], m_sel[keep[0] : keep[-1] + 1]
    df = np.diff(f_sel) / (f_sel[-1] - f_sel[0])
    return float(-np.sum(np.sqrt(df**2 + np.diff(m_sel) ** 2)))


def rms_jerk(p, t) -> float:
    if len(p) < 4:
        return float("nan")
    v = np.gradient(p, t, axis=0)
    a = np.gradient(v, t, axis=0)
    j = np.gradient(a, t, axis=0)
    return float(np.sqrt(np.mean(np.sum(j**2, axis=-1))))


def mean(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else float("nan")


def std(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.std(x, ddof=0)) if x.size else float("nan")


def decode(run, action, times, p0, R0, field, R_off):
    if run == "r1_world_world":
        return action[..., 0:3], rot6d_rows_to_R(action[..., 3:9])
    dt = np.diff(times, axis=1)
    dt = np.concatenate([dt, dt[:, -1:]], axis=1)
    dec = integrate_fixb(field, p0, action[..., 0:2], action[..., 2], dt)
    B = heading_branch(dec["R"][:, 0], R0, R_off)
    R = np.einsum("qhij,qjk->qhik", dec["R"], B)
    return dec["x_d"], R


def step_metrics(pos, R, pos_e, R_e, field):
    q_p = field.query(pos)
    q_e = field.query(pos_e)
    e = pos - pos_e
    e_n = np.sum(e * q_e["R"][..., 2], axis=-1)
    z_p, z_e = R[..., TOOL_Z], R_e[..., TOOL_Z]
    n_p, n_e = q_p["R"][..., 2], q_e["R"][..., 2]
    ang = lambda z, n: np.degrees(np.arccos(np.clip(np.sum(z * n, axis=-1), -1, 1)))
    return {
        "sd_err_mm": np.abs(q_p["sd"] - q_e["sd"]) * 1e3,
        "sd_bias_mm": (q_p["sd"] - q_e["sd"]) * 1e3,
        "tangential_mm": np.linalg.norm(e - e_n[..., None] * q_e["R"][..., 2], axis=-1) * 1e3,
        "normal_mm": np.abs(e_n) * 1e3,
        "normal_angle_deg": np.abs(ang(z_p, n_p) - ang(z_e, n_e)),
        "pos_mm": np.linalg.norm(e, axis=-1) * 1e3,
        "rot_deg": rot_angle_deg(R, R_e),
    }


def chunk_keep_mask(g, q_robot: np.ndarray, field: FixBMoldField) -> np.ndarray:
    """Keep chunks that are on-edge and have no normal flip in the obs window."""
    R_fb, p_fb = pose7_to_Rp(g["ts_pose_fb_0"])
    q = field.query(p_fb)
    n = q["R"][..., 2]
    sd = q["sd"]
    x_s = q["x_s"]
    dist = np.linalg.norm(p_fb - x_s, axis=1)
    in_plane = np.sqrt(np.maximum(dist**2 - sd**2, 0.0))
    flips = np.flatnonzero(np.sum(n[1:] * n[:-1], axis=1) < 0)

    # obs window indices (robot samples) for each query, matching trainer OBS_H/OBS_DS
    Q = len(q_robot)
    O = q_robot[:, None] - np.arange(OBS_H - 1, -1, -1)[None, :] * OBS_DS
    O = np.clip(O, 0, len(p_fb) - 1)
    # flip index i is between samples i and i+1; flag if it falls inside the obs window span
    if len(flips):
        obs_flip = (
            (flips[None, :] >= O.min(axis=1)[:, None]) & (flips[None, :] < O.max(axis=1)[:, None])
        ).any(axis=1)
    else:
        obs_flip = np.zeros(Q, dtype=bool)
    off_edge = in_plane[q_robot] > OFF_EDGE_M
    return ~(off_edge | obs_flip), off_edge, obs_flip


def load_episodes(ck_dir: Path):
    meta = json.loads((ck_dir / "meta.json").read_text())
    eps = []
    for name in meta["val_episodes"]:
        ep = int(name.split("_")[1])
        path = ck_dir / f"episode_{ep}.npz"
        if path.exists():
            eps.append(ep)
    return meta, eps


def evaluate_ckpt(args, ckpt_label: str, out: Path):
    _ensure_fixb()
    field = FixBMoldField(args.dataset_root / "shared" / "fixb_field.npz")
    meta_fixb = json.loads((args.dataset_root / "shared" / "fixb_meta.json").read_text())
    R_off = np.asarray(meta_fixb["R_off"], dtype=np.float64)
    expert_root = zarr.open(args.dataset_root / "r1_world_world" / "acp_replay_buffer.zarr", "r")["data"]

    rows = []
    pooled = {}
    per_ep = {r: {} for r in RUNS}
    filter_stats = {}
    path_bags = {r: {} for r in RUNS}  # for overlays

    for run in RUNS:
        ck = args.exports_root / run / ckpt_label
        if not (ck / "meta.json").exists():
            print(f"[eval] missing {ck}")
            continue
        exp_meta, episodes = load_episodes(ck)
        bag = {m: {"k0": [], "full": []} for m in STEP_METRICS}
        bag.update(
            {
                "stiffness_mse": [],
                "gripper_err": [],
                "grasp_force_err": [],
                "seam_mm": [],
                "seam_deg": [],
                "jerk": [],
                "sparc": [],
                "expert_jerk": [],
            }
        )
        n_total = n_keep = n_off = n_flip = 0
        ep_metric_lists = {k: [] for k in [
            "sd_err_mm_k0", "sd_err_mm_full", "tangential_mm_k0", "tangential_mm_full",
            "normal_angle_deg_k0", "normal_angle_deg_full", "seam_mm", "seam_deg",
            "chunk_jerk_rms", "chunk_sparc", "stiffness_mse", "gripper_err", "grasp_force_err",
        ]}

        for ep in episodes:
            d = np.load(ck / f"episode_{ep}.npz")
            g = expert_root[f"episode_{ep}"]
            keep, off_edge, obs_flip = chunk_keep_mask(g, d["query_robot_idx"], field)
            n_total += len(keep)
            n_keep += int(keep.sum())
            n_off += int(off_edge.sum())
            n_flip += int(obs_flip.sum())
            if not keep.any():
                continue

            action = d["raw_action"][keep]
            idx = d["action_robot_idx"][keep]
            times = d["action_target_time_s"][keep]
            nv = d["n_valid"][keep]
            q = d["query_robot_idx"][keep]
            rgb = d["rgb_id"][keep]
            R_fb, p_fb = pose7_to_Rp(g["ts_pose_fb_0"])
            stiff = np.asarray(g["stiffness_0"]).reshape(-1)
            grip = np.asarray(g["gripper_0"]).reshape(-1)
            # grasp force: stored as action dim 20; expert may be wrench-derived — use action GT
            gt = d["gt_action"][keep]

            pos, R = decode(run, action, times, p_fb[q], R_fb[q], field, R_off)
            pos_e = p_fb[idx]
            R_e = R_fb[idx]
            m = step_metrics(pos, R, pos_e, R_e, field)
            valid = np.arange(action.shape[1])[None, :] < nv[:, None]

            for key, val in m.items():
                bag[key]["k0"].append(val[:, 0])
                bag[key]["full"].append(val[valid])
            bag["stiffness_mse"].append((action[..., 18] - stiff[idx])[valid] ** 2)
            bag["gripper_err"].append(np.abs(action[..., 19] - grip[idx])[valid])
            bag["grasp_force_err"].append(np.abs(action[..., 20] - gt[..., 20])[valid])

            order = np.argsort(rgb)
            for a, b in zip(order[:-1], order[1:]):
                if rgb[b] - rgb[a] == args.exec_h and nv[a] > args.exec_h:
                    bag["seam_mm"].append(np.linalg.norm(pos[b, 0] - pos[a, args.exec_h]) * 1e3)
                    bag["seam_deg"].append(float(rot_angle_deg(R[b, 0:1], R[a, args.exec_h : args.exec_h + 1])[0]))

            for qi in range(len(nv)):
                if nv[qi] < 6:
                    continue
                t = times[qi, : nv[qi]]
                if np.any(np.diff(t) <= 0):
                    continue
                bag["jerk"].append(rms_jerk(pos[qi, : nv[qi]], t))
                bag["expert_jerk"].append(rms_jerk(pos_e[qi, : nv[qi]], t))
                speed = np.linalg.norm(np.gradient(pos[qi, : nv[qi]], t, axis=0), axis=-1)
                bag["sparc"].append(sparc(speed, 1.0 / float(np.median(np.diff(t)))))

            ep_row = {"run": run, "episode": int(ep), "n_chunks_kept": int(keep.sum()), "n_chunks_total": int(len(keep))}
            for key, val in m.items():
                ep_row[f"{key}_k0"] = mean(val[:, 0])
                ep_row[f"{key}_full"] = mean(val[valid])
            ep_row["stiffness_mse"] = mean((action[..., 18] - stiff[idx])[valid] ** 2)
            ep_row["gripper_err"] = mean(np.abs(action[..., 19] - grip[idx])[valid])
            ep_row["grasp_force_err"] = mean(np.abs(action[..., 20] - gt[..., 20])[valid])
            # per-ep seam/jerk from this episode only
            ep_seams_mm, ep_seams_deg, ep_jerks, ep_sparcs = [], [], [], []
            order = np.argsort(rgb)
            for a, b in zip(order[:-1], order[1:]):
                if rgb[b] - rgb[a] == args.exec_h and nv[a] > args.exec_h:
                    ep_seams_mm.append(np.linalg.norm(pos[b, 0] - pos[a, args.exec_h]) * 1e3)
                    ep_seams_deg.append(float(rot_angle_deg(R[b, 0:1], R[a, args.exec_h : args.exec_h + 1])[0]))
            for qi in range(len(nv)):
                if nv[qi] < 6:
                    continue
                t = times[qi, : nv[qi]]
                if np.any(np.diff(t) <= 0):
                    continue
                ep_jerks.append(rms_jerk(pos[qi, : nv[qi]], t))
                speed = np.linalg.norm(np.gradient(pos[qi, : nv[qi]], t, axis=0), axis=-1)
                ep_sparcs.append(sparc(speed, 1.0 / float(np.median(np.diff(t)))))
            ep_row["seam_mm"] = mean(ep_seams_mm)
            ep_row["seam_deg"] = mean(ep_seams_deg)
            ep_row["chunk_jerk_rms"] = mean(ep_jerks)
            ep_row["chunk_sparc"] = mean(ep_sparcs)
            rows.append(ep_row)
            per_ep[run][ep] = ep_row
            for k in ep_metric_lists:
                if k in ep_row:
                    ep_metric_lists[k].append(ep_row[k])

            # store executed path for overlays (exec_h steps)
            path_bags[run][ep] = {
                "pos": pos[:, : args.exec_h].copy(),
                "pos_e": pos_e[:, : args.exec_h].copy(),
                "rgb": rgb.copy(),
                "sd_err": m["sd_err_mm"][:, : args.exec_h].copy(),
                "q": q.copy(),
            }
            if run == "r3_surface_fixb":
                curv = field.query(pos[:, 0], with_curvature=True)["curv"]
                path_bags[run][ep]["curv_k1k2"] = curv[:, :2].copy()
                path_bags[run][ep]["sd_err_k0"] = m["sd_err_mm"][:, 0].copy()

            print(
                f"[eval] {run} ep {ep} kept {keep.sum()}/{len(keep)} "
                f"sd_k0 {ep_row['sd_err_mm_k0']:.1f} sd_full {ep_row['sd_err_mm_full']:.1f}"
            )

        filter_stats[run] = {
            "n_chunks_total": n_total,
            "n_chunks_kept": n_keep,
            "n_off_edge": n_off,
            "n_obs_flip": n_flip,
            "frac_kept": n_keep / max(n_total, 1),
        }
        pooled[run] = {
            **{f"{k}_k0": mean(np.concatenate(v["k0"])) for k, v in bag.items() if k in STEP_METRICS and v["k0"]},
            **{f"{k}_full": mean(np.concatenate(v["full"])) for k, v in bag.items() if k in STEP_METRICS and v["full"]},
            "stiffness_mse": mean(np.concatenate(bag["stiffness_mse"])) if bag["stiffness_mse"] else None,
            "gripper_err": mean(np.concatenate(bag["gripper_err"])) if bag["gripper_err"] else None,
            "grasp_force_err": mean(np.concatenate(bag["grasp_force_err"])) if bag["grasp_force_err"] else None,
            "seam_mm": mean(bag["seam_mm"]),
            "seam_deg": mean(bag["seam_deg"]),
            "chunk_jerk_rms": mean(bag["jerk"]),
            "expert_chunk_jerk_rms": mean(bag["expert_jerk"]),
            "chunk_sparc": mean(bag["sparc"]),
            **{f"{k}_ep_std": std(ep_metric_lists[k]) for k in ep_metric_lists},
            **{f"{k}_ep_mean": mean(ep_metric_lists[k]) for k in ep_metric_lists},
        }
        rows.append({"run": run, "episode": "pooled", **pooled[run]})

    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "ckpt": ckpt_label,
        "val_episodes": sorted({r["episode"] for r in rows if isinstance(r.get("episode"), int)}),
        "filter": filter_stats,
        "pooled": pooled,
        "rows": rows,
        "notes": [
            "Anchored decode only.",
            "Skipped off-edge (in-plane > 5 mm) and obs-window normal flips.",
            "Val shares the mold with training — fit, not transfer.",
            "Force contact quality cannot be compared offline.",
        ],
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2, default=float))
    if rows:
        with (out / "metrics.csv").open("w", newline="") as f:
            fields = list(dict.fromkeys(k for row in rows for k in row))
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    plot_bars(pooled, per_ep, out)
    plot_chunk_viewer(path_bags, args, out)
    plot_overlay_and_heatmap(path_bags, field, args, out)
    plot_r3_curvature(path_bags, out)
    return payload


def plot_bars(pooled, per_ep, out: Path):
    if not pooled:
        return
    keys = [
        ("sd_err_mm_k0", "SD err first (mm)"),
        ("sd_err_mm_full", "SD err full (mm)"),
        ("tangential_mm_full", "Tangential full (mm)"),
        ("normal_angle_deg_full", "Normal angle full (°)"),
        ("seam_mm", "Seam jump (mm)"),
        ("seam_deg", "Seam jump (°)"),
        ("chunk_jerk_rms", "Jerk RMS"),
        ("chunk_sparc", "SPARC"),
        ("stiffness_mse", "Stiffness MSE"),
        ("gripper_err", "Gripper |err|"),
        ("grasp_force_err", "Grasp force |err| (N)"),
    ]
    runs = [r for r in RUNS if r in pooled]
    fig, axs = plt.subplots(3, 4, figsize=(14, 9))
    axs = axs.ravel()
    for ax, (key, title) in zip(axs, keys):
        means, errs = [], []
        for r in runs:
            ep_vals = [per_ep[r][e][key] for e in per_ep[r] if key in per_ep[r][e]]
            means.append(mean(ep_vals) if ep_vals else pooled[r].get(key, np.nan))
            errs.append(std(ep_vals) if len(ep_vals) > 1 else 0.0)
        ax.bar(range(len(runs)), means, yerr=errs, color=[RUN_COLORS[r] for r in runs], capsize=3)
        ax.set_xticks(range(len(runs)))
        ax.set_xticklabels(["r1", "r2", "r3"], rotation=0)
        ax.set_title(title, fontsize=10)
    for ax in axs[len(keys) :]:
        ax.axis("off")
    fig.suptitle("Scanner Fix B open-loop (anchored); error bars = std across val episodes")
    fig.tight_layout()
    fig.savefig(out / "bars.png", dpi=130)
    plt.close(fig)


def plot_chunk_viewer(path_bags, args, out: Path):
    # pick up to 3 val episodes present in all runs
    common = None
    for r in RUNS:
        eps = set(path_bags[r])
        common = eps if common is None else common & eps
    if not common:
        return
    episodes = sorted(common)[:3]
    for ep in episodes:
        fig = make_subplots(
            rows=1, cols=1, specs=[[{"type": "scatter3d"}]],
            subplot_titles=[f"episode_{ep}: predicted vs expert (exec_h={args.exec_h})"],
        )
        # expert once
        pe = path_bags[RUNS[0]][ep]["pos_e"].reshape(-1, 3) * 1e3
        fig.add_trace(
            go.Scatter3d(
                x=pe[:, 0], y=pe[:, 1], z=pe[:, 2], mode="lines+markers",
                line=dict(color="black", width=3), marker=dict(size=2), name="expert",
            )
        )
        for run in RUNS:
            if ep not in path_bags[run]:
                continue
            p = path_bags[run][ep]["pos"].reshape(-1, 3) * 1e3
            fig.add_trace(
                go.Scatter3d(
                    x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="lines+markers",
                    line=dict(color=RUN_COLORS[run], width=2), marker=dict(size=2),
                    name=run,
                )
            )
        fig.update_layout(height=700, scene=dict(aspectmode="data"))
        fig.write_html(out / f"chunk_viewer_ep{ep}.html", include_plotlyjs="cdn")


def plot_overlay_and_heatmap(path_bags, field: FixBMoldField, args, out: Path):
    common = None
    for r in RUNS:
        eps = set(path_bags[r])
        common = eps if common is None else common & eps
    if not common:
        return
    # overlay: concatenate all val eps for each run
    fig = go.Figure()
    rng = np.random.default_rng(0)
    sel = rng.choice(len(field.vertices), min(args.mold_points, len(field.vertices)), replace=False)
    mp = field.vertices[sel] * 1e3
    fig.add_trace(
        go.Scatter3d(
            x=mp[:, 0], y=mp[:, 1], z=mp[:, 2], mode="markers",
            marker=dict(size=1.5, color="lightgray", opacity=0.3), name="mold",
        )
    )
    for run in RUNS:
        pts, err = [], []
        for ep in sorted(common):
            bag = path_bags[run][ep]
            pts.append(bag["pos"].reshape(-1, 3))
            err.append(bag["sd_err"].reshape(-1))
        if not pts:
            continue
        P = np.concatenate(pts, axis=0) * 1e3
        E = np.concatenate(err, axis=0)
        fig.add_trace(
            go.Scatter3d(
                x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers",
                marker=dict(size=3, color=E, colorscale="Turbo", cmin=0, cmax=args.sd_range_mm, colorbar=dict(title="SD err mm") if run == RUNS[0] else None),
                name=run,
            )
        )
    fig.update_layout(title="Three-run overlay (color = SD error mm)", height=800, scene=dict(aspectmode="data"))
    fig.write_html(out / "overlay_mesh.html", include_plotlyjs="cdn")

    # heatmap: project SD error onto mold vertices (nearest path point)
    for run in RUNS:
        pts, err = [], []
        for ep in sorted(common):
            bag = path_bags[run].get(ep)
            if bag is None:
                continue
            pts.append(bag["pos"].reshape(-1, 3))
            err.append(bag["sd_err"].reshape(-1))
        if not pts:
            continue
        P = np.concatenate(pts, axis=0)
        E = np.concatenate(err, axis=0)
        # subsample path for speed
        if len(P) > 5000:
            idx = rng.choice(len(P), 5000, replace=False)
            P, E = P[idx], E[idx]
        # assign each mold vertex the error of nearest path sample (within 30 mm)
        from scipy.spatial import cKDTree

        tree = cKDTree(P)
        d, ii = tree.query(field.vertices, k=1)
        heat = np.full(len(field.vertices), np.nan)
        ok = d < 0.03
        heat[ok] = E[ii[ok]]
        fig2 = go.Figure(
            data=[
                go.Mesh3d(
                    x=field.vertices[:, 0] * 1e3,
                    y=field.vertices[:, 1] * 1e3,
                    z=field.vertices[:, 2] * 1e3,
                    i=field.faces[:, 0],
                    j=field.faces[:, 1],
                    k=field.faces[:, 2],
                    intensity=np.nan_to_num(heat, nan=0.0),
                    colorscale="Turbo",
                    cmin=0,
                    cmax=args.sd_range_mm,
                    colorbar=dict(title="SD err mm"),
                    name=run,
                )
            ]
        )
        fig2.update_layout(title=f"Error heatmap on mold — {run}", height=750, scene=dict(aspectmode="data"))
        fig2.write_html(out / f"heatmap_{run}.html", include_plotlyjs="cdn")


def plot_r3_curvature(path_bags, out: Path):
    bag = path_bags.get("r3_surface_fixb", {})
    if not bag:
        return
    kmag, sd = [], []
    for ep, d in bag.items():
        if "curv_k1k2" not in d:
            continue
        km = np.mean(np.abs(d["curv_k1k2"]), axis=1)
        kmag.append(km)
        sd.append(d["sd_err_k0"])
    if not kmag:
        return
    kmag = np.concatenate(kmag)
    sd = np.concatenate(sd)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(kmag, sd, s=8, alpha=0.4, c=RUN_COLORS["r3_surface_fixb"])
    # binned means
    if len(kmag) > 10:
        edges = np.quantile(kmag, np.linspace(0, 1, 6))
        edges = np.unique(edges)
        centers, means = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (kmag >= a) & (kmag <= b)
            if m.any():
                centers.append(0.5 * (a + b))
                means.append(mean(sd[m]))
        ax.plot(centers, means, "k-o", lw=2, label="binned mean")
    ax.set_xlabel("|curvature| (1/m), mean(|k1|,|k2|)")
    ax.set_ylabel("SD err first step (mm)")
    ax.set_title("r3: SD error vs local curvature")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "r3_sd_vs_curvature.png", dpi=130)
    plt.close(fig)


def plot_training_curves(ckpt_root: Path, out: Path):
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    summary = {}
    for ax, run in zip(axs, RUNS):
        log = ckpt_root / run / "logs.json.txt"
        epochs, train, val = [], [], []
        best = None
        for line in log.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "train_loss" in r and "epoch" in r:
                epochs.append(r["epoch"])
                train.append(r["train_loss"])
                val.append(r.get("val_loss", np.nan))
                if r.get("val_loss") is not None and np.isfinite(r["val_loss"]):
                    if best is None or r["val_loss"] < best[0]:
                        best = (r["val_loss"], r["epoch"])
        ax.plot(epochs, train, label="train", lw=1)
        ax.plot(epochs, val, label="val", lw=1)
        if best:
            ax.axvline(best[1], color="gray", ls="--", lw=1, label=f"best val @{best[1]}")
            # late rise?
            late = [v for e, v in zip(epochs, val) if e >= best[1] and np.isfinite(v)]
            rose = bool(late) and mean(late[-max(1, len(late)//5):]) > best[0] * 1.05
        else:
            rose = None
        summary[run] = {"best_val_loss": best[0] if best else None, "best_epoch": best[1] if best else None, "val_rose_after_best": rose, "last_epoch": max(epochs) if epochs else None}
        ax.set_title(run.replace("_", "\n"), fontsize=9)
        ax.set_xlabel("epoch")
        ax.legend(fontsize=7)
    axs[0].set_ylabel("loss")
    fig.suptitle("Training curves (OOM-stopped; none reached 200)")
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=130)
    plt.close(fig)
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def write_report(out_root: Path, results: dict, train_summary: dict):
    lines = [
        "# Scanner Fix B post-training open-loop eval",
        "",
        "## Setup",
        "- Dataset: `stringer_scanner_v2_fixb` (September).",
        "- Primary ckpt: `epoch0035` (matched). Best-val check: `epoch0020` (lowest val_loss in all three logs).",
        "- Decode: anchored only. r1 = fixture cmd pose; r2/r3 = integrate (v∥, δn) from measured pose at chunk start.",
        "- Val episodes (from training `get_validation_dataset`, seed=42, val_ratio=0.1, 75 eps, lexicographic buffer order):",
        f"  `{results.get('val_episodes')}`",
        "- Filters: drop off-edge (in-plane > 5 mm) and obs-window normal flips.",
        "",
        "## Context",
        "- Val shares the mold with training → measures **fit**, not transfer.",
        "- Force / contact quality cannot be compared offline; hardware + profilometer still decide.",
        "- All three Hyak jobs died `OUT_OF_MEMORY` before epoch 200 (r1@35, r3@55, r2@65).",
        "",
        "## Training curves",
    ]
    for run, s in train_summary.items():
        lines.append(
            f"- **{run}**: best val_loss={s['best_val_loss']} at epoch {s['best_epoch']}; "
            f"last epoch {s['last_epoch']}; val rose after best={s['val_rose_after_best']}"
        )
    lines += ["", "## Filter counts (epoch0035)", ""]
    for run, f in results.get("filter", {}).items():
        lines.append(
            f"- **{run}**: kept {f['n_chunks_kept']}/{f['n_chunks_total']} "
            f"(off-edge {f['n_off_edge']}, obs-flip {f['n_obs_flip']})"
        )
    lines += ["", "## Pooled metrics (epoch0035, anchored)", ""]
    pooled = results.get("pooled", {})
    if pooled:
        keys = [
            "sd_err_mm_k0", "sd_err_mm_full", "tangential_mm_full", "normal_angle_deg_full",
            "seam_mm", "seam_deg", "chunk_jerk_rms", "chunk_sparc",
            "stiffness_mse", "gripper_err", "grasp_force_err",
        ]
        hdr = "| metric | " + " | ".join(pooled) + " |"
        sep = "|---|---|---|---|" if len(pooled) == 3 else "|" + "---|" * (len(pooled) + 1)
        lines += [hdr, sep]
        for k in keys:
            vals = []
            for r in pooled:
                v = pooled[r].get(k)
                vals.append(f"{v:.3g}" if isinstance(v, (int, float)) and v is not None else "—")
            lines.append(f"| {k} | " + " | ".join(vals) + " |")
    lines += [
        "",
        "## Artifacts",
        f"- `{out_root}/epoch0035/`",
        f"- `{out_root}/epoch0020/`",
        f"- `{out_root}/training_curves.png`",
        "",
    ]
    (out_root / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exports_root", type=Path, default=Path("/mnt/data/umidata/ft/training_outputs/scanner_fixb/eval/exports"))
    ap.add_argument("--dataset_root", type=Path, default=Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_scanner_v2_fixb"))
    ap.add_argument("--ckpt_root", type=Path, default=Path("/mnt/data/umidata/ft/training_outputs/scanner_fixb"))
    ap.add_argument("--out_dir", type=Path, default=Path("/mnt/data/umidata/ft/training_outputs/scanner_fixb/eval"))
    ap.add_argument("--ckpt", action="append", default=None, help="export label, e.g. epoch0035 (repeatable)")
    ap.add_argument("--exec_h", type=int, default=4)
    ap.add_argument("--sd_range_mm", type=float, default=60.0)
    ap.add_argument("--mold_points", type=int, default=5000)
    args = ap.parse_args()
    labels = args.ckpt or ["epoch0035", "epoch0020"]

    train_out = args.out_dir
    train_out.mkdir(parents=True, exist_ok=True)
    train_summary = plot_training_curves(args.ckpt_root, train_out)

    primary = None
    for label in labels:
        out = args.out_dir / label
        print(f"==== {label} ====")
        payload = evaluate_ckpt(args, label, out)
        if label == "epoch0035":
            primary = payload

    if primary is None and labels:
        primary = json.loads((args.out_dir / labels[0] / "metrics.json").read_text())
    write_report(args.out_dir, primary or {}, train_summary)
    print(f"[eval] wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""A–D on stringer_scanner_v2: surface-label consistency, heading, clean labels, rebuild.

Reads the September zarr and the existing Fix B mesh. Writes plots/json under
eval/scanner_v2_fixb/. Optional --build writes a new dataset directory (does not
touch stringer_data_v2_full_fixb). Laser scan arrays may be hardlinked in the
zarr; they are not in the policy shape_meta.

  python fixb_scanner_v2_ready.py
  python fixb_scanner_v2_ready.py --build
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr
from scipy.signal import find_peaks

from fixb_build_datasets import MOLD_KEYS, VARIANTS, link_tree, replace_array
from fixb_data_check import VAL_EPISODES, frame_step_deg
from fixb_field import FixBMoldField, integrate_fixb
from fixb_labels import RZ_PI, episode_arrays, fit_heading_offset, fixb_pose7_command, identity_pose7, pose7_to_Rp
from fixb_labels_tracked import episode_arrays_tracked
from fixb_pretrain_viz import (
    FORCE_PEAK_DIST,
    FORCE_PEAK_N,
    H,
    LABEL_HOP_EXCESS_M,
    NEAR_M,
    OFF_EDGE_M,
    SD_JUMP_M,
    V_CLIP,
    ScanOnlySD,
    chunk_grid,
    geodesic_deg,
    pct,
    smooth_positions,
)

SCANNER_ZARR = Path("/mnt/data/umidata/ft/datasets/stringer_scanner_v2/acp_replay_buffer.zarr")
FIELD_NPZ = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_data_v2_full_fixb/shared/fixb_field.npz")
OUT_DATA = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_scanner_v2_fixb")
SMOOTH_S = 0.1
N_BIN = 8
N_STROKE = 24
TRAVEL = 0.03
LABEL_VERSION = "fixb_v4_tracked"


def pearson(a, b) -> float:
    a, b = np.asarray(a, float).reshape(-1), np.asarray(b, float).reshape(-1)
    if len(a) < 4 or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def aligned_corr(a, b) -> float:
    r = pearson(a, b)
    r2 = pearson(a, -np.asarray(b, float).reshape(-1))
    if np.isnan(r):
        return r2
    if np.isnan(r2):
        return r
    return r if abs(r) >= abs(r2) else r2


def resample(x, n=N_STROKE):
    x = np.asarray(x, float)
    if len(x) == n:
        return x
    t = np.linspace(0, 1, len(x))
    tt = np.linspace(0, 1, n)
    if x.ndim == 1:
        return np.interp(tt, t, x)
    return np.stack([np.interp(tt, t, x[:, d]) for d in range(x.shape[1])], 1)


def pick_strokes(near, moving, xs, spd, min_len=20):
    mask = near & moving
    d = np.diff(mask.astype(np.int8), prepend=0, append=0)
    segs = [(int(s), int(t)) for s, t in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)) if t - s >= min_len]
    if len(segs) < 2:
        return segs
    means = np.array([xs[a:b].mean(0) for a, b in segs])
    best, pair = -1.0, (segs[0], segs[1])
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            si, sj = np.median(spd[segs[i][0] : segs[i][1]]), np.median(spd[segs[j][0] : segs[j][1]])
            score = float(np.linalg.norm(means[i] - means[j])) * (min(si, sj) / max(si, sj, 1e-6))
            if score > best:
                best, pair = score, (segs[i], segs[j])
    return list(pair)


def hops_on_rgb(rt, rgb, p_fb, x_s):
    """Hops of the foot used in the labels (x_s), not a fresh closest-point query."""
    ir = np.clip(np.searchsorted(rt, rgb), 0, len(rt) - 1)
    p_s = smooth_positions(rt, p_fb, SMOOTH_S)
    c = x_s[ir]
    j1 = np.minimum(np.arange(len(ir)) + 1, len(ir) - 1)
    excess = np.linalg.norm(c[j1] - c, axis=1) - np.linalg.norm(p_s[ir[j1]] - p_s[ir], axis=1)
    hop = excess > LABEL_HOP_EXCESS_M
    dist = np.linalg.norm(p_fb - x_s, axis=1)
    near = dist[ir] < NEAR_M
    return hop, near, ir


def score_episode(ep, arr, field, scan_sd, R_off):
    rt = np.asarray(ep["robot_time_stamps_0"], float).reshape(-1)
    rgb = np.asarray(ep["rgb_time_stamps_0"], float).reshape(-1)
    wt = np.asarray(ep["wrench_time_stamps_0"], float).reshape(-1)
    q, A, _, n_cand = chunk_grid(rt, rgb, wt)
    R_fb, p_fb = pose7_to_Rp(ep["ts_pose_fb_0"])
    x_s = np.asarray(arr["mold_x_s_0"], float)
    Rf = np.asarray(arr["mold_R_field_0"], float).reshape(-1, 3, 3)
    sd = np.asarray(arr["mold_signed_distance_0"], float).reshape(-1)
    v = np.asarray(arr["fixb_v_par_0"], float)
    dn = np.asarray(arr["fixb_delta_n_0"], float).reshape(-1)
    dist = np.linalg.norm(p_fb - x_s, axis=1)
    in_plane = np.sqrt(np.maximum(dist**2 - sd**2, 0.0))
    n = Rf[:, :, 2]
    hop, near_rgb, ir = hops_on_rgb(rt, rgb, p_fb, x_s)
    steps = np.minimum(q[:, None] + np.arange(H), len(rgb) - 1)
    clip_rgb = (np.abs(v[ir]) >= V_CLIP - 1e-6).any(1)
    n_flip = np.flatnonzero(np.sum(n[1:] * n[:-1], axis=1) < 0)
    n_jump = np.flatnonzero(np.abs(np.diff(sd)) > SD_JUMP_M)
    fl_in = ((n_flip[None, :] >= A[:, :1]) & (n_flip[None, :] < A[:, -1:])) if len(n_flip) else np.zeros((len(q), 0), bool)
    dt = np.diff(rt[A], axis=1)
    dt = np.concatenate([dt, dt[:, -1:]], 1)
    dec = integrate_fixb(field, p_fb[A[:, 0]], v[A], dn[A], dt)
    err = np.linalg.norm(dec["x_d"] - p_fb[A], axis=-1) * 1e3
    near_ch = (dist[A] < NEAR_M).all(1)
    off_edge = in_plane[A[:, 0]] > OFF_EDGE_M
    F = np.linalg.norm(np.asarray(ep["wrench_concat_0"])[:, 0:3], axis=1)
    pk, _ = find_peaks(F, height=FORCE_PEAK_N, distance=FORCE_PEAK_DIST)
    pk_i = np.clip(np.searchsorted(rt, wt[pk]), 0, len(rt) - 1) if len(pk) else np.zeros(0, int)
    M = np.einsum("nji,njk->nik", Rf[ir], R_fb[ir]) @ R_off.T
    psi = np.degrees(np.arctan2(M[:, 1, 0] - M[:, 0, 1], M[:, 0, 0] + M[:, 1, 1]))
    tilt = np.degrees(np.arccos(np.clip(M[:, 2, 2], -1, 1)))
    tz_n = np.degrees(np.arccos(np.clip(np.sum(R_fb[ir, :, 2] * n[ir], 1), -1, 1)))
    flipped = (M[:, 0, 0] + M[:, 1, 1]) < 0
    B0 = np.where(flipped[q][:, None, None], (RZ_PI @ R_off)[None], R_off[None])
    head = geodesic_deg(np.einsum("qij,qjk->qik", Rf[A[:, 0]], B0), R_fb[A[:, 0]])
    v_ir, n_ir = v[ir], n[ir]
    moving = np.linalg.norm(v_ir, axis=1) > TRAVEL
    dp = np.diff(p_fb[ir], axis=0, append=p_fb[ir][-1:])
    segs = pick_strokes(near_rgb, moving, x_s[ir], np.linalg.norm(v_ir, axis=1))
    stroke = None
    if len(segs) == 2:
        (a0, a1), (b0, b1) = segs
        va, vb = resample(v_ir[a0:a1]), resample(v_ir[b0:b1])
        da, db = resample(dp[a0:a1]), resample(dp[b0:b1])
        na, nb = n_ir[a0:a1].mean(0), n_ir[b0:b1].mean(0)
        na, nb = na / np.linalg.norm(na), nb / np.linalg.norm(nb)
        stroke = {
            "corr_vt1": aligned_corr(va[:, 0], vb[:, 0]),
            "corr_vt2": aligned_corr(va[:, 1], vb[:, 1]),
            "corr_dpx": aligned_corr(da[:, 0], db[:, 0]),
            "corr_dpy": aligned_corr(da[:, 1], db[:, 1]),
            "corr_dpz": aligned_corr(da[:, 2], db[:, 2]),
            "n_angle_deg": float(np.degrees(np.arccos(np.clip(na @ nb, -1, 1)))),
            "len_a": a1 - a0,
            "len_b": b1 - b0,
        }
    near_i = near_rgb
    return {
        "n_chunks": len(q),
        "n_cand": n_cand,
        "frac_flip_span": float((fl_in.any(1)).mean()) if fl_in.size else 0.0,
        "n_flips": int(len(n_flip)),
        "n_sd_jumps": int(len(n_jump)),
        "frac_hop": float(hop.mean()),
        "frac_hop_near": float((hop & near_rgb).mean()) if near_rgb.any() else 0.0,
        "frac_chunks_hop": float(hop[steps].any(1).mean()),
        "frac_chunks_hop_near": float((hop[steps] & near_rgb[steps]).any(1).mean()),
        "frac_clip": float(clip_rgb.mean()),
        "frac_chunks_clip": float(clip_rgb[steps].any(1).mean()),
        "frac_near": float(near_ch.mean()),
        "frac_off_edge": float(off_edge.mean()),
        "k23_near_p50": float(np.median(err[near_ch, -1])) if near_ch.any() else None,
        "k23_near_p90": float(np.percentile(err[near_ch, -1], 90)) if near_ch.any() else None,
        "k23_all_p50": float(np.median(err[:, -1])),
        "sd_peak": (sd[pk_i] * 1e3).tolist(),
        "sd_peak_scan": (scan_sd(p_fb[pk_i]) * 1e3).tolist() if len(pk_i) else [],
        "tilt_near": tilt[near_i],
        "psi_near": psi[near_i],
        "tzn_near": tz_n[near_i],
        "head_p50": float(np.median(head)),
        "stroke": stroke,
        "near_dp": dp[near_i],
        "near_v": v_ir[near_i],
        "near_dn": dn[ir][near_i],
        "near_n": n_ir[near_i],
        "fstep_p90": float(np.percentile(frame_step_deg(Rf), 90)) if len(Rf) > 1 else 0.0,
    }


def val_ids_n(n: int, ratio=0.1, seed=42):
    n_val = min(max(1, round(n * ratio)), n - 1)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=n_val, replace=False))


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def paged_series(eps, ys, out, title, ylabel, per=12):
    n = len(eps)
    pages = (n + per - 1) // per
    for p in range(pages):
        sl = eps[p * per : (p + 1) * per]
        r, c = 4, 3
        fig, axs = plt.subplots(r, c, figsize=(16, 12), sharex=False)
        axs = axs.ravel()
        for ax, e in zip(axs, sl):
            y = ys(e)
            ax.plot(np.linspace(0, 1, len(y)), y, lw=0.6)
            ax.set_title(f"ep {e['id']}", fontsize=9)
            ax.set_ylabel(ylabel, fontsize=8)
        for ax in axs[len(sl) :]:
            ax.axis("off")
        fig.suptitle(f"{title} (page {p + 1}/{pages})")
        fig.tight_layout()
        fig.savefig(out / f"{title.replace(' ', '_')}_p{p + 1}.png", dpi=90)
        plt.close(fig)


def run_mode(names, src, field, scan_sd, R_off, cache_dir, tracked: bool, max_episodes=None):
    fn = episode_arrays_tracked if tracked else episode_arrays
    tag = "tracked" if tracked else "raw"
    rows, strokes, near_dp, near_v, near_dn, near_n = [], [], [], [], [], []
    tilt_ep, psi_ep, tzn_ep, sd_peak_ep = [], [], [], []
    cache_dir.mkdir(parents=True, exist_ok=True)
    use = names[:max_episodes] if max_episodes else names
    for i, name in enumerate(use):
        eid = int(name.split("_")[-1])
        cache = cache_dir / f"{name}.npz"
        if cache.exists():
            arr = {k: np.load(cache)[k] for k in np.load(cache).files}
        else:
            arr = fn(field, src["data"][name], smooth_window_s=SMOOTH_S, v_par_mode="foot", delta_n_source="fb")
            np.savez_compressed(cache, **arr)
        s = score_episode(src["data"][name], arr, field, scan_sd, R_off)
        s["id"] = eid
        s["name"] = name
        s["arr"] = arr
        rows.append(s)
        if s["stroke"] and s["stroke"]["n_angle_deg"] >= 0:
            strokes.append({"id": eid, **s["stroke"]})
        if len(s["near_dp"]):
            near_dp.append(s["near_dp"])
            near_v.append(s["near_v"])
            near_dn.append(s["near_dn"])
            near_n.append(s["near_n"])
        tilt_ep.append(s["tilt_near"])
        psi_ep.append(s["psi_near"])
        tzn_ep.append(s["tzn_near"])
        sd_peak_ep.append(np.asarray(s["sd_peak"], float))
        print(
            f"[{tag} {i + 1}/{len(use)}] {name}: chunks {s['n_chunks']} hop {s['frac_hop']:.3f} "
            f"near-hop-chunk {s['frac_chunks_hop_near']:.3f} k23-near {s['k23_near_p50']} mm "
            f"off_edge {s['frac_off_edge']:.3f}"
        )
        s["tilt_ts"] = s["tilt_near"]
        s["psi_ts"] = s["psi_near"]
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros((0, 3))
    return {
        "rows": rows,
        "strokes": strokes,
        "near_dp": cat(near_dp),
        "near_v": cat(near_v),
        "near_dn": np.concatenate(near_dn) if near_dn else np.zeros(0),
        "near_n": cat(near_n),
        "tilt_ep": tilt_ep,
        "psi_ep": psi_ep,
        "tzn_ep": tzn_ep,
        "sd_peak_ep": sd_peak_ep,
        "tag": tag,
    }


def summarize_mode(M, field, out: Path):
    rows = M["rows"]
    tag = M["tag"]
    d = out / tag
    d.mkdir(parents=True, exist_ok=True)
    n_tr = sum(r["n_chunks"] for r in rows)
    scalar = {
        "episodes": len(rows),
        "chunks": n_tr,
        "frac_chunks_flip_span": round(sum(r["frac_flip_span"] * r["n_chunks"] for r in rows) / max(n_tr, 1), 4),
        "frac_rgb_hop": round(float(np.mean([r["frac_hop"] for r in rows])), 4),
        "frac_chunks_hop": round(sum(r["frac_chunks_hop"] * r["n_chunks"] for r in rows) / max(n_tr, 1), 4),
        "frac_chunks_hop_near": round(sum(r["frac_chunks_hop_near"] * r["n_chunks"] for r in rows) / max(n_tr, 1), 4),
        "frac_chunks_clip": round(sum(r["frac_chunks_clip"] * r["n_chunks"] for r in rows) / max(n_tr, 1), 4),
        "frac_chunks_off_edge": round(sum(r["frac_off_edge"] * r["n_chunks"] for r in rows) / max(n_tr, 1), 4),
        "frac_chunks_near": round(sum(r["frac_near"] * r["n_chunks"] for r in rows) / max(n_tr, 1), 4),
        "k23_near_p50_mm": pct(np.concatenate([np.array([r["k23_near_p50"]]) for r in rows if r["k23_near_p50"] is not None])),
        "k23_near_median_of_episode_p50_mm": round(
            float(np.median([r["k23_near_p50"] for r in rows if r["k23_near_p50"] is not None])), 2
        ),
    }
    peaks = np.concatenate([p for p in M["sd_peak_ep"] if len(p)]) if any(len(p) for p in M["sd_peak_ep"]) else np.zeros(0)
    scalar["force_peak_sd_pooled_p10_p50_p90_mm"] = pct(peaks) if len(peaks) else None
    scalar["force_peak_episode_median_p10_p50_p90_mm"] = (
        pct([np.median(p) for p in M["sd_peak_ep"] if len(p)]) if any(len(p) for p in M["sd_peak_ep"]) else None
    )
    scalar["force_peak_n"] = int(len(peaks))

    n = M["near_n"]
    dp, v, dn = M["near_dp"], M["near_v"], M["near_dn"]
    axes = "xyz"
    corr = {
        "r1": {f"dp_{a}_vs_n_{b}": round(pearson(dp[:, i], n[:, j]), 3) for i, a in enumerate(axes) for j, b in enumerate(axes)},
        "r2": {f"{lab}_vs_n_{b}": round(pearson(src, n[:, j]), 3)
               for lab, src in (("v_t1", v[:, 0] if len(v) else np.zeros(0)),
                                ("v_t2", v[:, 1] if len(v) else np.zeros(0)),
                                ("delta_n", dn))
               for j, b in enumerate(axes)},
        "n_near": int(len(n)),
    }
    scalar["corr"] = corr

    st = M["strokes"]
    useful = [s for s in st if s["n_angle_deg"] >= 25]
    scalar["strokes"] = {
        "n_pairs": len(st),
        "n_pairs_n_angle_ge_25deg": len(useful),
        "mean_n_angle_deg": round(float(np.mean([s["n_angle_deg"] for s in st])), 1) if st else None,
        "mean_corr_r2_vt_all": round(float(np.nanmean([0.5 * (s["corr_vt1"] + s["corr_vt2"]) for s in st])), 3) if st else None,
        "mean_corr_r1_dp_all": round(float(np.nanmean([np.mean([s["corr_dpx"], s["corr_dpy"], s["corr_dpz"]]) for s in st])), 3) if st else None,
        "mean_corr_r2_vt_angled": round(float(np.nanmean([0.5 * (s["corr_vt1"] + s["corr_vt2"]) for s in useful])), 3) if useful else None,
        "mean_corr_r1_dp_angled": round(float(np.nanmean([np.mean([s["corr_dpx"], s["corr_dpy"], s["corr_dpz"]]) for s in useful])), 3) if useful else None,
    }

    def wmean(xs):
        xs = [x for x in xs if len(x)]
        if not xs:
            return float("nan"), float("nan")
        means = [float(np.mean(x)) for x in xs]
        stds = [float(np.std(x)) for x in xs]
        return float(np.mean(stds)), float(np.std(means))

    tw, tb = wmean(M["tilt_ep"])
    pw, pb = wmean(M["psi_ep"])
    zw, zb = wmean(M["tzn_ep"])
    scalar["heading"] = {
        "tilt_after_Roff_within_std_deg": round(tw, 2),
        "tilt_after_Roff_between_std_deg": round(tb, 2),
        "tilt_after_Roff_mean_deg": round(float(np.mean([np.mean(x) for x in M["tilt_ep"] if len(x)])), 2) if any(len(x) for x in M["tilt_ep"]) else None,
        "psi_within_std_deg": round(pw, 2),
        "psi_between_std_deg": round(pb, 2),
        "toolz_vs_n_within_std_deg": round(zw, 2),
        "toolz_vs_n_between_std_deg": round(zb, 2),
        "toolz_vs_n_mean_deg": round(float(np.mean([np.mean(x) for x in M["tzn_ep"] if len(x)])), 2) if any(len(x) for x in M["tzn_ep"]) else None,
        "decoded_heading_p50_mean_over_eps_deg": round(float(np.mean([r["head_p50"] for r in rows])), 1),
    }
    # constant-tilt rule of thumb: within << between or within < 8 deg and mean offset > 10
    tilt_mean = scalar["heading"]["tilt_after_Roff_mean_deg"] or 0
    scalar["heading"]["calibrate_constant_tilt"] = bool(tilt_mean > 8 and tw < 10)

    # plots
    ids = [r["id"] for r in rows]
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs[0, 0].bar(ids, [r["frac_chunks_hop_near"] for r in rows], color="C0")
    axs[0, 0].set_title("near-surface label-hop fraction of chunks")
    axs[0, 0].set_xlabel("episode")
    axs[0, 1].bar(ids, [r["k23_near_p50"] or np.nan for r in rows], color="C2")
    axs[0, 1].set_title("round-trip k23 p50, near chunks [mm]")
    axs[0, 1].set_xlabel("episode")
    med_pk = [float(np.median(p)) if len(p) else np.nan for p in M["sd_peak_ep"]]
    axs[1, 0].bar(ids, med_pk, color="C1")
    axs[1, 0].axhline(0, color="k", lw=0.6)
    axs[1, 0].set_title("force-peak field SD median [mm]")
    axs[1, 0].set_xlabel("episode")
    axs[1, 1].bar(ids, [r["frac_off_edge"] for r in rows], color="C3")
    axs[1, 1].set_title("off-edge chunk fraction")
    axs[1, 1].set_xlabel("episode")
    for ax in axs.ravel():
        ax.set_xlim(-1, max(ids) + 1)
    fig.suptitle(f"scanner_v2 {tag}: per-episode hop / round-trip / force-peak SD / off-edge")
    fig.tight_layout()
    fig.savefig(d / "C_per_episode.png", dpi=100)
    plt.close(fig)

    if st:
        fig, ax = plt.subplots(figsize=(7, 6))
        c = [s["n_angle_deg"] for s in st]
        x = [0.5 * (s["corr_vt1"] + s["corr_vt2"]) for s in st]
        y = [np.mean([s["corr_dpx"], s["corr_dpy"], s["corr_dpz"]]) for s in st]
        sc = ax.scatter(y, x, c=c, cmap="viridis", vmin=0, vmax=90)
        ax.plot([-1, 1], [-1, 1], "k--", lw=0.7)
        ax.set_xlabel("r1 world-step corr (mean of x,y,z), sign-aligned")
        ax.set_ylabel("r2 (v_t1, v_t2) corr, sign-aligned")
        ax.set_title("stroke A vs B: surface labels should sit above the diagonal")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        fig.colorbar(sc, ax=ax, label="angle between mean n [deg]")
        fig.tight_layout()
        fig.savefig(d / "A_stroke_corr.png", dpi=110)
        plt.close(fig)

    if len(n) > 50:
        bins = np.linspace(-1, 1, N_BIN + 1)
        bid = np.digitize(n[:, 2], bins) - 1
        bid = np.clip(bid, 0, N_BIN - 1)
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
        ctr = 0.5 * (bins[:-1] + bins[1:])
        for ax, (lab, series) in zip(axs, (("r1 Δx,Δy,Δz [mm]", dp * 1e3), ("r2 v_t1, v_t2 [mm/s]", v * 1e3), ("r2 δn [mm]", dn.reshape(-1, 1) * 1e3))):
            for j, name in enumerate(("x / t1", "y / t2", "z / n")[: series.shape[1]]):
                mu, sg = [], []
                for b in range(N_BIN):
                    sl = series[bid == b, j] if series.shape[1] > 1 else series[bid == b]
                    mu.append(float(np.mean(sl)) if len(sl) else np.nan)
                    sg.append(float(np.std(sl)) if len(sl) else np.nan)
                ax.errorbar(ctr, mu, yerr=sg, marker="o", capsize=3, label=name)
            ax.set_xlabel("n_z bin center")
            ax.set_ylabel("mean ± std")
            ax.set_title(lab)
            ax.legend(fontsize=8)
        fig.suptitle(f"{tag}: near-contact labels vs surface-normal n_z (flat r2 + drifting r1 = surface frame helps)")
        fig.tight_layout()
        fig.savefig(d / "A_nz_bins.png", dpi=110)
        plt.close(fig)

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        im0 = axs[0].imshow([[corr["r1"][f"dp_{a}_vs_n_{b}"] for b in axes] for a in axes], vmin=-1, vmax=1, cmap="RdBu_r")
        axs[0].set_xticks(range(3), list(axes))
        axs[0].set_yticks(range(3), [f"Δ{a}" for a in axes])
        axs[0].set_title("r1 world step vs n")
        fig.colorbar(im0, ax=axs[0], fraction=0.046)
        r2m = np.array([[corr["r2"][f"{lab}_vs_n_{b}"] for b in axes] for lab in ("v_t1", "v_t2", "delta_n")])
        im1 = axs[1].imshow(r2m, vmin=-1, vmax=1, cmap="RdBu_r")
        axs[1].set_xticks(range(3), list(axes))
        axs[1].set_yticks(range(3), ["v_t1", "v_t2", "δn"])
        axs[1].set_title("r2 local vs n")
        fig.colorbar(im1, ax=axs[1], fraction=0.046)
        fig.suptitle(f"{tag}: Pearson r, near-contact RGB steps, all loaded episodes")
        fig.tight_layout()
        fig.savefig(d / "A_corr_heatmap.png", dpi=110)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(["tilt within", "tilt between", "ψ within", "ψ between", "toolz∠n within", "toolz∠n between"],
           [tw, tb, pw, pb, zw, zb], color=["C0", "C0", "C1", "C1", "C2", "C2"])
    ax.set_ylabel("deg")
    ax.set_title(f"{tag}: heading variation (near contact). Constant offset ⇒ within << between or within small.")
    fig.tight_layout()
    fig.savefig(d / "B_within_between.png", dpi=110)
    plt.close(fig)

    dummy = [{"id": r["id"], "tilt_near": r["tilt_near"], "psi_near": r["psi_near"]} for r in rows]
    paged_series(dummy, lambda e: e["tilt_near"] if len(e["tilt_near"]) else np.zeros(1), d, f"B_tilt_{tag}", "tilt after R_off [deg]")
    paged_series(dummy, lambda e: e["psi_near"] if len(e["psi_near"]) else np.zeros(1), d, f"B_twist_{tag}", "twist ψ [deg]")

    csv_rows = []
    for r in rows:
        csv_rows.append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                         if k not in ("arr", "near_dp", "near_v", "near_dn", "near_n", "tilt_near", "psi_near",
                                      "tzn_near", "tilt_ts", "psi_ts", "stroke", "sd_peak", "sd_peak_scan")})
    write_csv(d / "per_episode.csv", csv_rows)
    write_csv(d / "strokes.csv", st)
    return scalar


def build_dataset(names, src, field, cache_tracked: Path, R_off, out_root: Path, meta_extra: dict):
    shared = out_root / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    shutil.copy2(FIELD_NPZ, shared / "fixb_field.npz")
    arrays = {}
    for name in names:
        arrays[name] = {k: np.load(cache_tracked / f"{name}.npz")[k] for k in np.load(cache_tracked / f"{name}.npz").files}
    meta = {
        "label_version": LABEL_VERSION,
        "v_par_mode": "foot",
        "delta_n_source": "fb",
        "smooth_window_s": SMOOTH_S,
        "foot": "tracked (independent closest point, hops repaired from previous foot + tool step)",
        "source_zarr": str(SCANNER_ZARR),
        "n_episodes": len(names),
        "R_off": R_off.tolist(),
        "scan_in_policy": False,
        "scan_arrays_hardlinked_but_not_in_shape_meta": ["scan_heatmap_0", "scan_gap_intervals_0", "map_to_scan_idx_0"],
        **meta_extra,
        "decode": "x_s0 = tracked_foot(p_meas); x_s+ = foot(x_s + R[:, :2] v dt); x_d = x_s + delta_n n; R_d = R(x_s) B",
    }
    (shared / "fixb_meta.json").write_text(json.dumps(meta, indent=2, default=float))
    for variant in VARIANTS:
        zpath = out_root / variant / "acp_replay_buffer.zarr"
        print(f"[{variant}] linking {SCANNER_ZARR} -> {zpath}")
        link_tree(SCANNER_ZARR, zpath)
        root = zarr.open(str(zpath), "a")
        root.attrs.update({"mold_aware_variant": variant, "fixb": meta, "world_frame": "fixture", "label_version": LABEL_VERSION})
        for name in names:
            ep = root["data"][name]
            arr = arrays[name]
            for key in MOLD_KEYS:
                replace_array(ep, key, arr[key])
            ep.attrs.update({"pose_variant": variant, "label_version": LABEL_VERSION, "world_frame": "fixture"})
            if variant == "r1_world_world":
                continue
            replace_array(ep, "ts_pose_command_0", fixb_pose7_command(arr["fixb_v_par_0"], arr["fixb_delta_n_0"]), ep["ts_pose_command_0"])
            replace_array(ep, "ts_pose_virtual_target_0", identity_pose7(len(arr["fixb_v_par_0"])), ep["ts_pose_virtual_target_0"])
            if variant == "r3_surface_fixb":
                R_tool, p = pose7_to_Rp(ep["ts_pose_fb_0"])
                R_f = arr["mold_R_field_0"].astype(np.float64).reshape(-1, 3, 3)
                p_loc = np.einsum("nji,nj->ni", R_f, p - arr["mold_x_s_0"].astype(np.float64))
                R_loc = np.einsum("nji,njk->nik", R_f, R_tool)
                from fixb_labels import Rp_to_pose7
                replace_array(ep, "ts_pose_fb_0", Rp_to_pose7(R_loc, p_loc), ep["ts_pose_fb_0"])
        print(f"[{variant}] wrote {len(names)} episodes")
    return meta


def confirm_r2_r3(out_root: Path, names):
    r2 = zarr.open(str(out_root / "r2_world_fixb" / "acp_replay_buffer.zarr"), "r")["data"]
    r3 = zarr.open(str(out_root / "r3_surface_fixb" / "acp_replay_buffer.zarr"), "r")["data"]
    r1 = zarr.open(str(out_root / "r1_world_world" / "acp_replay_buffer.zarr"), "r")["data"]
    max_cmd = max_mold = max_fb = 0.0
    for name in names[:5] + names[-2:]:
        for k in ("fixb_v_par_0", "fixb_delta_n_0", "ts_pose_command_0"):
            max_cmd = max(max_cmd, float(np.max(np.abs(np.asarray(r2[name][k]) - np.asarray(r3[name][k])))))
        for k in ("mold_signed_distance_0", "mold_normal_0", "mold_R_field_rot6d_0", "mold_curvature_0"):
            max_mold = max(max_mold, float(np.max(np.abs(np.asarray(r1[name][k]) - np.asarray(r2[name][k])))))
            max_mold = max(max_mold, float(np.max(np.abs(np.asarray(r2[name][k]) - np.asarray(r3[name][k])))))
        max_fb = max(max_fb, float(np.max(np.abs(np.asarray(r1[name]["ts_pose_fb_0"]) - np.asarray(r2[name]["ts_pose_fb_0"])))))
    return {"r2_vs_r3_action_max_abs": max_cmd, "mold_channels_max_abs_across_runs": max_mold, "r1_vs_r2_fb_max_abs": max_fb}


def catalog():
    return {
        "policy_obs_not_including_scan": True,
        "obs": {
            "rgb_0": {"shape": [3, 224, 224], "frame": "wrist", "units": "uint8 RGB", "runs": "all"},
            "depth_0": {"shape": [3, 224, 224], "frame": "wrist", "units": "m, clipped at 0.5", "runs": "all"},
            "robot0_eef_pos": {"shape": [3], "frame": "r1/r2 fixture; r3 local R^T(p-x_s)", "units": "m", "runs": "all"},
            "robot0_eef_rot_axis_angle": {"shape": [6], "frame": "same as eef_pos (rot6d of first two rows)", "units": "-", "runs": "all"},
            "robot0_eef_wrench_left": {"shape": [6], "frame": "tcp", "units": "N, N·m", "runs": "all"},
            "robot0_eef_wrench_right": {"shape": [6], "frame": "tcp", "units": "N, N·m", "runs": "all"},
            "robot0_mold_geom": {"shape": [10], "frame": "fixture field", "units": "[sd_m, n_xyz, R_field_rot6d]", "runs": "all"},
            "robot0_mold_curv": {"shape": [4], "frame": "surface [t1,t2]", "units": "[k1,k2,a cos2θ, a sin2θ] 1/m", "runs": "r3 only"},
        },
        "action21": {
            "r1": "cmd_xyz_m (0:3), cmd_rot6d (3:9), vt_xyz_m (9:12), vt_rot6d (12:18), stiffness (18), gripper (19), grasp_force_N (20). cmd == measured fb in this zarr family.",
            "r2_r3": "v_t1_mps (0), v_t2_mps (1), delta_n_m (2), identity rot6d (3:9), unused vt zeros (9:18), stiffness (18), gripper (19), grasp_force_N (20).",
            "stiffness_and_gripper_still_in_action": True,
            "heading_is_not_a_label": "R_d = R(x_s) B, B in {R_off, diag(-1,-1,1) R_off} from measured tool at chunk start",
        },
        "horizons": {"rgb_obs": 5, "lowdim_obs": 5, "mold_obs": 5, "wrench_obs": 32, "action": 24, "query_stride_rgb": 2},
    }


def write_yamls(out_root: Path, scripts: Path):
    cfg = scripts / "config" / "task"
    cfg.mkdir(parents=True, exist_ok=True)
    specs = [
        ("umift_scanner_fixb_r1.yaml", "SCANNER-FIXB-R1-WORLD-WORLD", "r1_world_world", False),
        ("umift_scanner_fixb_r2.yaml", "SCANNER-FIXB-R2-WORLD-FIXB", "r2_world_fixb", False),
        ("umift_scanner_fixb_r3.yaml", "SCANNER-FIXB-R3-SURFACE-FIXB", "r3_surface_fixb", True),
    ]
    for fn, name, variant, curv in specs:
        extra = ""
        if curv:
            extra = """
dataset:
  _target_: fixb_dataset.UmiFTDatasetFixB

shape_meta:
  obs:
    robot0_mold_curv:
      shape: [4]
      type: low_dim
  sample:
    obs:
      sparse:
        robot0_mold_curv:
          horizon: ${task.sparse_obs_mold_horizon}
          down_sample_steps: ${task.sparse_obs_mold_down_sample_steps}
"""
        (cfg / fn).write_text(
            f"""# Fix B on stringer_scanner_v2 (September). Laser scan is not a policy input.
defaults:
  - umift_mold_aware
  - _self_

name: {name}

dataset_path: {out_root / variant / "acp_replay_buffer.zarr"}
{extra}"""
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=SCANNER_ZARR)
    ap.add_argument("--field", type=Path, default=FIELD_NPZ)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "eval" / "scanner_v2_fixb")
    ap.add_argument("--out_data", type=Path, default=OUT_DATA)
    ap.add_argument("--max_episodes", type=int, default=None)
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    field = FixBMoldField(args.field)
    npz = np.load(args.field)
    field.is_apron = npz["is_apron"].astype(bool) if "is_apron" in npz.files else None
    scan_sd = ScanOnlySD(field)
    src = zarr.open(str(args.source), "r")
    names = sorted((k for k in src["data"].keys() if k.startswith("episode_")), key=lambda s: int(s.split("_")[-1]))
    if args.max_episodes:
        names = names[: args.max_episodes]

    # heading from raw near-band samples of the first pass (fit on tracked later too)
    print("[pass] raw labels")
    cache_raw = args.out / "cache_raw"
    # temporary R_off = I until we have samples; score heading after fit
    # cheap heading samples without full score: load/create arrays
    L = []
    for name in names:
        cache = cache_raw / f"{name}.npz"
        cache_raw.mkdir(parents=True, exist_ok=True)
        if cache.exists():
            arr = {k: np.load(cache)[k] for k in np.load(cache).files}
        else:
            arr = episode_arrays(field, src["data"][name], smooth_window_s=SMOOTH_S, v_par_mode="foot", delta_n_source="fb")
            np.savez_compressed(cache, **arr)
        R_tool, _ = pose7_to_Rp(src["data"][name]["ts_pose_fb_0"])
        R_f = arr["mold_R_field_0"].astype(np.float64).reshape(-1, 3, 3)
        band = np.abs(arr["mold_signed_distance_0"][:, 0]) < NEAR_M
        if band.any():
            L.append(np.einsum("nji,njk->nik", R_f[band][::20], R_tool[band][::20]))
    R_off, flip, head_err = fit_heading_offset(np.concatenate(L))
    print(f"[heading raw] flip {flip.mean():.2f} residual p50/p90 {np.percentile(head_err, [50,90]).round(1)} "
          f"R_off tool-z vs -n {np.degrees(np.arccos(np.clip(-R_off[2,2],-1,1))):.1f} deg")

    raw = run_mode(names, src, field, scan_sd, R_off, cache_raw, False, args.max_episodes)
    print("[pass] tracked feet")
    cache_trk = args.out / "cache_tracked"
    trk = run_mode(names, src, field, scan_sd, R_off, cache_trk, True, args.max_episodes)

    s_raw = summarize_mode(raw, field, args.out)
    s_trk = summarize_mode(trk, field, args.out)

    off = s_trk["frac_chunks_off_edge"]
    hop_n = s_trk["frac_chunks_hop_near"]
    decision = {
        "off_edge": (
            "drop off-edge chunks (in-plane > 5 mm): δn cannot represent a side-of-mesh offset. "
            "The flange apron is already on the mesh; more apron would not fix tools that miss the part. "
            f"tracked off-edge fraction = {off:.3f}."
        ),
        "hops": (
            "use tracked feet for labels and decode (fixb_v4_tracked). "
            f"near-hop chunk fraction raw {s_raw['frac_chunks_hop_near']:.4f} -> tracked {hop_n:.4f}."
        ),
        "tilt": s_trk["heading"],
    }

    n_ep = len(names)
    val = val_ids_n(n_ep)
    summary = {
        "source": str(args.source),
        "n_episodes": n_ep,
        "elapsed_s": round(time.time() - t0, 1),
        "R_off_tool_z_tilt_from_minus_n_deg": float(np.degrees(np.arccos(np.clip(-R_off[2, 2], -1, 1)))),
        "A_surface_consistency": {"raw": s_raw["strokes"] | {"corr": s_raw["corr"]}, "tracked": s_trk["strokes"] | {"corr": s_trk["corr"]}},
        "B_heading": {"raw": s_raw["heading"], "tracked": s_trk["heading"]},
        "C_clean": {"raw": {k: s_raw[k] for k in s_raw if k not in ("corr", "strokes", "heading")},
                    "tracked": {k: s_trk[k] for k in s_trk if k not in ("corr", "strokes", "heading")}},
        "decisions": decision,
        "D_split": {
            "n_episodes": n_ep,
            "val_ratio": 0.1,
            "seed": 42,
            "val_ids_local_numpy": val.tolist(),
            "july_52_ep_val_ids_hyak": list(VAL_EPISODES),
            "same_mold": True,
            "metrics_measure": "fit, not transfer",
            "note": "Hyak numpy may pick a different 8-episode set; this list is local default_rng(42).",
        },
        "D_catalog": catalog(),
        "decode": "already: integrate_fixb from measured pose at chunk start (fixb_open_loop_metrics.decode)",
    }

    if args.build:
        extra = {"heading_flipped_frac": float(flip.mean()),
                 "heading_residual_deg_p50_p90": np.percentile(head_err, [50, 90]).tolist(),
                 "R_off_convention": "R_tool ~= R_field(x_s) @ B, B in {R_off, diag(-1,-1,1)@R_off} from measured tool at chunk start"}
        print("[build] writing", args.out_data)
        meta = build_dataset(names, src, field, cache_trk, R_off, args.out_data, extra)
        summary["D_build"] = {"root": str(args.out_data), "meta_keys": list(meta), "r2_r3_check": confirm_r2_r3(args.out_data, names)}
        write_yamls(args.out_data, Path(__file__).resolve().parent)
        # drop-off-edge note: chunk filter belongs in a dataset subclass; not applied in the zarr
        summary["D_build"]["off_edge_filter"] = "not written into the zarr; drop in a dataset index filter if used in training"

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: summary[k] for k in ("A_surface_consistency", "B_heading", "C_clean", "decisions", "D_split") if k in summary}, indent=2))
    print(f"wrote {args.out} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

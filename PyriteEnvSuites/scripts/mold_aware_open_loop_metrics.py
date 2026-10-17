#!/usr/bin/env python3
"""Fixture-frame open-loop metrics for the mold-aware ablation runs.

Reads mold_aware_open_loop_export.py outputs, decodes every prediction to
fixture-frame poses (both "oracle" and "anchored" decode, see
mold_aware_eval_common.py) and scores them against the expert:

  sd_err_mm       |SD(pred) - SD(expert)|, signed distance to the mold (main metric)
  sd_bias_mm      mean signed SD(pred) - SD(expert)
  tangential_mm   in-plane part of (pred - expert), plane normal at the expert foot
  normal_mm       |normal part| of (pred - expert)
  normal_angle_deg |angle(tool z, mesh n) at pred - same at expert|
  pos_mm / rot_deg  full position / geodesic rotation error
  stiffness_mse   (pred - expert)^2, frame independent

each at the first chunk step (k0) and over the full horizon (full), per val
episode and pooled over all val queries. Smoothness: seam jump between
consecutive chunks (same-time disagreement of chunk q+1 step 0 vs chunk q step
exec_h), RMS jerk and SPARC of the executed trajectory, and RMS jerk inside each
full-horizon chunk, with the expert's values at the same timestamps.

Usage:
  PYTHONPATH=<UMI-FT>/PyriteUtility python mold_aware_open_loop_metrics.py \
    --exports_root /mnt/data/umidata/ft/training_outputs/mold_aware/eval/exports \
    --out_dir /mnt/data/umidata/ft/training_outputs/mold_aware/eval/metrics
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mold_aware_eval_common import (
    DECODE_MODES,
    RUNS,
    TARGETS,
    TOOL_NORMAL_AXIS,
    MoldField,
    decode_actions,
    load_expert_episode,
    load_export,
    load_surface_scale,
    rot_angle_deg,
)

STEP_METRICS = (
    "sd_err_mm",
    "sd_bias_mm",
    "tangential_mm",
    "normal_mm",
    "normal_angle_deg",
    "pos_mm",
    "rot_deg",
)
SMOOTH_METRICS = (
    "seam_pos_mm",
    "seam_rot_deg",
    "exec_jerk_rms",
    "exec_sparc",
    "chunk_jerk_rms",
)


def sparc(speed: np.ndarray, fs: float, padlevel: int = 4, fc: float = 10.0, amp_th: float = 0.05) -> float:
    """Spectral arc length (Balasubramanian et al. 2015); closer to 0 = smoother."""
    n = len(speed)
    nfft = int(2 ** (np.ceil(np.log2(n)) + padlevel))
    f = np.arange(0, fs, fs / nfft)
    mf = np.abs(np.fft.fft(speed, nfft))
    mf = mf / max(mf.max(), 1e-12)
    sel = f <= min(fc, fs / 2.0)
    f_sel, m_sel = f[sel], mf[sel]
    keep = np.nonzero(m_sel >= amp_th)[0]
    if len(keep) < 2:
        return 0.0
    f_sel = f_sel[keep[0] : keep[-1] + 1]
    m_sel = m_sel[keep[0] : keep[-1] + 1]
    df = np.diff(f_sel) / (f_sel[-1] - f_sel[0])
    return float(-np.sum(np.sqrt(df**2 + np.diff(m_sel) ** 2)))


def rms_jerk(p: np.ndarray, t: np.ndarray) -> float:
    if len(p) < 4:
        return float("nan")
    v = np.gradient(p, t, axis=0)
    a = np.gradient(v, t, axis=0)
    j = np.gradient(a, t, axis=0)
    return float(np.sqrt(np.mean(np.sum(j**2, axis=-1))))


def contiguous_segments(rgb_id: np.ndarray, step: int) -> List[np.ndarray]:
    breaks = np.nonzero(np.diff(rgb_id) != step)[0] + 1
    return [s for s in np.split(np.arange(len(rgb_id)), breaks) if len(s) > 0]


def step_metrics(
    T_pred: np.ndarray, T_exp: np.ndarray, field: MoldField
) -> Dict[str, np.ndarray]:
    p, pe = T_pred[..., :3, 3], T_exp[..., :3, 3]
    sd_p, n_p = field.query(p)
    sd_e, n_e = field.query(pe)
    e = p - pe
    e_n = np.sum(e * n_e, axis=-1)
    tan = np.linalg.norm(e - e_n[..., None] * n_e, axis=-1)
    z_p = T_pred[..., :3, TOOL_NORMAL_AXIS]
    z_e = T_exp[..., :3, TOOL_NORMAL_AXIS]
    ang_p = np.degrees(np.arccos(np.clip(np.sum(z_p * n_p, axis=-1), -1, 1)))
    ang_e = np.degrees(np.arccos(np.clip(np.sum(z_e * n_e, axis=-1), -1, 1)))
    return {
        "sd_err_mm": np.abs(sd_p - sd_e) * 1e3,
        "sd_bias_mm": (sd_p - sd_e) * 1e3,
        "tangential_mm": tan * 1e3,
        "normal_mm": np.abs(e_n) * 1e3,
        "normal_angle_deg": np.abs(ang_p - ang_e),
        "pos_mm": np.linalg.norm(e, axis=-1) * 1e3,
        "rot_deg": rot_angle_deg(T_pred[..., :3, :3], T_exp[..., :3, :3]),
    }


def smoothness(
    T_pred: np.ndarray,
    T_exp: np.ndarray,
    times: np.ndarray,
    rgb_id: np.ndarray,
    n_valid: np.ndarray,
    exec_h: int,
    min_seg: int,
) -> Dict[str, Dict[str, List[float]]]:
    out = {m: {"pred": [], "expert": []} for m in SMOOTH_METRICS}
    for seg in contiguous_segments(rgb_id, exec_h):
        for a, b in zip(seg[:-1], seg[1:]):
            if n_valid[a] > exec_h:
                out["seam_pos_mm"]["pred"].append(
                    float(np.linalg.norm(T_pred[b, 0, :3, 3] - T_pred[a, exec_h, :3, 3]) * 1e3)
                )
                out["seam_rot_deg"]["pred"].append(
                    float(rot_angle_deg(T_pred[b, 0, :3, :3], T_pred[a, exec_h, :3, :3]))
                )
        if len(seg) * exec_h < min_seg:
            continue
        ex = np.concatenate([np.arange(exec_h) for _ in seg])
        qs = np.repeat(seg, exec_h)
        t = times[qs, ex]
        order = np.argsort(t, kind="stable")
        t = t[order]
        uniq = np.concatenate([[True], np.diff(t) > 1e-6])
        t = t[uniq]
        fs = 1.0 / float(np.median(np.diff(t)))
        for who, T in (("pred", T_pred), ("expert", T_exp)):
            p = T[qs, ex, :3, 3][order][uniq]
            out["exec_jerk_rms"][who].append(rms_jerk(p, t))
            speed = np.linalg.norm(np.gradient(p, t, axis=0), axis=-1)
            out["exec_sparc"][who].append(sparc(speed, fs))
    for q in range(T_pred.shape[0]):
        nv = int(n_valid[q])
        if nv < 6:
            continue
        t = times[q, :nv]
        if np.any(np.diff(t) <= 0):
            continue
        out["chunk_jerk_rms"]["pred"].append(rms_jerk(T_pred[q, :nv, :3, 3], t))
        out["chunk_jerk_rms"]["expert"].append(rms_jerk(T_exp[q, :nv, :3, 3], t))
    return out


def mean_or_nan(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--exports_root",
        type=Path,
        default=Path("/mnt/data/umidata/ft/training_outputs/mold_aware/eval/exports"),
    )
    ap.add_argument(
        "--dataset_root",
        type=Path,
        default=Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_data_v2_full"),
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/mnt/data/umidata/ft/training_outputs/mold_aware/eval/metrics"),
    )
    ap.add_argument("--ckpts", nargs="+", default=["latest", "ep40"])
    ap.add_argument("--min_segment_samples", type=int, default=24)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    field = MoldField(args.dataset_root / "shared" / "merged_fixture.ply")
    scale = load_surface_scale(args.dataset_root)
    meta0 = json.loads(
        (args.exports_root / RUNS[0] / args.ckpts[0] / "meta.json").read_text()
    )
    episodes = meta0["val_episodes"]
    exec_h = int(meta0["sparse_execution_horizon"])
    experts = {ep: load_expert_episode(args.dataset_root, ep) for ep in episodes}

    results: Dict = {}
    checks: Dict = {}
    rows = []
    for run in RUNS:
        for ck in args.ckpts:
            ck_dir = args.exports_root / run / ck
            if not (ck_dir / "meta.json").exists():
                print(f"[metrics] missing {ck_dir}, skipping")
                continue
            for mode in DECODE_MODES:
                res = results.setdefault(run, {}).setdefault(ck, {}).setdefault(mode, {})
                pooled_step = {t: {m: {"k0": [], "full": []} for m in STEP_METRICS} for t in TARGETS}
                pooled_stiff = {"k0": [], "full": []}
                pooled_smooth = {t: {m: {"pred": [], "expert": []} for m in SMOOTH_METRICS} for t in TARGETS}
                for ep in episodes:
                    d = load_export(ck_dir / f"{ep}.npz")
                    ex = experts[ep]
                    idx = d["action_robot_idx"]
                    valid = np.arange(idx.shape[1])[None, :] < d["n_valid"][:, None]
                    dec = decode_actions(
                        d["raw_action"], run, ex, idx, d["query_robot_idx"], scale, mode
                    )
                    if mode == "oracle" and ck == args.ckpts[0]:
                        gt = decode_actions(
                            d["gt_action"], run, ex, idx, d["query_robot_idx"], scale, mode
                        )
                        checks.setdefault(run, {})[ep] = {
                            t: float(
                                np.abs(gt[t][..., :3, 3] - ex[t][idx][..., :3, 3]).max() * 1e3
                            )
                            for t in TARGETS
                        }
                    ep_res = {}
                    for t in TARGETS:
                        T_exp = ex[t][idx]
                        sm = step_metrics(dec[t], T_exp, field)
                        ep_res[t] = {}
                        for m, v in sm.items():
                            k0 = v[:, 0]
                            full = v[valid]
                            ep_res[t][m] = {"k0": mean_or_nan(k0), "full": mean_or_nan(full)}
                            pooled_step[t][m]["k0"].append(k0)
                            pooled_step[t][m]["full"].append(full)
                        smo = smoothness(
                            dec[t],
                            T_exp,
                            d["action_target_time_s"],
                            d["rgb_id"],
                            d["n_valid"],
                            exec_h,
                            args.min_segment_samples,
                        )
                        for m, v in smo.items():
                            ep_res[t][m] = {
                                "pred": mean_or_nan(v["pred"]),
                                "expert": mean_or_nan(v["expert"]),
                            }
                            pooled_smooth[t][m]["pred"].extend(v["pred"])
                            pooled_smooth[t][m]["expert"].extend(v["expert"])
                    s_err = (dec["stiffness"] - d["gt_action"][..., 18]) ** 2
                    ep_res["stiffness_mse"] = {
                        "k0": mean_or_nan(s_err[:, 0]),
                        "full": mean_or_nan(s_err[valid]),
                    }
                    pooled_stiff["k0"].append(s_err[:, 0])
                    pooled_stiff["full"].append(s_err[valid])
                    ep_res["num_queries"] = int(idx.shape[0])
                    res[ep] = ep_res

                pooled = {}
                for t in TARGETS:
                    pooled[t] = {
                        m: {k: mean_or_nan(np.concatenate(v[k])) for k in ("k0", "full")}
                        for m, v in pooled_step[t].items()
                    }
                    for m, v in pooled_smooth[t].items():
                        pooled[t][m] = {k: mean_or_nan(v[k]) for k in ("pred", "expert")}
                pooled["stiffness_mse"] = {
                    k: mean_or_nan(np.concatenate(v)) for k, v in pooled_stiff.items()
                }
                pooled["num_queries"] = int(sum(res[e]["num_queries"] for e in episodes))
                res["pooled"] = pooled

                for scope in episodes + ["pooled"]:
                    r = res[scope]
                    for t in TARGETS:
                        for m in STEP_METRICS:
                            for k in ("k0", "full"):
                                rows.append([run, ck, mode, t, m, k, scope, r[t][m][k]])
                        for m in SMOOTH_METRICS:
                            for k in ("pred", "expert"):
                                rows.append([run, ck, mode, t, m, k, scope, r[t][m][k]])
                    for k in ("k0", "full"):
                        rows.append([run, ck, mode, "-", "stiffness_mse", k, scope, r["stiffness_mse"][k]])

    (args.out_dir / "metrics.json").write_text(
        json.dumps({"episodes": episodes, "gt_decode_max_err_mm": checks, "results": results}, indent=2)
    )
    with open(args.out_dir / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "ckpt", "decode", "target", "metric", "stat", "scope", "value"])
        w.writerows(rows)
    print("[metrics] gt decode max err (mm):", json.dumps(checks))

    plot_bars(results, episodes, args.ckpts, args.out_dir)
    print(f"[metrics] wrote {args.out_dir}")
    return 0


def _episode_spread(results, run, ck, mode, getter, episodes) -> float:
    vals = [getter(results[run][ck][mode][e]) for e in episodes]
    vals = np.asarray(vals, dtype=float)
    return float(np.nanstd(vals))


def plot_bars(results: Dict, episodes: List[str], ckpts: List[str], out_dir: Path) -> None:
    runs = [r for r in RUNS if r in results]
    x = np.arange(len(runs))
    colors = {"k0": "#4c72b0", "full": "#dd8452"}
    plot_dir = out_dir / "bars"
    plot_dir.mkdir(exist_ok=True)

    def bar_panel(ax, mode, getter, keys, title, expert_getter=None):
        n = len(ckpts) * len(keys)
        w = 0.8 / n
        i = 0
        for ck in ckpts:
            for k in keys:
                vals, errs = [], []
                for r in runs:
                    if ck not in results[r]:
                        vals.append(np.nan)
                        errs.append(0)
                        continue
                    vals.append(getter(results[r][ck][mode]["pooled"], k))
                    errs.append(
                        _episode_spread(results, r, ck, mode, lambda e: getter(e, k), episodes)
                    )
                hatch = "" if ck == ckpts[0] else "//"
                ax.bar(
                    x + (i - (n - 1) / 2) * w,
                    vals,
                    w,
                    yerr=errs,
                    capsize=2,
                    color=colors.get(k, "#55a868"),
                    hatch=hatch,
                    edgecolor="k",
                    linewidth=0.4,
                    label=f"{ck} {k}",
                )
                i += 1
        if expert_getter is not None:
            for j, r in enumerate(runs):
                ev = expert_getter(results[r][ckpts[0]][mode]["pooled"])
                ax.hlines(ev, j - 0.4, j + 0.4, colors="k", linestyles="--", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(runs, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3, axis="y")

    for metric in STEP_METRICS:
        fig, axs = plt.subplots(len(TARGETS), len(DECODE_MODES), figsize=(12, 7), squeeze=False)
        for i, t in enumerate(TARGETS):
            for j, mode in enumerate(DECODE_MODES):
                bar_panel(
                    axs[i, j],
                    mode,
                    lambda r, k, t=t, m=metric: r[t][m][k],
                    ("k0", "full"),
                    f"{t} target, {mode} decode",
                )
        axs[0, 0].legend(fontsize=7)
        fig.suptitle(f"{metric} (pooled over val queries; error bar = std across val episodes)")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{metric}.png", dpi=110)
        plt.close(fig)

    for metric in SMOOTH_METRICS:
        fig, axs = plt.subplots(len(TARGETS), len(DECODE_MODES), figsize=(12, 7), squeeze=False)
        for i, t in enumerate(TARGETS):
            for j, mode in enumerate(DECODE_MODES):
                bar_panel(
                    axs[i, j],
                    mode,
                    lambda r, k, t=t, m=metric: r[t][m][k],
                    ("pred",),
                    f"{t} target, {mode} decode (dashed = expert)",
                    expert_getter=None
                    if metric.startswith("seam")
                    else (lambda r, t=t, m=metric: r[t][m]["expert"]),
                )
        axs[0, 0].legend(fontsize=7)
        fig.suptitle(metric)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{metric}.png", dpi=110)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    bar_panel(ax, "oracle", lambda r, k: r["stiffness_mse"][k], ("k0", "full"), "stiffness_mse")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plot_dir / "stiffness_mse.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())

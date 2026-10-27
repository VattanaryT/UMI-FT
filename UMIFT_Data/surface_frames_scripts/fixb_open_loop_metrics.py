#!/usr/bin/env python3
"""Open-loop metrics for the Fix B runs.

Reads exports from the existing mold_aware_open_loop_export.py (do not modify
that script; run it with PYTHONPATH including this directory so run 3's
UmiFTDatasetFixB loads). Decodes anchored only:

  r1_world_world   absolute fixture pose in the command slot
  r2 / r3          integrate v_par, delta_n from the measured pose at the
                   chunk start. Heading is R(x_s) times the fixed offset,
                   180 deg about the normal chosen from the measured tool.

Scores against the expert tool pose (fixture-frame fb): signed-distance error
(main), tangential error, normal-axis angle error, chunk seam, jerk/SPARC,
stiffness MSE. First step and full horizon, per val episode and pooled.

Usage:
  python fixb_open_loop_metrics.py \
    --exports_root /mnt/data/umidata/ft/training_outputs/fixb/eval/exports
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
import zarr
from scipy.spatial.transform import Rotation

from fixb_field import FixBMoldField, integrate_fixb
from fixb_labels import heading_branch, make_R_field_continuous, pose7_to_Rp

TOOL_Z = 2
STEP_METRICS = (
    "sd_err_mm",
    "sd_bias_mm",
    "tangential_mm",
    "normal_mm",
    "normal_angle_deg",
    "pos_mm",
    "rot_deg",
    "tilt_err_deg",
)
RUNS = ("r1_world_world", "r2_world_fixb", "r3_surface_fixb", "r4_world_fixb_tilt")


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


def decode(run, action, times, p0, R0, field, R_off):
    """action (Q,H,21|27), times (Q,H), p0/R0 measured tool at the chunk start."""
    Q, H, D = action.shape
    if run == "r1_world_world":
        pos = action[..., 0:3]
        R = rot6d_rows_to_R(action[..., 3:9])
        return pos, R
    dt = np.diff(times, axis=1)
    dt = np.concatenate([dt, dt[:, -1:]], axis=1)
    dec = integrate_fixb(field, p0, action[..., 0:2], action[..., 2], dt)
    if run == "r4_world_fixb_tilt":
        if D < 27:
            raise ValueError(f"r4 decode needs action dim >= 27, got {D}")
        dR = rot6d_rows_to_R(action[..., 21:27])
        R_f = dec["R"]
        # Per-query field continuity (same rule as labels / runtime).
        R_out = np.empty_like(R_f)
        for qi in range(Q):
            R_out[qi], _ = make_R_field_continuous(R_f[qi])
        R = np.einsum("qhij,qhjk->qhik", R_out, dR)
        return dec["x_d"], R
    B = heading_branch(dec["R"][:, 0], R0, R_off)
    R = np.einsum("qhij,qjk->qhik", dec["R"], B)
    return dec["x_d"], R


def _tilt_from_minus_n(R, n) -> np.ndarray:
    z = R[..., TOOL_Z]
    return np.degrees(np.arccos(np.clip(np.sum(z * (-n), axis=-1), -1.0, 1.0)))


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
        "tilt_err_deg": np.abs(
            _tilt_from_minus_n(R, n_p) - _tilt_from_minus_n(R_e, n_e)
        ),
    }


def mean(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exports_root", type=Path, required=True)
    ap.add_argument(
        "--dataset_root",
        type=Path,
        default=Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_data_v2_full_fixb"),
    )
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--ckpt", default="latest")
    ap.add_argument("--exec_h", type=int, default=4)
    args = ap.parse_args()
    out = args.out_dir or (args.exports_root.parent / "metrics")
    out.mkdir(parents=True, exist_ok=True)

    field = FixBMoldField(args.dataset_root / "shared" / "fixb_field.npz")
    meta = json.loads((args.dataset_root / "shared" / "fixb_meta.json").read_text())
    R_off = np.asarray(meta["R_off"], dtype=np.float64)
    expert_root = zarr.open(args.dataset_root / "r1_world_world" / "acp_replay_buffer.zarr", "r")["data"]

    rows = []
    pooled = {}
    for run in RUNS:
        ck = args.exports_root / run / args.ckpt
        if not (ck / "meta.json").exists():
            print(f"[metrics] missing {ck}")
            continue
        exp_meta = json.loads((ck / "meta.json").read_text())
        episodes = exp_meta.get("episodes") or exp_meta.get("val_episodes")
        bag = {m: {"k0": [], "full": []} for m in STEP_METRICS}
        bag["stiffness_mse"] = []
        bag["seam_mm"] = []
        bag["jerk"] = []
        bag["sparc"] = []
        bag["expert_jerk"] = []
        for ep in episodes:
            d = np.load(ck / f"episode_{ep}.npz")
            nv = d["n_valid"]
            action = d["raw_action"]
            idx = d["action_robot_idx"]
            times = d["action_target_time_s"]
            g = expert_root[f"episode_{ep}"]
            R_fb, p_fb = pose7_to_Rp(g["ts_pose_fb_0"])
            stiff = np.asarray(g["stiffness_0"]).reshape(-1)
            q = d["query_robot_idx"]
            pos, R = decode(run, action, times, p_fb[q], R_fb[q], field, R_off)
            pos_e = p_fb[idx]
            R_e = R_fb[idx]
            # padded tail is repeated; score only valid steps
            m = step_metrics(pos, R, pos_e, R_e, field)
            valid = np.arange(action.shape[1])[None, :] < nv[:, None]
            for key, val in m.items():
                bag[key]["k0"].append(val[:, 0])
                bag[key]["full"].append(val[valid])
            bag["stiffness_mse"].append((action[..., 18] - stiff[idx])[valid] ** 2)
            rgb = d["rgb_id"]
            order = np.argsort(rgb)
            for a, b in zip(order[:-1], order[1:]):
                if rgb[b] - rgb[a] == args.exec_h and nv[a] > args.exec_h:
                    bag["seam_mm"].append(np.linalg.norm(pos[b, 0] - pos[a, args.exec_h]) * 1e3)
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
            ep_row = {"run": run, "episode": int(ep)}
            for key, val in m.items():
                ep_row[f"{key}_k0"] = mean(val[:, 0])
                ep_row[f"{key}_full"] = mean(val[valid])
            rows.append(ep_row)
            print(f"[metrics] {run} ep {ep} sd_k0 {ep_row['sd_err_mm_k0']:.1f} sd_full {ep_row['sd_err_mm_full']:.1f}")
        pooled[run] = {
            **{f"{k}_k0": mean(np.concatenate(v["k0"])) for k, v in bag.items() if k in STEP_METRICS},
            **{f"{k}_full": mean(np.concatenate(v["full"])) for k, v in bag.items() if k in STEP_METRICS},
            "stiffness_mse": mean(np.concatenate(bag["stiffness_mse"])) if bag["stiffness_mse"] else None,
            "seam_mm": mean(bag["seam_mm"]),
            "chunk_jerk_rms": mean(bag["jerk"]),
            "expert_chunk_jerk_rms": mean(bag["expert_jerk"]),
            "chunk_sparc": mean(bag["sparc"]),
        }
        rows.append({"run": run, "episode": "pooled", **{k: pooled[run][k] for k in pooled[run]}})

    (out / "metrics.json").write_text(json.dumps({"pooled": pooled, "rows": rows}, indent=2))
    if rows:
        with (out / "metrics.csv").open("w", newline="") as f:
            fields = list(dict.fromkeys(k for row in rows for k in row))
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    if pooled:
        keys = [
            "sd_err_mm_k0",
            "sd_err_mm_full",
            "tangential_mm_full",
            "normal_angle_deg_full",
            "tilt_err_deg_full",
            "seam_mm",
            "chunk_jerk_rms",
        ]
        fig, axs = plt.subplots(2, 4, figsize=(14, 6))
        for ax, key in zip(axs.ravel(), keys):
            ax.bar(list(pooled), [pooled[r].get(key, np.nan) for r in pooled])
            ax.set_title(key)
            ax.tick_params(axis="x", labelrotation=15)
        for ax in axs.ravel()[len(keys) :]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out / "bars.png", dpi=110)
    print(json.dumps(pooled, indent=2))
    print(f"[metrics] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

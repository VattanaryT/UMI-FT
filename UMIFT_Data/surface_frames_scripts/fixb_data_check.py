#!/usr/bin/env python3
"""Data check and label integration test for the Fix B datasets.

On train episodes (val split seed 42, ratio 0.1 — the same episodes the
trainer holds out):

  * normal sign flips, frame step vs the tool's frame step, signed-distance
    jumps, curvature step size
  * run 3 fb is in the surface frame and differs from run 2; mold channels match
  * integrating v_par, delta_n from the measured pose reproduces the demo
  * range-normalizer scales of the three Fix B action dims

Writes <out>/data_check.json and a few PNGs.

Usage:
  python fixb_data_check.py \
    --out_root /mnt/data/umidata/ft/mold_aware_datasets/stringer_data_v2_full_fixb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from fixb_field import FixBMoldField, integrate_fixb

H = 24
SD_JUMP_M = 0.050
# Held out by the training image (seed 42, val_ratio 0.1, 52 episodes).
# Local numpy 2.x Generator.choice does not reproduce this; the Hyak image does.
VAL_EPISODES = (9, 12, 29, 38, 42)


def val_ids(n: int) -> np.ndarray:
    ids = np.asarray([i for i in VAL_EPISODES if i < n], dtype=int)
    if len(ids) == 0:
        raise RuntimeError(f"no val episodes in 0..{n - 1}")
    return ids


def frame_step_deg(R: np.ndarray) -> np.ndarray:
    M = np.einsum("nji,njk->nik", R[:-1], R[1:])
    c = np.clip((M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2] - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(c))


def tool_R(pose7: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(np.asarray(pose7)[:, [4, 5, 6, 3]]).as_matrix()


def pct(x, qs=(50, 90, 99)):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return [None] * len(qs)
    return np.percentile(x, qs).round(3).tolist()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out_root",
        type=Path,
        default=Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_data_v2_full_fixb"),
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--query_stride", type=int, default=2, help="RGB frames between chunks, matches training")
    ap.add_argument("--plot_episode", type=int, default=0)
    args = ap.parse_args()
    out = args.out or (args.out_root / "checks")
    out.mkdir(parents=True, exist_ok=True)

    field = FixBMoldField(args.out_root / "shared" / "fixb_field.npz")
    meta = json.loads((args.out_root / "shared" / "fixb_meta.json").read_text())
    src = zarr.open(meta["source_zarr"], "r")["data"]
    r2 = zarr.open(args.out_root / "r2_world_fixb" / "acp_replay_buffer.zarr", "r")["data"]
    r3 = zarr.open(args.out_root / "r3_surface_fixb" / "acp_replay_buffer.zarr", "r")["data"]
    names = sorted(src.keys(), key=lambda s: int(s.split("_")[-1]))
    held_out = set(int(i) for i in val_ids(len(names)))
    train = [n for n in names if int(n.split("_")[-1]) not in held_out]

    faces = field.faces
    # boundary vertices: edges used by exactly one triangle
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges = edges[order]
    same = np.all(edges[1:] == edges[:-1], axis=1)
    once = np.ones(len(edges), dtype=bool)
    once[1:][same] = False
    once[:-1][same] = False
    bverts = np.unique(edges[once])
    btree = cKDTree(field.vertices[bverts])

    flips = jumps = 0
    n_steps = 0
    frame_rgb, tool_rgb, curv_step, sd_all = [], [], [], []
    v_all, dn_all = [], []
    obs_pos_diff, mold_diff = [], []
    err_rows = []

    for name in train:
        ep, a2, a3 = src[name], r2[name], r3[name]
        R = np.asarray(a2["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3)
        n = R[:, :, 2]
        sd = np.asarray(a2["mold_signed_distance_0"], dtype=np.float64).reshape(-1)
        dots = np.sum(n[1:] * n[:-1], axis=1)
        flips += int((dots < 0).sum())
        dsd = np.abs(np.diff(sd))
        jumps += int((dsd > SD_JUMP_M).sum())
        n_steps += len(dsd)
        rt = np.asarray(ep["robot_time_stamps_0"]).reshape(-1)
        rgb = np.asarray(ep["rgb_time_stamps_0"]).reshape(-1)
        ir = np.clip(np.searchsorted(rt, rgb), 0, len(rt) - 1)
        frame_rgb.append(frame_step_deg(R[ir]))
        tool_rgb.append(frame_step_deg(tool_R(ep["ts_pose_fb_0"])[ir]))
        curv = np.asarray(a2["mold_curvature_0"], dtype=np.float64)
        curv_step.append(np.linalg.norm(np.diff(curv[ir][:, :2], axis=0), axis=1))
        sd_all.append(sd)
        v_all.append(np.asarray(a2["fixb_v_par_0"], dtype=np.float64))
        dn_all.append(np.asarray(a2["fixb_delta_n_0"], dtype=np.float64).reshape(-1))

        fb2 = np.asarray(a2["ts_pose_fb_0"])
        fb3 = np.asarray(a3["ts_pose_fb_0"])
        obs_pos_diff.append(np.linalg.norm(fb2[:, :3] - fb3[:, :3], axis=1))
        mold_diff.append(
            np.max(np.abs(np.asarray(a2["mold_signed_distance_0"]) - np.asarray(a3["mold_signed_distance_0"])))
            + np.max(np.abs(np.asarray(a2["mold_normal_0"]) - np.asarray(a3["mold_normal_0"])))
            + np.max(np.abs(np.asarray(a2["mold_R_field_rot6d_0"]) - np.asarray(a3["mold_R_field_rot6d_0"])))
        )

        # integration on the training query grid
        q = np.arange(0, len(rgb) - H, args.query_stride)
        idx = np.clip(np.searchsorted(rt, rgb[q[:, None] + np.arange(H)]), 0, len(rt) - 1)
        p = fb2[:, :3]
        dt = np.diff(rt[idx], axis=1)
        dt = np.concatenate([dt, dt[:, -1:]], axis=1)
        dec = integrate_fixb(
            field,
            p[idx[:, 0]],
            np.asarray(a2["fixb_v_par_0"])[idx].astype(np.float64),
            np.asarray(a2["fixb_delta_n_0"])[idx, 0].astype(np.float64),
            dt,
        )
        err = np.linalg.norm(dec["x_d"] - p[idx], axis=-1) * 1e3
        xs = np.asarray(a2["mold_x_s_0"], dtype=np.float64)[idx]
        db = btree.query(xs.reshape(-1, 3))[0].reshape(idx.shape) * 1e3
        jump = np.linalg.norm(np.diff(xs, axis=1), axis=-1).max(axis=1) * 1e3
        near = (np.abs(dn_all[-1][idx]) * 1e3 < 30).all(axis=1)
        interior = (db > 3).all(axis=1)
        err_rows.append((err, interior, jump < 15, near))
        print(f"[check] {name}: flips {(dots < 0).sum()} sd_jumps {(dsd > SD_JUMP_M).sum()} k23 p50 {np.median(err[:, -1]):.2f} mm")

    frame_rgb = np.concatenate(frame_rgb)
    tool_rgb = np.concatenate(tool_rgb)
    curv_step = np.concatenate(curv_step)
    v = np.concatenate(v_all)
    dn = np.concatenate(dn_all)
    labels = np.concatenate([v, dn[:, None]], axis=1)
    lo, hi = labels.min(0), labels.max(0)
    span = np.maximum(hi - lo, 1e-9)
    normed = (labels - lo) / span * 2 - 1
    err = np.concatenate([r[0] for r in err_rows])
    interior = np.concatenate([r[1] for r in err_rows])
    nojump = np.concatenate([r[2] for r in err_rows])
    near = np.concatenate([r[3] for r in err_rows])

    def bucket(mask):
        e = err[mask]
        return {"n_chunks": int(mask.sum()), "k0_mm_p50_p90": pct(e[:, 0], (50, 90)), "k23_mm_p50_p90_p99": pct(e[:, -1]), "full_mean_mm": round(float(e.mean()), 3) if len(e) else None}

    report = {
        "train_episodes": [int(n.split("_")[-1]) for n in train],
        "val_episodes": sorted(held_out),
        "normal_sign_flips": flips,
        "normal_sign_flips_per_1e5_steps": round(flips / max(n_steps, 1) * 1e5, 3),
        "sd_jumps_over_50mm": jumps,
        "frame_step_deg_rgb_p50_p90_p99": pct(frame_rgb),
        "tool_step_deg_rgb_p50_p90_p99": pct(tool_rgb),
        "curvature_step_1_per_m_p50_p90_p99": pct(curv_step),
        "r3_fb_differs_from_r2_mm_p50": pct(np.concatenate(obs_pos_diff) * 1e3, (50,))[0],
        "mold_channel_max_abs_diff_r2_r3": float(np.max(mold_diff)),
        "label_raw_min": lo.round(5).tolist(),
        "label_raw_max": hi.round(5).tolist(),
        "label_raw_std": labels.std(0).round(5).tolist(),
        "label_normalized_std": normed.std(0).round(4).tolist(),
        "label_dims": ["v_t1_m_per_s", "v_t2_m_per_s", "delta_n_m"],
        "integration_mm": {
            "all": bucket(np.ones(len(err), dtype=bool)),
            "foot_interior": bucket(interior),
            "interior_no_foot_jump": bucket(interior & nojump),
            "interior_no_jump_near_surface": bucket(interior & nojump & near),
        },
        "heading": {
            "flipped_frac": meta["heading_flipped_frac"],
            "residual_deg_p50_p90": meta["heading_residual_deg_p50_p90"],
            "tool_z_tilt_deg": meta["R_off_tool_z_tilt_from_minus_n_deg"],
        },
    }
    (out / "data_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    # plots for one train episode
    name = f"episode_{args.plot_episode}"
    ep, a2 = src[name], r2[name]
    rt = np.asarray(ep["robot_time_stamps_0"]).reshape(-1)
    t = rt - rt[0]
    rgb = np.asarray(ep["rgb_time_stamps_0"]).reshape(-1)
    ir = np.clip(np.searchsorted(rt, rgb), 0, len(rt) - 1)
    R = np.asarray(a2["mold_R_field_0"]).reshape(-1, 3, 3)
    fig, axs = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    axs[0].plot(t, np.asarray(a2["mold_signed_distance_0"]).reshape(-1) * 1e3)
    axs[0].set_ylabel("sd [mm]")
    axs[0].set_title(f"{name} mold obs and Fix B labels")
    axs[1].plot(t, np.asarray(a2["fixb_v_par_0"])[:, 0], label="v_t1")
    axs[1].plot(t, np.asarray(a2["fixb_v_par_0"])[:, 1], label="v_t2")
    axs[1].plot(t, np.asarray(a2["fixb_delta_n_0"]).reshape(-1), label="delta_n [m]")
    axs[1].legend(fontsize=8)
    axs[1].set_ylabel("label")
    axs[2].plot(rgb[1:] - rt[0], frame_step_deg(R[ir]), label="R(x_s)")
    axs[2].plot(rgb[1:] - rt[0], frame_step_deg(tool_R(ep["ts_pose_fb_0"])[ir]), label="tool", alpha=0.7)
    axs[2].set_ylabel("frame step [deg]")
    axs[2].legend(fontsize=8)
    curv = np.asarray(a2["mold_curvature_0"])
    axs[3].plot(t, curv[:, 0], label="k1")
    axs[3].plot(t, curv[:, 1], label="k2")
    axs[3].set_ylabel("curvature [1/m]")
    axs[3].set_xlabel("t [s]")
    axs[3].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"{name}_labels.png", dpi=100)

    fig, ax = plt.subplots(figsize=(7, 4))
    for mask, label in (
        (np.ones(len(err), bool), "all"),
        (interior & nojump, "interior, no foot jump"),
        (interior & nojump & near, "those, |dn|<30 mm"),
    ):
        ax.hist(err[mask, -1], bins=60, histtype="step", density=True, label=label)
    ax.set_xlabel("integrated path error at step 23 [mm]")
    ax.set_xlim(0, 80)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "integration_error.png", dpi=100)
    print(f"[check] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

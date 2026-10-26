#!/usr/bin/env python3
"""Build r4_world_fixb_tilt from the existing r2_world_fixb zarr.

Copies r2 unchanged (v_par, δn, mold obs, chunking) and adds per-step
  ΔR_t = R(x_s)^T R_tool,t
as rot6d in fixb_delta_R_rot6d_0. Twist branch is made continuous within each
episode (180° about n); if the pad looks 180°-symmetric we fold to R_off instead.

Action packing (via raw_to_action21 append): action27 =
  [v_t1, v_t2, δn, id_rot6d(6), vt_unused(9), stiff, grip, grasp, ΔR_rot6d(6)]

Usage:
  python fixb_build_r4_dataset.py
  python fixb_build_r4_dataset.py --checks_only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import zarr

from fixb_field import FixBMoldField
from fixb_labels import (
    RZ_PI,
    delta_R_from_field_tool,
    fold_delta_R_to_R_off_branch,
    make_delta_R_continuous,
    pad_symmetric_under_twist,
    pose7_to_Rp,
    rot_angle_deg,
    so3_to_rot6d_rows,
)

ROOT = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_scanner_v2_fixb")
R2 = ROOT / "r2_world_fixb" / "acp_replay_buffer.zarr"
R4 = ROOT / "r4_world_fixb_tilt" / "acp_replay_buffer.zarr"
SHARED = ROOT / "shared"
LABEL_VERSION = "fixb_v4_tracked_r4_tilt"


def episode_names(group) -> list[str]:
    return sorted(
        (k for k in group.keys() if k.startswith("episode_")),
        key=lambda s: int(s.split("_")[-1]),
    )


def link_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    for dirpath, _, filenames in os.walk(src):
        rel = Path(dirpath).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for fn in filenames:
            s, d = Path(dirpath) / fn, dst / rel / fn
            if fn.startswith("."):
                shutil.copy2(s, d)
            else:
                os.link(s, d)


def replace_array(group, key: str, data: np.ndarray, like=None) -> None:
    kw = {} if like is None else {"dtype": like.dtype, "compressor": like.compressor}
    if key in group:
        del group[key]
    group.create_dataset(
        key, data=np.asarray(data, dtype=kw.get("dtype", data.dtype)), chunks=data.shape, **kw
    )


def compute_episode_delta_R(ep, R_off: np.ndarray, fold: bool):
    R_tool, _ = pose7_to_Rp(ep["ts_pose_fb_0"])
    R_f = np.asarray(ep["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3)
    raw = delta_R_from_field_tool(R_f, R_tool)
    if fold:
        dR = fold_delta_R_to_R_off_branch(raw, R_off)
        flipped = np.zeros(len(dR), dtype=bool)
    else:
        dR, flipped = make_delta_R_continuous(raw)
    return dR, flipped, raw


def build(out: Path = R4) -> dict:
    t0 = time.time()
    meta = json.loads((SHARED / "fixb_meta.json").read_text())
    R_off = np.asarray(meta["R_off"], dtype=np.float64)
    src = zarr.open(str(R2), "r")
    names = episode_names(src["data"])

    # Probe pad symmetry on a subsample of raw ΔR (before continuity).
    probe = []
    for name in names[:: max(1, len(names) // 15)]:
        R_tool, _ = pose7_to_Rp(src["data"][name]["ts_pose_fb_0"])
        R_f = np.asarray(src["data"][name]["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3)
        probe.append(delta_R_from_field_tool(R_f, R_tool)[::50])
    probe_dR = np.concatenate(probe, axis=0)
    fold = pad_symmetric_under_twist(probe_dR, R_off)
    branch_mode = "fold_to_R_off" if fold else "continuous_per_episode"
    print(f"[r4] twist branch mode: {branch_mode} (pad_symmetric={fold})")

    print(f"[r4] linking {R2} -> {out}")
    link_tree(R2, out)
    root = zarr.open(str(out), "a")

    n_flip = 0
    n_tot = 0
    for i, name in enumerate(names):
        ep = root["data"][name]
        dR, flipped, _ = compute_episode_delta_R(ep, R_off, fold=fold)
        r6 = so3_to_rot6d_rows(dR).astype(np.float32)
        replace_array(ep, "fixb_delta_R_rot6d_0", r6)
        replace_array(ep, "fixb_delta_R_flipped_0", flipped.astype(np.uint8).reshape(-1, 1))
        ep.attrs["pose_variant"] = "r4_world_fixb_tilt"
        ep.attrs["label_version"] = LABEL_VERSION
        ep.attrs["delta_R_branch_mode"] = branch_mode
        n_flip += int(flipped.sum())
        n_tot += len(flipped)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"[r4 {i + 1}/{len(names)}] {name}: N={len(r6)} flipped={int(flipped.sum())}")

    r4_meta = {
        **meta,
        "label_version": LABEL_VERSION,
        "variant": "r4_world_fixb_tilt",
        "parent_variant": "r2_world_fixb",
        "delta_R": "R(x_s)^T R_tool (demo TCP / runner TCP), rot6d rows",
        "delta_R_branch_mode": branch_mode,
        "delta_R_pad_symmetric_under_180_about_n": bool(fold),
        "delta_R_flipped_frac": float(n_flip / max(n_tot, 1)),
        "action_packing": {
            "layout": (
                "action27: v_t1, v_t2, delta_n, id_rot6d[6], vt_unused[9], "
                "stiffness, gripper, grasp_force, delta_R_rot6d[6]"
            ),
            "dims": 27,
            "ts_pose_command_0": "[v_t1, v_t2, delta_n, quat identity] (same as r2)",
            "fixb_delta_R_rot6d_0": "appended as action[..., 21:27] by raw_to_action21",
        },
        "decode": (
            "x_s0 = tracked_foot(p_meas); integrate v_par, delta_n; "
            "R_d = R(x_s) project(delta_R_rot6d); hover keeps predicted delta_R"
        ),
    }
    root.attrs.update(
        {
            "mold_aware_variant": "r4_world_fixb_tilt",
            "fixb": r4_meta,
            "world_frame": "fixture",
        }
    )
    (out.parent / "r4_meta.json").write_text(json.dumps(r4_meta, indent=2))
    # Keep shared meta's R_off; write r4-specific sidecar next to shared too.
    (SHARED / "fixb_r4_meta.json").write_text(json.dumps(r4_meta, indent=2))
    print(f"[r4] done in {time.time() - t0:.0f}s -> {out} (flipped_frac={n_flip / max(n_tot, 1):.4f})")
    return r4_meta


def run_checks(out: Path = R4) -> dict:
    """Pre-train checks: round-trip, tilt hist, continuity, rotate-mold invariance."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    field = FixBMoldField(SHARED / "fixb_field.npz")
    meta = json.loads((SHARED / "fixb_meta.json").read_text())
    R_off = np.asarray(meta["R_off"], dtype=np.float64)
    root = zarr.open(str(out), "r")
    names = episode_names(root["data"])
    branch_mode = root.attrs.get("fixb", {}).get("delta_R_branch_mode")
    if branch_mode is None:
        # zarr may store nested dict as plain; fall back to sidecar
        side = json.loads((out.parent / "r4_meta.json").read_text())
        branch_mode = side["delta_R_branch_mode"]

    rt_err = []
    tilt_all = []
    tilt_ep_std = []
    jump_max = []
    n_jump_gt30 = 0
    n_steps = 0
    rot_mold_err = []

    # Fixed random rotation of the mold (and demos) for invariance test.
    rng = np.random.default_rng(0)
    R_m = __import__("scipy.spatial.transform", fromlist=["Rotation"]).Rotation.random(
        random_state=int(rng.integers(1e9))
    ).as_matrix()

    for name in names:
        ep = root["data"][name]
        R_tool, p_tool = pose7_to_Rp(ep["ts_pose_fb_0"])
        R_f = np.asarray(ep["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3)
        r6 = np.asarray(ep["fixb_delta_R_rot6d_0"], dtype=np.float64).reshape(-1, 6)
        # project rot6d → SO(3) (same as runtime)
        from mold_aware_runtime import rot6d_rows_to_R

        dR = rot6d_rows_to_R(r6)
        R_hat = np.einsum("nij,njk->nik", R_f, dR)
        err = rot_angle_deg(R_hat, R_tool)
        rt_err.append(err)

        # tool axis = column 2; angle to surface normal
        z = R_tool[:, :, 2]
        n = R_f[:, :, 2]
        ang = np.degrees(np.arccos(np.clip(np.sum(z * n, axis=-1), -1.0, 1.0)))
        # tilt from -n (press into mold) matches prior ~28° convention when z ~ -n
        tilt = np.degrees(np.arccos(np.clip(np.sum(z * (-n), axis=-1), -1.0, 1.0)))
        tilt_all.append(tilt)
        tilt_ep_std.append(float(np.std(tilt)))

        if len(dR) > 1:
            jumps = rot_angle_deg(dR[1:], dR[:-1])
            jump_max.append(float(np.max(jumps)))
            n_jump_gt30 += int(np.sum(jumps > 30.0))
            n_steps += len(jumps)

        # Rotate mold + demos together: ΔR' = (R_m R_f)^T (R_m R_tool) = ΔR
        R_f2 = np.einsum("ij,njk->nik", R_m, R_f)
        R_t2 = np.einsum("ij,njk->nik", R_m, R_tool)
        dR2 = delta_R_from_field_tool(R_f2, R_t2)
        # apply same branch rule as build
        if branch_mode == "fold_to_R_off":
            # R_off also rotates: R_off' = R_m R_off? For invariance of ΔR under joint
            # rotation, raw ΔR is invariant before folding. Folding uses fixed R_off
            # in fixture frame, so fold mode is NOT rotate-invariant. Continuous is.
            dR2_c = dR2
        else:
            dR2_c, _ = make_delta_R_continuous(dR2)
        rot_mold_err.append(rot_angle_deg(dR2_c, dR))

    rt = np.concatenate(rt_err)
    tilt = np.concatenate(tilt_all)
    rot_e = np.concatenate(rot_mold_err)

    report = {
        "branch_mode": branch_mode,
        "round_trip_deg": {
            "max": float(np.max(rt)),
            "p50": float(np.median(rt)),
            "p99": float(np.percentile(rt, 99)),
            "pass_0p5": bool(np.max(rt) < 0.5),
        },
        "tilt_from_minus_n_deg": {
            "median": float(np.median(tilt)),
            "mean": float(np.mean(tilt)),
            "within_ep_std_median": float(np.median(tilt_ep_std)),
            "within_ep_std_mean": float(np.mean(tilt_ep_std)),
        },
        "continuity": {
            "max_jump_deg": float(np.max(jump_max)) if jump_max else None,
            "n_jumps_gt_30deg": int(n_jump_gt30),
            "n_steps": int(n_steps),
            "pass_no_gt30": n_jump_gt30 == 0,
        },
        "rotate_mold": {
            "max_deltaR_change_deg": float(np.max(rot_e)),
            "p99_deg": float(np.percentile(rot_e, 99)),
            "pass_invariant": bool(np.max(rot_e) < 0.5),
            "note": (
                "raw ΔR = R_f^T R_tool is SO(3)-invariant under joint R_m; "
                "continuous branch preserves it; fold-to-R_off would not."
            ),
        },
        "action_layout": {
            "dims": 27,
            "slots": {
                "0": "v_t1 (m/s)",
                "1": "v_t2 (m/s)",
                "2": "delta_n (m)",
                "3:9": "identity rot6d (unused heading slot)",
                "9:18": "unused VT pose9",
                "18": "stiffness",
                "19": "gripper",
                "20": "grasp_force",
                "21:27": "delta_R rot6d (= R(x_s)^T R_tool)",
            },
        },
    }

    out_dir = out.parent / "checks"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(tilt, bins=60, color="steelblue", edgecolor="none")
    ax.axvline(np.median(tilt), color="crimson", ls="--", label=f"median {np.median(tilt):.1f}°")
    ax.set_xlabel("tool-z tilt from −n (deg)")
    ax.set_ylabel("count")
    ax.set_title("r4 demo tilt distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "tilt_histogram.png", dpi=120)
    plt.close(fig)
    (out_dir / "pretrain_checks.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote {out_dir}")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=R4)
    ap.add_argument("--checks_only", action="store_true")
    ap.add_argument("--skip_checks", action="store_true")
    args = ap.parse_args()
    if not args.checks_only:
        build(args.out)
    if not args.skip_checks:
        run_checks(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

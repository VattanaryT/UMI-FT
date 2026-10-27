#!/usr/bin/env python3
"""Build r4_world_fixb_tilt from the existing r2_world_fixb zarr.

Copies r2 unchanged (v_par, δn, mold_R_field, chunking) and adds per-step
  ΔR_t = R(x_s)^T R_tool,t
as rot6d in fixb_delta_R_rot6d_0, using the *same* mesh field frame as r2 / as
integrate_fixb at decode. No field-frame rewrite: flipping R_f without
negating v_par would reverse the slide direction.

Twist continuity: pad is not 180°-symmetric, so we keep the raw ΔR (exact
round-trip with mesh R_f). Remaining >30° jumps are mesh R_f discontinuities;
training keeps r2 chunks and zeros ΔR loss after a jump (runtime holds R_d).

Action packing (via raw_to_action21 append): action27 =
  [v_t1, v_t2, δn, id_rot6d(6), vt_unused(9), stiff, grip, grasp, ΔR_rot6d(6)]

Usage:
  python fixb_build_r4_dataset.py
  python fixb_build_r4_dataset.py --relabel_only
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

from fixb_labels import (
    delta_R_from_field_tool,
    fold_delta_R_to_R_off_branch,
    pad_symmetric_under_twist,
    pose7_to_Rp,
    rot_angle_deg,
    so3_to_rot6d_rows,
)

ROOT = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_scanner_v2_fixb")
R2 = ROOT / "r2_world_fixb" / "acp_replay_buffer.zarr"
R4 = ROOT / "r4_world_fixb_tilt" / "acp_replay_buffer.zarr"
SHARED = ROOT / "shared"
LABEL_VERSION = "fixb_v4_tracked_r4_tilt_meshRf"
OFF_EDGE_M = 0.005


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


def restore_array_from_src(ep_dst, ep_src, key: str) -> None:
    """Re-hardlink key from r2 if we previously broke the link by rewriting."""
    if key not in ep_src:
        return
    src_arr = ep_src[key]
    data = np.asarray(src_arr[:])
    if key in ep_dst:
        del ep_dst[key]
    ep_dst.create_dataset(
        key,
        data=data,
        chunks=src_arr.chunks,
        dtype=src_arr.dtype,
        compressor=src_arr.compressor,
    )


def replace_array(group, key: str, data: np.ndarray, like=None) -> None:
    kw = {} if like is None else {"dtype": like.dtype, "compressor": like.compressor}
    if key in group:
        del group[key]
    group.create_dataset(
        key, data=np.asarray(data, dtype=kw.get("dtype", data.dtype)), chunks=data.shape, **kw
    )


def build(out: Path = R4, relabel_only: bool = False) -> dict:
    t0 = time.time()
    meta = json.loads((SHARED / "fixb_meta.json").read_text())
    R_off = np.asarray(meta["R_off"], dtype=np.float64)
    src = zarr.open(str(R2), "r")
    names = episode_names(src["data"])

    # Probe pad symmetry on raw ΔR (mesh R_f, same as r2).
    probe = []
    for name in names[:: max(1, len(names) // 15)]:
        R_tool, _ = pose7_to_Rp(src["data"][name]["ts_pose_fb_0"])
        R_f = np.asarray(src["data"][name]["mold_R_field_0"], dtype=np.float64).reshape(
            -1, 3, 3
        )
        probe.append(delta_R_from_field_tool(R_f, R_tool)[::50])
    probe_dR = np.concatenate(probe, axis=0)
    fold = pad_symmetric_under_twist(probe_dR, R_off)
    branch_mode = "fold_to_R_off" if fold else "mesh_Rf_raw"
    print(f"[r4] twist branch mode: {branch_mode} (pad_symmetric={fold})")

    if not relabel_only or not out.exists():
        print(f"[r4] linking {R2} -> {out}")
        link_tree(R2, out)
    else:
        print(f"[r4] relabel_only: rewriting ΔR; restoring mold_R_field from r2")

    root = zarr.open(str(out), "a")

    for i, name in enumerate(names):
        ep = root["data"][name]
        ep_src = src["data"][name]
        # Ensure mold field / v_par match r2 (undo any prior continuous-field rewrite).
        for key in (
            "mold_R_field_0",
            "mold_R_field_rot6d_0",
            "mold_normal_0",
            "ts_pose_command_0",
            "fixb_v_par_0",
        ):
            restore_array_from_src(ep, ep_src, key)

        R_tool, _ = pose7_to_Rp(ep["ts_pose_fb_0"])
        R_f = np.asarray(ep["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3)
        dR = delta_R_from_field_tool(R_f, R_tool)
        if fold:
            dR = fold_delta_R_to_R_off_branch(dR, R_off)
        r6 = so3_to_rot6d_rows(dR).astype(np.float32)
        replace_array(ep, "fixb_delta_R_rot6d_0", r6)
        # No per-step flip mask when using mesh R_f raw.
        replace_array(
            ep, "fixb_delta_R_flipped_0", np.zeros((len(r6), 1), dtype=np.uint8)
        )
        ep.attrs["pose_variant"] = "r4_world_fixb_tilt"
        ep.attrs["label_version"] = LABEL_VERSION
        ep.attrs["delta_R_branch_mode"] = branch_mode
        if (i + 1) % 10 == 0 or i == 0:
            print(f"[r4 {i + 1}/{len(names)}] {name}: N={len(r6)}")

    r4_meta = {
        **meta,
        "label_version": LABEL_VERSION,
        "variant": "r4_world_fixb_tilt",
        "parent_variant": "r2_world_fixb",
        "delta_R": "R(x_s_mesh)^T R_tool (same R_f as r2 / integrate_fixb)",
        "delta_R_branch_mode": branch_mode,
        "delta_R_pad_symmetric_under_180_about_n": bool(fold),
        "runtime_twist": (
            "No episode-level twist resolution. Decode uses mesh R_f from "
            "integrate_fixb at the anchored foot; ΔR is defined in that same frame. "
            "v_par stays in mesh t1/t2 (r2 labels). Initial R_f = mesh query (no flip)."
        ),
        "action_packing": {
            "layout": (
                "action27: v_t1, v_t2, delta_n, id_rot6d[6], vt_unused[9], "
                "stiffness, gripper, grasp_force, delta_R_rot6d[6]"
            ),
            "dims": 27,
            "vt_unused": (
                "pose9 virtual-target slot from ACP schema; Fix B leaves it "
                "identity/unused (same as r2). Not a second motion command."
            ),
            "id_rot6d": (
                "constant identity rot6d in the command pose9 rotation slots; "
                "heading lives in ΔR. Identity-normalized like r2 rot slots."
            ),
        },
        "decode": (
            "x_s0 = tracked_foot(p_meas); integrate v_par, delta_n with mesh R; "
            "R_d = R_mesh project(delta_R_rot6d); hover keeps predicted delta_R"
        ),
        "train_filter_note": (
            "Same sampler chunks as r2 (scan-gap + action-gap + end pads). "
            "UmiFTDatasetR4 zeros ΔR loss (dims 21:27) after a >30° mesh R_f jump "
            f"inside the chunk; off-edge (in-plane>{OFF_EDGE_M*1e3:.0f}mm) is not filtered."
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
    (SHARED / "fixb_r4_meta.json").write_text(json.dumps(r4_meta, indent=2))
    print(f"[r4] done in {time.time() - t0:.0f}s -> {out}")
    return r4_meta


def run_checks(out: Path = R4) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mold_aware_runtime import rot6d_rows_to_R

    root = zarr.open(str(out), "r")
    names = episode_names(root["data"])
    side = json.loads((out.parent / "r4_meta.json").read_text())
    branch_mode = side["delta_R_branch_mode"]

    rt_err = []
    tilt_all = []
    tilt_contact = []
    tilt_approach = []
    tilt_ep_std = []
    tilt_ep_std_contact = []
    jump_max = []
    n_jump_gt30 = 0
    n_steps = 0
    n_jump_off_edge = 0
    rot_mold_err = []
    CONTACT_SD = 0.003

    rng = np.random.default_rng(0)
    R_m = __import__("scipy.spatial.transform", fromlist=["Rotation"]).Rotation.random(
        random_state=int(rng.integers(1e9))
    ).as_matrix()

    for name in names:
        ep = root["data"][name]
        R_tool, p = pose7_to_Rp(ep["ts_pose_fb_0"])
        R_f = np.asarray(ep["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3)
        x = np.asarray(ep["mold_x_s_0"], dtype=np.float64).reshape(-1, 3)
        n = R_f[:, :, 2]
        sd = np.sum((p - x) * n, axis=-1)
        ip = np.linalg.norm(
            (p - x) - sd[:, None] * n, axis=-1
        )
        r6 = np.asarray(ep["fixb_delta_R_rot6d_0"], dtype=np.float64).reshape(-1, 6)
        dR = rot6d_rows_to_R(r6)
        R_hat = np.einsum("nij,njk->nik", R_f, dR)
        rt_err.append(rot_angle_deg(R_hat, R_tool))

        tilt = np.degrees(np.arccos(np.clip(np.sum(R_tool[:, :, 2] * (-n), axis=-1), -1, 1)))
        contact = np.abs(sd) < CONTACT_SD
        tilt_all.append(tilt)
        tilt_contact.append(tilt[contact])
        tilt_approach.append(tilt[~contact])
        tilt_ep_std.append(float(np.std(tilt)))
        if contact.any():
            tilt_ep_std_contact.append(float(np.std(tilt[contact])))

        if len(dR) > 1:
            jumps = rot_angle_deg(dR[1:], dR[:-1])
            jump_max.append(float(np.max(jumps)))
            big = jumps > 30.0
            n_jump_gt30 += int(big.sum())
            n_steps += len(jumps)
            # jump at index t connects t-1 → t; flag if either sample off-edge
            off = ip > OFF_EDGE_M
            n_jump_off_edge += int(np.sum(big & (off[1:] | off[:-1])))

        R_f2 = np.einsum("ij,njk->nik", R_m, R_f)
        R_t2 = np.einsum("ij,njk->nik", R_m, R_tool)
        dR2 = delta_R_from_field_tool(R_f2, R_t2)
        rot_mold_err.append(rot_angle_deg(dR2, dR))

    def cat(xs):
        xs = [x for x in xs if len(x)]
        return np.concatenate(xs) if xs else np.array([np.nan])

    rt = np.concatenate(rt_err)
    tilt = cat(tilt_all)
    tc = cat(tilt_contact)
    ta = cat(tilt_approach)
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
            "all_median": float(np.median(tilt)),
            "all_within_ep_std_median": float(np.median(tilt_ep_std)),
            "contact_median": float(np.nanmedian(tc)),
            "contact_within_ep_std_median": float(np.median(tilt_ep_std_contact))
            if tilt_ep_std_contact
            else None,
            "approach_depart_median": float(np.nanmedian(ta)),
        },
        "continuity": {
            "max_jump_deg": float(np.max(jump_max)) if jump_max else None,
            "n_jumps_gt_30deg": int(n_jump_gt30),
            "n_jumps_gt_30deg_touching_off_edge_sample": int(n_jump_off_edge),
            "n_steps": int(n_steps),
            "pass_no_gt30": n_jump_gt30 == 0,
            "note": (
                "Mesh-R_f ΔR; jumps are field discontinuities. Train: same chunks "
                "as r2; mask ΔR loss after jump. Runtime: hold R_d on mesh jump."
            ),
        },
        "rotate_mold": {
            "max_deltaR_change_deg": float(np.max(rot_e)),
            "p99_deg": float(np.percentile(rot_e, 99)),
            "pass_invariant": bool(np.max(rot_e) < 0.5),
        },
        "action_layout": {
            "dims": 27,
            "slots": {
                "0": "v_t1 (m/s) in mesh t1",
                "1": "v_t2 (m/s) in mesh t2",
                "2": "delta_n (m)",
                "3:9": "identity rot6d (unused; heading in ΔR)",
                "9:18": "unused VT pose9 (ACP schema placeholder)",
                "18": "stiffness",
                "19": "gripper",
                "20": "grasp_force",
                "21:27": "delta_R rot6d (= R_mesh^T R_tool)",
            },
        },
    }

    out_dir = out.parent / "checks"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(tilt, bins=60, color="steelblue", edgecolor="none", alpha=0.5, label="all")
    ax.hist(tc, bins=60, color="crimson", edgecolor="none", alpha=0.5, label="contact")
    ax.axvline(np.median(tc), color="crimson", ls="--", label=f"contact med {np.median(tc):.1f}°")
    ax.set_xlabel("tool-z tilt from −n (deg)")
    ax.set_ylabel("count")
    ax.set_title("r4 demo tilt (all vs contact)")
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
    ap.add_argument("--relabel_only", action="store_true")
    args = ap.parse_args()
    if not args.checks_only:
        build(args.out, relabel_only=args.relabel_only)
    if not args.skip_checks:
        run_checks(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

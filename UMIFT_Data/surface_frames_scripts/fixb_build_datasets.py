#!/usr/bin/env python3
"""Build the three Fix B mold-aware datasets from the source zarr.

Variants (same mold obs channels and per-step mold arrays in all three):
  r1_world_world   obs fixture-frame fb, actions = original fixture-frame cmd/vt
  r2_world_fixb    obs fixture-frame fb, actions = Fix B (v_par, delta_n)
  r3_surface_fixb  obs fb in the local surface frame [R^T(p - x_s), R^T R_fb],
                   actions = Fix B; curvature obs (robot0_mold_curv, run 3 config only)

Fix B labels are packed into the existing action21 slots:
  ts_pose_command_0        = [v_t1, v_t2, delta_n, 1, 0, 0, 0]   (m/s, m/s, m)
  ts_pose_virtual_target_0 = [0, 0, 0, 1, 0, 0, 0]
  stiffness/gripper/force unchanged. Heading is not a label: R_tool = R(x_s) R_off.

Per-episode arrays written to every variant (robot rate, from the measured fb):
  mold_x_s_0 (N,3), mold_R_field_0 (N,9, row-major, columns [t1 t2 n]),
  mold_R_field_rot6d_0 (N,6), mold_normal_0 (N,3), mold_signed_distance_0 (N,1),
  mold_curvature_0 (N,4) [k1, k2, a cos2th, a sin2th], fixb_v_par_0 (N,2),
  fixb_delta_n_0 (N,1), fixb_dt_next_0 (N,1); attrs mold_id, mesh_version,
  field_version, fixb label conventions.

Unchanged arrays are hardlinked from the source zarr (zarr writes via
temp-file + rename, so writes never reach the source).

Usage:
  python fixb_build_datasets.py \
    --out_root /mnt/data/umidata/ft/mold_aware_datasets/stringer_data_v2_full_fixb
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
    Rp_to_pose7,
    episode_arrays,
    fit_heading_offset,
    fixb_pose7_command,
    identity_pose7,
    pose7_to_Rp,
)

VARIANTS = ("r1_world_world", "r2_world_fixb", "r3_surface_fixb")
MOLD_KEYS = (
    "mold_x_s_0",
    "mold_R_field_0",
    "mold_R_field_rot6d_0",
    "mold_normal_0",
    "mold_signed_distance_0",
    "mold_curvature_0",
    "fixb_v_par_0",
    "fixb_delta_n_0",
    "fixb_dt_next_0",
)
LABEL_VERSION = "fixb_v3"


def episode_names(group) -> list[str]:
    return sorted((k for k in group.keys() if k.startswith("episode_")), key=lambda s: int(s.split("_")[-1]))


def link_tree(src: Path, dst: Path) -> None:
    """Hardlink chunk files, copy metadata files (.zarray/.zattrs/.zgroup)."""
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
    group.create_dataset(key, data=np.asarray(data, dtype=kw.get("dtype")), chunks=data.shape, **kw)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source", type=Path, default=Path("/mnt/data/umidata/ft/datasets/stringer_data_v2_full/acp_replay_buffer.zarr")
    )
    ap.add_argument("--out_root", type=Path, required=True)
    ap.add_argument("--field", type=Path, default=None, help="default <out_root>/shared/fixb_field.npz")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--v_par_mode", choices=["foot", "tool"], default="foot")
    ap.add_argument("--delta_n_source", choices=["fb", "vt"], default="fb")
    ap.add_argument("--smooth_window_s", type=float, default=0.1)
    ap.add_argument("--heading_band_m", type=float, default=0.03, help="|sd| band used to fit R_off")
    ap.add_argument("--max_episodes", type=int, default=None)
    args = ap.parse_args()

    t0 = time.time()
    shared = args.out_root / "shared"
    field_path = args.field or shared / "fixb_field.npz"
    field = FixBMoldField(field_path)
    fmeta = field.meta
    cache_dir = shared / "episode_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    label_cfg = {
        "label_version": LABEL_VERSION,
        "v_par_mode": args.v_par_mode,
        "delta_n_source": args.delta_n_source,
        "smooth_window_s": args.smooth_window_s,
        "mesh_version": fmeta["mesh_version"],
        "field_version": fmeta["field_version"],
    }

    src = zarr.open(str(args.source), "r")
    names = episode_names(src["data"])[: args.max_episodes]

    # Pass 1: mold arrays + labels per episode, heading samples.
    L_samples = []
    arrays = {}
    for i, name in enumerate(names):
        cache = cache_dir / f"{name}.npz"
        if cache.exists() and json.loads(str(np.load(cache)["label_cfg"])) == label_cfg:
            arr = {k: v for k, v in np.load(cache).items() if k != "label_cfg"}
        else:
            arr = episode_arrays(
                field,
                src["data"][name],
                smooth_window_s=args.smooth_window_s,
                v_par_mode=args.v_par_mode,
                delta_n_source=args.delta_n_source,
            )
            np.savez_compressed(cache, label_cfg=json.dumps(label_cfg), **arr)
        R_tool, _ = pose7_to_Rp(src["data"][name]["ts_pose_fb_0"])
        R_f = arr["mold_R_field_0"].astype(np.float64).reshape(-1, 3, 3)
        band = np.abs(arr["mold_signed_distance_0"][:, 0]) < args.heading_band_m
        L_samples.append(np.einsum("nji,njk->nik", R_f[band][::10], R_tool[band][::10]))
        arrays[name] = arr
        print(f"[pass1 {i + 1}/{len(names)}] {name}: N={len(arr['mold_x_s_0'])}")

    R_off, flip, head_err = fit_heading_offset(np.concatenate(L_samples))
    tilt = np.degrees(np.arccos(np.clip(-R_off[2, 2], -1.0, 1.0)))
    meta = {
        **label_cfg,
        "mold_id": fmeta["mold_id"],
        "source_zarr": str(args.source),
        "n_episodes": len(names),
        "R_off": R_off.tolist(),
        "R_off_convention": (
            "R_tool ~= R_field(x_s) @ B, B in {R_off, diag(-1,-1,1) @ R_off} (180 deg about n); "
            "B is chosen from the measured tool at the chunk start"
        ),
        "R_off_tool_z_tilt_from_minus_n_deg": float(tilt),
        "heading_flipped_frac": float(flip.mean()),
        "heading_residual_deg_p50_p90": np.percentile(head_err, [50, 90]).tolist(),
        "action_packing": {
            "ts_pose_command_0": "[v_t1 m/s, v_t2 m/s, delta_n m, quat wxyz identity]",
            "ts_pose_virtual_target_0": "identity (unused)",
        },
        "decode": "x_s0 = foot(p_meas); x_s+ = foot(x_s + R[:, :2] v dt); x_d = x_s + delta_n n; R_d = R(x_s) B",
    }
    (shared / "fixb_meta.json").write_text(json.dumps(meta, indent=2))
    print(
        f"[heading] R_off tool-z tilt {tilt:.1f} deg, flipped {flip.mean():.2f}; "
        f"residual p50/p90 {np.percentile(head_err, [50, 90]).round(1)} deg"
    )

    # Pass 2: materialise variants.
    for variant in args.variants:
        zpath = args.out_root / variant / "acp_replay_buffer.zarr"
        print(f"[{variant}] linking {args.source} -> {zpath}")
        link_tree(args.source, zpath)
        root = zarr.open(str(zpath), "a")
        root.attrs.update(
            {
                "mold_aware_variant": variant,
                "mold_id": fmeta["mold_id"],
                "mesh_version": fmeta["mesh_version"],
                "field_version": fmeta["field_version"],
                "fixb": meta,
                "world_frame": "fixture",
            }
        )
        for name in names:
            ep = root["data"][name]
            arr = arrays[name]
            for key in MOLD_KEYS:
                replace_array(ep, key, arr[key])
            ep.attrs.update(
                {
                    "mold_id": fmeta["mold_id"],
                    "mesh_version": fmeta["mesh_version"],
                    "field_version": fmeta["field_version"],
                    "label_version": LABEL_VERSION,
                    "pose_variant": variant,
                    "world_frame": "fixture",
                }
            )
            if variant == "r1_world_world":
                continue
            cmd_like = ep["ts_pose_command_0"]
            replace_array(
                ep, "ts_pose_command_0", fixb_pose7_command(arr["fixb_v_par_0"], arr["fixb_delta_n_0"]), cmd_like
            )
            vt_like = ep["ts_pose_virtual_target_0"]
            replace_array(ep, "ts_pose_virtual_target_0", identity_pose7(len(arr["fixb_v_par_0"])), vt_like)
            if variant == "r3_surface_fixb":
                R_tool, p = pose7_to_Rp(ep["ts_pose_fb_0"])
                R_f = arr["mold_R_field_0"].astype(np.float64).reshape(-1, 3, 3)
                p_loc = np.einsum("nji,nj->ni", R_f, p - arr["mold_x_s_0"].astype(np.float64))
                R_loc = np.einsum("nji,njk->nik", R_f, R_tool)
                replace_array(ep, "ts_pose_fb_0", Rp_to_pose7(R_loc, p_loc), ep["ts_pose_fb_0"])
        print(f"[{variant}] wrote {len(names)} episodes")

    print(f"done in {time.time() - t0:.0f}s -> {args.out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

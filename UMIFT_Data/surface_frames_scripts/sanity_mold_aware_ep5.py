#!/usr/bin/env python3
"""≤10 min sanity: ep5 through live obs + decode path offline.

Checks:
  1) mold obs (sd, normal, rot6d) from MoldAwareRuntime vs dataset episode_5
  2) r2 anchored decode of exported policy actions vs offline eval decode

Uses training stringer FixB field. Exit non-zero on failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import zarr

from fixb_field import FixBMoldField, integrate_fixb
from fixb_labels import heading_branch, pose7_to_Rp, so3_to_rot6d_rows
from fixb_open_loop_metrics import decode as offline_decode
from mold_aware_runtime import MoldAwareRuntime

ROOT = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_scanner_v2_fixb")
EXPORT = Path(
    "/mnt/data/umidata/ft/training_outputs/scanner_fixb/eval/exports/"
    "r2_world_fixb/epoch0035/episode_5.npz"
)
EP = "episode_5"
TOL_OBS = 2e-3  # meters / unitless — live tracked foot vs stored (allow small track drift)
TOL_DECODE = 1e-3  # meters vs offline anchored decode on same actions


def main() -> int:
    field = FixBMoldField(ROOT / "shared" / "fixb_field.npz")
    meta = json.loads((ROOT / "shared" / "fixb_meta.json").read_text())
    R_off = np.asarray(meta["R_off"], dtype=np.float64)
    g = zarr.open(ROOT / "r2_world_fixb" / "acp_replay_buffer.zarr", "r")["data"][EP]

    # --- 1) obs match ---
    rt = MoldAwareRuntime.from_assets("r2_world_fixb", dry_run=False)
    obs_raw = {"ts_pose_fb_0": np.asarray(g["ts_pose_fb_0"])}
    live = rt.fill_obs_raw(obs_raw, 0)

    sd_ds = np.asarray(g["mold_signed_distance_0"], dtype=np.float64).reshape(-1)
    n_ds = np.asarray(g["mold_normal_0"], dtype=np.float64).reshape(-1, 3)
    r6_ds = np.asarray(g["mold_R_field_rot6d_0"], dtype=np.float64).reshape(-1, 6)
    geom_ds = np.concatenate(
        [sd_ds.reshape(-1, 1), n_ds, r6_ds], axis=-1
    )  # same packing as umift_type_conversions → robot0_mold_geom (pre-normalizer)

    # Compare on a mid-episode window (avoid start transients of tracked foot)
    sl = slice(200, 800)
    sd_err = float(np.max(np.abs(live["sd"][sl] - sd_ds[sl])))
    n_err = float(np.max(np.linalg.norm(live["normal"][sl] - n_ds[sl], axis=1)))
    r6_err = float(np.max(np.linalg.norm(obs_raw["mold_R_field_rot6d_0"][sl] - r6_ds[sl], axis=1)))
    geom_live = np.concatenate(
        [
            np.asarray(obs_raw["mold_signed_distance_0"], dtype=np.float64).reshape(-1, 1),
            np.asarray(obs_raw["mold_normal_0"], dtype=np.float64).reshape(-1, 3),
            np.asarray(obs_raw["mold_R_field_rot6d_0"], dtype=np.float64).reshape(-1, 6),
        ],
        axis=-1,
    )
    geom_err = float(np.max(np.abs(geom_live[sl] - geom_ds[sl])))
    # Also vs independent field.query on dataset feet (tighter reference)
    x_ds = field.project(pose7_to_Rp(g["ts_pose_fb_0"])[1])
    q_ref = field.query(x_ds)
    # live uses tool sd; dataset stores tool sd — compare to recompute from same poses
    R_fb, p_fb = pose7_to_Rp(g["ts_pose_fb_0"])
    x_live = live["x_s"]
    sd_re = np.sum((p_fb - x_live) * live["normal"], axis=1)
    sd_ds_re = np.sum((p_fb - x_ds) * q_ref["R"][..., 2], axis=1)

    print(f"[obs] max|sd_live - sd_dataset|[{sl.start}:{sl.stop}] = {sd_err:.4e} m")
    print(f"[obs] max|n_live - n_dataset| = {n_err:.4e}")
    print(f"[obs] max|r6_live - r6_dataset| = {r6_err:.4e}")
    print(f"[obs] max|mold_geom pack live - dataset| = {geom_err:.4e}")
    print(f"[obs] max|sd_recompute_live - sd_recompute_closest| = {float(np.max(np.abs(sd_re[sl]-sd_ds_re[sl]))):.4e}")

    obs_ok = sd_err < TOL_OBS and n_err < 5e-2 and r6_err < 5e-2 and geom_err < TOL_OBS
    # Note: tracked foot can differ from independent closest used in the zarr; report both.
    if not obs_ok:
        print("[obs] FAIL vs stored dataset mold_* (tracked foot may differ).")
        print("[obs] Checking recompute-from-closest consistency on stored x_s if present…")
        if "mold_x_s_0" in g:
            x_stored = np.asarray(g["mold_x_s_0"], dtype=np.float64)
            q_s = field.query(x_stored)
            sd_s = np.sum((p_fb - x_stored) * q_s["R"][..., 2], axis=1)
            print(f"  max|sd_dataset - sd(from stored x_s)| = {float(np.max(np.abs(sd_ds-sd_s))):.4e}")

    # --- 2) r2 decode match offline eval (no live safety clamps) ---
    d = np.load(EXPORT)
    # one query near the middle
    qi = min(100, d["raw_action"].shape[0] - 1)
    action = d["raw_action"][qi : qi + 1]
    times = d["action_target_time_s"][qi : qi + 1]
    q = int(d["query_robot_idx"][qi])
    pos_off, R_off_dec = offline_decode(
        "r2_world_fixb", action, times, p_fb[q : q + 1], R_fb[q : q + 1], field, R_off
    )

    dt = np.diff(times, axis=1)
    dt = np.concatenate([dt, dt[:, -1:]], axis=1)
    live_dec = rt.decode_chunk(
        action[0], p_fb[q], R_fb[q], dt[0], apply_safety=False
    )
    pos_err = float(np.max(np.linalg.norm(live_dec["x_d"] - pos_off[0], axis=1)))
    # rotation angle
    rel = np.einsum("...ji,...jk->...ik", live_dec["R_d"], R_off_dec[0])
    from scipy.spatial.transform import Rotation

    ang = np.degrees(np.linalg.norm(Rotation.from_matrix(rel).as_rotvec(), axis=-1))
    rot_err = float(np.max(ang))
    print(f"[decode] query {qi}: max|x_d live - offline| = {pos_err:.4e} m")
    print(f"[decode] query {qi}: max angle(R_d) = {rot_err:.4e} deg")

    decode_ok = pos_err < TOL_DECODE and rot_err < 0.5
    if not decode_ok:
        print("[decode] FAIL")
        return 1
    if not obs_ok:
        print("[obs] FAIL (hard gate)")
        return 1

    print("SANITY PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

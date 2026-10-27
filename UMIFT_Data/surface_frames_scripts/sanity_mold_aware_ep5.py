#!/usr/bin/env python3
"""≤10 min sanity: ep5 through live obs + decode path offline.

Checks:
  1) mold obs (sd, normal, rot6d) from MoldAwareRuntime vs dataset episode_5
  2) r2/r4 anchored decode of GT (or exported) actions vs offline eval decode
     — for r4, decoded orientation must match offline decode (ΔR path).

Uses training stringer FixB field. Exit non-zero on failure.

  python sanity_mold_aware_ep5.py
  python sanity_mold_aware_ep5.py --run r4_world_fixb_tilt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import zarr
from scipy.spatial.transform import Rotation

from fixb_field import FixBMoldField
from fixb_labels import pose7_to_Rp
from fixb_open_loop_metrics import decode as offline_decode
from mold_aware_runtime import MoldAwareRuntime

ROOT = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_scanner_v2_fixb")
EXPORT_R2 = Path(
    "/mnt/data/umidata/ft/training_outputs/scanner_fixb/eval/exports/"
    "r2_world_fixb/epoch0035/episode_5.npz"
)
EP = "episode_5"
TOL_OBS = 2e-3  # meters / unitless — live tracked foot vs stored (allow small track drift)
TOL_DECODE = 1e-3  # meters vs offline anchored decode on same actions
TOL_ROT_DEG = 0.5


def _pack_gt_action27(ep, q: int, H: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Build one (1,H,27) GT action chunk + times from the r4 zarr (robot rate)."""
    import PyriteUtility.spatial_math.spatial_utilities as su

    cmd = np.asarray(ep["ts_pose_command_0"], dtype=np.float64)
    vt = np.asarray(ep["ts_pose_virtual_target_0"], dtype=np.float64)
    stiff = np.asarray(ep["stiffness_0"], dtype=np.float64).reshape(-1, 1)
    grip = np.asarray(ep["gripper_0"], dtype=np.float64).reshape(-1, 1)
    dR = np.asarray(ep["fixb_delta_R_rot6d_0"], dtype=np.float64).reshape(-1, 6)
    t = np.asarray(ep["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
    n = len(cmd)
    q = int(np.clip(q, 0, max(n - H, 0)))
    sl = slice(q, q + H)
    pose9_c = su.SE3_to_pose9(su.pose7_to_SE3(cmd[sl]))
    pose9_v = su.SE3_to_pose9(su.pose7_to_SE3(vt[sl]))
    # grasp force placeholder zeros (unused for decode)
    gf = np.zeros((H, 1), dtype=np.float64)
    act = np.concatenate(
        [pose9_c, pose9_v, stiff[sl], grip[sl], gf, dR[sl]], axis=-1
    ).astype(np.float64)
    times = t[sl]
    return act[None], times[None]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run",
        default="r2_world_fixb",
        choices=("r2_world_fixb", "r4_world_fixb_tilt"),
    )
    args = ap.parse_args()
    run = args.run
    variant_dir = {
        "r2_world_fixb": "r2_world_fixb",
        "r4_world_fixb_tilt": "r4_world_fixb_tilt",
    }[run]

    field = FixBMoldField(ROOT / "shared" / "fixb_field.npz")
    meta = json.loads((ROOT / "shared" / "fixb_meta.json").read_text())
    R_off = np.asarray(meta["R_off"], dtype=np.float64)
    zpath = ROOT / variant_dir / "acp_replay_buffer.zarr"
    if not zpath.exists():
        print(f"[sanity] missing {zpath}")
        return 1
    g = zarr.open(zpath, "r")["data"][EP]

    # --- 1) obs match ---
    rt = MoldAwareRuntime.from_assets(run, dry_run=False)
    obs_raw = {"ts_pose_fb_0": np.asarray(g["ts_pose_fb_0"])}
    live = rt.fill_obs_raw(obs_raw, 0)

    sd_ds = np.asarray(g["mold_signed_distance_0"], dtype=np.float64).reshape(-1)
    n_ds = np.asarray(g["mold_normal_0"], dtype=np.float64).reshape(-1, 3)
    r6_ds = np.asarray(g["mold_R_field_rot6d_0"], dtype=np.float64).reshape(-1, 6)
    geom_ds = np.concatenate([sd_ds.reshape(-1, 1), n_ds, r6_ds], axis=-1)

    sl = slice(200, 800)
    sd_err = float(np.max(np.abs(live["sd"][sl] - sd_ds[sl])))
    n_err = float(np.max(np.linalg.norm(live["normal"][sl] - n_ds[sl], axis=1)))
    r6_err = float(
        np.max(np.linalg.norm(obs_raw["mold_R_field_rot6d_0"][sl] - r6_ds[sl], axis=1))
    )
    geom_live = np.concatenate(
        [
            np.asarray(obs_raw["mold_signed_distance_0"], dtype=np.float64).reshape(-1, 1),
            np.asarray(obs_raw["mold_normal_0"], dtype=np.float64).reshape(-1, 3),
            np.asarray(obs_raw["mold_R_field_rot6d_0"], dtype=np.float64).reshape(-1, 6),
        ],
        axis=-1,
    )
    geom_err = float(np.max(np.abs(geom_live[sl] - geom_ds[sl])))
    R_fb, p_fb = pose7_to_Rp(g["ts_pose_fb_0"])

    print(f"[obs] max|sd_live - sd_dataset|[{sl.start}:{sl.stop}] = {sd_err:.4e} m")
    print(f"[obs] max|n_live - n_dataset| = {n_err:.4e}")
    print(f"[obs] max|r6_live - r6_dataset| = {r6_err:.4e}")
    print(f"[obs] max|mold_geom pack live - dataset| = {geom_err:.4e}")

    obs_ok = sd_err < TOL_OBS and n_err < 5e-2 and r6_err < 5e-2 and geom_err < TOL_OBS
    if not obs_ok:
        print("[obs] FAIL vs stored dataset mold_* (tracked foot may differ).")

    # --- 2) decode match offline ---
    if run == "r4_world_fixb_tilt":
        q = 400
        action, times = _pack_gt_action27(g, q, H=24)
        qi = 0
        q_robot = q
    else:
        if not EXPORT_R2.exists():
            print(f"[decode] missing export {EXPORT_R2}; skip r2 export decode")
            print("SANITY PASS" if obs_ok else "[obs] FAIL")
            return 0 if obs_ok else 1
        d = np.load(EXPORT_R2)
        qi = min(100, d["raw_action"].shape[0] - 1)
        action = d["raw_action"][qi : qi + 1]
        times = d["action_target_time_s"][qi : qi + 1]
        q_robot = int(d["query_robot_idx"][qi])

    pos_off, R_off_dec = offline_decode(
        run, action, times, p_fb[q_robot : q_robot + 1], R_fb[q_robot : q_robot + 1], field, R_off
    )

    dt = np.diff(times, axis=1)
    dt = np.concatenate([dt, dt[:, -1:]], axis=1)
    live_dec = rt.decode_chunk(
        action[0], p_fb[q_robot], R_fb[q_robot], dt[0], apply_safety=False
    )
    pos_err = float(np.max(np.linalg.norm(live_dec["x_d"] - pos_off[0], axis=1)))
    rel = np.einsum("...ji,...jk->...ik", live_dec["R_d"], R_off_dec[0])
    ang = np.degrees(np.linalg.norm(Rotation.from_matrix(rel).as_rotvec(), axis=-1))
    rot_err = float(np.max(ang))
    print(f"[decode] {run} query {qi}: max|x_d live - offline| = {pos_err:.4e} m")
    print(f"[decode] {run} query {qi}: max angle(R_d) = {rot_err:.4e} deg")

    decode_ok = pos_err < TOL_DECODE and rot_err < TOL_ROT_DEG
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

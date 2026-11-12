#!/usr/bin/env python3
"""Relabel October stringer_v3.5 from the odom-corrected TCP sidecars.

Writes a new dataset root. Does not open the source zarr or the existing
r1 / r4 trees for writing.

  r4     tracked foot, v1/v2, mold_signed_distance, delta_n from the corrected TCP
  r4vt   same foot and v1/v2; delta_n = (p_vt - x_s) · n
  r1rel  same corrected TCP as r4, with the source virtual-target offset
         carried in the tool frame

The source virtual target is the tool-frame offset of ts_pose_virtual_target_0
from the old TCP. It is applied to the corrected TCP. Subtracting the old
world-frame VT from the new foot would put the press target on the teleported branch.

Recording clock is seconds after the first live frame (zarr time minus the 5 s pad).
Bad spans, saved for a later filter and not applied here:
  episode 47, 31–47 s
  episode 38, t >= 50 s
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import zarr
from scipy.spatial.transform import Rotation, Slerp

from fixb_build_datasets import MOLD_KEYS, link_tree, replace_array
from fixb_field import FixBMoldField
from fixb_labels import (
    delta_R_from_field_tool,
    fixb_pose7_command,
    identity_pose7,
    pose7_to_Rp,
    so3_to_rot6d_rows,
)
from fixb_labels_tracked import episode_arrays_tracked

SRC = Path("/mnt/data/umidata/ft/datasets/stringer_v3.5/acp_replay_buffer.zarr")
OLD_ROOT = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb")
OLD_R1 = OLD_ROOT / "r1_world_world" / "acp_replay_buffer.zarr"
OLD_R4 = OLD_ROOT / "r4_world_fixb_tilt" / "acp_replay_buffer.zarr"
FIELD_NPZ = OLD_ROOT / "shared" / "fixb_field.npz"
TAG_DIR = Path("/mnt/data/umidata/ft/datasets/stringer_v3.5/tag_traj")
CORR_DIR = Path("/mnt/data/umidata/ft/datasets/stringer_v3.5/tag_traj_odom_corr")
OUT = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr")
N_OCT = 50
PAD_S = 5.0
SMOOTH_S = 0.1
CONTACT_N = 2.0
# Recording seconds. Open end means through the last robot sample.
BAD_SPANS = {38: [(50.0, None)], 47: [(31.0, 47.0)]}


def recording_names() -> list[str]:
    return [p.name.replace("_ee_master.npz", "") for p in sorted(TAG_DIR.glob("*_ee_master.npz"))]


def resample_pose7(st: np.ndarray, pose7: np.ndarray, robot_t: np.ndarray) -> np.ndarray:
    """Slerp the sidecar pose onto zarr robot times. Sidecar t=0 is zarr t=PAD_S."""
    st = np.asarray(st, dtype=np.float64).reshape(-1)
    pose7 = np.asarray(pose7, dtype=np.float64).reshape(-1, 7)
    keep = np.ones(len(st), dtype=bool)
    keep[1:] = np.diff(st) > 1e-9
    st, pose7 = st[keep], pose7[keep]
    tq = np.clip(np.asarray(robot_t, dtype=np.float64).reshape(-1) - PAD_S, float(st[0]), float(st[-1]))
    rot = Slerp(st, Rotation.from_quat(pose7[:, [4, 5, 6, 3]]))
    R = rot(tq).as_matrix()
    p = np.column_stack([np.interp(tq, st, pose7[:, i]) for i in range(3)])
    q = Rotation.from_matrix(R).as_quat()
    out = np.empty((len(tq), 7), dtype=np.float64)
    out[:, :3] = p
    out[:, 3] = q[:, 3]
    out[:, 4:] = q[:, :3]
    return out


def transport_vt(fb_old: np.ndarray, vt_old: np.ndarray, fb_new: np.ndarray) -> np.ndarray:
    """Keep the source tool-frame VT offset, expressed on the corrected TCP."""
    R_old, p_old = pose7_to_Rp(fb_old)
    _, p_vt = pose7_to_Rp(vt_old)
    R_new, p_new = pose7_to_Rp(fb_new)
    offset = np.einsum("nji,nj->ni", R_old, p_vt - p_old)
    p = p_new + np.einsum("nij,nj->ni", R_new, offset)
    q = Rotation.from_matrix(R_new).as_quat()
    out = np.empty((len(p), 7), dtype=np.float64)
    out[:, :3] = p
    out[:, 3] = q[:, 3]
    out[:, 4:] = q[:, :3]
    return out


def bad_mask(robot_t: np.ndarray, spans: list[tuple[float, float | None]]) -> np.ndarray:
    rel = np.asarray(robot_t, dtype=np.float64).reshape(-1) - PAD_S
    mask = np.zeros(len(rel), dtype=np.uint8)
    for lo, hi in spans:
        hit = rel >= lo if hi is None else (rel >= lo) & (rel <= hi)
        mask[hit] = 1
    return mask


def contact_mask(ep, robot_t: np.ndarray) -> np.ndarray:
    wt = np.asarray(ep["wrench_time_stamps_0"], dtype=np.float64).reshape(-1)
    force = np.linalg.norm(np.asarray(ep["wrench_left_0"], dtype=np.float64)[:, :3], axis=1)
    order = np.argsort(wt, kind="mergesort")
    wt, force = wt[order], force[order]
    keep = np.ones(len(wt), dtype=bool)
    keep[1:] = np.diff(wt) > 1e-9
    return np.interp(robot_t, wt[keep], force[keep]) > CONTACT_N


def median_or_none(values: np.ndarray, mask: np.ndarray) -> float | None:
    if not np.any(mask):
        return None
    return float(np.median(values[mask]))


def label_stats(sd, v_par, x_s, n, p, dn_vt, contact) -> dict:
    sd = np.asarray(sd, dtype=np.float64).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1, 3)
    x_s = np.asarray(x_s, dtype=np.float64).reshape(-1, 3)
    n = np.asarray(n, dtype=np.float64).reshape(-1, 3)
    raw = np.sum((p - x_s) * n, axis=1)
    ip = np.sqrt(np.maximum(np.sum((p - x_s) ** 2, axis=1) - raw**2, 0.0))
    speed = np.linalg.norm(np.asarray(v_par, dtype=np.float64).reshape(-1, 2), axis=1)
    dn_vt = np.asarray(dn_vt, dtype=np.float64).reshape(-1)
    return {
        "n": int(len(sd)),
        "n_contact": int(np.sum(contact)),
        "sd_contact_median_mm": None if median_or_none(sd, contact) is None else round(median_or_none(sd, contact) * 1e3, 2),
        "dn_vt_contact_median_mm": None if median_or_none(dn_vt, contact) is None else round(median_or_none(dn_vt, contact) * 1e3, 2),
        "sd_min_mm": round(float(sd.min()) * 1e3, 2),
        "sd_max_mm": round(float(sd.max()) * 1e3, 2),
        "sd_abs_p50_mm": round(float(np.median(np.abs(sd))) * 1e3, 2),
        "in_plane_p50_mm": round(float(np.median(ip)) * 1e3, 2),
        "v_abs_p50_mm_s": round(float(np.median(speed)) * 1e3, 2),
        "frac_abs_sd_gt_5cm": round(float(np.mean(np.abs(sd) > 0.05)), 4),
    }


def episode_names(group) -> list[str]:
    return sorted(
        (k for k in group.keys() if str(k).startswith("episode_")),
        key=lambda s: int(str(s).split("_")[-1]),
    )


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    names = recording_names()
    if not names[49].startswith("2026-10-") or names[50].startswith("2026-10-"):
        raise SystemExit(f"October cutoff is not episode 49: {names[49]} / {names[50]}")
    src = zarr.open(str(SRC), "r")
    old_r4 = zarr.open(str(OLD_R4), "r")
    old_r1 = zarr.open(str(OLD_R1), "r")
    src_names = episode_names(src["data"])
    if episode_names(old_r4["data"]) != src_names[:N_OCT] or episode_names(old_r1["data"]) != src_names[:N_OCT]:
        raise SystemExit("r1/r4 episode lists do not match the first 50 source episodes")

    field = FixBMoldField(FIELD_NPZ)
    t0 = time.time()
    rows = []
    built = []
    for i, name in enumerate(names[:N_OCT]):
        ep_name = f"episode_{i}"
        src_ep = src["data"][ep_name]
        old_ep = old_r4["data"][ep_name]
        robot_t = np.asarray(src_ep["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
        rgb_t = np.asarray(src_ep["rgb_time_stamps_0"], dtype=np.float64).reshape(-1)
        fb_old = np.asarray(src_ep["ts_pose_fb_0"], dtype=np.float64)
        vt_old = np.asarray(src_ep["ts_pose_virtual_target_0"], dtype=np.float64)
        if not np.allclose(fb_old, np.asarray(old_r1["data"][ep_name]["ts_pose_fb_0"])):
            raise SystemExit(f"{ep_name} r1 pose is not the source pose")
        side = np.load(CORR_DIR / f"{name}_odom_corr.npz")
        fb_new = resample_pose7(side["timestamp_sec_unified"], side["pose7"], robot_t)
        vt_new = transport_vt(fb_old, vt_old, fb_new)
        view = {
            "robot_time_stamps_0": robot_t,
            "rgb_time_stamps_0": rgb_t,
            "ts_pose_fb_0": fb_new,
        }
        arr = episode_arrays_tracked(
            field, view, smooth_window_s=SMOOTH_S, v_par_mode="foot", delta_n_source="fb"
        )
        x_s = np.asarray(arr["mold_x_s_0"], dtype=np.float64)
        n = np.asarray(arr["mold_normal_0"], dtype=np.float64)
        _, p_vt = pose7_to_Rp(vt_new)
        dn_vt = np.sum((p_vt - x_s) * n, axis=1).astype(np.float32)
        R_tool, p_new = pose7_to_Rp(fb_new)
        R_f = np.asarray(arr["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3)
        dR = so3_to_rot6d_rows(delta_R_from_field_tool(R_f, R_tool)).astype(np.float32)
        mask = bad_mask(robot_t, BAD_SPANS.get(i, []))
        contact = contact_mask(src_ep, robot_t)
        x_old = np.asarray(old_ep["mold_x_s_0"], dtype=np.float64)
        n_old = np.asarray(old_ep["mold_normal_0"], dtype=np.float64)
        _, p_vt_old = pose7_to_Rp(vt_old)
        before = label_stats(
            old_ep["mold_signed_distance_0"],
            old_ep["fixb_v_par_0"],
            x_old,
            n_old,
            fb_old[:, :3],
            np.sum((p_vt_old - x_old) * n_old, axis=1),
            contact,
        )
        after = label_stats(
            arr["mold_signed_distance_0"],
            arr["fixb_v_par_0"],
            x_s,
            n,
            p_new,
            dn_vt,
            contact,
        )
        built.append(
            {
                "fb": fb_new.astype(np.float32),
                "vt": vt_new.astype(np.float32),
                "arr": arr,
                "dn_vt": dn_vt.reshape(-1, 1),
                "dR": dR,
                "bad": mask,
                "cmd_r4": fixb_pose7_command(arr["fixb_v_par_0"], arr["fixb_delta_n_0"]).astype(np.float32),
                "cmd_vt": fixb_pose7_command(arr["fixb_v_par_0"], dn_vt).astype(np.float32),
            }
        )
        rows.append(
            {
                "episode": i,
                "name": name,
                "before": before,
                "after": after,
                "n_bad": int(mask.sum()),
                "pose_shift_p50_mm": round(float(np.median(np.linalg.norm(p_new - fb_old[:, :3], axis=1))) * 1e3, 2),
                "pose_shift_max_mm": round(float(np.max(np.linalg.norm(p_new - fb_old[:, :3], axis=1))) * 1e3, 2),
            }
        )
        print(
            f"[{i:02d}] sd {before['sd_contact_median_mm']} -> {after['sd_contact_median_mm']} mm  "
            f"vt {before['dn_vt_contact_median_mm']} -> {after['dn_vt_contact_median_mm']} mm  "
            f"shift p50 {rows[-1]['pose_shift_p50_mm']} mm  ({time.time() - t0:.0f}s)",
            flush=True,
        )

    print("[write] linking r1 and r4 into the new root", flush=True)
    (OUT / "shared").mkdir(parents=True)
    os.link(FIELD_NPZ, OUT / "shared" / "fixb_field.npz")
    link_tree(OLD_R1, OUT / "r1rel_world" / "acp_replay_buffer.zarr")
    link_tree(OLD_R4, OUT / "r4_world_fixb_tilt" / "acp_replay_buffer.zarr")
    link_tree(OLD_R4, OUT / "r4vt_world_fixb_tilt" / "acp_replay_buffer.zarr")
    trees = {
        "r1rel_world": zarr.open(str(OUT / "r1rel_world" / "acp_replay_buffer.zarr"), "a"),
        "r4_world_fixb_tilt": zarr.open(str(OUT / "r4_world_fixb_tilt" / "acp_replay_buffer.zarr"), "a"),
        "r4vt_world_fixb_tilt": zarr.open(str(OUT / "r4vt_world_fixb_tilt" / "acp_replay_buffer.zarr"), "a"),
    }
    for variant, root in trees.items():
        root.attrs["mold_aware_variant"] = variant
        root.attrs["pose_source"] = "tag_traj_odom_corr"
        root.attrs["october_episodes_only"] = N_OCT
        root.attrs["bad_spans_recording_s"] = {
            str(k): [[lo, hi] for lo, hi in spans] for k, spans in BAD_SPANS.items()
        }
    for i, pack in enumerate(built):
        ep_name = f"episode_{i}"
        rel = trees["r1rel_world"]["data"][ep_name]
        r4 = trees["r4_world_fixb_tilt"]["data"][ep_name]
        vt = trees["r4vt_world_fixb_tilt"]["data"][ep_name]
        for ep, pose_cmd, dn in (
            (rel, pack["fb"], pack["arr"]["fixb_delta_n_0"]),
            (r4, pack["cmd_r4"], pack["arr"]["fixb_delta_n_0"]),
            (vt, pack["cmd_vt"], pack["dn_vt"]),
        ):
            replace_array(ep, "ts_pose_fb_0", pack["fb"], ep["ts_pose_fb_0"])
            replace_array(ep, "ts_pose_command_0", pose_cmd, ep["ts_pose_command_0"])
            for key in MOLD_KEYS:
                value = pack["dn_vt"] if ep is vt and key == "fixb_delta_n_0" else pack["arr"][key]
                replace_array(ep, key, value, ep[key] if key in ep else None)
            replace_array(ep, "bad_span_0", pack["bad"].reshape(-1, 1))
            ep.attrs["pose_source"] = "tag_traj_odom_corr"
            ep.attrs["bad_spans_recording_s"] = [[lo, hi] for lo, hi in BAD_SPANS.get(i, [])]
        replace_array(rel, "ts_pose_virtual_target_0", pack["vt"], rel["ts_pose_virtual_target_0"])
        for ep in (r4, vt):
            replace_array(ep, "fixb_delta_R_rot6d_0", pack["dR"], ep["fixb_delta_R_rot6d_0"])
            replace_array(ep, "ts_pose_virtual_target_0", identity_pose7(len(pack["fb"])).astype(np.float32), ep["ts_pose_virtual_target_0"])
        rel.attrs["label_version"] = "odom_corr_tcp"
        r4.attrs["label_version"] = "fixb_v4_tracked_odom_corr"
        vt.attrs["label_version"] = "fixb_v4_tracked_odom_corr_r4vt"
        # r4vt keeps the TCP signed distance; only the action delta_n is the VT depth.
        if not np.allclose(vt["mold_signed_distance_0"], r4["mold_signed_distance_0"]):
            raise SystemExit(f"{ep_name} r4vt signed distance diverged")
        if not np.allclose(vt["fixb_v_par_0"], r4["fixb_v_par_0"]):
            raise SystemExit(f"{ep_name} r4vt v_par diverged")
        if not np.allclose(rel["ts_pose_fb_0"], r4["ts_pose_fb_0"]):
            raise SystemExit(f"{ep_name} r1rel TCP diverged from r4")

    spans = {}
    for ep, intervals in BAD_SPANS.items():
        rel_t = np.asarray(src["data"][f"episode_{ep}"]["robot_time_stamps_0"]).reshape(-1) - PAD_S
        spans[str(ep)] = {
            "name": names[ep],
            "recording_s": [[lo, (float(rel_t[-1]) if hi is None else hi)] for lo, hi in intervals],
            "zarr_s": [[lo + PAD_S, (float(rel_t[-1]) + PAD_S if hi is None else hi + PAD_S)] for lo, hi in intervals],
            "n_samples": rows[ep]["n_bad"],
        }
    meta = {
        "source_zarr": str(SRC),
        "corrected_poses": str(CORR_DIR),
        "episodes": "0-49 October only",
        "clock": "recording seconds after the first live frame; zarr time is recording plus the 5 s pad",
        "virtual_target": "source tool-frame offset applied to the corrected TCP",
        "bad_spans": spans,
        "per_episode": rows,
    }
    (OUT / "shared" / "label_stats.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] {OUT} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

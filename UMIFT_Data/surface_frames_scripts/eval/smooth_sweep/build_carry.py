#!/usr/bin/env python3
"""New causal-foot zarr: chosen smooth field, 30° frame carry, δn at 3000 N/m.

Does not modify r4vt_causal_foot, its normalizer, or fixb_smooth_frames.npz.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import zarr
from scipy.spatial import cKDTree

ROOT = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames")
SCRIPTS = ROOT / "UMIFT_Data/surface_frames_scripts"
sys.path[:0] = [str(SCRIPTS), str(ROOT / "PyriteML"), str(ROOT / "PyriteConfig"), str(ROOT / "PyriteUtility"), str(ROOT)]

from causal_foot import CausalFoot
from fixb_field import FixBMoldField
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from fixb_build_datasets import MOLD_KEYS, link_tree, replace_array
from fixb_labels import delta_R_from_field_tool, fixb_pose7_command, pose7_to_Rp, so3_to_rot6d_rows
from fixb_smooth_frame_labels import EXCLUDE, LONG_DT
from fixb_tcp_track import project_tcp_velocity
from fixb_tcp_track_clean import TRACK, _kept
from heading_tilt import geodesic_deg
from PyriteUtility.data_pipeline.wrench_baseline_utils import time_based_moving_average

DATA = TRACK.parents[1]
SRC = DATA / "r4vt_smooth_tcp_track_clean" / "acp_replay_buffer.zarr"
FIELD = DATA / "shared" / "fixb_field.npz"
OUT = DATA / "r4vt_causal_foot_carry"
NORM = SCRIPTS / "assets" / "oct_v35_h4s_r4vt_action_normalizer_minmax_causal_foot_carry.pkl"
K = 3000.0
CARRY_DEG = 30.0


def _check() -> None:
    eye = np.repeat(np.eye(3)[None], 3, axis=0)
    hopped = eye.copy()
    hopped[1] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    tool = np.repeat(np.eye(3)[None], 3, axis=0)
    held = carry_frames(hopped, tool)
    if not np.allclose(held[1], held[0]):
        raise RuntimeError("a 90° frame hop with a still tool was not carried")
    turned = tool.copy()
    turned[1] = hopped[1]
    kept = carry_frames(hopped, turned)
    if np.allclose(kept[1], kept[0]):
        raise RuntimeError("a hop was carried while the tool turned with it")


def carry_frames(frames: np.ndarray, tool: np.ndarray) -> np.ndarray:
    """Hold the previous frame when it jumps more than 30° and the tool does not."""
    out = np.array(frames, dtype=np.float64, copy=True)
    for i in range(1, len(out)):
        ang = float(geodesic_deg(out[i - 1][None], out[i][None])[0])
        moved = float(geodesic_deg(tool[i - 1][None], tool[i][None])[0])
        if ang > CARRY_DEG and moved < CARRY_DEG:
            out[i] = out[i - 1]
    return out


def jump_counts(frames: np.ndarray, robot_t: np.ndarray) -> tuple[int, int, int]:
    if len(frames) < 2:
        return 0, 0, 0
    ang = geodesic_deg(frames[:-1], frames[1:])
    ok = np.diff(robot_t) < 0.05
    return int(np.sum((ang > 20.0) & ok)), int(np.sum((ang > 30.0) & ok)), int(ok.sum())


def smooth_force(group, robot_t):
    wrench_t = np.asarray(group["wrench_time_stamps_left_0"]).reshape(-1)
    wrench = np.asarray(group["wrench_left_0"], dtype=np.float64)
    smooth = time_based_moving_average(wrench, wrench_t, 1.0)
    j = np.clip(np.searchsorted(wrench_t, robot_t), 1, len(wrench_t) - 1)
    left = np.abs(robot_t - wrench_t[j - 1]) <= np.abs(robot_t - wrench_t[j])
    return smooth[np.where(left, j - 1, j), :3]


def fit(rows: np.ndarray) -> str:
    if NORM.exists() and NORM.stat().st_size > 0:
        raise SystemExit(f"refusing to overwrite {NORM}")
    lo = rows.min(0).astype(np.float32)
    hi = rows.max(0).astype(np.float32)
    span = np.maximum(hi - lo, np.float32(1e-7))
    scale = (np.float32(2.0) / span).astype(np.float32)
    offset = (np.float32(-1.0) - scale * lo).astype(np.float32)
    stats = {
        "min": lo,
        "max": hi,
        "mean": rows.mean(0).astype(np.float32),
        "std": rows.std(0).astype(np.float32),
    }
    lin = LinearNormalizer()
    lin["action"] = SingleFieldLinearNormalizer.create_manual(scale, offset, stats)
    NORM.parent.mkdir(parents=True, exist_ok=True)
    with NORM.open("wb") as handle:
        pickle.dump(lin, handle)
    digest = hashlib.sha256()
    with NORM.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    _check()
    if len(sys.argv) != 2:
        raise SystemExit("usage: build_carry.py FRAMES.npz")
    frames_path = Path(sys.argv[1])
    out_zarr = OUT / "acp_replay_buffer.zarr"
    if out_zarr.exists() or NORM.exists():
        raise SystemExit(f"refusing to overwrite {out_zarr} or {NORM}")
    if not frames_path.is_file():
        raise SystemExit(f"missing {frames_path}")
    long_blob = np.load(LONG_DT)
    long_idx: dict[int, set[int]] = {}
    for ep, idx in zip(long_blob["flagged_episode"], long_blob["flagged_index"]):
        long_idx.setdefault(int(ep), set()).add(int(idx))
    tracker = CausalFoot(FIELD, frames_path)
    link_tree(SRC, out_zarr)
    root = zarr.open(str(out_zarr), "a")
    root.attrs["label_version"] = "causal_foot_carry_v1"
    root.attrs["smooth_frames"] = str(frames_path)
    root.attrs["delta_n"] = "-(F_smooth · n_out) / 3000, wrench moving average 1 s"
    root.attrs["frame_carry_deg"] = CARRY_DEG
    rows = []
    before = [0, 0, 0]
    after = [0, 0, 0]
    dn_by = {name: [] for name in ("crown", "fillet", "wall", "flange")}
    mesh = FixBMoldField(FIELD)
    apron = cKDTree(mesh.vertices[np.asarray(np.load(FIELD)["is_apron"]).astype(bool)])
    up = np.array([0.0, -1.0, 0.0])
    for ep_id in range(50):
        ep = root["data"][f"episode_{ep_id}"]
        robot_t = np.asarray(ep["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
        rgb_t = np.asarray(ep["rgb_time_stamps_0"], dtype=np.float64).reshape(-1)
        rotation, position = pose7_to_Rp(np.asarray(ep["ts_pose_fb_0"], dtype=np.float64))
        foot, frame = tracker.track(position)
        b20, b30, nstep = jump_counts(frame, robot_t)
        carried = carry_frames(frame, rotation)
        a20, a30, _n = jump_counts(carried, robot_t)
        if ep_id not in EXCLUDE:
            before[0] += b20
            before[1] += b30
            before[2] += nstep
            after[0] += a20
            after[1] += a30
        outward = -carried[:, :, 2]
        force = smooth_force(ep, robot_t)
        fn = np.einsum("ni,ni->n", np.einsum("nij,nj->ni", rotation, force), outward)
        depth = -fn / K
        velocity, dt = project_tcp_velocity(carried, position, robot_t, rgb_t)
        delta_r = so3_to_rot6d_rows(delta_R_from_field_tool(carried, rotation))
        curved = tracker.field.query(foot, with_curvature=True)
        pack = {
            "mold_x_s_0": foot.astype(np.float32),
            "mold_R_field_0": carried.reshape(-1, 9).astype(np.float32),
            "mold_R_field_rot6d_0": so3_to_rot6d_rows(carried).astype(np.float32),
            "mold_normal_0": outward.astype(np.float32),
            "mold_signed_distance_0": np.sum((position - foot) * outward, axis=1).astype(np.float32).reshape(-1, 1),
            "mold_curvature_0": np.asarray(curved["curv"], dtype=np.float32),
            "fixb_v_par_0": velocity.astype(np.float32),
            "fixb_delta_n_0": depth.reshape(-1, 1).astype(np.float32),
            "fixb_dt_next_0": dt.reshape(-1, 1).astype(np.float32),
        }
        for name in MOLD_KEYS:
            replace_array(ep, name, pack[name], ep[name])
        replace_array(ep, "ts_pose_command_0", fixb_pose7_command(velocity, depth).astype(np.float32), ep["ts_pose_command_0"])
        replace_array(ep, "fixb_delta_R_rot6d_0", delta_r.astype(np.float32), ep["fixb_delta_R_rot6d_0"])
        ep.attrs["label_version"] = "causal_foot_carry_v1"
        if ep_id not in EXCLUDE:
            use = _kept(ep, ep_id, long_idx, np.zeros(len(position), dtype=bool))
            if np.any(use):
                stiff = np.asarray(ep["stiffness_0"], dtype=np.float64).reshape(-1)
                rows.append(np.column_stack([
                    velocity[use, 0], velocity[use, 1], depth[use], stiff[use], delta_r[use],
                ]))
            hit = np.linalg.norm(force, axis=1) >= 8.0
            if ep_id in {9, 12, 28, 37, 41}:
                hit = np.zeros(len(hit), dtype=bool)
            if np.any(hit):
                _h, vid, w = mesh.closest(position[hit])
                raw = np.einsum("qk,qkd->qd", w, mesh.normals[vid])
                raw /= np.clip(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12, None)
                tilt = np.degrees(np.arccos(np.clip(raw @ up, -1.0, 1.0)))
                name = np.full(int(hit.sum()), "wall", dtype=object)
                name[tilt < 25.0] = "crown"
                name[(tilt >= 25.0) & (tilt < 55.0)] = "fillet"
                name[apron.query(foot[hit], workers=-1)[0] < 0.015] = "flange"
                mm = depth[hit] * 1e3
                for region in dn_by:
                    sel = name == region
                    if np.any(sel):
                        dn_by[region].append(mm[sel])
        print(f"  ep {ep_id:02d}", flush=True)
    got = np.concatenate(rows)
    sha = fit(got)
    def _dn_stats(parts):
        x = np.concatenate(parts)
        return {
            "n": int(len(x)),
            "median_mm": round(float(np.median(x)), 2),
            "p10_mm": round(float(np.percentile(x, 10)), 2),
            "p90_mm": round(float(np.percentile(x, 90)), 2),
        }

    report = {
        "zarr": str(out_zarr),
        "frames": str(frames_path),
        "normalizer": str(NORM),
        "sha256": sha,
        "kept_rows": int(len(got)),
        "jumps_before_gt20": before[0],
        "jumps_before_gt30": before[1],
        "jumps_after_gt20": after[0],
        "jumps_after_gt30": after[1],
        "steps": before[2],
        "dn_contact_mm": {name: _dn_stats(parts) for name, parts in dn_by.items()},
        "normalizer_min": got.min(0).tolist(),
        "normalizer_max": got.max(0).tolist(),
    }
    (OUT / "carry_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Which stored pose sits on the vertical thin wall. Does not edit the mesh or the viewer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import zarr

ROOT = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames")
SCRIPTS = ROOT / "UMIFT_Data/surface_frames_scripts"
sys.path[:0] = [str(SCRIPTS / "eval" / "frame_flip"), str(SCRIPTS), str(ROOT)]

from fixb_field import FixBMoldField
import register_contact as reg

OUT = Path(__file__).resolve().parent
BASE = Path("/mnt/data/umidata/ft/held_out_mold/thin_stringer/transform_check")
WALL = BASE / "wall_fix/fixb_field_wall_vertical.npz"
REMAP = BASE / "remap_to_training/acp_replay_buffer.zarr"
PUBLISHED = BASE / "as_published/acp_replay_buffer.zarr"
TAG0 = BASE / "thin_tag0/2026-11-19-15.49.52_ee_master.npz"
STAMP = "2026-11-19-15.49.52"


def _check() -> None:
    a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    got = reg.dist_mm(type("F", (), {"query": lambda _s, p: {"x_s": np.zeros_like(p)}})(), a)
    if abs(got[1] - 1000.0) > 1e-6:
        raise RuntimeError("distance unit is not millimetres")


def stats(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if len(x) == 0:
        return {"n": 0}
    return {
        "n": int(len(x)),
        "median_mm": round(float(np.median(x)), 1),
        "p90_mm": round(float(np.percentile(x, 90)), 1),
    }


def pose_xyz(group):
    return np.asarray(group["ts_pose_fb_0"], dtype=np.float64)[:, :3]


def tag_on_robot(robot_t):
    side = np.load(TAG0, allow_pickle=True)
    t = np.asarray(side["timestamp_sec_unified"], dtype=np.float64).reshape(-1)
    p = np.asarray(side["T_master_tcp"], dtype=np.float64)[:, :3, 3]
    j = reg.nearest(t, robot_t)
    return p[j], str(side["world_frame"]), str(side["body_frame"])


def main() -> None:
    _check()
    field = FixBMoldField(WALL)
    ep = zarr.open(REMAP, "r")["data"]["episode_0"]
    robot_t = np.asarray(ep["robot_time_stamps_0"]).reshape(-1)
    rgb_t = np.asarray(ep["rgb_time_stamps_0"]).reshape(-1)
    wrench_t = np.asarray(ep["wrench_time_stamps_left_0"]).reshape(-1)
    force = np.linalg.norm(np.asarray(ep["wrench_left_0"], dtype=np.float64)[:, :3], axis=1)
    t0 = float(rgb_t[0])
    cutoff = t0 + 0.5 * float(rgb_t[-1] - t0)
    contact = force[reg.nearest(wrench_t, robot_t)] >= reg.FORCE_N
    spans = reg.windows(robot_t, rgb_t, t0, cutoff)
    tag, world, body = tag_on_robot(robot_t)
    published = pose_xyz(zarr.open(PUBLISHED, "r")["data"]["episode_0"])
    sources = {
        "remap_training_mount": pose_xyz(ep),
        "as_published": published,
        "thin_tag0": tag,
    }
    info = json.loads(zarr.open(REMAP, "r").attrs["conversion_info"])
    out = {
        "viewer_pose": "remap_to_training ts_pose_fb_0, no extra matrix",
        "viewer_mesh": str(WALL),
        "mesh_frame": "thin fixture. Source field T_fixture_from_scanner is identity. Yaw 5.1, tilt 0, offset 7.5 mm is baked into these vertices.",
        "remap_note": info["transform_check_note"],
        "tag0_world_frame": world,
        "tag0_body_frame": body,
        "force_n": reg.FORCE_N,
        "poses": {},
    }
    for name, xyz in sources.items():
        gap = reg.dist_mm(field, xyz)
        rows = []
        for i, (tq, t_a, t_b) in enumerate(spans, start=1):
            sel = contact & (robot_t >= t_a) & (robot_t <= t_b)
            rows.append({"window": i, "t_query_s": round(tq, 2), **stats(gap[sel])})
        out["poses"][name] = {
            "whole_episode_mm": stats(gap),
            "contact_mm": stats(gap[contact]),
            "first_half_contact_mm": stats(gap[contact & (robot_t < cutoff)]),
            "windows": rows,
        }
    (OUT / "frames.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

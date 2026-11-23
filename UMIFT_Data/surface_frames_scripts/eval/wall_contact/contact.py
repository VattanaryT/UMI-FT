#!/usr/bin/env python3
"""TCP-to-mesh distance on the adjusted thin wall. Contact only. No rescore."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames")
SCRIPTS = ROOT / "UMIFT_Data/surface_frames_scripts"
sys.path[:0] = [str(SCRIPTS / "eval" / "frame_flip"), str(SCRIPTS), str(ROOT)]

import zarr

from fixb_field import FixBMoldField
import register_contact as reg

WALL = Path(
    "/mnt/data/umidata/ft/held_out_mold/thin_stringer/transform_check/"
    "wall_fix/fixb_field_wall_vertical.npz"
)
OUT = Path(__file__).resolve().parent / "contact.json"


def main() -> None:
    meta = json.loads(WALL.with_suffix(".json").read_text())
    field = FixBMoldField(WALL)
    ep = zarr.open(reg.SRC, "r")["data"]["episode_0"]
    robot_t = np.asarray(ep["robot_time_stamps_0"]).reshape(-1)
    rgb_t = np.asarray(ep["rgb_time_stamps_0"]).reshape(-1)
    tcp = np.asarray(ep["ts_pose_fb_0"], dtype=np.float64)[:, :3]
    wrench_t = np.asarray(ep["wrench_time_stamps_left_0"]).reshape(-1)
    force = np.linalg.norm(np.asarray(ep["wrench_left_0"], dtype=np.float64)[:, :3], axis=1)
    t0 = float(rgb_t[0])
    cutoff = t0 + 0.5 * float(rgb_t[-1] - t0)
    rel = robot_t - t0
    aligned = force[reg.nearest(wrench_t, robot_t)]
    contact = aligned >= reg.FORCE_N
    gap = reg.dist_mm(field, tcp)
    rows = []
    for i, (tq, t_a, t_b) in enumerate(reg.windows(robot_t, rgb_t, t0, cutoff), start=1):
        sel = contact & (rel >= (t_a - t0)) & (rel <= (t_b - t0))
        rows.append({
            "window": i,
            "t_query_s": round(tq, 2),
            "t_end_s": round(t_b - t0, 2),
            **reg.stats(gap[sel]),
        })
    report = {
        "mesh": str(WALL),
        "yaw_deg": meta.get("yaw_deg"),
        "tilt_deg": meta.get("tilt_deg"),
        "offset_mm": meta.get("offset_mm"),
        "length_mm": meta.get("length_mm"),
        "cap_radius_mm": meta.get("cap_radius_mm"),
        "force_n": reg.FORCE_N,
        "episode_median_mm": float(np.median(gap)),
        "contact": reg.stats(gap[contact]),
        "first_half_contact": reg.stats(gap[contact & (rel < (cutoff - t0))]),
        "windows": rows,
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

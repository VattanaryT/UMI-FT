#!/usr/bin/env python3
"""Does causal-foot's stored ΔR hop while the tool does not?

The runtime foot is CausalFoot.step, which is the tracker that wrote these
labels. A hop here is a hop in that decode.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import zarr
from scipy.spatial.transform import Rotation

ROOT = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames")
SCRIPTS = ROOT / "UMIFT_Data/surface_frames_scripts"
sys.path[:0] = [str(SCRIPTS)]

from fixb_labels import pose7_to_Rp  # noqa: E402
from mold_aware_runtime import rot6d_rows_to_R  # noqa: E402

CAUSAL = (
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "r4vt_causal_foot/acp_replay_buffer.zarr"
)
DATASETS = {
    "heading_tilt": "r4vt_heading_tilt",
    "tilt_1_r4vt": "r4vt_world_fixb_tilt",
    "tilt_1_r4": "r4_world_fixb_tilt",
    "causal_foot": "r4vt_causal_foot",
}
BASE = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr")
OUT = SCRIPTS / "eval/causal_foot_hops.json"


def _angle(rotation):
    rel = np.einsum("nji,njk->nik", rotation[:-1], rotation[1:])
    return np.degrees(np.linalg.norm(Rotation.from_matrix(rel).as_rotvec(), axis=1))


def _normal_angle(normal):
    signed = np.array(normal, copy=True)
    for i in range(1, len(signed)):
        if float(np.dot(signed[i], signed[i - 1])) < 0.0:
            signed[i] *= -1.0
    return np.degrees(np.arccos(np.clip(np.sum(signed[:-1] * signed[1:], axis=1), -1.0, 1.0)))


def _count(zarr_path: str) -> dict:
    data = zarr.open(zarr_path, "r")["data"]
    tool_big = chart_big = normal_big = steps = 0
    for episode in range(50):
        group = data[f"episode_{episode}"]
        pose = np.asarray(group["ts_pose_fb_0"], np.float64)
        time = np.asarray(group["robot_time_stamps_0"], np.float64).reshape(-1)
        rotation, _position = pose7_to_Rp(pose)
        delta = rot6d_rows_to_R(np.asarray(group["fixb_delta_R_rot6d_0"], np.float64))
        normal = np.asarray(group["mold_normal_0"], np.float64)
        n = min(len(rotation), len(delta), len(normal), len(time))
        dt = np.diff(time[:n])
        neighbor = (dt > 1e-4) & (dt < 0.1)
        tool = _angle(rotation[:n])
        chart = _angle(delta[:n])
        hop = _normal_angle(normal[:n])
        steps += int(neighbor.sum())
        still = neighbor & (tool <= 45.0)
        tool_big += int((neighbor & (tool > 45.0)).sum())
        chart_big += int((still & (chart > 45.0)).sum())
        normal_big += int((still & (hop > 20.0)).sum())
    return {
        "steps": steps,
        "tool_over_45": tool_big,
        "rot6d_over_45_tool_under_45": chart_big,
        "normal_over_20_tool_under_45": normal_big,
    }


def _uses_heading(name: str) -> dict:
    path = BASE / name / "acp_replay_buffer.zarr"
    group = zarr.open(str(path), "r")["data"]["episode_0"]
    keys = set(group.keys())
    return {
        "heading_tilt_labels": "fixb_heading_tilt_0" in keys,
        "rot6d_labels": "fixb_delta_R_rot6d_0" in keys,
        "label_version": group.attrs.get("label_version"),
    }


def main() -> None:
    report = {
        "which_labels": {key: _uses_heading(name) for key, name in DATASETS.items()},
        "causal_foot": _count(CAUSAL),
        "runtime": (
            "Causal-foot decode calls CausalFoot.step, the same tracker that wrote "
            "fixb_delta_R_rot6d_0. It does not repack from a fresh closest-point normal."
        ),
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

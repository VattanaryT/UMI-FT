#!/usr/bin/env python3
"""Offline validation for TCP frame consistency in converted UMIFT zarr."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import zarr

UMIFT_DATA_ROOT = Path(__file__).resolve().parents[1]
if str(UMIFT_DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(UMIFT_DATA_ROOT))

from umift.utils.frame_conventions import FRAME_CONVENTION_VERSION, check_tcp_pose_consistency

# Import push-direction check from PyriteUtility when available
try:
    from PyriteUtility.data_pipeline.wrench_baseline_utils import (
        check_virtual_target_push_direction,
    )
except ImportError:
    check_virtual_target_push_direction = None


def validate_episode(ep_data, robot_id: int = 0) -> dict:
    ts_pose_fb = np.array(ep_data[f"ts_pose_fb_{robot_id}"])
    wrench = np.array(ep_data[f"wrench_left_{robot_id}"])
    result = {
        "tcp_pose_consistency": check_tcp_pose_consistency(ts_pose_fb),
        "wrench_force_norm_mean": float(np.mean(np.linalg.norm(wrench[:, :3], axis=1))),
    }
    if f"ts_pose_virtual_target_{robot_id}" in ep_data:
        ts_vt = np.array(ep_data[f"ts_pose_virtual_target_{robot_id}"])
        if check_virtual_target_push_direction is not None:
            result["virtual_target_push"] = check_virtual_target_push_direction(
                ts_pose_fb, ts_vt, wrench
            )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate UMIFT zarr frame conventions")
    parser.add_argument("--dataset_path", required=True, type=str)
    parser.add_argument("--robot_id", default=0, type=int)
    parser.add_argument("--max_episodes", default=3, type=int, help="Limit episodes printed")
    args = parser.parse_args()

    root = zarr.open(args.dataset_path, mode="r")
    conversion_info = root.attrs.get("conversion_info", "{}")
    if isinstance(conversion_info, str):
        info = json.loads(conversion_info)
    else:
        info = dict(conversion_info)
    print(f"conversion_info version: {info.get('frame_convention_version', 'unknown')}")
    print(f"expected version: {FRAME_CONVENTION_VERSION}")

    for i, (ep_name, ep_data) in enumerate(root["data"].items()):
        if i >= args.max_episodes:
            break
        metrics = validate_episode(ep_data, robot_id=args.robot_id)
        print(f"{ep_name}: {metrics}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Capture a free-space wrench offset for session_tare baseline mode."""

import argparse
import json
from pathlib import Path

import numpy as np
import zarr

from PyriteUtility.data_pipeline.wrench_baseline_utils import (
    compute_first_n_sample_count,
    save_session_wrench_offset,
)


def capture_from_zarr(
    dataset_path: Path,
    episode_key: str,
    robot_id: int,
    duration_s: float,
    output_path: Path,
) -> np.ndarray:
    root = zarr.open(str(dataset_path), mode="r")
    ep = root["data"][episode_key]
    wrench = np.array(ep[f"wrench_left_{robot_id}"])
    timestamps = np.array(ep[f"wrench_time_stamps_left_{robot_id}"]).reshape(-1)
    n = compute_first_n_sample_count(timestamps, duration_s)
    offset = np.mean(wrench[:n], axis=0)
    save_session_wrench_offset(
        offset,
        output_path,
        metadata={
            "source_dataset": str(dataset_path),
            "episode": episode_key,
            "robot_id": robot_id,
            "baseline_duration_s": duration_s,
            "num_samples": n,
        },
    )
    return offset


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture session wrench offset from zarr")
    parser.add_argument("--dataset_path", required=True, type=str)
    parser.add_argument("--episode", default="episode_0", type=str)
    parser.add_argument("--robot_id", default=0, type=int)
    parser.add_argument("--duration_s", default=1.0, type=float)
    parser.add_argument("--output", required=True, type=str)
    args = parser.parse_args()

    offset = capture_from_zarr(
        Path(args.dataset_path),
        args.episode,
        args.robot_id,
        args.duration_s,
        Path(args.output),
    )
    print(f"Saved session offset to {args.output}: {offset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

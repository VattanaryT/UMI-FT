#!/usr/bin/env python3
"""
Convert unified T265 + wrist FT + D405 recordings into UMIFT-compatible zarr.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import zarr
from numcodecs import Blosc

UMIFT_DATA_ROOT = Path(__file__).resolve().parents[1]
if str(UMIFT_DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(UMIFT_DATA_ROOT))

from umift.utils.frame_conventions import (
    check_tcp_pose_consistency,
    conversion_info_json,
    ft_poses_to_tcp_poses,
    ft_wrench_to_tcp_wrench,
)


def _ensure_zarr_v2():
    version = getattr(zarr, "__version__", "unknown")
    if not hasattr(zarr, "DirectoryStore"):
        raise RuntimeError(
            "This converter is implemented for zarr v2 APIs (UMIFT-consistent). "
            f"Current zarr={version} does not expose DirectoryStore. "
            "Install zarr==2.18.3 and numcodecs==0.15.1."
        )


def _as_col(ts: np.ndarray) -> np.ndarray:
    return np.asarray(ts, dtype=np.float64).reshape(-1, 1)


def normalize_timestamps(streams: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Shift all stream timestamps by the smallest first sample."""
    first_samples = []
    for arr in streams.values():
        if arr.size == 0:
            continue
        first_samples.append(float(arr.reshape(-1)[0]))
    if not first_samples:
        return streams

    offset = min(first_samples)
    return {k: np.asarray(v, dtype=np.float64) - offset for k, v in streams.items()}


def build_ts_pose_fb(
    odom: Dict[str, np.ndarray],
    sensor_offset_xyz_m: np.ndarray = np.array([0.0, 0.0, -0.02], dtype=np.float64),
) -> np.ndarray:
    """Build UMIFT TCP pose [x, y, z, qw, qx, qy, qz] from publisher-corrected FT odom."""
    positions = np.column_stack([odom["pos_x"], odom["pos_y"], odom["pos_z"]])
    orientations_wxyz = np.column_stack(
        [odom["orient_w"], odom["orient_x"], odom["orient_y"], odom["orient_z"]]
    )
    return ft_poses_to_tcp_poses(
        positions_ft=positions,
        orientations_wxyz=orientations_wxyz,
        sensor_offset_xyz_m=sensor_offset_xyz_m,
    )


def _apply_wrench_point_offset(
    wrench: np.ndarray, sensor_offset_xyz_m: np.ndarray
) -> np.ndarray:
    """
    Shift wrench from sensor point to target point using:
        tau_target = tau_sensor + r x F
    where r is sensor offset vector (target -> sensor) in meters.
    """
    r = np.asarray(sensor_offset_xyz_m, dtype=np.float64).reshape(3)
    if np.allclose(r, 0.0):
        return wrench
    adjusted = np.array(wrench, copy=True)
    force = adjusted[:, 0:3]
    adjusted[:, 3:6] = adjusted[:, 3:6] + np.cross(np.broadcast_to(r, force.shape), force)
    return adjusted


def build_wrench_keys(
    ft: Dict[str, np.ndarray],
    sensor_offset_xyz_m: np.ndarray = np.zeros(3),
) -> Dict[str, np.ndarray]:
    """Build UMIFT wrench keys from wrist FT sensor readings (FT frame -> TCP frame)."""
    wrench_ft = np.column_stack(
        [
            ft["force_x"],
            ft["force_y"],
            ft["force_z"],
            ft["torque_x"],
            ft["torque_y"],
            ft["torque_z"],
        ]
    ).astype(np.float64)
    left = ft_wrench_to_tcp_wrench(wrench_ft)
    left = _apply_wrench_point_offset(left, sensor_offset_xyz_m=sensor_offset_xyz_m)
    right = np.zeros_like(left)
    concat = np.hstack((left, right))

    return {
        "wrench_left_0": left,
        "wrench_right_0": right,
        "wrench_concat_0": concat,
        "wrench_left_coinft_0": left,
        "wrench_right_coinft_0": right,
        "wrench_concat_coinft_0": concat,
    }


def convert_depth_to_umift_depth(
    depth_mm: np.ndarray, depth_scale: float = 0.0001, depth_clip: float = 0.5
) -> np.ndarray:
    """
    Convert depth from uint16 mm to float16 meters and expand to 3 channels.
    """
    depth_m = np.asarray(depth_mm, dtype=np.float32) * depth_scale
    depth_m = np.clip(depth_m, 0.0, depth_clip).astype(np.float16)
    return np.repeat(depth_m[..., None], 3, axis=-1)


def _read_ts(group, rel_path: str) -> np.ndarray:
    arr = np.asarray(group[rel_path][:], dtype=np.float64).reshape(-1)
    return arr


def _read_frame_ts(camera_group) -> np.ndarray:
    if "timestamp_sec_relative" in camera_group:
        return _read_ts(camera_group, "timestamp_sec_relative")
    if "timestamp_sec" in camera_group:
        return _read_ts(camera_group, "timestamp_sec")
    if "timestamp_ms" in camera_group:
        return _read_ts(camera_group, "timestamp_ms") * 1e-3
    raise KeyError("camera stream must contain timestamp_sec_relative/sec/ms")


def _build_index_map(source_ts: np.ndarray, target_ts: np.ndarray) -> np.ndarray:
    if len(target_ts) == 0:
        return np.zeros((len(source_ts), 1), dtype=np.int64)
    idx = np.searchsorted(target_ts, source_ts, side="left")
    idx = np.clip(idx, 0, len(target_ts) - 1)
    return idx.reshape(-1, 1).astype(np.int64)


def _open_unified_root(path: Path):
    # Works with zarr v3 environments where these unified datasets are produced.
    try:
        return zarr.open_group(str(path), mode="r")
    except Exception as exc:
        raise RuntimeError(
            f"Unable to open unified zarr at {path}. "
            "This dataset format requires a zarr runtime that can read zarr v3 groups "
            "(for example: pip install 'zarr>=3')."
        ) from exc


def _read_episode_from_unified(
    path: Path,
    sensor_offset_xyz_m: np.ndarray = np.zeros(3),
) -> Dict[str, np.ndarray]:
    root = _open_unified_root(path)

    color_group = root["camera"]["color"]
    depth_group = root["camera"]["depth"]
    odom_group = root["sensors"]["t265_odom"]
    ft_group = root["sensors"]["ft_wrench"]

    rgb = np.asarray(color_group["frames"][:], dtype=np.uint8)
    ultrawide = np.asarray(color_group["frames"][:], dtype=np.uint8)
    depth = convert_depth_to_umift_depth(np.asarray(depth_group["frames"][:], dtype=np.uint16))

    rgb_ts = _read_frame_ts(color_group)
    uw_ts = np.copy(rgb_ts)
    depth_ts = _read_frame_ts(depth_group)
    robot_ts = _read_ts(odom_group, "timestamp_sec_relative")
    wrench_ts = _read_ts(ft_group, "timestamp_sec_relative")

    odom = {k: np.asarray(odom_group[k][:], dtype=np.float64) for k in [
        "pos_x",
        "pos_y",
        "pos_z",
        "orient_x",
        "orient_y",
        "orient_z",
        "orient_w",
    ]}
    pose = build_ts_pose_fb(odom, sensor_offset_xyz_m=sensor_offset_xyz_m)
    consistency = check_tcp_pose_consistency(pose)
    print(f"[convert] TCP pose consistency ({path.name}): {consistency}")

    ft = {k: np.asarray(ft_group[k][:], dtype=np.float64) for k in [
        "force_x",
        "force_y",
        "force_z",
        "torque_x",
        "torque_y",
        "torque_z",
    ]}
    wrench_keys = build_wrench_keys(ft, sensor_offset_xyz_m=sensor_offset_xyz_m)

    ts_norm = normalize_timestamps(
        {
            "rgb_time_stamps_0": rgb_ts,
            "ultrawide_time_stamps_0": uw_ts,
            "depth_time_stamps_0": depth_ts,
            "robot_time_stamps_0": robot_ts,
            "wrench_time_stamps_left_0": wrench_ts,
            "wrench_time_stamps_right_0": wrench_ts,
        }
    )

    robot_len = pose.shape[0]
    gripper = np.zeros((robot_len, 1), dtype=np.float64)
    gripper_ts = _as_col(ts_norm["robot_time_stamps_0"])

    return {
        "rgb_0": rgb,
        "ultrawide_0": ultrawide,
        "depth_0": depth,
        "map_to_uw_idx_0": _build_index_map(ts_norm["rgb_time_stamps_0"], ts_norm["ultrawide_time_stamps_0"]),
        "map_to_d_idx_0": _build_index_map(ts_norm["rgb_time_stamps_0"], ts_norm["depth_time_stamps_0"]),
        "ts_pose_fb_0": pose,
        "gripper_0": gripper,
        "rgb_time_stamps_0": _as_col(ts_norm["rgb_time_stamps_0"]),
        "ultrawide_time_stamps_0": _as_col(ts_norm["ultrawide_time_stamps_0"]),
        "depth_time_stamps_0": _as_col(ts_norm["depth_time_stamps_0"]),
        "robot_time_stamps_0": _as_col(ts_norm["robot_time_stamps_0"]),
        "gripper_time_stamps_0": gripper_ts,
        "wrench_time_stamps_left_0": _as_col(ts_norm["wrench_time_stamps_left_0"]),
        "wrench_time_stamps_right_0": _as_col(ts_norm["wrench_time_stamps_right_0"]),
        **wrench_keys,
    }


def _episode_len(cumulative: List[int], this_len: int) -> None:
    cumulative.append((cumulative[-1] if cumulative else 0) + int(this_len))


def _input_unified_paths(input_path: Path) -> List[Path]:
    if input_path.is_file() and input_path.name.endswith(".zarr"):
        return [input_path]
    if input_path.is_dir() and input_path.name.endswith(".zarr"):
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.glob("*_unified.zarr"))
    return []


def convert(
    input_path: Path,
    output_path: Path,
    sensor_offset_xyz_m: np.ndarray = np.zeros(3),
) -> None:
    _ensure_zarr_v2()

    unified_paths = _input_unified_paths(input_path)
    if not unified_paths:
        raise FileNotFoundError(f"No unified zarr found under: {input_path}")

    numeric_compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)
    image_compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)

    store = zarr.DirectoryStore(str(output_path))
    root = zarr.group(store=store, overwrite=True)
    data_group = root.create_group("data")
    meta_group = root.create_group("meta")

    episode_rgb_len = []
    episode_uw_len = []
    episode_depth_len = []
    episode_robot_len = []
    episode_gripper_len = []
    episode_wrench_len = []

    for ep_idx, unified_path in enumerate(unified_paths):
        ep_data = _read_episode_from_unified(
            unified_path,
            sensor_offset_xyz_m=sensor_offset_xyz_m,
        )
        ep_group = data_group.create_group(f"episode_{ep_idx}")

        for key, value in ep_data.items():
            compressor = image_compressor if key in {"rgb_0", "ultrawide_0"} else numeric_compressor
            ep_group.create_dataset(
                name=key,
                data=value,
                shape=value.shape,
                chunks=True,
                dtype=value.dtype,
                compressor=compressor,
            )

        _episode_len(episode_rgb_len, ep_data["rgb_0"].shape[0])
        _episode_len(episode_uw_len, ep_data["ultrawide_0"].shape[0])
        _episode_len(episode_depth_len, ep_data["depth_0"].shape[0])
        _episode_len(episode_robot_len, ep_data["ts_pose_fb_0"].shape[0])
        _episode_len(episode_gripper_len, ep_data["gripper_0"].shape[0])
        _episode_len(episode_wrench_len, ep_data["wrench_left_0"].shape[0])

        print(f"[convert] episode_{ep_idx} <- {unified_path.name}")

    meta_group.create_dataset(
        "episode_rgb0_len",
        data=np.asarray(episode_rgb_len, dtype=np.int64),
        compressor=numeric_compressor,
    )
    meta_group.create_dataset(
        "episode_ultrawide0_len",
        data=np.asarray(episode_uw_len, dtype=np.int64),
        compressor=numeric_compressor,
    )
    meta_group.create_dataset(
        "episode_depth0_len",
        data=np.asarray(episode_depth_len, dtype=np.int64),
        compressor=numeric_compressor,
    )
    meta_group.create_dataset(
        "episode_robot0_len",
        data=np.asarray(episode_robot_len, dtype=np.int64),
        compressor=numeric_compressor,
    )
    meta_group.create_dataset(
        "episode_gripper0_len",
        data=np.asarray(episode_gripper_len, dtype=np.int64),
        compressor=numeric_compressor,
    )
    meta_group.create_dataset(
        "episode_wrench0_len",
        data=np.asarray(episode_wrench_len, dtype=np.int64),
        compressor=numeric_compressor,
    )

    root.attrs["conversion_info"] = conversion_info_json(
        extra={
            "source_type": "unified_t265_wild",
            "pose_source": "t265_odom_ft_to_tcp",
            "gripper_source": "synthetic_zero",
            "wrench_source": "single_wrist_ft_mapped_to_left_tcp_frame",
            "sensor_offset_xyz_m": np.asarray(sensor_offset_xyz_m, dtype=np.float64).tolist(),
            "ultrawide_source": "alias_of_d405_color",
            "depth_source": "d405_depth_mm_to_m_float16_3ch_clipped_0p5",
        },
    )

    print(f"[convert] wrote UMIFT-style dataset to: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert unified T265 recordings to UMIFT zarr schema")
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Input unified zarr path or directory containing *_unified.zarr files",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Output episode-structured zarr path",
    )
    parser.add_argument(
        "--sensor_offset_z_m",
        type=float,
        default=0.0,
        help="Sensor offset along z (meters). Example: -0.03 for 3cm in -z.",
    )
    parser.add_argument(
        "--sensor_offset_xyz_m",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Full sensor offset vector in meters. Overrides --sensor_offset_z_m.",
    )
    args = parser.parse_args()

    if args.sensor_offset_xyz_m is not None:
        sensor_offset_xyz_m = np.asarray(args.sensor_offset_xyz_m, dtype=np.float64)
    else:
        sensor_offset_xyz_m = np.array([0.0, 0.0, args.sensor_offset_z_m], dtype=np.float64)

    convert(
        Path(args.input),
        Path(args.output),
        sensor_offset_xyz_m=sensor_offset_xyz_m,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

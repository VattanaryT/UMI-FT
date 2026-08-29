#!/usr/bin/env python3
"""
Convert LeRobot v3 teleop dataset (geometry only) into UMIFT episode zarr.

Does not write virtual-target / stiffness (use existing postprocess).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import zarr
from numcodecs import Blosc

UMIFT_DATA_ROOT = Path(__file__).resolve().parents[1]
if str(UMIFT_DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(UMIFT_DATA_ROOT))

from umift.processing.pad_episode_start import pad_episode_start  # noqa: E402
from umift.processing.umift_image_prep import resize_hwc_uint8  # noqa: E402
from umift.utils.frame_conventions import rotation_matrix_to_quaternion_wxyz  # noqa: E402

WRIST_CAM_KEY = "observation.images.wrist_cam_0_color_image_rect_raw_compressed"


def _ensure_zarr_v2() -> None:
    version = getattr(zarr, "__version__", "unknown")
    if not hasattr(zarr, "DirectoryStore"):
        raise RuntimeError(
            "This converter is implemented for zarr v2 APIs (UMIFT-consistent). "
            f"Current zarr={version} does not expose DirectoryStore. "
            "Install zarr==2.18.3 and numcodecs==0.15.1."
        )


def _as_col(ts: np.ndarray) -> np.ndarray:
    return np.asarray(ts, dtype=np.float64).reshape(-1, 1)


def _normalize_rows(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.maximum(n, eps)
    return v / n


def rot6d_to_quat_wxyz(rot6: np.ndarray) -> np.ndarray:
    """Convert Zhou-style rot6d (first two rotation rows) to quaternion wxyz.

    Accepts shape (..., 6) and returns (..., 4).
    """
    rot6 = np.asarray(rot6, dtype=np.float64)
    single = rot6.ndim == 1
    if single:
        rot6 = rot6.reshape(1, 6)
    if rot6.shape[-1] != 6:
        raise ValueError(f"rot6 last dim must be 6, got {rot6.shape}")

    a1 = rot6[..., :3]
    a2 = rot6[..., 3:]
    b1 = _normalize_rows(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = _normalize_rows(b2)
    b3 = np.cross(b1, b2, axis=-1)
    R = np.stack((b1, b2, b3), axis=-2)  # (..., 3, 3)

    flat = R.reshape(-1, 3, 3)
    quats = np.stack(
        [rotation_matrix_to_quaternion_wxyz(Ri) for Ri in flat], axis=0
    ).reshape(R.shape[:-2] + (4,))
    if single:
        return quats[0]
    return quats


def right_state_to_pose7_wrench(
    state30: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Unpack right arm from observation.state (T, 30) → pose7 (T,7), wrench (T,6).

    Layout: [pos3, rot6d, wrench6, left...]
    """
    state30 = np.asarray(state30)
    if state30.ndim != 2 or state30.shape[1] != 30:
        raise ValueError(f"state30 must be (T, 30), got {state30.shape}")
    pos = state30[:, 0:3].astype(np.float64)
    quat = rot6d_to_quat_wxyz(state30[:, 3:9])
    pose7 = np.concatenate([pos, quat], axis=-1)
    wrench = state30[:, 9:15].astype(np.float64)
    return pose7, wrench


def build_episode_arrays_from_tables(
    timestamps: np.ndarray,
    observation_state: np.ndarray,
    rgb_frames: np.ndarray,
    image_size: int = 224,
) -> Dict[str, np.ndarray]:
    """Assemble UMIFT episode arrays (geometry only; no VT/stiffness)."""
    timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    observation_state = np.asarray(observation_state)
    rgb_frames = np.asarray(rgb_frames, dtype=np.uint8)
    T = len(timestamps)
    if observation_state.shape[0] != T:
        raise ValueError(
            f"state length {observation_state.shape[0]} != timestamps {T}"
        )
    if rgb_frames.shape[0] != T:
        raise ValueError(f"rgb length {rgb_frames.shape[0]} != timestamps {T}")

    out_hw = (int(image_size), int(image_size))
    rgb = resize_hwc_uint8(rgb_frames, out_hw)
    # Schema placeholder only; training does not use ultrawide (same as depth zeros).
    ultrawide = np.zeros((T, out_hw[0], out_hw[1], 3), dtype=np.uint8)
    depth = np.zeros((T, out_hw[0], out_hw[1], 3), dtype=np.float16)

    pose7, wrench_left = right_state_to_pose7_wrench(observation_state)
    wrench_right = np.zeros_like(wrench_left)
    wrench_concat = np.hstack((wrench_left, wrench_right))
    gripper = np.zeros((T, 1), dtype=np.float64)
    ts_col = _as_col(timestamps)
    identity_map = np.arange(T, dtype=np.int64).reshape(-1, 1)

    return {
        "rgb_0": rgb,
        "ultrawide_0": ultrawide,
        "depth_0": depth,
        "map_to_uw_idx_0": identity_map,
        "map_to_d_idx_0": identity_map.copy(),
        "ts_pose_fb_0": pose7,
        "ts_pose_command_0": pose7.copy(),
        "gripper_0": gripper,
        "rgb_time_stamps_0": ts_col.copy(),
        "ultrawide_time_stamps_0": ts_col.copy(),
        "depth_time_stamps_0": ts_col.copy(),
        "robot_time_stamps_0": ts_col.copy(),
        "gripper_time_stamps_0": ts_col.copy(),
        "wrench_time_stamps_left_0": ts_col.copy(),
        "wrench_time_stamps_right_0": ts_col.copy(),
        "wrench_left_0": wrench_left,
        "wrench_right_0": wrench_right,
        "wrench_concat_0": wrench_concat,
        "wrench_left_coinft_0": wrench_left.copy(),
        "wrench_right_coinft_0": wrench_right.copy(),
        "wrench_concat_coinft_0": wrench_concat.copy(),
    }


def _load_info(root: Path) -> dict:
    return json.loads((root / "meta" / "info.json").read_text())


def _load_episodes_table(root: Path):
    import pyarrow.parquet as pq

    ep_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not ep_files:
        raise FileNotFoundError(f"No episode parquet under {root / 'meta' / 'episodes'}")
    tables = [pq.read_table(p) for p in ep_files]
    import pyarrow as pa

    return pa.concat_tables(tables).to_pandas()


def _stack_state_column(series) -> np.ndarray:
    rows = []
    for item in series:
        rows.append(np.asarray(item, dtype=np.float32).reshape(-1))
    return np.stack(rows, axis=0)


def _load_data_slice(
    root: Path,
    dataset_from_index: int,
    dataset_to_index: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (timestamps, observation_state) for [from, to)."""
    import pyarrow.parquet as pq

    data_files = sorted((root / "data").rglob("*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No data parquet under {root / 'data'}")

    # Fast path: single parquet covering full dataset (common for this teleop set).
    if len(data_files) == 1:
        table = pq.read_table(
            data_files[0],
            columns=["timestamp", "observation.state", "index"],
        )
        pdf = table.to_pandas()
        mask = (pdf["index"] >= dataset_from_index) & (pdf["index"] < dataset_to_index)
        pdf = pdf.loc[mask].sort_values("index")
        if len(pdf) != (dataset_to_index - dataset_from_index):
            raise RuntimeError(
                f"Expected {dataset_to_index - dataset_from_index} rows, got {len(pdf)} "
                f"for indices [{dataset_from_index}, {dataset_to_index})"
            )
        ts = np.asarray(pdf["timestamp"].to_numpy(), dtype=np.float64)
        state = _stack_state_column(pdf["observation.state"])
        return ts, state

    rows_index: List[int] = []
    rows_ts: List[float] = []
    rows_state: List[np.ndarray] = []
    for path in data_files:
        table = pq.read_table(path, columns=["timestamp", "observation.state", "index"])
        pdf = table.to_pandas()
        mask = (pdf["index"] >= dataset_from_index) & (pdf["index"] < dataset_to_index)
        if not mask.any():
            continue
        pdf = pdf.loc[mask]
        for idx, ts, item in zip(
            pdf["index"], pdf["timestamp"], pdf["observation.state"]
        ):
            rows_index.append(int(idx))
            rows_ts.append(float(ts))
            rows_state.append(np.asarray(item, dtype=np.float32).reshape(-1))
    if len(rows_ts) != (dataset_to_index - dataset_from_index):
        raise RuntimeError(
            f"Expected {dataset_to_index - dataset_from_index} rows across data files, "
            f"got {len(rows_ts)}"
        )
    order = np.argsort(rows_index)
    ts = np.asarray([rows_ts[i] for i in order], dtype=np.float64)
    state = np.stack([rows_state[i] for i in order], axis=0)
    return ts, state


def _video_path(root: Path, info: dict, video_key: str, chunk_index: int, file_index: int) -> Path:
    rel = info["video_path"].format(
        video_key=video_key,
        chunk_index=chunk_index,
        file_index=file_index,
    )
    return root / rel


def decode_video_segment(
    video_path: Path,
    from_timestamp: float,
    to_timestamp: float,
    expected_frames: Optional[int] = None,
) -> np.ndarray:
    """Decode [from_timestamp, to_timestamp) from an mp4 via PyAV (AV1-safe)."""
    import av

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    # Seek slightly before start; drop frames with t < from_timestamp.
    seek_target = max(0.0, float(from_timestamp) - 1.0 / max(float(stream.average_rate or 10.0), 1.0))
    try:
        container.seek(int(seek_target * av.time_base), any_frame=False, backward=True)
    except av.AVError:
        container.seek(0)

    frames: List[np.ndarray] = []
    for frame in container.decode(video=0):
        if frame.pts is None:
            continue
        t = float(frame.pts * stream.time_base)
        if t < float(from_timestamp) - 1e-6:
            continue
        if t >= float(to_timestamp) - 1e-9:
            break
        frames.append(frame.to_ndarray(format="rgb24"))
    container.close()

    if not frames:
        raise RuntimeError(
            f"No frames decoded from {video_path} in [{from_timestamp}, {to_timestamp})"
        )
    arr = np.stack(frames, axis=0)
    if expected_frames is not None and arr.shape[0] != expected_frames:
        raise RuntimeError(
            f"Decoded {arr.shape[0]} frames from {video_path.name} "
            f"[{from_timestamp}, {to_timestamp}) but expected {expected_frames}"
        )
    return arr


def _episode_len(cumulative: List[int], this_len: int) -> None:
    cumulative.append((cumulative[-1] if cumulative else 0) + int(this_len))


def convert(
    input_path: Path,
    output_path: Path,
    image_size: int = 224,
    max_episodes: Optional[int] = None,
    start_pad_seconds: float = 5.0,
) -> None:
    _ensure_zarr_v2()
    input_path = Path(input_path)
    output_path = Path(output_path)

    info = _load_info(input_path)
    fps = float(info.get("fps", 10))
    episodes = _load_episodes_table(input_path)
    episodes = episodes.sort_values("episode_index").reset_index(drop=True)
    if max_episodes is not None:
        episodes = episodes.iloc[: int(max_episodes)].copy()

    numeric_compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)
    image_compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)

    store = zarr.DirectoryStore(str(output_path))
    root = zarr.group(store=store, overwrite=True)
    data_group = root.create_group("data")
    meta_group = root.create_group("meta")

    episode_rgb_len: List[int] = []
    episode_uw_len: List[int] = []
    episode_depth_len: List[int] = []
    episode_robot_len: List[int] = []
    episode_gripper_len: List[int] = []
    episode_wrench_len: List[int] = []

    video_chunk_col = f"videos/{WRIST_CAM_KEY}/chunk_index"
    video_file_col = f"videos/{WRIST_CAM_KEY}/file_index"
    video_from_col = f"videos/{WRIST_CAM_KEY}/from_timestamp"
    video_to_col = f"videos/{WRIST_CAM_KEY}/to_timestamp"

    for write_idx, (_, row) in enumerate(episodes.iterrows()):
        i0 = int(row["dataset_from_index"])
        i1 = int(row["dataset_to_index"])
        length = int(row["length"])
        if i1 - i0 != length:
            raise RuntimeError(
                f"episode {row['episode_index']}: length={length} but "
                f"dataset indices [{i0}, {i1}) span {i1 - i0}"
            )

        timestamps, state = _load_data_slice(input_path, i0, i1)
        # Episode-relative clock for UMIFT (start at 0); keep 10 Hz spacing.
        timestamps = timestamps - float(timestamps[0])

        vpath = _video_path(
            input_path,
            info,
            WRIST_CAM_KEY,
            chunk_index=int(row[video_chunk_col]),
            file_index=int(row[video_file_col]),
        )
        rgb = decode_video_segment(
            vpath,
            from_timestamp=float(row[video_from_col]),
            to_timestamp=float(row[video_to_col]),
            expected_frames=length,
        )

        ep_data = build_episode_arrays_from_tables(
            timestamps=timestamps,
            observation_state=state,
            rgb_frames=rgb,
            image_size=image_size,
        )
        ep_data = pad_episode_start(ep_data, pad_duration_s=start_pad_seconds)

        ep_group = data_group.create_group(f"episode_{write_idx}")
        ep_group.attrs["start_pad_seconds"] = float(start_pad_seconds)
        for key, value in ep_data.items():
            compressor = (
                image_compressor if key in {"rgb_0", "ultrawide_0"} else numeric_compressor
            )
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

        print(
            f"[convert] episode_{write_idx} <- lerobot ep {int(row['episode_index'])} "
            f"T={length} fps={fps}"
        )

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

    root.attrs["image_size_hw"] = [int(image_size), int(image_size)]
    root.attrs["resample_hz"] = float(fps)
    root.attrs["start_pad_seconds"] = float(start_pad_seconds)
    root.attrs["source_type"] = "lerobot_teleop"
    root.attrs["arm"] = "right"
    root.attrs["depth_source"] = "zeros"
    root.attrs["ultrawide_source"] = "zeros"
    root.attrs["conversion_info"] = json.dumps(
        {
            "source_type": "lerobot_teleop",
            "arm": "right",
            "depth_source": "zeros",
            "ultrawide_source": "zeros",
            "image_size_hw": [int(image_size), int(image_size)],
            "resample_hz": float(fps),
            "wrist_cam_key": WRIST_CAM_KEY,
            "num_episodes": int(len(episodes)),
            "vt_stiffness": "deferred_to_postprocess",
            "start_pad_seconds": float(start_pad_seconds),
        }
    )

    print(f"[convert] wrote UMIFT-style dataset to: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert LeRobot teleop dataset to UMIFT episode zarr (geometry only)"
    )
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="LeRobot dataset root (v3 parquet + mp4)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Output episode-structured zarr path",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Output RGB/depth spatial size (default 224)",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap on number of episodes (smoke / debug)",
    )
    parser.add_argument(
        "--start-pad-seconds",
        type=float,
        default=5.0,
        help=(
            "Prepend this many seconds of repeated first-frame data to every "
            "episode, so training queries can be sampled from the true reset "
            "state instead of always requiring real backward history "
            "(see SequenceSamplerUmiFT's obs horizons). Should cover at least "
            "the largest configured obs horizon (currently ~3.2s). Set to 0 "
            "to disable."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    convert(
        Path(args.input),
        Path(args.output),
        image_size=int(args.image_size),
        max_episodes=args.max_episodes,
        start_pad_seconds=args.start_pad_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

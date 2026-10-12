#!/usr/bin/env python3
"""Convert a UMIFT episode zarr into a LeRobot v3 dataset for viewing.

Frames are aligned to ``rgb_time_stamps_0`` (vision clock). Robot / wrench /
gripper streams are gathered by nearest timestamp so every RGB frame carries
pose and force. Vision modalities (rgb, ultrawide, depth, scan heatmap) are
kept; depth and scan heatmaps are stored both as viewable videos and as
float arrays so values are not discarded.

Episode-level arrays that are not per-frame (e.g. ``scan_gap_intervals_0``)
are written to ``meta/umift_episode_extras.json``.

Example:
  python UMIFT_Data/scripts/convert_umift_zarr_to_lerobot.py \\
    --input  /mnt/data/umidata/ft/datasets/stringer_scanner/acp_replay_buffer.zarr \\
    --output /mnt/data/umidata/ft/datasets/stringer_scanner/lerobot_acp_replay_buffer \\
    --repo-id local/stringer_scanner_tag
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import zarr
from tqdm import tqdm

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "lerobot is required. Use a Python env with lerobot installed "
        f"(import failed: {e})."
    ) from e


def _as_1d(ts: np.ndarray) -> np.ndarray:
    return np.asarray(ts, dtype=np.float64).reshape(-1)


def _nearest_indices(src_t: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    """For each query time, index of nearest sample in src_t (sorted ascending)."""
    src_t = _as_1d(src_t)
    query_t = _as_1d(query_t)
    if src_t.size == 0:
        raise ValueError("empty source timestamps")
    idx = np.searchsorted(src_t, query_t, side="left")
    idx = np.clip(idx, 0, len(src_t) - 1)
    left = np.clip(idx - 1, 0, len(src_t) - 1)
    choose_left = np.abs(src_t[left] - query_t) <= np.abs(src_t[idx] - query_t)
    return np.where(choose_left, left, idx).astype(np.int64)


def _estimate_fps(rgb_t: np.ndarray) -> int:
    rgb_t = _as_1d(rgb_t)
    if rgb_t.size < 2:
        return 10
    dt = float(np.median(np.diff(rgb_t)))
    if dt <= 0:
        return 10
    return max(1, int(round(1.0 / dt)))


def _depth_to_uint8(depth_hwc: np.ndarray, clip_m: float = 0.5) -> np.ndarray:
    """Viewable RGB from meters float depth (uses channel 0; channels are identical)."""
    d = np.asarray(depth_hwc[..., 0], dtype=np.float32)
    d = np.nan_to_num(d, nan=0.0, posinf=clip_m, neginf=0.0)
    d = np.clip(d, 0.0, clip_m) / max(clip_m, 1e-6)
    u8 = (d * 255.0).astype(np.uint8)
    return np.stack([u8, u8, u8], axis=-1)


def _heatmap_to_uint8(hm_hwc: np.ndarray, clip_mm: float = 5.0) -> np.ndarray:
    """Viewable RGB from scan heatmap (mm). NaNs → 0."""
    x = np.asarray(hm_hwc, dtype=np.float32)
    finite = np.isfinite(x)
    out = np.zeros(x.shape, dtype=np.uint8)
    if not np.any(finite):
        return out
    # Map signed mm into mid-gray centered viz: 0 → 128, ±clip → 0/255.
    scaled = np.clip(x, -clip_mm, clip_mm)
    scaled = (scaled / max(clip_mm, 1e-6) + 1.0) * 0.5  # [0,1]
    u8 = (scaled * 255.0).astype(np.uint8)
    out[finite] = u8[finite]
    return out


def _zero_heatmap(shape_hwc: Tuple[int, int, int]) -> np.ndarray:
    return np.zeros(shape_hwc, dtype=np.float32)


def _episode_names(root: zarr.Group) -> List[str]:
    names = [k for k in root["data"].group_keys() if str(k).startswith("episode")]
    def _key(n: str) -> Tuple[int, str]:
        try:
            return (int(str(n).split("_", 1)[1]), str(n))
        except Exception:
            return (10**12, str(n))
    return sorted(names, key=_key)


def _build_features(
    ep0: zarr.Group,
    include_scan: bool,
) -> Dict[str, Dict[str, Any]]:
    rgb = ep0["rgb_0"]
    _, h, w, c = rgb.shape
    features: Dict[str, Dict[str, Any]] = {
        "observation.images.rgb_0": {
            "dtype": "video",
            "shape": (c, h, w),
            "names": ["channel", "height", "width"],
        },
        "observation.images.ultrawide_0": {
            "dtype": "video",
            "shape": (c, h, w),
            "names": ["channel", "height", "width"],
        },
        "observation.images.depth_0": {
            "dtype": "video",
            "shape": (c, h, w),
            "names": ["channel", "height", "width"],
        },
        # Lossless-ish meters (channel 0 only; other channels are duplicates).
        "observation.depth_m": {
            "dtype": "float32",
            "shape": (h, w),
            "names": None,
        },
        # Packed low-dim: pose_fb7, pose_cmd7, pose_vt7, grip1, stiff1,
        # wrench_left6, wrench_right6, wrench_concat12, coinft_left6,
        # coinft_right6, coinft_concat12, map_d1, map_uw1, map_scan1,
        # rgb_t1, robot_t1, wrench_t1
        "observation.state": {
            "dtype": "float32",
            # pose_fb7 + cmd7 + vt7 + grip1 + stiff1
            # + wrench L/R/concat 6+6+12 + coinft L/R/concat 6+6+12
            # + map_d/uw/scan 1+1+1 + rgb_t/robot_t/wrench_t 1+1+1
            "shape": (77,),
            "names": None,
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": None,
        },
    }
    if include_scan:
        hm = ep0["scan_heatmap_0"]
        _, hh, ww, cc = hm.shape
        features["observation.images.scan_heatmap_0"] = {
            "dtype": "video",
            "shape": (cc, hh, ww),
            "names": ["channel", "height", "width"],
        }
        features["observation.scan_heatmap"] = {
            "dtype": "float32",
            "shape": (hh, ww, cc),
            "names": None,
        }
    return features


def _load_episode_arrays(ep: zarr.Group) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for key in ep.array_keys():
        out[key] = np.asarray(ep[key])
    return out


def convert_zarr(
    input_zarr: Path,
    output_root: Path,
    repo_id: str,
    task: str,
    overwrite: bool,
    max_episodes: Optional[int],
    depth_clip_m: float,
    heatmap_clip_mm: float,
    image_writer_threads: int,
) -> None:
    root = zarr.open(str(input_zarr), mode="r")
    if "data" not in root:
        raise SystemExit(f"No data/ group in {input_zarr}")

    names = _episode_names(root)
    if not names:
        raise SystemExit(f"No episodes in {input_zarr}")
    if max_episodes is not None:
        names = names[: max(0, max_episodes)]

    ep0 = root["data"][names[0]]
    include_scan = "scan_heatmap_0" in ep0
    rgb_t0 = _as_1d(ep0["rgb_time_stamps_0"])
    fps = _estimate_fps(rgb_t0)
    features = _build_features(ep0, include_scan=include_scan)

    if output_root.exists():
        if not overwrite:
            raise SystemExit(
                f"Output exists: {output_root} (pass --overwrite to replace)"
            )
        shutil.rmtree(output_root)

    print(
        f"Creating LeRobot dataset at {output_root}\n"
        f"  source={input_zarr}\n"
        f"  episodes={len(names)} fps={fps} scan={include_scan}"
    )
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=str(output_root),
        features=features,
        use_videos=True,
        image_writer_threads=image_writer_threads,
        robot_type="umift",
    )

    extras: Dict[str, Any] = {
        "source_zarr": str(input_zarr),
        "timeline": "rgb_time_stamps_0",
        "fps": fps,
        "include_scan": include_scan,
        "depth_clip_m": depth_clip_m,
        "heatmap_clip_mm": heatmap_clip_mm,
        "state_layout": [
            "ts_pose_fb_0[7]",
            "ts_pose_command_0[7]",
            "ts_pose_virtual_target_0[7]",
            "gripper_0[1]",
            "stiffness_0[1]",
            "wrench_left_0[6]",
            "wrench_right_0[6]",
            "wrench_concat_0[12]",
            "wrench_left_coinft_0[6]",
            "wrench_right_coinft_0[6]",
            "wrench_concat_coinft_0[12]",
            "map_to_d_idx_0[1]",
            "map_to_uw_idx_0[1]",
            "map_to_scan_idx_0[1]",
            "rgb_time_stamps_0[1]",
            "robot_time_stamps_0[1]",
            "wrench_time_stamps_0[1]",
        ],
        "action": "ts_pose_virtual_target_0",
        "episodes": {},
        "zarr_root_attrs": dict(root.attrs),
    }

    hm_shape = (80, 160, 3)
    if include_scan:
        hm_shape = tuple(int(x) for x in ep0["scan_heatmap_0"].shape[1:])

    for ep_name in tqdm(names, desc="episodes"):
        arrays = _load_episode_arrays(root["data"][ep_name])
        rgb_t = _as_1d(arrays["rgb_time_stamps_0"])
        n = len(rgb_t)
        robot_t = _as_1d(arrays["robot_time_stamps_0"])
        wrench_t = _as_1d(arrays["wrench_time_stamps_0"])
        robot_idx = _nearest_indices(robot_t, rgb_t)
        wrench_idx = _nearest_indices(wrench_t, rgb_t)

        rgb = arrays["rgb_0"]
        uw = arrays["ultrawide_0"]
        depth = arrays["depth_0"]
        map_d = arrays["map_to_d_idx_0"].reshape(-1).astype(np.int64)
        map_uw = arrays["map_to_uw_idx_0"].reshape(-1).astype(np.int64)
        if len(map_d) != n or len(map_uw) != n:
            raise ValueError(
                f"{ep_name}: map_to_* length mismatch "
                f"(rgb={n}, map_d={len(map_d)}, map_uw={len(map_uw)})"
            )

        if include_scan:
            map_scan = arrays["map_to_scan_idx_0"].reshape(-1).astype(np.int64)
            heatmaps = arrays["scan_heatmap_0"].astype(np.float32)
            if len(map_scan) != n:
                raise ValueError(
                    f"{ep_name}: map_to_scan_idx_0 length {len(map_scan)} != rgb {n}"
                )
        else:
            map_scan = np.full((n,), -1, dtype=np.int64)
            heatmaps = np.zeros((0,) + hm_shape, dtype=np.float32)

        pose_fb = arrays["ts_pose_fb_0"]
        pose_cmd = arrays["ts_pose_command_0"]
        pose_vt = arrays["ts_pose_virtual_target_0"]
        grip = arrays["gripper_0"].reshape(-1, 1)
        stiff = arrays["stiffness_0"].reshape(-1, 1)
        w_left = arrays["wrench_left_0"]
        w_right = arrays["wrench_right_0"]
        w_cat = arrays["wrench_concat_0"]
        w_left_c = arrays["wrench_left_coinft_0"]
        w_right_c = arrays["wrench_right_coinft_0"]
        w_cat_c = arrays["wrench_concat_coinft_0"]

        ep_extra: Dict[str, Any] = {
            "source_episode": ep_name,
            "n_rgb_frames": int(n),
            "n_robot_frames": int(len(robot_t)),
            "n_wrench_frames": int(len(wrench_t)),
            "array_shapes": {k: list(v.shape) for k, v in arrays.items()},
        }
        if "scan_gap_intervals_0" in arrays:
            ep_extra["scan_gap_intervals_0"] = arrays["scan_gap_intervals_0"].tolist()
        if include_scan:
            ep_extra["n_scan_heatmaps"] = int(heatmaps.shape[0])
        extras["episodes"][ep_name] = ep_extra

        for i in range(n):
            ri = int(robot_idx[i])
            wi = int(wrench_idx[i])
            di = int(np.clip(map_d[i], 0, len(depth) - 1))
            ui = int(np.clip(map_uw[i], 0, len(uw) - 1))

            depth_frame = depth[di]
            depth_m = np.asarray(depth_frame[..., 0], dtype=np.float32)
            depth_viz = _depth_to_uint8(depth_frame, clip_m=depth_clip_m)

            si = int(map_scan[i])
            if include_scan and si >= 0 and si < len(heatmaps):
                hm = heatmaps[si]
            else:
                hm = _zero_heatmap(hm_shape)
            hm_viz = _heatmap_to_uint8(hm, clip_mm=heatmap_clip_mm)

            state = np.concatenate(
                [
                    pose_fb[ri].astype(np.float32),
                    pose_cmd[ri].astype(np.float32),
                    pose_vt[ri].astype(np.float32),
                    grip[ri].astype(np.float32),
                    stiff[ri].astype(np.float32),
                    w_left[wi].astype(np.float32),
                    w_right[wi].astype(np.float32),
                    w_cat[wi].astype(np.float32),
                    w_left_c[wi].astype(np.float32),
                    w_right_c[wi].astype(np.float32),
                    w_cat_c[wi].astype(np.float32),
                    np.array([map_d[i]], dtype=np.float32),
                    np.array([map_uw[i]], dtype=np.float32),
                    np.array([map_scan[i]], dtype=np.float32),
                    np.array([rgb_t[i]], dtype=np.float32),
                    np.array([robot_t[ri]], dtype=np.float32),
                    np.array([wrench_t[wi]], dtype=np.float32),
                ],
                axis=0,
            )

            frame: Dict[str, Any] = {
                "observation.images.rgb_0": np.asarray(rgb[i], dtype=np.uint8),
                "observation.images.ultrawide_0": np.asarray(uw[ui], dtype=np.uint8),
                "observation.images.depth_0": depth_viz,
                "observation.depth_m": depth_m,
                "observation.state": state,
                "action": pose_vt[ri].astype(np.float32),
                "task": task,
            }
            if include_scan:
                frame["observation.images.scan_heatmap_0"] = hm_viz
                frame["observation.scan_heatmap"] = np.asarray(hm, dtype=np.float32)

            dataset.add_frame(frame)

        dataset.save_episode()

    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    extras_path = meta_dir / "umift_episode_extras.json"
    extras_path.write_text(json.dumps(extras, indent=2) + "\n")
    print(f"Wrote {extras_path}")
    print(f"Done. View with LeRobotDataset(repo_id={repo_id!r}, root={str(output_root)!r})")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="UMIFT acp_replay_buffer*.zarr")
    p.add_argument("--output", type=Path, required=True, help="LeRobot dataset root")
    p.add_argument("--repo-id", type=str, default="local/umift_view")
    p.add_argument("--task", type=str, default="umift demonstration")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--depth-clip-m", type=float, default=0.5)
    p.add_argument("--heatmap-clip-mm", type=float, default=5.0)
    p.add_argument("--image-writer-threads", type=int, default=4)
    args = p.parse_args(argv)

    convert_zarr(
        input_zarr=args.input,
        output_root=args.output,
        repo_id=args.repo_id,
        task=args.task,
        overwrite=args.overwrite,
        max_episodes=args.max_episodes,
        depth_clip_m=args.depth_clip_m,
        heatmap_clip_mm=args.heatmap_clip_mm,
        image_writer_threads=args.image_writer_threads,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

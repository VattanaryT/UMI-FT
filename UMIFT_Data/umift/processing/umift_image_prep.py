"""Shared image resize and episode resampling helpers for UMIFT converters."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_UMI_FT_ROOT = Path(__file__).resolve().parents[3]
_PYRITE_UTILITY_ROOT = _UMI_FT_ROOT / "PyriteUtility"
if str(_PYRITE_UTILITY_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYRITE_UTILITY_ROOT))

from PyriteUtility.computer_vision.computer_vision_utility import (  # noqa: E402
    get_image_transform_with_border,
)


def resize_hwc_uint8(
    frames: np.ndarray, out_hw: tuple[int, int] = (224, 224)
) -> np.ndarray:
    """Resize (T,H,W,3) uint8 RGB using square-border + INTER_AREA (policy path)."""
    if frames.size == 0:
        return np.zeros((0, out_hw[0], out_hw[1], 3), dtype=np.uint8)
    _, hi, wi, _ = frames.shape
    oh, ow = out_hw
    transform = get_image_transform_with_border(
        input_res=(wi, hi), output_res=(ow, oh), bgr_to_rgb=False
    )
    return np.stack([transform(frame) for frame in frames], axis=0)


def resize_hwc_float(
    frames: np.ndarray, out_hw: tuple[int, int] = (224, 224)
) -> np.ndarray:
    """Resize (T,H,W,3) depth float16/float32 with nearest-neighbor interpolation."""
    if frames.size == 0:
        return np.zeros((0, out_hw[0], out_hw[1], 3), dtype=frames.dtype)
    oh, ow = out_hw
    out_dtype = frames.dtype
    resized = np.stack(
        [
            cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_NEAREST)
            for frame in frames
        ],
        axis=0,
    )
    if out_dtype == np.float16:
        resized = resized.astype(np.float16)
    return resized


def make_uniform_time_grid(t0: float, t1: float, hz: float) -> np.ndarray:
    """Return 1D seconds grid from t0 to t1 inclusive at uniform hz."""
    if hz <= 0:
        raise ValueError("hz must be positive")
    dt = 1.0 / hz
    n = int(np.round((t1 - t0) / dt)) + 1
    return t0 + np.arange(n, dtype=np.float64) * dt


def nearest_indices(src_t: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    """Map each query time to the nearest source index."""
    src_t = np.asarray(src_t, dtype=np.float64).reshape(-1)
    query_t = np.asarray(query_t, dtype=np.float64).reshape(-1)
    if src_t.size == 0:
        raise ValueError("src_t must be non-empty")
    idx = np.searchsorted(src_t, query_t, side="left")
    idx = np.clip(idx, 0, len(src_t) - 1)
    left = np.maximum(idx - 1, 0)
    right = idx
    pick_right = np.abs(src_t[right] - query_t) <= np.abs(src_t[left] - query_t)
    return np.where(pick_right, right, left).astype(np.int64)


def _resample_pose7(
    src_t: np.ndarray,
    pose7: np.ndarray,
    query_t: np.ndarray,
) -> np.ndarray:
    """Linear pos + SLERP quat onto query times. pose7 is [x,y,z,qw,qx,qy,qz]."""
    from scipy.spatial.transform import Rotation, Slerp

    src_t = np.asarray(src_t, dtype=np.float64).reshape(-1)
    query_t = np.asarray(query_t, dtype=np.float64).reshape(-1)
    pose7 = np.asarray(pose7, dtype=np.float64)
    out = np.zeros((len(query_t), 7), dtype=np.float64)
    for j in range(3):
        out[:, j] = np.interp(query_t, src_t, pose7[:, j])
    quat_xyzw = np.column_stack([pose7[:, 4], pose7[:, 5], pose7[:, 6], pose7[:, 3]])
    rots = Rotation.from_quat(quat_xyzw)
    t0, t1 = float(src_t[0]), float(src_t[-1])
    t_clip = np.clip(query_t, t0, t1)
    slerp = Slerp(src_t, rots)
    q = slerp(t_clip).as_quat()
    out[:, 3] = q[:, 3]
    out[:, 4] = q[:, 0]
    out[:, 5] = q[:, 1]
    out[:, 6] = q[:, 2]
    return out


def _as_col(ts: np.ndarray) -> np.ndarray:
    ts = np.asarray(ts, dtype=np.float64).reshape(-1)
    return ts.reshape(-1, 1)


def _nearest_sample(arr: np.ndarray, src_t: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    idx = nearest_indices(src_t, query_t)
    return np.asarray(arr[idx])


def build_map_to_scan_idx(
    ref_ts: np.ndarray, scan_gap_intervals: np.ndarray
) -> np.ndarray:
    """
    (T, 1) int64 array mapping each `ref_ts` timestep to the row index into
    scan_heatmap_0 of the most-recently-*completed* scan as of that timestep
    (largest row whose scan_gap_intervals[row, 1] <= ref_ts[i]), or -1 if no
    scan has completed yet.

    Mirrors `_build_map_to_scan_idx` in
    Robotiq-portable-real2sim/python/convert_recording_to_zarr.py (stage 1),
    adapted to this converter's (T, 1) int64 index-map convention (matches
    `_build_index_map`'s output shape for map_to_uw_idx_0/map_to_d_idx_0).
    `scan_gap_intervals` must already be expressed in the same time
    convention as `ref_ts` (callers are responsible for any rebasing).

    Kept here (rather than inline in the converter) so it can be reused both
    at initial conversion time and again after resample_episode_to_hz() moves
    the reference timeline onto a new grid.
    """
    ref = np.asarray(ref_ts, dtype=np.float64).reshape(-1)
    scan_gap_intervals = np.asarray(scan_gap_intervals, dtype=np.float64).reshape(-1, 2)
    n_scans = scan_gap_intervals.shape[0]
    if n_scans == 0:
        return np.full((ref.shape[0], 1), -1, dtype=np.int64)

    end_times = scan_gap_intervals[:, 1]
    order = np.argsort(end_times, kind="stable")
    sorted_ends = end_times[order]

    pos = np.searchsorted(sorted_ends, ref, side="right") - 1
    result = np.full(ref.shape[0], -1, dtype=np.int64)
    valid = pos >= 0
    result[valid] = order[pos[valid]]
    return result.reshape(-1, 1)


def resample_episode_to_hz(ep: dict[str, Any], hz: float = 10.0) -> dict[str, Any]:
    """
    Resample a unified-converter episode dict onto a uniform hz grid.

    Images and wrench use nearest-neighbor sampling; pose uses linear+SLERP.
    All timestamp keys are rewritten to the grid; map_to_* indices become identity.
    """
    rgb_ts = np.asarray(ep["rgb_time_stamps_0"], dtype=np.float64).reshape(-1)
    t0, t1 = float(rgb_ts[0]), float(rgb_ts[-1])
    grid = make_uniform_time_grid(t0, t1, hz)
    grid_col = _as_col(grid)
    n = len(grid)

    out = dict(ep)

    for key in ("rgb_0", "ultrawide_0", "depth_0"):
        ts_key = key.replace("_0", "_time_stamps_0")
        src_t = np.asarray(ep[ts_key], dtype=np.float64).reshape(-1)
        out[key] = _nearest_sample(ep[key], src_t, grid)

    robot_ts = np.asarray(ep["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
    out["ts_pose_fb_0"] = _resample_pose7(robot_ts, ep["ts_pose_fb_0"], grid)

    gripper_ts = np.asarray(ep["gripper_time_stamps_0"], dtype=np.float64).reshape(-1)
    out["gripper_0"] = _nearest_sample(ep["gripper_0"], gripper_ts, grid)

    wrench_ts_left = np.asarray(
        ep["wrench_time_stamps_left_0"], dtype=np.float64
    ).reshape(-1)
    wrench_ts_right = np.asarray(
        ep["wrench_time_stamps_right_0"], dtype=np.float64
    ).reshape(-1)
    wrench_key_ts = {
        "wrench_left_0": wrench_ts_left,
        "wrench_left_coinft_0": wrench_ts_left,
        "wrench_right_0": wrench_ts_right,
        "wrench_right_coinft_0": wrench_ts_right,
        "wrench_concat_0": wrench_ts_left,
        "wrench_concat_coinft_0": wrench_ts_left,
    }
    for key, src_t in wrench_key_ts.items():
        if key in ep:
            out[key] = _nearest_sample(ep[key], src_t, grid)

    timestamp_keys = [
        "rgb_time_stamps_0",
        "ultrawide_time_stamps_0",
        "depth_time_stamps_0",
        "robot_time_stamps_0",
        "gripper_time_stamps_0",
        "wrench_time_stamps_left_0",
        "wrench_time_stamps_right_0",
    ]
    for ts_key in timestamp_keys:
        if ts_key in out:
            out[ts_key] = grid_col.copy()

    out["map_to_uw_idx_0"] = np.arange(n, dtype=np.int64).reshape(-1, 1)
    out["map_to_d_idx_0"] = np.arange(n, dtype=np.int64).reshape(-1, 1)

    # scan_heatmap_0 / scan_gap_intervals_0 (if present) pass through unchanged above
    # (via `out = dict(ep)`): scan_heatmap_0 is indexed by scan count, not by time, and
    # scan_gap_intervals_0's [start,end] values don't depend on the sampling grid's
    # spacing, only on the (unchanged) [t0, t1] range. map_to_scan_idx_0 DOES depend on
    # the per-timestep reference grid though (its length and the times it's queried at
    # both change here), so it must be rebuilt fresh against the new `grid` -- otherwise
    # it would keep its pre-resample length/values, desyncing from every other key this
    # function just resampled to length `n`.
    if "map_to_scan_idx_0" in ep and "scan_gap_intervals_0" in ep:
        out["map_to_scan_idx_0"] = build_map_to_scan_idx(grid, ep["scan_gap_intervals_0"])

    return out

#!/usr/bin/env python3
"""Plotly 3D comparison of expert vs open-loop policy trajectories.

Overlays nominal + virtual-target paths from a training episode and the matching
``episode_N_policy`` open-loop export. Start/end time sliders select the visible
window; query markers and base→action arrows come from
``policy_open_loop_val_queries.json``. Expert poses inside scan-pause intervals
are omitted (paths may jump across gaps). Each policy chunk is drawn as its own
polyline (no connector to the next chunk).

Usage:
  python plot_open_loop_vs_expert_3d.py \
    --training_dataset_path /mnt/data/umidata/ft/stringer_data_v2_full/acp_replay_buffer_10hz_224.zarr \
    --open_loop_dataset_path /mnt/data/umidata/ft/training_outputs/data_source_test_bad_tags/umi/policy_open_loop_val.zarr \
    --episode 42

Expert-only (no policy export; scrub command + virtual target from the dataset):
  python plot_open_loop_vs_expert_3d.py \
    --training_dataset_path /path/to/acp_replay_buffer.zarr \
    --episode 0 \
    --expert_only \
    --show
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
import zarr

from PyriteUtility.spatial_math import spatial_utilities as su


AXIS_COLORS = ("#e74c3c", "#2ecc71", "#3498db")  # RGB = XYZ

# #region agent log
_DEBUG_LOG_PATH = "/home/boeing/umi_ft_ws/.cursor/debug-c83f21.log"


def _debug_log(hypothesis_id: str, message: str, data: dict) -> None:
    import time as _time

    entry = {
        "sessionId": "c83f21",
        "id": f"log_{int(_time.time() * 1000)}_{hypothesis_id}_{message[:8]}",
        "timestamp": int(_time.time() * 1000),
        "location": "plot_open_loop_vs_expert_3d.py",
        "message": message,
        "data": data,
        "hypothesisId": hypothesis_id,
    }
    try:
        with open(_DEBUG_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


# #endregion agent log


def parse_episode_id(episode: str) -> int:
    text = str(episode).strip()
    if text.isdigit():
        return int(text)
    match = re.search(r"(\d+)", text)
    if not match:
        raise ValueError(f"Could not parse episode id from {episode!r}")
    return int(match.group(1))


def resolve_episode_names(episode: str) -> Tuple[str, str]:
    epi_id = parse_episode_id(episode)
    return f"episode_{epi_id}", f"episode_{epi_id}_policy"


def scene_xyz_ranges(
    pts: np.ndarray, pad_frac: float = 0.15, min_half_extent: float = 0.02
) -> Tuple[List[float], List[float], List[float]]:
    lo = pts.min(axis=0).astype(np.float64)
    hi = pts.max(axis=0).astype(np.float64)
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    half = np.maximum(half * (1.0 + pad_frac), min_half_extent)
    half_eq = float(np.max(half))
    return (
        [float(center[0] - half_eq), float(center[0] + half_eq)],
        [float(center[1] - half_eq), float(center[1] + half_eq)],
        [float(center[2] - half_eq), float(center[2] + half_eq)],
    )


def sample_slider_indices(n: int, max_steps: int) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    max_steps = max(2, int(max_steps))
    if n <= max_steps:
        return np.arange(n, dtype=np.int64)
    idx = np.linspace(0, n - 1, max_steps)
    return np.unique(np.round(idx).astype(np.int64))


def _as_pose7(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim == 1:
        out = out.reshape(1, -1)
    if out.shape[-1] != 7:
        raise ValueError(f"Expected pose7 with last dim 7, got {out.shape}")
    return out


def _as_times(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64).reshape(-1)


def load_episode_poses(
    dataset_path: str,
    episode_name: str,
    robot_id: int = 0,
    prefer_raw_policy: bool = False,
) -> Dict[str, np.ndarray]:
    root = zarr.open(dataset_path, mode="r")
    if "data" not in root or episode_name not in root["data"]:
        raise KeyError(f"{episode_name} not found in {dataset_path}")
    ep = root["data"][episode_name]
    keys = set(ep.keys())

    cmd_key = f"ts_pose_command_{robot_id}"
    vt_key = f"ts_pose_virtual_target_{robot_id}"
    raw_cmd_key = f"ts_pose_command_raw_{robot_id}"
    raw_vt_key = f"ts_pose_virtual_target_raw_{robot_id}"
    fb_key = f"ts_pose_fb_{robot_id}"
    time_key = f"robot_time_stamps_{robot_id}"

    used_raw = False
    if prefer_raw_policy and raw_cmd_key in keys:
        command = _as_pose7(ep[raw_cmd_key][:])
        used_raw = True
        if raw_vt_key in keys:
            virtual = _as_pose7(ep[raw_vt_key][:])
        else:
            virtual = command.copy()
    elif cmd_key in keys:
        command = _as_pose7(ep[cmd_key][:])
        if vt_key in keys:
            virtual = _as_pose7(ep[vt_key][:])
        else:
            virtual = command.copy()
    elif fb_key in keys:
        command = _as_pose7(ep[fb_key][:])
        virtual = command.copy()
    else:
        raise KeyError(
            f"{episode_name} missing {cmd_key} (and {fb_key}) in {dataset_path}"
        )

    if time_key not in keys:
        raise KeyError(f"{episode_name} missing {time_key} in {dataset_path}")
    times = _as_times(ep[time_key][:])
    n = min(command.shape[0], virtual.shape[0], times.shape[0])
    return {
        "command": command[:n],
        "virtual": virtual[:n],
        "times_s": times[:n],
        "used_raw_policy": used_raw,
    }


def policy_poses_from_query_records(
    queries: Sequence[Dict[str, Any]], n_policy: int, key: str
) -> Optional[np.ndarray]:
    """Build (N, 7) policy pose array from per-query raw pose lists in metadata."""
    if not queries or key not in queries[0]:
        return None
    out = np.full((n_policy, 7), np.nan, dtype=np.float64)
    for q in queries:
        start = int(q["open_loop_start_idx"])
        end = int(q["open_loop_end_idx"])
        poses = np.asarray(q[key], dtype=np.float64)
        if poses.ndim != 2 or poses.shape[1] != 7:
            return None
        n = min(end - start, poses.shape[0])
        out[start : start + n] = poses[:n]
    if np.any(~np.isfinite(out)):
        return None
    return out


def default_queries_path(open_loop_dataset_path: str) -> str:
    parent = os.path.dirname(open_loop_dataset_path.rstrip("/"))
    return os.path.join(parent, "policy_open_loop_val_queries.json")


def load_query_records(
    queries_path: str, policy_episode_name: str
) -> List[Dict[str, Any]]:
    if not os.path.isfile(queries_path):
        raise FileNotFoundError(
            f"Query metadata not found: {queries_path}\n"
            "Re-run run_policy_open_loop_val_export.py to write "
            "policy_open_loop_val_queries.json."
        )
    with open(queries_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    episodes = payload.get("episodes", {})
    if policy_episode_name not in episodes:
        raise KeyError(
            f"{policy_episode_name} missing from {queries_path}. "
            f"Available: {sorted(episodes.keys())}"
        )
    return list(episodes[policy_episode_name])


def build_open_loop_reveal_times(
    queries: Sequence[Dict[str, Any]], n_policy: int
) -> np.ndarray:
    """Expert-clock reveal time for each open-loop waypoint.

    Open-loop export only keeps ``exec_h`` sparse steps per query, while the
    next val-sampler query is typically much later on the expert timeline
    (often ~0.2s+). Revealing with raw ``action_offsets_s`` (starting at 0 with
    spacing ``action_ds * raw_dt``) therefore freezes the policy tip for most of
    each inter-query gap and makes the policy look lagged behind the expert.

    For scrubbing, spread each query's executed waypoints evenly across
    ``[expert_robot_time_s, next_query_time)`` so the policy advances with the
    expert between replans. Physical ``action_offsets_s`` remain in the JSON for
    scheduling truth; this remapping is display-only.
    """
    reveal = np.full(n_policy, np.inf, dtype=np.float64)
    if not queries:
        return reveal

    q_times = [float(q["expert_robot_time_s"]) for q in queries]
    n_q = len(queries)

    for qi, q in enumerate(queries):
        start = int(q["open_loop_start_idx"])
        end = int(q["open_loop_end_idx"])
        n_steps = end - start
        if n_steps <= 0:
            continue

        t0 = q_times[qi]
        if qi + 1 < n_q:
            t1 = q_times[qi + 1]
        elif qi > 0:
            # Last query: reuse previous inter-query spacing.
            t1 = t0 + max(q_times[qi] - q_times[qi - 1], 1e-6)
        else:
            offsets = np.asarray(
                q.get("action_offsets_s", []), dtype=np.float64
            ).reshape(-1)
            if offsets.size >= 2:
                t1 = t0 + float(offsets[-1] + (offsets[1] - offsets[0]))
            elif offsets.size == 1:
                t1 = t0 + max(float(offsets[0]), 1e-3)
            else:
                t1 = t0 + 1e-3 * n_steps

        span = max(float(t1 - t0), 1e-9)
        for j, ol_idx in enumerate(range(start, end)):
            if ol_idx < 0 or ol_idx >= n_policy:
                continue
            # First step at t0; last step just before next query.
            reveal[ol_idx] = t0 + (j / float(n_steps)) * span
    return reveal


def load_scan_gap_intervals(
    dataset_path: str, episode_name: str
) -> np.ndarray:
    """Return (N, 2) [start, end] scan-pause intervals, or empty (0, 2)."""
    root = zarr.open(dataset_path, mode="r")
    if "data" not in root or episode_name not in root["data"]:
        return np.zeros((0, 2), dtype=np.float64)
    ep = root["data"][episode_name]
    for container in (ep, ep.get("obs")):
        if container is None:
            continue
        if "scan_gap_intervals_0" in container:
            gaps = np.asarray(container["scan_gap_intervals_0"], dtype=np.float64)
            if gaps.size == 0:
                return np.zeros((0, 2), dtype=np.float64)
            return gaps.reshape(-1, 2)
    return np.zeros((0, 2), dtype=np.float64)


def time_in_scan_gap(t: float, scan_gap_intervals: np.ndarray) -> bool:
    if scan_gap_intervals.size == 0:
        return False
    for start, end in scan_gap_intervals.reshape(-1, 2):
        if float(start) <= t <= float(end):
            return True
    return False


def path_xyz_time_window(
    poses: np.ndarray,
    times: np.ndarray,
    t_start: float,
    t_end: float,
    scan_gap_intervals: Optional[np.ndarray] = None,
) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    """Pose polyline for ``[t_start, t_end]``, skipping scan-pause samples."""
    xs: List[Optional[float]] = []
    ys: List[Optional[float]] = []
    zs: List[Optional[float]] = []
    if poses.shape[0] == 0 or times.size == 0:
        return xs, ys, zs

    gaps = (
        np.zeros((0, 2), dtype=np.float64)
        if scan_gap_intervals is None
        else np.asarray(scan_gap_intervals, dtype=np.float64).reshape(-1, 2)
    )
    after_gap = False
    for i in range(min(poses.shape[0], times.size)):
        t = float(times[i])
        if t < t_start or t > t_end:
            continue
        if gaps.size and time_in_scan_gap(t, gaps):
            after_gap = True
            continue
        if after_gap and xs and xs[-1] is not None:
            xs.append(None)
            ys.append(None)
            zs.append(None)
        after_gap = False
        xs.append(float(poses[i, 0]))
        ys.append(float(poses[i, 1]))
        zs.append(float(poses[i, 2]))
    return xs, ys, zs


def path_xyz_policy_chunks(
    poses: np.ndarray,
    queries: Sequence[Dict[str, Any]],
    reveal_times: np.ndarray,
    t_start: float,
    t_end: float,
) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    """Policy polyline with None breaks between open-loop chunks."""
    xs: List[Optional[float]] = []
    ys: List[Optional[float]] = []
    zs: List[Optional[float]] = []
    if poses.shape[0] == 0:
        return xs, ys, zs

    for q in queries:
        t_query = float(q["expert_robot_time_s"])
        if t_query < t_start or t_query > t_end:
            continue
        start = int(q["open_loop_start_idx"])
        end = int(q["open_loop_end_idx"])
        chunk_x: List[float] = []
        chunk_y: List[float] = []
        chunk_z: List[float] = []
        for ol_idx in range(start, end):
            if ol_idx < 0 or ol_idx >= poses.shape[0]:
                break
            rt = float(reveal_times[ol_idx]) if ol_idx < reveal_times.size else np.nan
            if not np.isfinite(rt) or rt < t_start or rt > t_end:
                continue
            chunk_x.append(float(poses[ol_idx, 0]))
            chunk_y.append(float(poses[ol_idx, 1]))
            chunk_z.append(float(poses[ol_idx, 2]))
        if not chunk_x:
            continue
        xs.extend(chunk_x)
        ys.extend(chunk_y)
        zs.extend(chunk_z)
        xs.append(None)
        ys.append(None)
        zs.append(None)

    if xs and xs[-1] is None:
        xs.pop()
        ys.pop()
        zs.pop()
    return xs, ys, zs


def tip_pose_in_window(
    poses: np.ndarray,
    times: np.ndarray,
    t_end: float,
    scan_gap_intervals: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    if poses.shape[0] == 0 or times.size == 0:
        return None
    gaps = (
        np.zeros((0, 2), dtype=np.float64)
        if scan_gap_intervals is None
        else np.asarray(scan_gap_intervals, dtype=np.float64).reshape(-1, 2)
    )
    best_i = None
    best_t = -np.inf
    for i in range(min(poses.shape[0], times.size)):
        t = float(times[i])
        if t > t_end:
            break
        if gaps.size and time_in_scan_gap(t, gaps):
            continue
        if t >= best_t:
            best_t = t
            best_i = i
    if best_i is None:
        return None
    return np.asarray(poses[best_i], dtype=np.float64)


def policy_tip_pose_in_window(
    poses: np.ndarray,
    reveal_times: np.ndarray,
    t_end: float,
) -> Optional[np.ndarray]:
    if poses.shape[0] == 0 or reveal_times.size == 0:
        return None
    visible = np.where(np.isfinite(reveal_times) & (reveal_times <= t_end))[0]
    if visible.size == 0:
        return None
    return np.asarray(poses[int(visible[-1])], dtype=np.float64)


def visible_queries_in_window(
    queries: Sequence[Dict[str, Any]], t_start: float, t_end: float
) -> List[Dict[str, Any]]:
    return [
        q
        for q in queries
        if t_start <= float(q["expert_robot_time_s"]) <= t_end
    ]


def make_time_slider_steps(times: np.ndarray, max_steps: int) -> List[Dict[str, Any]]:
    indices = sample_slider_indices(len(times), max_steps)
    steps = []
    for idx in indices:
        t = float(times[int(idx)])
        # No-op step; plotly_sliderchange handler performs the real update.
        steps.append(
            {
                "args": [{"uirevision": "open-loop-vs-expert"}],
                "label": f"{t:.2f}s",
                "method": "relayout",
            }
        )
    return steps


def downsample_indices(n: int, max_points: int) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    max_points = max(2, int(max_points))
    if n <= max_points:
        return np.arange(n, dtype=np.int64)
    idx = np.linspace(0, n - 1, max_points)
    return np.unique(np.round(idx).astype(np.int64))


def triad_segments_to_trace_lists(
    pose7: Optional[np.ndarray], axis_length: float
) -> List[Dict[str, List[float]]]:
    out: List[Dict[str, List[float]]] = []
    for axis_i in range(3):
        xyz = triad_trace_xyz(pose7, axis_length, axis_i)
        out.append({"x": xyz[0], "y": xyz[1], "z": xyz[2]})
    return out


def build_interactive_payload(
    expert: Dict[str, np.ndarray],
    policy: Dict[str, np.ndarray],
    queries: Sequence[Dict[str, Any]],
    reveal_times: np.ndarray,
    scan_gap_intervals: np.ndarray,
    time_slider_idx: np.ndarray,
    axis_length: float,
    title: str,
    show_policy: bool,
    max_display_points: int = 1500,
) -> Dict[str, Any]:
    """Compact plot source data; the HTML slider filters this client-side."""
    expert_times = expert["times_s"]
    ds_idx = downsample_indices(len(expert_times), max_display_points)
    slider_times = [float(expert_times[int(i)]) for i in time_slider_idx]

    policy_chunks: List[Dict[str, Any]] = []
    for q in queries:
        start = int(q["open_loop_start_idx"])
        end = int(q["open_loop_end_idx"])
        chunk_times: List[float] = []
        chunk_cmd: List[List[float]] = []
        chunk_vt: List[List[float]] = []
        for ol_idx in range(start, end):
            if ol_idx < 0 or ol_idx >= policy["command"].shape[0]:
                break
            rt = float(reveal_times[ol_idx]) if ol_idx < reveal_times.size else np.nan
            if not np.isfinite(rt):
                continue
            chunk_times.append(rt)
            chunk_cmd.append(
                np.asarray(policy["command"][ol_idx, :3], dtype=np.float64).tolist()
            )
            chunk_vt.append(
                np.asarray(policy["virtual"][ol_idx, :3], dtype=np.float64).tolist()
            )
        if chunk_times:
            policy_chunks.append(
                {
                    "queryTime": float(q["expert_robot_time_s"]),
                    "revealTimes": chunk_times,
                    "command": chunk_cmd,
                    "virtual": chunk_vt,
                }
            )

    query_markers = []
    for q in queries:
        pose = np.asarray(q["base_pose7"], dtype=np.float64).reshape(-1)
        query_markers.append(
            {
                "queryTime": float(q["expert_robot_time_s"]),
                "queryIdx": int(q["query_idx"]),
                "x": float(pose[0]),
                "y": float(pose[1]),
                "z": float(pose[2]),
                "hover": (
                    f"query {q['query_idx']}<br>"
                    f"t={float(q['expert_robot_time_s']):.3f}s"
                ),
            }
        )

    arrow_queries: List[Dict[str, Any]] = []
    n_expert = int(expert["command"].shape[0])
    frame_mismatch_err_m: List[float] = []
    for q in queries:
        # Arrow origin must match the SE3 base used to decode policy actions
        # (query base_pose7), not whatever training zarr the plot loaded for the
        # expert path. Mixing T265 vs tag-corrected ACP frames produces the
        # classic "starburst" of long green arrows.
        base = np.asarray(q["base_pose7"], dtype=np.float64).reshape(-1)
        base_cmd = base[:3].copy()
        base_vt = base[:3].copy()
        pose_idx = int(q.get("expert_pose_index", -1))
        if 0 <= pose_idx < n_expert:
            expert_cmd = np.asarray(expert["command"][pose_idx, :3], dtype=np.float64)
            frame_mismatch_err_m.append(float(np.linalg.norm(expert_cmd - base_cmd)))
            # Prefer expert virtual target at the same index when it agrees with
            # the decode base frame (small translation error).
            if frame_mismatch_err_m[-1] < 0.02:
                base_vt = np.asarray(expert["virtual"][pose_idx, :3], dtype=np.float64)

        start = int(q["open_loop_start_idx"])
        end = int(q["open_loop_end_idx"])
        step_times: List[float] = []
        targets_cmd: List[List[float]] = []
        targets_vt: List[List[float]] = []
        raw_cmd = q.get("policy_pose_command_raw")
        raw_vt = q.get("policy_pose_virtual_target_raw")
        for ol_idx in range(start, end):
            if ol_idx < 0 or ol_idx >= policy["command"].shape[0]:
                break
            rt = float(reveal_times[ol_idx]) if ol_idx < reveal_times.size else np.nan
            if not np.isfinite(rt):
                continue
            step_i = ol_idx - start
            if raw_cmd is not None and step_i < len(raw_cmd):
                target_cmd = np.asarray(raw_cmd[step_i], dtype=np.float64)[:3]
            else:
                target_cmd = np.asarray(policy["command"][ol_idx, :3], dtype=np.float64)
            if raw_vt is not None and step_i < len(raw_vt):
                target_vt = np.asarray(raw_vt[step_i], dtype=np.float64)[:3]
            else:
                target_vt = np.asarray(policy["virtual"][ol_idx, :3], dtype=np.float64)
            step_times.append(rt)
            targets_cmd.append(target_cmd.tolist())
            targets_vt.append(target_vt.tolist())
        if step_times:
            arrow_queries.append(
                {
                    "queryTime": float(q["expert_robot_time_s"]),
                    "baseCmd": base_cmd.tolist(),
                    "baseVt": base_vt.tolist(),
                    "revealTimes": step_times,
                    "targetsCmd": targets_cmd,
                    "targetsVt": targets_vt,
                }
            )

    gaps = np.asarray(scan_gap_intervals, dtype=np.float64).reshape(-1, 2)
    tips_by_end_idx: List[List[Dict[str, List[float]]]] = []
    for t_end in slider_times:
        tips_by_end_idx.append(
            triad_segments_to_trace_lists(
                tip_pose_in_window(expert["command"], expert_times, t_end, gaps),
                axis_length,
            )
            + triad_segments_to_trace_lists(
                policy_tip_pose_in_window(policy["command"], reveal_times, t_end),
                axis_length,
            )
            + triad_segments_to_trace_lists(
                tip_pose_in_window(expert["virtual"], expert_times, t_end, gaps),
                axis_length,
            )
            + triad_segments_to_trace_lists(
                policy_tip_pose_in_window(policy["virtual"], reveal_times, t_end),
                axis_length,
            )
        )

    if frame_mismatch_err_m:
        err = np.asarray(frame_mismatch_err_m, dtype=np.float64)
        if float(np.median(err)) > 0.05:
            print(
                "[plot] WARNING: expert path frame disagrees with open-loop decode "
                f"bases (median ||expert[pose_idx]-base_pose7||={float(np.median(err)):.3f}m, "
                f"max={float(np.max(err)):.3f}m). Pass the same --training_dataset_path "
                "used for export (see policy_open_loop_val_metrics.json). Arrows use "
                "base_pose7; the grey expert path may still be in the wrong frame."
            )

    return {
        "sliderTimes": slider_times,
        "expertTimes": np.asarray(expert_times[ds_idx], dtype=np.float64).tolist(),
        "expertCommand": np.asarray(expert["command"][ds_idx, :3], dtype=np.float64).tolist(),
        "expertVirtual": np.asarray(expert["virtual"][ds_idx, :3], dtype=np.float64).tolist(),
        "scanGaps": gaps.tolist(),
        "policyChunks": policy_chunks,
        "queryMarkers": query_markers,
        "arrowQueries": arrow_queries,
        "tipsByEndIdx": tips_by_end_idx,
        "baseTitle": title,
        "showPolicy": bool(show_policy),
        "traceIds": list(range(19)),
    }


def build_window_update_script() -> str:
    return """
(function() {
  var DATA = window.OPEN_LOOP_PLOT_DATA;
  if (!DATA) { return; }

  function boot() {
    var gd = document.getElementsByClassName('plotly-graph-div')[0];
    if (!gd || !gd.layout || !gd.layout.sliders || gd.layout.sliders.length < 2) {
      setTimeout(boot, 50);
      return;
    }
    var traceIds = DATA.traceIds;

  function inScanGap(t) {
    var gaps = DATA.scanGaps || [];
    for (var i = 0; i < gaps.length; i++) {
      if (t >= gaps[i][0] && t <= gaps[i][1]) { return true; }
    }
    return false;
  }

  function expertPath(times, poses, tStart, tEnd) {
    var xs = [], ys = [], zs = [], afterGap = false;
    for (var i = 0; i < times.length; i++) {
      var t = times[i];
      if (t < tStart || t > tEnd) { continue; }
      if (inScanGap(t)) { afterGap = true; continue; }
      if (afterGap && xs.length && xs[xs.length - 1] !== null) {
        xs.push(null); ys.push(null); zs.push(null);
      }
      afterGap = false;
      xs.push(poses[i][0]); ys.push(poses[i][1]); zs.push(poses[i][2]);
    }
    return {x: xs, y: ys, z: zs};
  }

  function policyPath(chunks, tStart, tEnd, key) {
    var xs = [], ys = [], zs = [];
    for (var ci = 0; ci < chunks.length; ci++) {
      var ch = chunks[ci];
      if (ch.queryTime < tStart || ch.queryTime > tEnd) { continue; }
      var cx = [], cy = [], cz = [];
      for (var j = 0; j < ch.revealTimes.length; j++) {
        var rt = ch.revealTimes[j];
        if (rt < tStart || rt > tEnd) { continue; }
        var p = ch[key][j];
        cx.push(p[0]); cy.push(p[1]); cz.push(p[2]);
      }
      if (!cx.length) { continue; }
      xs = xs.concat(cx); ys = ys.concat(cy); zs = zs.concat(cz);
      xs.push(null); ys.push(null); zs.push(null);
    }
    if (xs.length && xs[xs.length - 1] === null) {
      xs.pop(); ys.pop(); zs.pop();
    }
    return {x: xs, y: ys, z: zs};
  }

  function queryMarkers(tStart, tEnd) {
    var xs = [], ys = [], zs = [], text = [], hovertext = [];
    var markers = DATA.queryMarkers || [];
    for (var i = 0; i < markers.length; i++) {
      var q = markers[i];
      if (q.queryTime < tStart || q.queryTime > tEnd) { continue; }
      xs.push(q.x); ys.push(q.y); zs.push(q.z);
      text.push(String(q.queryIdx));
      hovertext.push(q.hover);
    }
    return {x: xs, y: ys, z: zs, text: text, hovertext: hovertext};
  }

  function actionArrows(tStart, tEnd, keyBase, keyTargets) {
    var xs = [], ys = [], zs = [];
    var arrows = DATA.arrowQueries || [];
    for (var i = 0; i < arrows.length; i++) {
      var q = arrows[i];
      if (q.queryTime < tStart || q.queryTime > tEnd) { continue; }
      var base = q[keyBase];
      var targets = q[keyTargets];
      for (var j = 0; j < q.revealTimes.length; j++) {
        var rt = q.revealTimes[j];
        if (rt < tStart || rt > tEnd) { continue; }
        var target = targets[j];
        xs.push(base[0], target[0], null);
        ys.push(base[1], target[1], null);
        zs.push(base[2], target[2], null);
      }
    }
    return {x: xs, y: ys, z: zs};
  }

  function applyWindow(startIdx, endIdx) {
    var times = DATA.sliderTimes || [];
    if (!times.length) { return; }
    startIdx = Math.max(0, Math.min(times.length - 1, startIdx));
    endIdx = Math.max(0, Math.min(times.length - 1, endIdx));
    if (endIdx < startIdx) { endIdx = startIdx; }

    var tStart = times[startIdx];
    var tEnd = times[endIdx];
    var expertNom = expertPath(DATA.expertTimes, DATA.expertCommand, tStart, tEnd);
    var expertVt = expertPath(DATA.expertTimes, DATA.expertVirtual, tStart, tEnd);
    var policyNom = policyPath(DATA.policyChunks || [], tStart, tEnd, 'command');
    var policyVt = policyPath(DATA.policyChunks || [], tStart, tEnd, 'virtual');
    var queries = queryMarkers(tStart, tEnd);
    var axNom = actionArrows(tStart, tEnd, 'baseCmd', 'targetsCmd');
    var axVt = actionArrows(tStart, tEnd, 'baseVt', 'targetsVt');
    var tips = (DATA.tipsByEndIdx && DATA.tipsByEndIdx[endIdx]) || [];

    var traceUpdates = [
      expertNom, policyNom, expertVt, policyVt, queries, axNom, axVt
    ];
    for (var ti = 0; ti < tips.length; ti++) {
      traceUpdates.push(tips[ti]);
    }
    while (traceUpdates.length < traceIds.length) {
      traceUpdates.push({x: [], y: [], z: []});
    }

    var restyle = {x: [], y: [], z: []};
    for (var i = 0; i < traceIds.length; i++) {
      var tr = traceUpdates[i] || {x: [], y: [], z: []};
      restyle.x.push(tr.x || []);
      restyle.y.push(tr.y || []);
      restyle.z.push(tr.z || []);
    }
    Plotly.restyle(gd, restyle, traceIds);
    Plotly.restyle(
      gd,
      {text: [queries.text], hovertext: [queries.hovertext]},
      [traceIds[4]]
    );
    Plotly.relayout(gd, {
      title: {text: DATA.baseTitle + ' [' + tStart.toFixed(2) + 's – ' + tEnd.toFixed(2) + 's]'}
    });
  }

  function onSliderChange() {
    var startIdx = gd.layout.sliders[0].active || 0;
    var endIdx = gd.layout.sliders[1].active || (DATA.sliderTimes.length - 1);
    applyWindow(startIdx, endIdx);
  }

  gd.on('plotly_sliderchange', onSliderChange);
  gd.on('plotly_relayout', function(evt) {
    if (!evt) { return; }
    var keys = Object.keys(evt);
    if (keys.length === 1 && evt.uirevision === 'open-loop-vs-expert') {
      onSliderChange();
    }
  });
  onSliderChange();
  }

  boot();
})();
"""


def write_figure_html(fig: go.Figure, output_path: str, payload: Dict[str, Any]) -> None:
    payload_json = json.dumps(payload, separators=(",", ":"))
    post_script = (
        f"window.OPEN_LOOP_PLOT_DATA = {payload_json};\n"
        + build_window_update_script()
    )
    fig.write_html(
        output_path,
        include_plotlyjs="cdn",
        auto_open=False,
        post_script=post_script,
    )
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[plot] HTML size: {size_mb:.2f} MiB")



def triad_segments(
    pose7: np.ndarray, axis_length: float
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return [(origin, end_x), (origin, end_y), (origin, end_z)]."""
    se3 = su.pose7_to_SE3(np.asarray(pose7, dtype=np.float64).reshape(1, 7))[0]
    origin = se3[:3, 3]
    segs = []
    for i in range(3):
        end = origin + axis_length * se3[:3, i]
        segs.append((origin.copy(), end.copy()))
    return segs


def empty_xyz() -> Tuple[List[float], List[float], List[float]]:
    return [], [], []


def path_xyz(poses: np.ndarray, n: int) -> Tuple[List[float], List[float], List[float]]:
    if n <= 0 or poses.shape[0] == 0:
        return empty_xyz()
    n = min(n, poses.shape[0])
    return (
        poses[:n, 0].tolist(),
        poses[:n, 1].tolist(),
        poses[:n, 2].tolist(),
    )


def query_marker_xyz(
    queries: Sequence[Dict[str, Any]],
) -> Tuple[List[float], List[float], List[float], List[str]]:
    xs, ys, zs, texts = [], [], [], []
    for q in queries:
        pose = np.asarray(q["base_pose7"], dtype=np.float64).reshape(-1)
        xs.append(float(pose[0]))
        ys.append(float(pose[1]))
        zs.append(float(pose[2]))
        texts.append(
            f"query {q['query_idx']}<br>t={float(q['expert_robot_time_s']):.3f}s"
        )
    return xs, ys, zs, texts


def action_arrow_xyz(
    queries: Sequence[Dict[str, Any]],
    expert_poses: np.ndarray,
    policy_poses: np.ndarray,
    t_start: float,
    t_end: float,
    reveal_times: Optional[np.ndarray] = None,
    policy_pose_key: Optional[str] = None,
) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    """None-separated segments from expert pose at query → policy actions in window.

    ``expert_poses`` should be the matching stream (nominal or virtual) so arrows
    originate at the expert nominal / virtual pose for that query.

    When ``policy_pose_key`` is set (e.g. ``policy_pose_command_raw``), targets
    come from that per-query JSON field (raw policy decode) instead of
    ``policy_poses`` indexing — preferred so arrows show the true policy action.
    """
    xs: List[Optional[float]] = []
    ys: List[Optional[float]] = []
    zs: List[Optional[float]] = []
    n_expert = int(expert_poses.shape[0])
    for q in queries:
        t_query = float(q["expert_robot_time_s"])
        if t_query < t_start or t_query > t_end:
            continue
        # Prefer decode base (base_pose7). Only fall back to expert_poses when
        # they agree, so a T265 expert zarr cannot pull arrows into the wrong frame.
        base = np.asarray(q["base_pose7"], dtype=np.float64).reshape(-1)
        pose_idx = int(q.get("expert_pose_index", -1))
        if 0 <= pose_idx < n_expert:
            expert_base = np.asarray(expert_poses[pose_idx], dtype=np.float64).reshape(-1)
            if np.linalg.norm(expert_base[:3] - base[:3]) < 0.02:
                base = expert_base
        start = int(q["open_loop_start_idx"])
        end = int(q["open_loop_end_idx"])
        offsets = np.asarray(q.get("action_offsets_s", []), dtype=np.float64).reshape(-1)

        raw_targets = None
        if policy_pose_key is not None and policy_pose_key in q:
            raw_targets = np.asarray(q[policy_pose_key], dtype=np.float64)
            if raw_targets.ndim != 2 or raw_targets.shape[1] < 3:
                raw_targets = None

        for i, ol_idx in enumerate(range(start, end)):
            if reveal_times is not None and ol_idx < len(reveal_times):
                rt = float(reveal_times[ol_idx])
                if not np.isfinite(rt) or rt < t_start or rt > t_end:
                    continue
            else:
                off = float(offsets[i]) if i < offsets.shape[0] else 0.0
                rt = t_query + off
                if rt < t_start or rt > t_end:
                    continue

            if raw_targets is not None and i < raw_targets.shape[0]:
                target = raw_targets[i, :3]
            elif 0 <= ol_idx < policy_poses.shape[0]:
                target = policy_poses[ol_idx, :3]
            else:
                continue
            xs.extend([float(base[0]), float(target[0]), None])
            ys.extend([float(base[1]), float(target[1]), None])
            zs.extend([float(base[2]), float(target[2]), None])
    return xs, ys, zs


def triad_trace_xyz(
    pose7: Optional[np.ndarray], axis_length: float, axis_i: int
) -> Tuple[List[float], List[float], List[float]]:
    if pose7 is None:
        return empty_xyz()
    segs = triad_segments(pose7, axis_length)
    a, b = segs[axis_i]
    return [float(a[0]), float(b[0])], [float(a[1]), float(b[1])], [
        float(a[2]),
        float(b[2]),
    ]


def estimate_axis_length(pts: np.ndarray, frac: float = 0.04) -> float:
    if pts.size == 0:
        return 0.02
    extent = float(np.max(pts.max(axis=0) - pts.min(axis=0)))
    return max(extent * frac, 0.01)


def empty_policy_poses() -> Dict[str, np.ndarray]:
    """Placeholder policy bundle for expert-only plotting."""
    return {
        "command": np.zeros((0, 7), dtype=np.float64),
        "virtual": np.zeros((0, 7), dtype=np.float64),
        "times_s": np.zeros((0,), dtype=np.float64),
        "used_raw_policy": False,
    }


def build_figure(
    expert: Dict[str, np.ndarray],
    policy: Dict[str, np.ndarray],
    queries: Sequence[Dict[str, Any]],
    title: str,
    max_slider_steps: int = 300,
    use_raw_policy_actions: bool = True,
    expert_only: bool = False,
    scan_gap_intervals: Optional[np.ndarray] = None,
    max_display_points: int = 1500,
) -> Tuple[go.Figure, Dict[str, Any]]:
    expert_times = expert["times_s"]
    if expert_only:
        policy = empty_policy_poses()
        queries = []
        use_raw_policy_actions = False
    gaps = (
        np.zeros((0, 2), dtype=np.float64)
        if scan_gap_intervals is None
        else np.asarray(scan_gap_intervals, dtype=np.float64).reshape(-1, 2)
    )
    reveal_times = build_open_loop_reveal_times(queries, policy["command"].shape[0])

    cmd_key = "policy_pose_command_raw" if use_raw_policy_actions else None
    vt_key = "policy_pose_virtual_target_raw" if use_raw_policy_actions else None
    if use_raw_policy_actions and not expert_only:
        raw_cmd = policy_poses_from_query_records(
            queries, policy["command"].shape[0], "policy_pose_command_raw"
        )
        raw_vt = policy_poses_from_query_records(
            queries, policy["command"].shape[0], "policy_pose_virtual_target_raw"
        )
        if raw_cmd is not None:
            policy = dict(policy)
            policy["command"] = raw_cmd
            if raw_vt is not None:
                policy["virtual"] = raw_vt
            print("[plot] Using raw policy poses from query metadata (pre-ensemble).")
        elif policy.get("used_raw_policy"):
            print("[plot] Using raw policy poses from zarr ts_pose_*_raw_*.")
        else:
            print(
                "[plot] WARNING: raw policy poses not found; plotting zarr "
                "ts_pose_command/virtual_target (may be temporally ensembled). "
                "Re-run export to log raw actions."
            )

    xyz_parts = [expert["command"][:, :3], expert["virtual"][:, :3]]
    if not expert_only and policy["command"].shape[0] > 0:
        xyz_parts.extend([policy["command"][:, :3], policy["virtual"][:, :3]])
    all_xyz = np.concatenate(xyz_parts, axis=0)
    x_range, y_range, z_range = scene_xyz_ranges(all_xyz)
    axis_length = estimate_axis_length(all_xyz)

    window_grid_steps = max(2, int(max_slider_steps))
    time_slider_idx = sample_slider_indices(len(expert_times), window_grid_steps)
    if time_slider_idx.size == 0:
        raise RuntimeError("Expert episode has no timestamps to scrub")
    time_labels = [f"{float(expert_times[int(i)]):.2f}s" for i in time_slider_idx]
    n_time_steps = int(time_slider_idx.size)

    def window_trace_payload(t_start: float, t_end: float) -> Dict[int, Dict[str, Any]]:
        q_vis = visible_queries_in_window(queries, t_start, t_end)
        qx, qy, qz, qtext = query_marker_xyz(q_vis)
        expert_nom_path = path_xyz_time_window(
            expert["command"], expert_times, t_start, t_end, gaps
        )
        expert_vt_path = path_xyz_time_window(
            expert["virtual"], expert_times, t_start, t_end, gaps
        )
        policy_nom_path = path_xyz_policy_chunks(
            policy["command"], queries, reveal_times, t_start, t_end
        )
        policy_vt_path = path_xyz_policy_chunks(
            policy["virtual"], queries, reveal_times, t_start, t_end
        )
        ax_n = action_arrow_xyz(
            queries,
            expert["command"],
            policy["command"],
            t_start,
            t_end,
            reveal_times=reveal_times,
            policy_pose_key=cmd_key,
        )
        ax_v = action_arrow_xyz(
            queries,
            expert["virtual"],
            policy["virtual"],
            t_start,
            t_end,
            reveal_times=reveal_times,
            policy_pose_key=vt_key,
        )
        expert_nom_tip = tip_pose_in_window(
            expert["command"], expert_times, t_end, gaps
        )
        expert_vt_tip = tip_pose_in_window(
            expert["virtual"], expert_times, t_end, gaps
        )
        policy_nom_tip = policy_tip_pose_in_window(
            policy["command"], reveal_times, t_end
        )
        policy_vt_tip = policy_tip_pose_in_window(
            policy["virtual"], reveal_times, t_end
        )

        payload: Dict[int, Dict[str, Any]] = {
            0: {"x": expert_nom_path[0], "y": expert_nom_path[1], "z": expert_nom_path[2]},
            1: {"x": policy_nom_path[0], "y": policy_nom_path[1], "z": policy_nom_path[2]},
            2: {"x": expert_vt_path[0], "y": expert_vt_path[1], "z": expert_vt_path[2]},
            3: {"x": policy_vt_path[0], "y": policy_vt_path[1], "z": policy_vt_path[2]},
            4: {
                "x": qx,
                "y": qy,
                "z": qz,
                "text": [str(q["query_idx"]) for q in q_vis],
                "hovertext": qtext,
            },
            5: {"x": ax_n[0], "y": ax_n[1], "z": ax_n[2]},
            6: {"x": ax_v[0], "y": ax_v[1], "z": ax_v[2]},
        }
        trace_i = 7
        for pose7 in (
            expert_nom_tip,
            policy_nom_tip,
            expert_vt_tip,
            policy_vt_tip,
        ):
            for axis_i in range(3):
                xyz = triad_trace_xyz(pose7, axis_length, axis_i)
                payload[trace_i] = {"x": xyz[0], "y": xyz[1], "z": xyz[2]}
                trace_i += 1
        return payload

    t_start0 = float(expert_times[int(time_slider_idx[0])])
    t_end0 = float(expert_times[int(time_slider_idx[-1])])
    initial = window_trace_payload(t_start0, t_end0)

    traces: List[Any] = []

    def add_path(name: str, color: str, dash: str, trace_id: int, legendgroup: str):
        tr = initial[trace_id]
        traces.append(
            go.Scatter3d(
                x=tr["x"],
                y=tr["y"],
                z=tr["z"],
                mode="lines",
                name=name,
                line=dict(color=color, width=5, dash=dash),
                legendgroup=legendgroup,
                legendgrouptitle_text=legendgroup,
                showlegend=True,
            )
        )

    def add_tip_triad(
        trace_start: int,
        name_prefix: str,
        legendgroup: str,
        showlegend: bool,
    ) -> None:
        for axis_i, color in enumerate(AXIS_COLORS):
            tr = initial[trace_start + axis_i]
            traces.append(
                go.Scatter3d(
                    x=tr["x"],
                    y=tr["y"],
                    z=tr["z"],
                    mode="lines",
                    name=f"{name_prefix} tip",
                    line=dict(color=color, width=6),
                    legendgroup=legendgroup,
                    legendgrouptitle_text=legendgroup,
                    showlegend=(showlegend and axis_i == 0),
                )
            )

    show_policy = not expert_only
    add_path("expert nominal", "#7f8c8d", "solid", 0, "Nominal")
    add_path("policy nominal", "#f1c40f", "solid", 1, "Nominal")
    if expert_only:
        traces[-1].update(showlegend=False, visible=False)
    add_path("expert virtual", "#95a5a6", "dot", 2, "Virtual")
    add_path("policy virtual", "#e67e22", "dash", 3, "Virtual")
    if expert_only:
        traces[-1].update(showlegend=False, visible=False)

    q0 = initial[4]
    traces.append(
        go.Scatter3d(
            x=q0["x"],
            y=q0["y"],
            z=q0["z"],
            mode="markers+text",
            name="query starts",
            text=q0.get("text", []),
            textposition="top center",
            marker=dict(
                size=6,
                color="#9b59b6",
                symbol="diamond",
                line=dict(width=1, color="#ffffff"),
            ),
            hovertext=q0.get("hovertext", []),
            hoverinfo="text",
            legendgroup="Queries",
            legendgrouptitle_text="Queries",
            showlegend=show_policy,
            visible=True if show_policy else False,
        )
    )
    for trace_id, name, color, dash, group in (
        (5, "action arrows (nominal)", "#27ae60", None, "Nominal"),
        (6, "action arrows (virtual)", "#16a085", "dot", "Virtual"),
    ):
        tr = initial[trace_id]
        line = dict(color=color, width=4 if dash is None else 3)
        if dash is not None:
            line["dash"] = dash
        traces.append(
            go.Scatter3d(
                x=tr["x"],
                y=tr["y"],
                z=tr["z"],
                mode="lines",
                name=name,
                line=line,
                legendgroup=group,
                legendgrouptitle_text=group,
                showlegend=show_policy,
                visible=True if show_policy else False,
            )
        )

    add_tip_triad(7, "expert nominal", "Nominal", showlegend=True)
    add_tip_triad(10, "policy nominal", "Nominal", showlegend=show_policy)
    if expert_only:
        for tr in traces[-3:]:
            tr.update(showlegend=False, visible=False)
    add_tip_triad(13, "expert virtual", "Virtual", showlegend=True)
    add_tip_triad(16, "policy virtual", "Virtual", showlegend=show_policy)
    if expert_only:
        for tr in traces[-3:]:
            tr.update(showlegend=False, visible=False)

    payload = build_interactive_payload(
        expert=expert,
        policy=policy,
        queries=queries,
        reveal_times=reveal_times,
        scan_gap_intervals=gaps,
        time_slider_idx=time_slider_idx,
        axis_length=axis_length,
        title=title,
        show_policy=show_policy,
        max_display_points=max_display_points,
    )

    range_title = f"{title} [{time_labels[0]} – {time_labels[-1]}]"
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=range_title, x=0.01, xanchor="left"),
        height=900,
        width=1200,
        margin=dict(t=70, r=240, b=140, l=40),
        uirevision="open-loop-vs-expert",
        legend=dict(
            orientation="v",
            x=1.02,
            y=1.0,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#cccccc",
            borderwidth=1,
            itemclick="toggle",
            itemdoubleclick="toggleothers",
            groupclick="togglegroup",
        ),
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="cube",
            uirevision="open-loop-vs-expert-3d",
            xaxis=dict(range=x_range, autorange=False),
            yaxis=dict(range=y_range, autorange=False),
            zaxis=dict(range=z_range, autorange=False),
        ),
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "start t = ", "xanchor": "left"},
                "pad": {"t": 40, "b": 0},
                "x": 0.05,
                "len": 0.75,
                "xanchor": "left",
                "y": 0.02,
                "yanchor": "bottom",
                "steps": make_time_slider_steps(expert_times, window_grid_steps),
            },
            {
                "active": n_time_steps - 1,
                "currentvalue": {"prefix": "end t = ", "xanchor": "left"},
                "pad": {"t": 10, "b": 10},
                "x": 0.05,
                "len": 0.75,
                "xanchor": "left",
                "y": 0.08,
                "yanchor": "bottom",
                "steps": make_time_slider_steps(expert_times, window_grid_steps),
            },
        ],
    )
    if gaps.size:
        print(
            f"[plot] Omitting expert poses inside {gaps.shape[0]} scan-gap interval(s)"
        )
    print(
        f"[plot] Expert display downsample: "
        f"{len(expert_times)} -> {len(payload['expertTimes'])} points, "
        f"{len(queries)} queries, {len(payload['policyChunks'])} policy chunks"
    )
    return fig, payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3D Plotly overlay of expert vs open-loop policy trajectories."
    )
    parser.add_argument("--training_dataset_path", required=True, type=str)
    parser.add_argument(
        "--open_loop_dataset_path",
        default="",
        type=str,
        help="Open-loop policy zarr (required unless --expert_only).",
    )
    parser.add_argument(
        "--episode",
        required=True,
        type=str,
        help="Episode id or name, e.g. 44 / episode_44 / episode_44_policy",
    )
    parser.add_argument(
        "--expert_only",
        action="store_true",
        help=(
            "Plot expert command + virtual target only (no open-loop zarr / "
            "queries). Uses the same Plotly scrubber."
        ),
    )
    parser.add_argument(
        "--queries_path",
        default="",
        type=str,
        help="Query metadata JSON (default: beside open-loop zarr).",
    )
    parser.add_argument(
        "--output_html",
        default="",
        type=str,
        help="Output HTML path (default: beside training or open-loop zarr).",
    )
    parser.add_argument("--robot_id", default=0, type=int)
    parser.add_argument("--max_slider_steps", default=80, type=int)
    parser.add_argument(
        "--max_display_points",
        default=1500,
        type=int,
        help="Max expert trajectory points embedded in HTML (display only).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the figure in a browser in addition to writing HTML.",
    )
    args = parser.parse_args(argv)
    if not args.expert_only and not args.open_loop_dataset_path:
        parser.error("--open_loop_dataset_path is required unless --expert_only")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    expert_name, policy_name = resolve_episode_names(args.episode)

    print(f"[plot] expert={expert_name} from {args.training_dataset_path}")
    expert = load_episode_poses(
        args.training_dataset_path, expert_name, robot_id=args.robot_id
    )
    scan_gap_intervals = load_scan_gap_intervals(
        args.training_dataset_path, expert_name
    )

    if args.expert_only:
        policy = empty_policy_poses()
        queries: List[Dict[str, Any]] = []
        print("[plot] expert_only=True (skipping open-loop / queries)")
        print(f"[plot] expert steps={expert['command'].shape[0]}")
        title = f"{expert_name}: expert command vs virtual target"
        out_html = args.output_html or os.path.join(
            os.path.dirname(os.path.abspath(args.training_dataset_path.rstrip("/"))),
            f"{expert_name}_expert_only_3d.html",
        )
    else:
        queries_path = args.queries_path or default_queries_path(
            args.open_loop_dataset_path
        )
        print(f"[plot] policy={policy_name} from {args.open_loop_dataset_path}")
        print(f"[plot] queries={queries_path}")
        policy = load_episode_poses(
            args.open_loop_dataset_path,
            policy_name,
            robot_id=args.robot_id,
            prefer_raw_policy=True,
        )
        queries = load_query_records(queries_path, policy_name)
        print(
            f"[plot] expert steps={expert['command'].shape[0]}, "
            f"policy steps={policy['command'].shape[0]}, queries={len(queries)}, "
            f"zarr_raw={policy.get('used_raw_policy', False)}"
        )
        title = (
            f"Open-loop vs expert: {expert_name} "
            f"(raw policy actions; per-chunk polylines; start/end window)"
        )
        out_html = args.output_html or os.path.join(
            os.path.dirname(os.path.abspath(args.open_loop_dataset_path.rstrip("/"))),
            f"open_loop_vs_expert_{expert_name}.html",
        )

        # #region agent log
        try:
            n_expert = int(expert["command"].shape[0])
            for q in queries:
                pose_idx = int(q.get("expert_pose_index", -1))
                if 0 <= pose_idx < n_expert:
                    base_cmd = np.asarray(
                        expert["command"][pose_idx], dtype=np.float64
                    )[:3]
                    base_source = "expert_command_at_pose_idx"
                else:
                    base_cmd = np.asarray(q["base_pose7"], dtype=np.float64)[:3]
                    base_source = "base_pose7_fallback"
                base_pose7 = np.asarray(q["base_pose7"], dtype=np.float64)[:3]

                raw_targets = np.asarray(q["policy_pose_command_raw"], dtype=np.float64)
                nominal_arrow_len_m = np.linalg.norm(
                    raw_targets[:, :3] - base_cmd[None, :], axis=-1
                ).tolist()

                raw_vt_targets = np.asarray(
                    q["policy_pose_virtual_target_raw"], dtype=np.float64
                )
                n_expert_v = int(expert["virtual"].shape[0])
                if 0 <= pose_idx < n_expert_v:
                    base_vt = np.asarray(
                        expert["virtual"][pose_idx], dtype=np.float64
                    )[:3]
                else:
                    base_vt = base_pose7
                vt_arrow_len_m = np.linalg.norm(
                    raw_vt_targets[:, :3] - base_vt[None, :], axis=-1
                ).tolist()

                _debug_log(
                    "H4_H5",
                    "per-query arrow length as actually drawn",
                    {
                        "query_idx": int(q["query_idx"]),
                        "expert_pose_index": pose_idx,
                        "base_source": base_source,
                        "base_pose7_feedback_m": base_pose7.tolist(),
                        "base_command_used_for_arrow_m": base_cmd.tolist(),
                        "base_feedback_vs_command_err_m": float(
                            np.linalg.norm(base_pose7 - base_cmd)
                        ),
                        "nominal_arrow_len_m": nominal_arrow_len_m,
                        "vt_arrow_len_m": vt_arrow_len_m,
                    },
                )
        except Exception as e:
            _debug_log("ERR", "plot instrumentation failure", {"error": str(e)})
        # #endregion agent log

    fig, payload = build_figure(
        expert=expert,
        policy=policy,
        queries=queries,
        title=title,
        max_slider_steps=args.max_slider_steps,
        use_raw_policy_actions=not args.expert_only,
        expert_only=args.expert_only,
        scan_gap_intervals=scan_gap_intervals,
        max_display_points=args.max_display_points,
    )

    os.makedirs(os.path.dirname(os.path.abspath(out_html)) or ".", exist_ok=True)
    write_figure_html(fig, out_html, payload)
    print(f"[plot] Wrote {out_html}")
    if args.show:
        fig.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

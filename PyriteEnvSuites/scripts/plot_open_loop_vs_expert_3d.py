#!/usr/bin/env python3
"""Plotly 3D comparison of expert vs open-loop policy trajectories.

Overlays nominal + virtual-target paths from a training episode and the matching
``episode_N_policy`` open-loop export. Expert timestamps drive the scrubber;
query markers and base→action arrows come from ``policy_open_loop_val_queries.json``.

Usage:
  python plot_open_loop_vs_expert_3d.py \\
    --training_dataset_path /path/to/train.zarr \\
    --open_loop_dataset_path /path/to/policy_open_loop_val.zarr \\
    --episode 44
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


def prefix_len_by_time(times_s: np.ndarray, t: float) -> int:
    if times_s.size == 0:
        return 0
    return int(np.searchsorted(times_s, t, side="right"))


def policy_prefix_len(reveal_times: np.ndarray, t: float) -> int:
    if reveal_times.size == 0:
        return 0
    visible = np.where(np.isfinite(reveal_times) & (reveal_times <= t))[0]
    if visible.size == 0:
        return 0
    return int(visible[-1]) + 1


def visible_queries(
    queries: Sequence[Dict[str, Any]], t: float
) -> List[Dict[str, Any]]:
    return [q for q in queries if float(q["expert_robot_time_s"]) <= t]


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
    t: float,
    reveal_times: Optional[np.ndarray] = None,
    policy_pose_key: Optional[str] = None,
) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    """None-separated segments from expert pose at query → revealed policy actions.

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
        if t_query > t:
            continue
        pose_idx = int(q.get("expert_pose_index", -1))
        if pose_idx < 0 or pose_idx >= n_expert:
            base = np.asarray(q["base_pose7"], dtype=np.float64).reshape(-1)
        else:
            base = np.asarray(expert_poses[pose_idx], dtype=np.float64).reshape(-1)
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
                if not np.isfinite(reveal_times[ol_idx]) or reveal_times[ol_idx] > t:
                    continue
            else:
                off = float(offsets[i]) if i < offsets.shape[0] else 0.0
                if t_query + off > t:
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


def build_figure(
    expert: Dict[str, np.ndarray],
    policy: Dict[str, np.ndarray],
    queries: Sequence[Dict[str, Any]],
    title: str,
    max_slider_steps: int = 300,
    use_raw_policy_actions: bool = True,
) -> go.Figure:
    expert_times = expert["times_s"]
    reveal_times = build_open_loop_reveal_times(queries, policy["command"].shape[0])

    # Prefer per-query raw decoded poses from metadata when plotting policy.
    cmd_key = "policy_pose_command_raw" if use_raw_policy_actions else None
    vt_key = "policy_pose_virtual_target_raw" if use_raw_policy_actions else None
    if use_raw_policy_actions:
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

    all_xyz = np.concatenate(
        [
            expert["command"][:, :3],
            expert["virtual"][:, :3],
            policy["command"][:, :3],
            policy["virtual"][:, :3],
        ],
        axis=0,
    )
    x_range, y_range, z_range = scene_xyz_ranges(all_xyz)
    axis_length = estimate_axis_length(all_xyz)

    slider_idx = sample_slider_indices(len(expert_times), max_slider_steps)
    if slider_idx.size == 0:
        raise RuntimeError("Expert episode has no timestamps to scrub")

    def state_at(expert_i: int):
        t = float(expert_times[expert_i])
        e_n = expert_i + 1
        p_n = policy_prefix_len(reveal_times, t)
        q_vis = visible_queries(queries, t)
        expert_nom_tip = expert["command"][expert_i]
        expert_vt_tip = expert["virtual"][expert_i]
        policy_nom_tip = policy["command"][p_n - 1] if p_n > 0 else None
        policy_vt_tip = policy["virtual"][p_n - 1] if p_n > 0 else None
        return (
            t,
            e_n,
            p_n,
            q_vis,
            expert_nom_tip,
            policy_nom_tip,
            expert_vt_tip,
            policy_vt_tip,
        )

    (
        t0,
        e0,
        p0,
        q0,
        expert_nom_tip0,
        policy_nom_tip0,
        expert_vt_tip0,
        policy_vt_tip0,
    ) = state_at(int(slider_idx[0]))
    qx, qy, qz, qtext = query_marker_xyz(q0)
    ax_n = action_arrow_xyz(
        queries,
        expert["command"],
        policy["command"],
        t0,
        reveal_times=reveal_times,
        policy_pose_key=cmd_key,
    )
    ax_v = action_arrow_xyz(
        queries,
        expert["virtual"],
        policy["virtual"],
        t0,
        reveal_times=reveal_times,
        policy_pose_key=vt_key,
    )

    traces: List[Any] = []

    def add_path(name: str, color: str, dash: str, xyz, legendgroup: str):
        traces.append(
            go.Scatter3d(
                x=xyz[0],
                y=xyz[1],
                z=xyz[2],
                mode="lines",
                name=name,
                line=dict(color=color, width=5, dash=dash),
                legendgroup=legendgroup,
                legendgrouptitle_text=legendgroup,
                showlegend=True,
            )
        )

    def add_tip_triad(
        pose7: Optional[np.ndarray],
        name_prefix: str,
        legendgroup: str,
        showlegend: bool,
    ) -> None:
        for axis_i, color in enumerate(AXIS_COLORS):
            xyz = triad_trace_xyz(pose7, axis_length, axis_i)
            traces.append(
                go.Scatter3d(
                    x=xyz[0],
                    y=xyz[1],
                    z=xyz[2],
                    mode="lines",
                    name=f"{name_prefix} tip",
                    line=dict(color=color, width=6),
                    legendgroup=legendgroup,
                    legendgrouptitle_text=legendgroup,
                    showlegend=(showlegend and axis_i == 0),
                )
            )

    # Group by nominal vs virtual so one legend click shows/hides both streams.
    add_path(
        "expert nominal",
        "#7f8c8d",
        "solid",
        path_xyz(expert["command"], e0),
        "Nominal",
    )
    add_path(
        "policy nominal",
        "#f1c40f",
        "solid",
        path_xyz(policy["command"], p0),
        "Nominal",
    )
    add_path(
        "expert virtual",
        "#95a5a6",
        "dot",
        path_xyz(expert["virtual"], e0),
        "Virtual",
    )
    add_path(
        "policy virtual",
        "#e67e22",
        "dash",
        path_xyz(policy["virtual"], p0),
        "Virtual",
    )

    traces.append(
        go.Scatter3d(
            x=qx,
            y=qy,
            z=qz,
            mode="markers+text",
            name="query starts",
            text=[str(q["query_idx"]) for q in q0],
            textposition="top center",
            marker=dict(
                size=6,
                color="#9b59b6",
                symbol="diamond",
                line=dict(width=1, color="#ffffff"),
            ),
            hovertext=qtext,
            hoverinfo="text",
            legendgroup="Queries",
            legendgrouptitle_text="Queries",
            showlegend=True,
        )
    )
    traces.append(
        go.Scatter3d(
            x=ax_n[0],
            y=ax_n[1],
            z=ax_n[2],
            mode="lines",
            name="action arrows (nominal)",
            line=dict(color="#27ae60", width=4),
            legendgroup="Nominal",
            legendgrouptitle_text="Nominal",
            showlegend=True,
        )
    )
    traces.append(
        go.Scatter3d(
            x=ax_v[0],
            y=ax_v[1],
            z=ax_v[2],
            mode="lines",
            name="action arrows (virtual)",
            line=dict(color="#16a085", width=3, dash="dot"),
            legendgroup="Virtual",
            legendgrouptitle_text="Virtual",
            showlegend=True,
        )
    )

    # Tips: RGB axes; legend entry once per tip under Nominal / Virtual.
    add_tip_triad(expert_nom_tip0, "expert nominal", "Nominal", showlegend=True)
    add_tip_triad(policy_nom_tip0, "policy nominal", "Nominal", showlegend=True)
    add_tip_triad(expert_vt_tip0, "expert virtual", "Virtual", showlegend=True)
    add_tip_triad(policy_vt_tip0, "policy virtual", "Virtual", showlegend=True)

    fig = go.Figure(data=traces)
    # Trace order: 0-3 paths, 4 queries, 5-6 arrows,
    # 7-9 expert nom tip, 10-12 policy nom tip,
    # 13-15 expert vt tip, 16-18 policy vt tip
    n_traces = 19
    frame_trace_ids = list(range(n_traces))

    def append_tip_frame_data(
        frame_data: List[Any], pose7: Optional[np.ndarray]
    ) -> None:
        for axis_i in range(3):
            xyz = triad_trace_xyz(pose7, axis_length, axis_i)
            frame_data.append(go.Scatter3d(x=xyz[0], y=xyz[1], z=xyz[2]))

    frames = []
    slider_steps = []
    for step_i, expert_i in enumerate(slider_idx):
        expert_i = int(expert_i)
        (
            t,
            e_n,
            p_n,
            q_vis,
            expert_nom_tip,
            policy_nom_tip,
            expert_vt_tip,
            policy_vt_tip,
        ) = state_at(expert_i)
        qx, qy, qz, qtext = query_marker_xyz(q_vis)
        ax_n = action_arrow_xyz(
            queries,
            expert["command"],
            policy["command"],
            t,
            reveal_times=reveal_times,
            policy_pose_key=cmd_key,
        )
        ax_v = action_arrow_xyz(
            queries,
            expert["virtual"],
            policy["virtual"],
            t,
            reveal_times=reveal_times,
            policy_pose_key=vt_key,
        )
        frame_data = [
            go.Scatter3d(**dict(zip(("x", "y", "z"), path_xyz(expert["command"], e_n)))),
            go.Scatter3d(**dict(zip(("x", "y", "z"), path_xyz(policy["command"], p_n)))),
            go.Scatter3d(**dict(zip(("x", "y", "z"), path_xyz(expert["virtual"], e_n)))),
            go.Scatter3d(**dict(zip(("x", "y", "z"), path_xyz(policy["virtual"], p_n)))),
            go.Scatter3d(
                x=qx,
                y=qy,
                z=qz,
                text=[str(q["query_idx"]) for q in q_vis],
                hovertext=qtext,
            ),
            go.Scatter3d(x=ax_n[0], y=ax_n[1], z=ax_n[2]),
            go.Scatter3d(x=ax_v[0], y=ax_v[1], z=ax_v[2]),
        ]
        append_tip_frame_data(frame_data, expert_nom_tip)
        append_tip_frame_data(frame_data, policy_nom_tip)
        append_tip_frame_data(frame_data, expert_vt_tip)
        append_tip_frame_data(frame_data, policy_vt_tip)

        frames.append(
            go.Frame(name=str(step_i), data=frame_data, traces=frame_trace_ids)
        )
        slider_steps.append(
            {
                "args": [
                    [str(step_i)],
                    {
                        "frame": {"duration": 0, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    },
                ],
                "label": f"{t:.2f}s",
                "method": "animate",
            }
        )

    fig.frames = frames
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left"),
        height=900,
        width=1200,
        margin=dict(t=70, r=240, b=120, l=40),
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
                "currentvalue": {"prefix": "expert t = ", "xanchor": "left"},
                "pad": {"t": 30, "b": 10},
                "x": 0.05,
                "len": 0.75,
                "xanchor": "left",
                "y": 0.0,
                "yanchor": "top",
                "steps": slider_steps,
            }
        ],
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "showactive": False,
                "x": 0.92,
                "y": 0.0,
                "xanchor": "right",
                "yanchor": "top",
                "pad": {"t": 30, "r": 10},
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 40, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
    )
    return fig


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3D Plotly overlay of expert vs open-loop policy trajectories."
    )
    parser.add_argument("--training_dataset_path", required=True, type=str)
    parser.add_argument("--open_loop_dataset_path", required=True, type=str)
    parser.add_argument(
        "--episode",
        required=True,
        type=str,
        help="Episode id or name, e.g. 44 / episode_44 / episode_44_policy",
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
        help="Output HTML path (default: beside open-loop zarr).",
    )
    parser.add_argument("--robot_id", default=0, type=int)
    parser.add_argument("--max_slider_steps", default=300, type=int)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the figure in a browser in addition to writing HTML.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    expert_name, policy_name = resolve_episode_names(args.episode)
    queries_path = args.queries_path or default_queries_path(args.open_loop_dataset_path)

    print(f"[plot] expert={expert_name} from {args.training_dataset_path}")
    print(f"[plot] policy={policy_name} from {args.open_loop_dataset_path}")
    print(f"[plot] queries={queries_path}")

    expert = load_episode_poses(
        args.training_dataset_path, expert_name, robot_id=args.robot_id
    )
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

    # #region agent log
    try:
        n_expert = int(expert["command"].shape[0])
        for q in queries:
            pose_idx = int(q.get("expert_pose_index", -1))
            if 0 <= pose_idx < n_expert:
                base_cmd = np.asarray(expert["command"][pose_idx], dtype=np.float64)[:3]
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
                base_vt = np.asarray(expert["virtual"][pose_idx], dtype=np.float64)[:3]
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

    fig = build_figure(
        expert=expert,
        policy=policy,
        queries=queries,
        title=(
            f"Open-loop vs expert: {expert_name} "
            f"(raw policy actions; scrub stretched across inter-query gaps)"
        ),
        max_slider_steps=args.max_slider_steps,
        use_raw_policy_actions=True,
    )

    output_html = args.output_html
    if not output_html:
        parent = os.path.dirname(args.open_loop_dataset_path.rstrip("/"))
        output_html = os.path.join(
            parent, f"open_loop_vs_expert_{expert_name}.html"
        )
    os.makedirs(os.path.dirname(os.path.abspath(output_html)) or ".", exist_ok=True)
    fig.write_html(output_html, include_plotlyjs="cdn", auto_open=False)
    print(f"[plot] Wrote {output_html}")
    if args.show:
        fig.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

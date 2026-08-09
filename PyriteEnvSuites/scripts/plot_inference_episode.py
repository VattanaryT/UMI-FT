"""Plot a full inference episode from per-horizon control logs (Plotly).

Chains executed nominal/virtual targets across horizons, overlays episode JSON
pose/wrench feedback, and shows policy RGB/depth for the selected time.

The bottom slider scrubs over action timestamps (smooth), holds RGB/depth from
the latest replan, and truncates the 3D trajectory to the current time.

Usage:
  export PYRITE_CONTROL_LOG_FOLDERS=$HOME/data/control_log
  python plot_inference_episode.py --log_subdir episode_1783810767
  python plot_inference_episode.py --log_path /path/to/episode_log --html /tmp/view.html

Also writes a side-by-side RGB+depth MP4 at 15 FPS next to the HTML output.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import plotly.graph_objects as go
import yaml
import zarr
from plotly.subplots import make_subplots

from PyriteUtility.spatial_math import spatial_utilities as su


def sorted_horizon_keys(log) -> List[str]:
    keys = [k for k in log.keys() if k.startswith("horizon_")]
    return sorted(keys, key=lambda k: int(k.split("_", 1)[1]))


def resolve_log_path(args) -> str:
    if args.log_path:
        return args.log_path
    if "PYRITE_CONTROL_LOG_FOLDERS" not in os.environ:
        raise ValueError(
            "Set PYRITE_CONTROL_LOG_FOLDERS or pass --log_path / --log_subdir"
        )
    return os.path.join(os.environ["PYRITE_CONTROL_LOG_FOLDERS"], args.log_subdir)


def _parse_manip_json_array(text: str) -> list:
    """Parse manip_server JSON array (trailing commas, possibly incomplete)."""
    text = re.sub(r",\s*([}\]])", r"\1", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        trimmed = text.rstrip().rstrip(",")
        if not trimmed.endswith("]"):
            last_close = trimmed.rfind("}")
            if last_close < 0:
                raise
            trimmed = trimmed[: last_close + 1] + "\n]"
        return json.loads(trimmed)


def load_pose7_offset_tc(hardware_config_path: Optional[str]) -> Optional[np.ndarray]:
    """Legacy SE3_TC from pose7_offset_TC, or None for umift_std_v2 (UR TCP only)."""
    if not hardware_config_path:
        print("[plot] No hardware_config_path; poses assumed TCP (umift_std_v2).")
        return None
    if not os.path.isfile(hardware_config_path):
        print(
            f"[plot] hardware_config_path not found: {hardware_config_path}; "
            "poses assumed TCP (umift_std_v2)."
        )
        return None
    try:
        with open(hardware_config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"[plot] Failed to read hardware config ({exc}); poses assumed TCP.")
        return None

    pose7 = cfg.get("pose7_offset_TC")
    if pose7 is None:
        print("[plot] pose7_offset_TC absent; poses assumed TCP (umift_std_v2).")
        return None

    pose7 = np.asarray(pose7, dtype=np.float64).reshape(-1)
    if pose7.size != 7:
        print(f"[plot] pose7_offset_TC length {pose7.size}; skipping.")
        return None

    # umift_std_v2: identity / zero translation — no software flange↔TCP.
    if np.linalg.norm(pose7[:3]) < 1e-6:
        print("[plot] pose7_offset_TC identity; UR/RTDE TCP only (umift_std_v2).")
        return None

    se3_tc = su.pose7_to_SE3(pose7)
    print(
        f"[plot] LEGACY: correcting JSON pose_fb using "
        f"pose7_offset_TC={pose7.tolist()} from {hardware_config_path}"
    )
    return se3_tc


def compliance_pose_to_tool_pose(
    pose_fb_compliance: np.ndarray, se3_tc: np.ndarray
) -> np.ndarray:
    """Convert logged compliance-frame poses (WC) back to tool-frame (WT).

    manip_server logs SE3_WC = SE3_WT @ SE3_TC after overwriting pose_fb.
    Invert: SE3_WT = SE3_WC @ inv(SE3_TC).
    """
    se3_wc = su.pose7_to_SE3(pose_fb_compliance)
    se3_ct = su.SE3_inv(se3_tc)
    se3_wt = se3_wc @ se3_ct
    return su.SE3_to_pose7(se3_wt)


def load_robot_episode_json(
    episode_folder: str,
    robot_id: int,
    se3_tc: Optional[np.ndarray] = None,
) -> Optional[dict]:
    if not episode_folder:
        return None
    robot_path = os.path.join(episode_folder, f"robot_data_{robot_id}.json")
    if not os.path.isfile(robot_path):
        return None
    with open(robot_path, "r", encoding="utf-8") as f:
        text = f.read()
    records = _parse_manip_json_array(text)
    if not records:
        return None

    # manip_server robot_time_stamps are milliseconds.
    times = np.asarray([r["robot_time_stamps"] for r in records], dtype=np.float64)
    times = times / 1000.0
    times = times - times[0]
    pose_fb = np.asarray([r["ts_pose_fb"] for r in records], dtype=np.float64)
    wrench = np.asarray([r["robot_wrench"] for r in records], dtype=np.float64)

    # JSON logs compliance-frame pose; policy targets use tool-frame get_pose().
    if se3_tc is not None:
        pose_fb = compliance_pose_to_tool_pose(pose_fb, se3_tc)

    return {"time_s": times, "pose_fb": pose_fb, "wrench": wrench}


def compliant_stiffness_from_blocks(stiffness_6x6n: np.ndarray) -> np.ndarray:
    """Min eigenvalue of each translational 3x3 block (= compliant-direction k)."""
    assert stiffness_6x6n.ndim == 2 and stiffness_6x6n.shape[0] == 6
    n = stiffness_6x6n.shape[1] // 6
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        s_trans = stiffness_6x6n[:3, 6 * i : 6 * i + 3]
        out[i] = float(np.min(np.linalg.eigvalsh(s_trans)))
    return out


def chain_horizon_actions(
    log,
    horizon_keys: List[str],
    robot_id: int,
    drop_boundary: bool = True,
) -> Dict[str, np.ndarray]:
    nominal_chunks = []
    virtual_chunks = []
    stiffness_chunks = []
    time_chunks = []
    horizon_start_idx = []
    horizon_times_abs = []

    for i, horizon_key in enumerate(horizon_keys):
        action = log[horizon_key]["action"]
        nom = np.asarray(action[f"ts_nominal_targets_{robot_id}"][:], dtype=np.float64)
        virt = np.asarray(action[f"ts_virtual_targets_{robot_id}"][:], dtype=np.float64)
        ts = np.asarray(action["timestamps_s"][:], dtype=np.float64).reshape(-1)
        stiff_key = f"ts_stiffnesses_{robot_id}"
        if stiff_key in action:
            stiff = np.asarray(action[stiff_key][:], dtype=np.float64)
            k_comp = compliant_stiffness_from_blocks(stiff)
        else:
            k_comp = np.full(nom.shape[0], np.nan, dtype=np.float64)

        horizon_times_abs.append(float(ts[0]) if ts.size else np.nan)
        if drop_boundary and i > 0 and nominal_chunks:
            nom = nom[1:]
            virt = virt[1:]
            ts = ts[1:]
            k_comp = k_comp[1:]

        horizon_start_idx.append(sum(c.shape[0] for c in nominal_chunks))
        nominal_chunks.append(nom)
        virtual_chunks.append(virt)
        stiffness_chunks.append(k_comp)
        time_chunks.append(ts)

    nominal = np.concatenate(nominal_chunks, axis=0)
    virtual = np.concatenate(virtual_chunks, axis=0)
    stiffness = np.concatenate(stiffness_chunks, axis=0)
    timestamps = np.concatenate(time_chunks, axis=0)
    t0 = float(timestamps[0]) if timestamps.size else 0.0
    timestamps = timestamps - t0
    horizon_times = np.asarray(horizon_times_abs, dtype=np.float64) - t0
    horizon_start_idx = np.asarray(horizon_start_idx, dtype=np.int64)
    horizon_end_idx = np.empty_like(horizon_start_idx)
    horizon_end_idx[:-1] = horizon_start_idx[1:]
    horizon_end_idx[-1] = nominal.shape[0]

    return {
        "nominal": nominal,
        "virtual": virtual,
        "stiffness": stiffness,
        "timestamps_s": timestamps,
        "horizon_times_s": horizon_times,
        "horizon_start_idx": horizon_start_idx,
        "horizon_end_idx": horizon_end_idx,
    }


def tensor_to_image(arr: np.ndarray) -> np.ndarray:
    """Convert policy tensor (T,C,H,W) or (C,H,W) to displayable RGB uint8."""
    if arr.ndim == 4:
        arr = arr[-1]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D or 4D image tensor, got shape {arr.shape}")

    if arr.shape[0] in (1, 3):
        img = np.transpose(arr, (1, 2, 0))
    else:
        img = arr

    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)

    img = img.astype(np.float32)
    if np.nanmax(img) <= 1.0:
        img = img * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def rgb_buffer_frame_to_image(frame: np.ndarray) -> np.ndarray:
    """Convert one RGB buffer frame (HWC uint8 or CHW float) to display RGB."""
    frame = np.asarray(frame)
    if frame.ndim == 3 and frame.shape[-1] == 3 and frame.dtype == np.uint8:
        return frame
    return tensor_to_image(frame)


def depth_buffer_frame_to_image(frame: np.ndarray, depth_clip: float = 0.5) -> np.ndarray:
    """Convert one depth buffer frame to grayscale RGB uint8."""
    depth = np.asarray(frame, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[0] in (1, 3):
        return tensor_to_image(depth)
    if depth.ndim == 3 and depth.shape[-1] == 3:
        depth = depth[..., 0]
    depth_norm = np.clip(depth / depth_clip, 0.0, 1.0)
    gray = (depth_norm * 255.0).astype(np.uint8)
    return np.repeat(gray[:, :, np.newaxis], 3, axis=2)


def _dedupe_vision_timeline(
    images: List[np.ndarray], times_s: List[float], eps: float = 1e-4
) -> Tuple[List[np.ndarray], np.ndarray]:
    if not images:
        return [], np.zeros(0, dtype=np.float64)
    order = np.argsort(times_s, kind="stable")
    out_images: List[np.ndarray] = []
    out_times: List[float] = []
    for idx in order:
        t = float(times_s[idx])
        if out_times and abs(t - out_times[-1]) < eps:
            out_images[-1] = images[idx]
            out_times[-1] = t
            continue
        out_images.append(images[idx])
        out_times.append(t)
    return out_images, np.asarray(out_times, dtype=np.float64)


def _append_horizon_policy_vision_fallback(
    obs_policy,
    horizon_time_s: float,
    episode_action_t0: float,
    rgb_images: List[np.ndarray],
    rgb_times: List[float],
    depth_images: List[np.ndarray],
    depth_times: List[float],
    depth_clip: float,
) -> None:
    t_rel = float(horizon_time_s - episode_action_t0)
    if "rgb_0" in obs_policy:
        arr = np.asarray(obs_policy["rgb_0"][:])
        rgb_images.append(tensor_to_image(arr))
        rgb_times.append(t_rel)
    elif "ultrawide_0" in obs_policy:
        arr = np.asarray(obs_policy["ultrawide_0"][:])
        rgb_images.append(tensor_to_image(arr))
        rgb_times.append(t_rel)
    if "depth_0" in obs_policy:
        arr = np.asarray(obs_policy["depth_0"][:])
        depth_images.append(depth_buffer_frame_to_image(arr[-1] if arr.ndim == 4 else arr, depth_clip))
        depth_times.append(t_rel)


def _append_horizon_raw_vision(
    obs_inference,
    robot_id: int,
    episode_action_t0: float,
    rgb_images: List[np.ndarray],
    rgb_times: List[float],
    depth_images: List[np.ndarray],
    depth_times: List[float],
    depth_clip: float,
) -> None:
    rgb_key = f"rgb_{robot_id}"
    ultrawide_key = f"ultrawide_{robot_id}"
    depth_key = f"depth_{robot_id}"

    image_key = (
        rgb_key
        if rgb_key in obs_inference
        else ultrawide_key
        if ultrawide_key in obs_inference
        else None
    )
    if image_key is not None:
        frames = np.asarray(obs_inference[image_key][:])
        ts_key = (
            f"rgb_time_stamps_{robot_id}"
            if image_key == rgb_key
            else f"ultrawide_time_stamps_{robot_id}"
        )
        if ts_key in obs_inference:
            stamps = np.asarray(obs_inference[ts_key][:], dtype=np.float64).reshape(-1)
            n = min(frames.shape[0], stamps.size)
            for i in range(n):
                rgb_images.append(rgb_buffer_frame_to_image(frames[i]))
                rgb_times.append(float(stamps[i] - episode_action_t0))

    if depth_key in obs_inference:
        frames = np.asarray(obs_inference[depth_key][:])
        ts_key = f"depth_time_stamps_{robot_id}"
        if ts_key in obs_inference:
            stamps = np.asarray(obs_inference[ts_key][:], dtype=np.float64).reshape(-1)
            n = min(frames.shape[0], stamps.size)
            for i in range(n):
                depth_images.append(depth_buffer_frame_to_image(frames[i], depth_clip))
                depth_times.append(float(stamps[i] - episode_action_t0))


def collect_vision_timeline(
    log,
    horizon_keys: List[str],
    robot_id: int,
    max_horizons: int,
    episode_action_t0: float,
    depth_clip: float = 0.5,
) -> Tuple[List[np.ndarray], np.ndarray, List[np.ndarray], np.ndarray]:
    """Flatten deduped RGB/depth frames across horizons with episode-relative times."""
    rgb_images: List[np.ndarray] = []
    rgb_times: List[float] = []
    depth_images: List[np.ndarray] = []
    depth_times: List[float] = []

    for horizon_key in horizon_keys[:max_horizons]:
        horizon = log[horizon_key]
        horizon_time_s = float(horizon["action"]["timestamps_s"][0])
        used_raw = False
        before_rgb = len(rgb_images)
        before_depth = len(depth_images)
        if "obs_inference" in horizon:
            oi = horizon["obs_inference"]
            _append_horizon_raw_vision(
                oi,
                robot_id,
                episode_action_t0,
                rgb_images,
                rgb_times,
                depth_images,
                depth_times,
                depth_clip,
            )
            used_raw = (len(rgb_images) > before_rgb) or (len(depth_images) > before_depth)
        if not used_raw and "obs_policy" in horizon:
            _append_horizon_policy_vision_fallback(
                horizon["obs_policy"],
                horizon_time_s,
                episode_action_t0,
                rgb_images,
                rgb_times,
                depth_images,
                depth_times,
                depth_clip,
            )

    rgb_images, rgb_times = _dedupe_vision_timeline(rgb_images, rgb_times)
    depth_images, depth_times = _dedupe_vision_timeline(depth_images, depth_times)
    # Camera frame timestamps can use a different clock domain (for example,
    # Unix time) than action timestamps. The viewer/video only need elapsed
    # episode time, so rebase each vision stream to its first available frame.
    if rgb_times.size:
        rgb_times = rgb_times - rgb_times[0]
    if depth_times.size:
        depth_times = depth_times - depth_times[0]
    return rgb_images, rgb_times, depth_images, depth_times


def collect_vision_frames(
    log, horizon_keys: List[str], max_horizons: int
) -> Tuple[List[Optional[np.ndarray]], List[Optional[np.ndarray]]]:
    """Backward-compatible one-image-per-horizon collector (last sparse policy frame)."""
    rgb_frames: List[Optional[np.ndarray]] = []
    depth_frames: List[Optional[np.ndarray]] = []

    for horizon_key in horizon_keys[:max_horizons]:
        if "obs_policy" not in log[horizon_key]:
            rgb_frames.append(None)
            depth_frames.append(None)
            continue
        obs_policy = log[horizon_key]["obs_policy"]

        if "rgb_0" in obs_policy:
            rgb_frames.append(tensor_to_image(np.asarray(obs_policy["rgb_0"][:])))
        elif "ultrawide_0" in obs_policy:
            rgb_frames.append(tensor_to_image(np.asarray(obs_policy["ultrawide_0"][:])))
        else:
            rgb_frames.append(None)

        if "depth_0" in obs_policy:
            depth_frames.append(tensor_to_image(np.asarray(obs_policy["depth_0"][:])))
        else:
            depth_frames.append(None)

    return rgb_frames, depth_frames


def _blank_rgb(h: int = 64, w: int = 64) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _y_span(ys: List[np.ndarray], pad_frac: float = 0.05) -> Tuple[float, float]:
    """Return (ymin, ymax) with padding; falls back to a small default span."""
    vals = [np.asarray(y, dtype=np.float64).reshape(-1) for y in ys if y is not None]
    vals = [v[np.isfinite(v)] for v in vals if v.size]
    if not vals:
        return (-1.0, 1.0)
    lo = float(np.min([v.min() for v in vals]))
    hi = float(np.max([v.max() for v in vals]))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return (-1.0, 1.0)
    if hi <= lo:
        pad = max(abs(lo) * 0.05, 1e-3)
        return (lo - pad, hi + pad)
    pad = pad_frac * (hi - lo)
    return (lo - pad, hi + pad)


def time_marker_scatter(t: float, y0: float, y1: float) -> go.Scatter:
    """Magenta scrubber line updated each animation frame (shapes are unreliable)."""
    return go.Scatter(
        x=[t, t],
        y=[y0, y1],
        mode="lines",
        line=dict(color="magenta", width=2, dash="dash"),
        name="time marker",
        showlegend=False,
        hoverinfo="skip",
    )


def scene_xyz_ranges(
    pts: np.ndarray, pad_frac: float = 0.2, min_half_extent: float = 0.02
) -> Tuple[List[float], List[float], List[float]]:
    """Tight cube ranges around trajectory AABB with padding."""
    lo = pts.min(axis=0).astype(np.float64)
    hi = pts.max(axis=0).astype(np.float64)
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    half = np.maximum(half * (1.0 + pad_frac), min_half_extent)
    # Equal cube extent so aspectmode=cube does not stretch empty space oddly.
    half_eq = float(np.max(half))
    return (
        [float(center[0] - half_eq), float(center[0] + half_eq)],
        [float(center[1] - half_eq), float(center[1] + half_eq)],
        [float(center[2] - half_eq), float(center[2] + half_eq)],
    )


def sample_slider_indices(n: int, max_steps: int) -> np.ndarray:
    """Uniformly sample indices in [0, n), always including first and last."""
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    max_steps = max(2, int(max_steps))
    if n <= max_steps:
        return np.arange(n, dtype=np.int64)
    idx = np.linspace(0, n - 1, max_steps)
    return np.unique(np.round(idx).astype(np.int64))


def horizon_index_for_time(t: float, horizon_times: np.ndarray) -> int:
    """Latest horizon with start time <= t (hold previous image between replans)."""
    if horizon_times.size == 0:
        return 0
    valid = np.where(np.isfinite(horizon_times) & (horizon_times <= t))[0]
    if valid.size == 0:
        return 0
    return int(valid[-1])


def vision_index_for_time(t: float, vision_times: np.ndarray) -> int:
    """Latest vision frame with timestamp <= t."""
    return horizon_index_for_time(t, vision_times)


DEFAULT_VISION_VIDEO_FPS = 15.0


def episode_video_duration_s(
    rgb_times: np.ndarray, depth_times: np.ndarray
) -> float:
    ends = [float(arr[-1]) for arr in (rgb_times, depth_times) if arr.size > 0]
    return max(ends) if ends else 0.0


def _to_bgr_uint8(image: np.ndarray) -> np.ndarray:
    """Convert RGB/RGBA uint8 or float image to BGR uint8 for OpenCV."""
    img = np.asarray(image)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def image_to_data_uri(
    image: np.ndarray,
    jpeg_quality: int = 60,
    max_dimension: int = 128,
) -> str:
    """Downsample and JPEG-compress an image for a compact Plotly payload."""
    bgr = _to_bgr_uint8(image)
    height, width = bgr.shape[:2]
    scale = min(1.0, float(max_dimension) / max(height, width))
    if scale < 1.0:
        bgr = cv2.resize(
            bgr,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(
        ".jpg",
        bgr,
        [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)],
    )
    if not ok:
        raise RuntimeError("Failed to JPEG-encode Plotly image")
    payload = base64.b64encode(encoded).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def composite_rgb_depth_side_by_side(
    rgb_image: np.ndarray, depth_image: np.ndarray
) -> np.ndarray:
    """Stack RGB (left) and depth (right) with matched height."""
    rgb_bgr = _to_bgr_uint8(rgb_image)
    depth_bgr = _to_bgr_uint8(depth_image)
    target_h, target_w = rgb_bgr.shape[:2]
    if depth_bgr.shape[:2] != (target_h, target_w):
        depth_bgr = cv2.resize(
            depth_bgr, (target_w, target_h), interpolation=cv2.INTER_NEAREST
        )
    return np.hstack([rgb_bgr, depth_bgr])


def build_vision_video_frames(
    rgb_images: List[np.ndarray],
    rgb_times: np.ndarray,
    depth_images: List[np.ndarray],
    depth_times: np.ndarray,
    fps: float = DEFAULT_VISION_VIDEO_FPS,
) -> List[np.ndarray]:
    """Sample side-by-side frames at fixed fps over the episode vision timeline."""
    if not rgb_images:
        raise ValueError("No RGB frames available for video export")
    duration_s = episode_video_duration_s(rgb_times, depth_times)
    fps = max(float(fps), 1e-3)
    n_frames = max(1, int(np.floor(duration_s * fps + 1e-9)) + 1)
    blank_depth = depth_images[0] if depth_images else np.zeros_like(rgb_images[0])

    frames: List[np.ndarray] = []
    for i in range(n_frames):
        t = i / fps
        rgb_idx = vision_index_for_time(t, rgb_times)
        if depth_times.size and depth_images:
            depth_idx = vision_index_for_time(t, depth_times)
            depth_frame = depth_images[depth_idx]
        else:
            depth_frame = blank_depth
        frames.append(composite_rgb_depth_side_by_side(rgb_images[rgb_idx], depth_frame))
    return frames


def write_mp4_video(path: str, frames: List[np.ndarray], fps: float) -> None:
    """Write BGR uint8 frames to an MP4 file."""
    if not frames:
        raise ValueError("Cannot write video with zero frames")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {path}")
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(
                    frame, (width, height), interpolation=cv2.INTER_NEAREST
                )
            writer.write(frame)
    finally:
        writer.release()


def export_episode_vision_video(
    rgb_images: List[np.ndarray],
    rgb_times: np.ndarray,
    depth_images: List[np.ndarray],
    depth_times: np.ndarray,
    video_path: str,
    fps: float = DEFAULT_VISION_VIDEO_FPS,
) -> str:
    """Stream sampled frames directly to MP4 without buffering the whole video."""
    if not rgb_images:
        raise ValueError("No RGB frames available for video export")
    fps = max(float(fps), 1e-3)
    duration_s = episode_video_duration_s(rgb_times, depth_times)
    n_frames = max(1, int(np.floor(duration_s * fps + 1e-9)) + 1)
    blank_depth = depth_images[0] if depth_images else np.zeros_like(rgb_images[0])
    first_depth = (
        depth_images[vision_index_for_time(0.0, depth_times)]
        if depth_images and depth_times.size
        else blank_depth
    )
    first = composite_rgb_depth_side_by_side(rgb_images[0], first_depth)

    os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {video_path}")
    try:
        for i in range(n_frames):
            t = i / fps
            rgb = rgb_images[vision_index_for_time(t, rgb_times)]
            depth = (
                depth_images[vision_index_for_time(t, depth_times)]
                if depth_images and depth_times.size
                else blank_depth
            )
            writer.write(composite_rgb_depth_side_by_side(rgb, depth))
    finally:
        writer.release()
    return video_path


def feedback_prefix_len(t: float, feedback_times: np.ndarray) -> int:
    if feedback_times.size == 0:
        return 0
    return int(np.searchsorted(feedback_times, t, side="right"))


def downsample_timed_series(
    times: np.ndarray,
    values: np.ndarray,
    max_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Uniformly bound an animated series while retaining its endpoints."""
    n = min(len(times), len(values))
    if n == 0:
        return np.asarray(times)[:0], np.asarray(values)[:0]
    if n <= max_points:
        return np.asarray(times)[:n], np.asarray(values)[:n]
    idx = sample_slider_indices(n, max_points)
    return np.asarray(times)[idx], np.asarray(values)[idx]


def plot_episode(
    log_path: str,
    robot_id: int = 0,
    save_dir: Optional[str] = None,
    html_path: Optional[str] = None,
    video_path: Optional[str] = None,
    max_vision_horizons: int = 64,
    max_slider_steps: int = 300,
    video_fps: float = DEFAULT_VISION_VIDEO_FPS,
    save_video: bool = True,
    show: bool = False,
) -> str:
    log = zarr.open(store=zarr.DirectoryStore(path=log_path), mode="r")
    horizon_keys = sorted_horizon_keys(log)
    if not horizon_keys:
        raise RuntimeError(f"No horizon_* groups in {log_path}")

    vision_keys = horizon_keys[: max(1, max_vision_horizons)]
    chained = chain_horizon_actions(log, horizon_keys, robot_id)
    episode_folder = log.attrs.get("episode_data_folder", "")
    se3_tc = load_pose7_offset_tc(log.attrs.get("hardware_config_path"))
    episode_data = load_robot_episode_json(episode_folder, robot_id, se3_tc=se3_tc)
    episode_action_t0 = float(
        np.asarray(log[horizon_keys[0]]["action"]["timestamps_s"][:], dtype=np.float64)[0]
    )
    rgb_images, rgb_times, depth_images, depth_times = collect_vision_timeline(
        log,
        vision_keys,
        robot_id=robot_id,
        max_horizons=len(vision_keys),
        episode_action_t0=episode_action_t0,
    )

    t_cmd = chained["timestamps_s"]
    nom = chained["nominal"]
    virt = chained["virtual"]
    stiff = chained["stiffness"]
    h_times = chained["horizon_times_s"][: len(vision_keys)]

    # Fallback blank images if missing
    sample_rgb = next((im for im in rgb_images if im is not None), _blank_rgb())
    sample_depth = next((im for im in depth_images if im is not None), _blank_rgb())
    if not rgb_images:
        rgb_images = [sample_rgb]
        rgb_times = np.asarray([0.0], dtype=np.float64)
    if not depth_images:
        depth_images = [sample_depth]
        depth_times = np.asarray([0.0], dtype=np.float64)
    rgb_sources = [image_to_data_uri(image) for image in rgb_images]
    depth_sources = [image_to_data_uri(image) for image in depth_images]

    slider_idx = sample_slider_indices(len(t_cmd), max_slider_steps)
    if slider_idx.size == 0:
        raise RuntimeError("No action timestamps to scrub")

    k0 = int(slider_idx[0])
    t0 = float(t_cmd[k0])
    v0_rgb = vision_index_for_time(t0, rgb_times)
    v0_depth = vision_index_for_time(t0, depth_times)
    fb_times = (
        episode_data["time_s"] if episode_data is not None else np.zeros(0, dtype=np.float64)
    )
    fb_pose = episode_data["pose_fb"] if episode_data is not None else None
    if fb_pose is not None:
        fb_3d_times, fb_3d_pose = downsample_timed_series(
            fb_times, fb_pose, max_points=1000
        )
    else:
        fb_3d_times = np.zeros(0, dtype=np.float64)
        fb_3d_pose = None
    fb0 = feedback_prefix_len(t0, fb_3d_times)

    fig = make_subplots(
        rows=6,
        cols=2,
        shared_xaxes=False,
        vertical_spacing=0.05,
        horizontal_spacing=0.06,
        column_widths=[0.34, 0.66],
        row_heights=[0.14, 0.14, 0.18, 0.18, 0.18, 0.18],
        specs=[
            [{"type": "xy"}, {"type": "scene", "rowspan": 2}],
            [{"type": "xy"}, None],
            [{"type": "xy", "colspan": 2}, None],
            [{"type": "xy", "colspan": 2}, None],
            [{"type": "xy", "colspan": 2}, None],
            [{"type": "xy", "colspan": 2}, None],
        ],
        subplot_titles=(
            "Policy RGB",
            "3D trajectory",
            "Policy depth",
            "Pose (m)",
            "Force (N)",
            "Stiffness (N/m)",
            "Torque (Nm)",
        ),
    )

    # --- Images (left column) ---
    fig.add_trace(go.Image(source=rgb_sources[v0_rgb], name="rgb"), row=1, col=1)
    fig.add_trace(
        go.Image(source=depth_sources[v0_depth], name="depth"), row=2, col=1
    )

    # --- Pose (row 3, full width): full history always visible in 2D ---
    pose_colors = {"x": "#1f77b4", "y": "#ff7f0e", "z": "#2ca02c"}
    for i, label in enumerate(("x", "y", "z")):
        name_nom = f"nom {label}"
        fig.add_trace(
            go.Scatter(
                x=t_cmd,
                y=nom[:, i],
                mode="lines",
                name=name_nom,
                line=dict(color="gold", dash="dash", width=1.5),
                legendgroup="Pose",
                legendgrouptitle_text="Pose",
                showlegend=True,
                visible=True,
            ),
            row=3,
            col=1,
        )
        name_virt = f"virt {label}"
        fig.add_trace(
            go.Scatter(
                x=t_cmd,
                y=virt[:, i],
                mode="lines",
                name=name_virt,
                line=dict(color="orangered", width=2),
                legendgroup="Pose",
                legendgrouptitle_text="Pose",
                showlegend=True,
                visible=True,
            ),
            row=3,
            col=1,
        )
        if episode_data is not None:
            name_fb = f"fb {label}"
            fig.add_trace(
                go.Scatter(
                    x=episode_data["time_s"],
                    y=episode_data["pose_fb"][:, i],
                    mode="lines",
                    name=name_fb,
                    line=dict(color=pose_colors[label], width=1.2),
                    opacity=0.85,
                    legendgroup="Pose",
                    legendgrouptitle_text="Pose",
                    showlegend=True,
                    visible=True,
                ),
                row=3,
                col=1,
            )

    # --- Force (row 4) ---
    if episode_data is not None:
        t_fb = episode_data["time_s"]
        wrench = episode_data["wrench"]
        for i, label in enumerate(("fx", "fy", "fz")):
            fig.add_trace(
                go.Scatter(
                    x=t_fb,
                    y=wrench[:, i],
                    mode="lines",
                    name=label,
                    legendgroup="Force",
                    legendgrouptitle_text="Force",
                    showlegend=True,
                ),
                row=4,
                col=1,
            )
    else:
        fig.add_annotation(
            text="no episode JSON wrench/pose feedback",
            xref="x3 domain",
            yref="y3 domain",
            x=0.5,
            y=0.5,
            showarrow=False,
        )

    # --- Stiffness (row 5): compliant-direction min eigenvalue ---
    fig.add_trace(
        go.Scatter(
            x=t_cmd,
            y=stiff,
            mode="lines",
            name="k_compliant",
            line=dict(color="#9467bd", width=2),
            legendgroup="Stiffness",
            legendgrouptitle_text="Stiffness",
            showlegend=True,
        ),
        row=5,
        col=1,
    )

    # --- Torque (row 6) ---
    if episode_data is not None:
        t_fb = episode_data["time_s"]
        wrench = episode_data["wrench"]
        for i, label in enumerate(("mx", "my", "mz")):
            fig.add_trace(
                go.Scatter(
                    x=t_fb,
                    y=wrench[:, i + 3],
                    mode="lines",
                    name=label,
                    legendgroup="Torque",
                    legendgrouptitle_text="Torque",
                    showlegend=True,
                ),
                row=6,
                col=1,
            )

    # Y spans for magenta scrubber traces (data coords, updated each frame)
    pose_ys = [nom[:, i] for i in range(3)] + [virt[:, i] for i in range(3)]
    if episode_data is not None:
        pose_ys += [episode_data["pose_fb"][:, i] for i in range(3)]
    pose_y0, pose_y1 = _y_span(pose_ys)
    if episode_data is not None:
        force_y0, force_y1 = _y_span([episode_data["wrench"][:, i] for i in range(3)])
        torque_y0, torque_y1 = _y_span(
            [episode_data["wrench"][:, i] for i in range(3, 6)]
        )
    else:
        force_y0, force_y1 = (-1.0, 1.0)
        torque_y0, torque_y1 = (-1.0, 1.0)
    stiff_y0, stiff_y1 = _y_span([stiff])

    fig.add_trace(time_marker_scatter(t0, pose_y0, pose_y1), row=3, col=1)
    fig.add_trace(time_marker_scatter(t0, force_y0, force_y1), row=4, col=1)
    fig.add_trace(time_marker_scatter(t0, stiff_y0, stiff_y1), row=5, col=1)
    fig.add_trace(time_marker_scatter(t0, torque_y0, torque_y1), row=6, col=1)
    idx_marker_pose = len(fig.data) - 4
    idx_marker_force = len(fig.data) - 3
    idx_marker_stiff = len(fig.data) - 2
    idx_marker_torque = len(fig.data) - 1

    # --- 3D trajectory prefixes only (updated by slider) ---
    fig.add_trace(
        go.Scatter3d(
            x=nom[: k0 + 1, 0],
            y=nom[: k0 + 1, 1],
            z=nom[: k0 + 1, 2],
            mode="lines",
            name="nominal 3d",
            line=dict(color="gold", width=4, dash="dash"),
            legendgroup="3D",
            legendgrouptitle_text="3D",
            showlegend=True,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter3d(
            x=virt[: k0 + 1, 0],
            y=virt[: k0 + 1, 1],
            z=virt[: k0 + 1, 2],
            mode="lines",
            name="virtual 3d",
            line=dict(color="orangered", width=4),
            legendgroup="3D",
            legendgrouptitle_text="3D",
            showlegend=True,
        ),
        row=1,
        col=2,
    )
    has_feedback_3d = fb_3d_pose is not None
    if has_feedback_3d:
        fig.add_trace(
            go.Scatter3d(
                x=fb_3d_pose[:fb0, 0],
                y=fb_3d_pose[:fb0, 1],
                z=fb_3d_pose[:fb0, 2],
                mode="lines",
                name="feedback 3d",
                line=dict(color="royalblue", width=3),
                opacity=0.7,
                legendgroup="3D",
                legendgrouptitle_text="3D",
                showlegend=True,
            ),
            row=1,
            col=2,
        )

    # Trace indices for animated updates
    idx_rgb, idx_depth = 0, 1
    n_traces = len(fig.data)
    marker_ids = [
        idx_marker_pose,
        idx_marker_force,
        idx_marker_stiff,
        idx_marker_torque,
    ]
    if has_feedback_3d:
        idx_nom3d, idx_virt3d, idx_fb3d = n_traces - 3, n_traces - 2, n_traces - 1
        frame_trace_ids = [
            idx_rgb,
            idx_depth,
            *marker_ids,
            idx_nom3d,
            idx_virt3d,
            idx_fb3d,
        ]
    else:
        idx_nom3d, idx_virt3d = n_traces - 2, n_traces - 1
        frame_trace_ids = [
            idx_rgb,
            idx_depth,
            *marker_ids,
            idx_nom3d,
            idx_virt3d,
        ]

    frames = []
    slider_steps = []
    for step_i, k in enumerate(slider_idx):
        k = int(k)
        t_i = float(t_cmd[k])
        v_rgb = vision_index_for_time(t_i, rgb_times)
        v_depth = vision_index_for_time(t_i, depth_times)
        frame_data = [
            go.Image(source=rgb_sources[v_rgb]),
            go.Image(source=depth_sources[v_depth]),
            time_marker_scatter(t_i, pose_y0, pose_y1),
            time_marker_scatter(t_i, force_y0, force_y1),
            time_marker_scatter(t_i, stiff_y0, stiff_y1),
            time_marker_scatter(t_i, torque_y0, torque_y1),
            go.Scatter3d(
                x=nom[: k + 1, 0].tolist(),
                y=nom[: k + 1, 1].tolist(),
                z=nom[: k + 1, 2].tolist(),
            ),
            go.Scatter3d(
                x=virt[: k + 1, 0].tolist(),
                y=virt[: k + 1, 1].tolist(),
                z=virt[: k + 1, 2].tolist(),
            ),
        ]
        if has_feedback_3d:
            fb_n = feedback_prefix_len(t_i, fb_3d_times)
            frame_data.append(
                go.Scatter3d(
                    x=fb_3d_pose[:fb_n, 0].tolist(),
                    y=fb_3d_pose[:fb_n, 1].tolist(),
                    z=fb_3d_pose[:fb_n, 2].tolist(),
                )
            )
        frames.append(
            go.Frame(
                name=str(step_i),
                data=frame_data,
                traces=frame_trace_ids,
            )
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
                "label": f"{t_i:.1f}s",
                "method": "animate",
            }
        )

    fig.frames = frames

    # Tight 3D framing from full trajectory; fixed while path prefix grows
    all_xyz = [nom[:, :3], virt[:, :3]]
    if fb_pose is not None and fb_pose.size:
        all_xyz.append(fb_pose[:, :3])
    pts = np.concatenate(all_xyz, axis=0)
    x_range, y_range, z_range = scene_xyz_ranges(pts)

    fig.update_layout(
        title=dict(
            text=f"Inference episode: {os.path.basename(log_path)}",
            x=0.01,
            xanchor="left",
            y=0.98,
            yanchor="top",
        ),
        height=1300,
        width=1500,
        margin=dict(t=90, r=220, b=130, l=80),
        uirevision="episode-viewer",
        legend=dict(
            orientation="v",
            x=1.01,
            y=1.0,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#cccccc",
            borderwidth=1,
            font=dict(size=11),
            itemclick="toggle",
            itemdoubleclick="toggleothers",
            groupclick="toggleitem",
            tracegroupgap=12,
        ),
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "t = ", "xanchor": "left"},
                "pad": {"t": 20, "b": 10},
                "x": 0.05,
                "len": 0.75,
                "xanchor": "left",
                "y": -0.02,
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
                "y": -0.02,
                "xanchor": "right",
                "yanchor": "top",
                "pad": {"t": 20, "r": 10},
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 50, "redraw": True},
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
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="cube",
            uirevision="episode-viewer-3d",
            xaxis=dict(range=x_range, autorange=False),
            yaxis=dict(range=y_range, autorange=False),
            zaxis=dict(range=z_range, autorange=False),
        ),
    )

    # Hide ticks on image panels
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=1)
    fig.update_yaxes(
        showticklabels=False, showgrid=False, zeroline=False, scaleanchor="x", row=1, col=1
    )
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=2, col=1)
    fig.update_yaxes(
        showticklabels=False, showgrid=False, zeroline=False, scaleanchor="x2", row=2, col=1
    )

    fig.update_xaxes(title_text="time (s)", row=6, col=1)
    fig.update_yaxes(title_text="pos (m)", title_standoff=8, row=3, col=1)
    fig.update_yaxes(title_text="force (N)", title_standoff=8, row=4, col=1)
    fig.update_yaxes(title_text="stiffness (N/m)", title_standoff=8, row=5, col=1)
    fig.update_yaxes(title_text="torque (Nm)", title_standoff=8, row=6, col=1)

    # Share time axis across pose/force/stiffness/torque for easier reading
    fig.update_xaxes(matches="x3", row=4, col=1)
    fig.update_xaxes(matches="x3", row=5, col=1)
    fig.update_xaxes(matches="x3", row=6, col=1)

    if html_path is None:
        os.makedirs(save_dir or ".", exist_ok=True)
        html_path = os.path.join(
            save_dir or ".",
            f"{os.path.basename(log_path)}_r{robot_id}_episode.html",
        )
    else:
        os.makedirs(os.path.dirname(html_path) or ".", exist_ok=True)

    fig.write_html(html_path, include_plotlyjs="cdn", auto_open=False)
    print(f"Saved: {html_path}")

    if save_video:
        if video_path is None:
            video_path = os.path.splitext(html_path)[0] + "_rgb_depth.mp4"
        export_episode_vision_video(
            rgb_images,
            rgb_times,
            depth_images,
            depth_times,
            video_path=video_path,
            fps=video_fps,
        )
        duration_s = episode_video_duration_s(rgb_times, depth_times)
        n_video_frames = max(1, int(np.floor(duration_s * video_fps + 1e-9)) + 1)
        print(
            f"Saved: {video_path} "
            f"({n_video_frames} frames @ {video_fps:g} fps, duration={duration_s:.2f}s)"
        )

    print(
        f"Vision frames: rgb={len(rgb_images)} depth={len(depth_images)} "
        f"(horizons loaded={len(vision_keys)})"
    )
    print(
        f"Slider steps: {len(slider_idx)} "
        f"(action samples={len(t_cmd)}, max_slider_steps={max_slider_steps})"
    )

    if show:
        fig.show()

    return html_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log_path", default=None, help="Absolute path to episode zarr log")
    parser.add_argument(
        "--log_subdir",
        default=None,
        help="Subfolder under PYRITE_CONTROL_LOG_FOLDERS",
    )
    parser.add_argument("--robot_id", type=int, default=0)
    parser.add_argument("--save_dir", default=None, help="Directory for output HTML")
    parser.add_argument(
        "--html",
        default=None,
        help="Explicit HTML output path (overrides --save_dir default name)",
    )
    parser.add_argument(
        "--max_vision_horizons",
        type=int,
        default=64,
        help="Max horizons for loading policy RGB/depth frames",
    )
    parser.add_argument(
        "--max_slider_steps",
        type=int,
        default=300,
        help="Max scrubber frames over action timestamps (default: 300)",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Explicit MP4 output path (default: alongside HTML as *_rgb_depth.mp4)",
    )
    parser.add_argument(
        "--video_fps",
        type=float,
        default=DEFAULT_VISION_VIDEO_FPS,
        help="FPS for exported RGB+depth video (default: 15)",
    )
    parser.add_argument(
        "--no_video",
        action="store_true",
        help="Skip writing the side-by-side RGB+depth MP4",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the interactive figure after saving (disabled by default)",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if not args.log_path and not args.log_subdir:
        parser.error("Provide --log_path or --log_subdir")

    log_path = resolve_log_path(args)
    print(f"Loading control log from {log_path}")
    plot_episode(
        log_path=log_path,
        robot_id=args.robot_id,
        save_dir=args.save_dir,
        html_path=args.html,
        video_path=args.video,
        max_vision_horizons=args.max_vision_horizons,
        max_slider_steps=args.max_slider_steps,
        video_fps=args.video_fps,
        save_video=not args.no_video,
        show=args.show and not args.no_show,
    )


if __name__ == "__main__":
    main()

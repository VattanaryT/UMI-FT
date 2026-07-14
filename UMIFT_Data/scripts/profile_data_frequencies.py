#!/usr/bin/env python3
"""
Profile per-stream sample rates for UMIFT training data (wild collection zarr)
and live layup inference (ManipServerLayupEnv).

Collection mode is offline (unified zarr + optional replay buffer).
Inference mode polls ManipServerLayupEnv live so depth (Python-only) is included.

python UMIFT_Data/scripts/profile_data_frequencies.py collection   --unified-zarr /mnt/data/umidata/ft/straight_convex_data_v3/zarr/2026-06-26-22.41.26_unified.zarr   --also-replay-buffer /mnt/data/umidata/ft/straight_convex_data_v3/acp_replay_buffer.zarr   --replay-episode episode_60
 python UMIFT_Data/scripts/profile_data_frequencies.py inference   --hardware-config $PYRITE_HARDWARE_CONFIG_FOLDERS/ur5e_wrist_ft.yaml   --realsense-serial $UMIFT_REALSENSE_SERIAL   --duration-s 10 --depth-clip 0.5 --poll-hz 1000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import zarr

SCRIPT_DIR = Path(__file__).resolve().parent
UMI_FT_ROOT = SCRIPT_DIR.parents[1]
PYRITE_ENVSUITES_ROOT = UMI_FT_ROOT / "PyriteEnvSuites"
if str(PYRITE_ENVSUITES_ROOT) not in sys.path:
    sys.path.insert(0, str(PYRITE_ENVSUITES_ROOT))

# Default policy window params (umift_zucchini.yaml)
DEFAULT_RGB_HORIZON = 2
DEFAULT_RGB_DS = 10
DEFAULT_ROBOT_HORIZON = 2
DEFAULT_ROBOT_DS = 5
DEFAULT_WRENCH_HORIZON = 32
DEFAULT_WRENCH_DS = 1

UNIFIED_STREAMS = {
    "rgb": ("camera", "color", "timestamp_sec_relative"),
    "depth": ("camera", "depth", "timestamp_sec_relative"),
    "robot": ("sensors", "t265_odom", "timestamp_sec_relative"),
    "wrench": ("sensors", "ft_wrench", "timestamp_sec_relative"),
}

REPLAY_STREAMS = {
    "rgb": "rgb_time_stamps_0",
    "depth": "depth_time_stamps_0",
    "robot": "robot_time_stamps_0",
    "wrench": "wrench_time_stamps_left_0",
}

LIVE_STREAM_MAP = {
    "rgb": "rgb_time_stamps_0",
    "depth": "depth_time_stamps_0",
    "robot": "robot_time_stamps_0",
    "wrench": "wrench_time_stamps_0",
}


@dataclass
class StreamStats:
    name: str
    n_samples: int
    duration_s: float
    median_dt: float
    mean_dt: float
    std_dt: float
    p5_dt: float
    p95_dt: float
    median_hz: float
    start_offset_s: float
    end_gap_s: float  # gap vs reference stream end time

    @property
    def valid(self) -> bool:
        return self.n_samples >= 2 and self.duration_s > 0


def as_1d_time(time_arr) -> np.ndarray:
    return np.asarray(time_arr, dtype=np.float64).reshape(-1)


def profile_timestamps(
    name: str,
    t_sec: np.ndarray,
    t_ref_start: Optional[float] = None,
    t_ref_end: Optional[float] = None,
) -> StreamStats:
    t = as_1d_time(t_sec)
    if len(t) < 2:
        return StreamStats(
            name=name,
            n_samples=len(t),
            duration_s=0.0,
            median_dt=float("nan"),
            mean_dt=float("nan"),
            std_dt=float("nan"),
            p5_dt=float("nan"),
            p95_dt=float("nan"),
            median_hz=float("nan"),
            start_offset_s=float("nan"),
            end_gap_s=float("nan"),
        )
    dt = np.diff(t)
    dt = dt[dt > 0]
    if len(dt) == 0:
        median_dt = float("nan")
    else:
        median_dt = float(np.median(dt))
    duration_s = float(t[-1] - t[0])
    start_offset_s = float(t[0] - t_ref_start) if t_ref_start is not None else 0.0
    end_gap_s = (
        float(t_ref_end - t[-1]) if t_ref_end is not None else float("nan")
    )
    median_hz = 1.0 / median_dt if median_dt and median_dt > 0 else float("nan")
    return StreamStats(
        name=name,
        n_samples=len(t),
        duration_s=duration_s,
        median_dt=median_dt,
        mean_dt=float(np.mean(dt)) if len(dt) else float("nan"),
        std_dt=float(np.std(dt)) if len(dt) else float("nan"),
        p5_dt=float(np.percentile(dt, 5)) if len(dt) else float("nan"),
        p95_dt=float(np.percentile(dt, 95)) if len(dt) else float("nan"),
        median_hz=median_hz,
        start_offset_s=start_offset_s,
        end_gap_s=end_gap_s,
    )


def profile_buffer_intra(name: str, t_sec: np.ndarray) -> StreamStats:
    """Profile spacing within a single buffer snapshot (all timestamps in one read)."""
    return profile_timestamps(f"{name}_buffer", t_sec)


def compute_reference_bounds(
    streams: Dict[str, np.ndarray],
) -> Tuple[float, float]:
    """Earliest start and latest end across streams."""
    starts = []
    ends = []
    for t in streams.values():
        t = as_1d_time(t)
        if len(t) >= 1:
            starts.append(float(t[0]))
            ends.append(float(t[-1]))
    if not starts:
        return 0.0, 0.0
    return min(starts), max(ends)


def profile_all_streams(
    streams: Dict[str, np.ndarray],
    ref_name: str = "rgb",
) -> Dict[str, StreamStats]:
    t_ref_start, t_ref_end = compute_reference_bounds(streams)
    if ref_name in streams and len(as_1d_time(streams[ref_name])) >= 1:
        t_ref = as_1d_time(streams[ref_name])
        t_ref_start = float(t_ref[0])
        t_ref_end = float(t_ref[-1])
    stats = {}
    for name, ts in streams.items():
        stats[name] = profile_timestamps(name, ts, t_ref_start, t_ref_end)
    return stats


def policy_window_span_s(
    query_len: int, median_hz: float, downsample_steps: int = 1
) -> float:
    """Wall-clock span of a policy observation window at measured rate.

    MPC fetches query_len = (horizon-1)*ds+1 raw samples, then strides by ds to
    produce `horizon` policy inputs. Wall-clock span is (horizon-1)*ds/hz, i.e.
  (query_len-1)/hz — do not multiply by ds again.
    """
    del downsample_steps  # kept for call-site compatibility
    if not np.isfinite(median_hz) or median_hz <= 0 or query_len < 2:
        return float("nan")
    return (query_len - 1) / median_hz


def print_policy_windows(
    stats: Dict[str, StreamStats],
    wrench_horizon: int = DEFAULT_WRENCH_HORIZON,
    wrench_ds: int = DEFAULT_WRENCH_DS,
    *,
    compare_train_deploy: bool = False,
    train_wrench_hz: float = 28.0,
) -> None:
    print("\n--- Policy observation windows (wall-clock span) ---")
    configs = [
        ("rgb", DEFAULT_RGB_HORIZON, DEFAULT_RGB_DS),
        ("robot", DEFAULT_ROBOT_HORIZON, DEFAULT_ROBOT_DS),
        ("wrench", wrench_horizon, wrench_ds),
        ("depth", DEFAULT_RGB_HORIZON, DEFAULT_RGB_DS),
    ]
    for name, horizon, ds in configs:
        if name not in stats or not stats[name].valid:
            continue
        qlen = (horizon - 1) * ds + 1
        span = policy_window_span_s(qlen, stats[name].median_hz, ds)
        print(
            f"  {name:8s} query_len={qlen:3d}  "
            f"@ {stats[name].median_hz:7.2f} Hz  ->  {span:.3f} s window"
        )
    if "wrench" in stats and stats["wrench"].valid:
        deploy_qlen = (wrench_horizon - 1) * wrench_ds + 1
        train_qlen = (wrench_horizon - 1) * DEFAULT_WRENCH_DS + 1
        measured_span = policy_window_span_s(
            deploy_qlen, stats["wrench"].median_hz, wrench_ds
        )
        train_span = policy_window_span_s(
            train_qlen, train_wrench_hz, DEFAULT_WRENCH_DS
        )
        if compare_train_deploy and np.isfinite(train_span) and np.isfinite(measured_span):
            print(
                f"\n  Wrench train/deploy: ~{train_span:.3f} s @ ~{train_wrench_hz:.0f} Hz "
                f"(collection ds={DEFAULT_WRENCH_DS}) vs {measured_span:.3f} s @ "
                f"{stats['wrench'].median_hz:.1f} Hz (deploy ds={wrench_ds})"
            )
            if measured_span < train_span * 0.5:
                print(
                    "  ** WARNING: inference wrench window is much shorter than training — "
                    "consider increasing wrench down_sample_steps at deploy time."
                )
            elif abs(measured_span - train_span) / train_span < 0.15:
                print(
                    "  OK: deploy wrench window matches collection span (within 15%)."
                )
        elif np.isfinite(measured_span) and np.isfinite(train_span):
            print(
                f"\n  Wrench window: {measured_span:.3f} s @ {stats['wrench'].median_hz:.1f} Hz "
                f"(training expects ~{train_span:.3f} s @ ~{train_wrench_hz:.0f} Hz)"
            )


def print_summary_table(
    stats: Dict[str, StreamStats], title: str = "Stream summary"
) -> None:
    print(f"\n=== {title} ===")
    header = (
        f"{'stream':<10} {'n':>6} {'dur_s':>8} {'med_hz':>8} "
        f"{'med_dt':>10} {'start_off':>10} {'end_gap':>10}"
    )
    print(header)
    print("-" * len(header))
    for name in sorted(stats.keys()):
        s = stats[name]
        if not s.valid and s.n_samples < 2:
            print(f"{name:<10} {s.n_samples:>6}  (insufficient samples)")
            continue
        print(
            f"{s.name:<10} {s.n_samples:>6} {s.duration_s:>8.2f} "
            f"{s.median_hz:>8.2f} {s.median_dt:>10.6f} "
            f"{s.start_offset_s:>10.3f} {s.end_gap_s:>10.3f}"
        )


def stats_to_dict(stats: Dict[str, StreamStats]) -> dict:
    return {k: asdict(v) for k, v in stats.items()}


def load_unified_zarr(path: Path) -> Dict[str, np.ndarray]:
    root = zarr.open(str(path), mode="r")
    out = {}
    for name, (grp1, grp2, ts_key) in UNIFIED_STREAMS.items():
        try:
            out[name] = as_1d_time(root[grp1][grp2][ts_key][:])
        except KeyError as exc:
            raise KeyError(f"Missing {grp1}/{grp2}/{ts_key} in {path}") from exc
    return out


def load_replay_episode(zarr_path: Path, episode_key: str) -> Dict[str, np.ndarray]:
    root = zarr.open(str(zarr_path), mode="r")
    ep = root["data"][episode_key]
    out = {}
    for name, key in REPLAY_STREAMS.items():
        if key not in ep:
            continue
        out[name] = as_1d_time(ep[key][:])
    return out


def list_replay_episodes(zarr_path: Path) -> List[str]:
    root = zarr.open(str(zarr_path), mode="r")
    return sorted(root["data"].group_keys(), key=lambda x: (len(x), x))


def list_unified_zarrs(directory: Path) -> List[Path]:
    return sorted(directory.glob("*_unified.zarr"))


def preflight_realsense_serial(serial: Optional[str]) -> None:
    """Verify pyrealsense2 is importable and optional serial is present."""
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise SystemExit(
            "pyrealsense2 is required for inference profiling. "
            "Install librealsense Python bindings in the active env."
        ) from exc
    if not serial:
        return
    ctx = rs.context()
    found = [
        d.get_info(rs.camera_info.serial_number)
        for d in ctx.query_devices()
    ]
    if serial not in found:
        raise SystemExit(
            f"RealSense serial {serial!r} not found. Connected: {found or '(none)'}"
        )


def sample_live_layup(
    hardware_config: Path,
    duration_s: float,
    depth_clip: float,
    poll_hz: float,
    rgb_query: int = 30,
    depth_query: int = 30,
    pose_query: int = 200,
    wrench_query: int = 200,
    camera_res_hw: Tuple[int, int] = (224, 224),
    realsense_serial: Optional[str] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, StreamStats], Dict[str, StreamStats]]:
    preflight_realsense_serial(realsense_serial)
    from PyriteEnvSuites.envs.task.manip_server_layup_env import ManipServerLayupEnv

    query_sizes = {
        "sparse": {
            "rgb": rgb_query,
            "depth": depth_query,
            "ts_pose_fb": pose_query,
            "wrench": wrench_query,
        }
    }
    env = ManipServerLayupEnv(
        camera_res_hw=list(camera_res_hw),
        hardware_config_path=str(hardware_config),
        query_sizes=query_sizes,
        compliant_dimensionality=3,
        depth_clip=depth_clip,
        vision_key=["rgb_0", "depth_0"],
        realsense_serial=realsense_serial,
    )
    env.reset()
    print("[inference] Waiting for hardware server...")
    while not env.server.is_ready():
        time.sleep(0.1)

    poll_series: Dict[str, List[float]] = {k: [] for k in LIVE_STREAM_MAP}
    poll_wall: List[float] = []
    t_end = time.time() + duration_s
    period = 1.0 / poll_hz if poll_hz > 0 else 0.0
    n_polls = 0
    obs_final: Dict = {}
    print(f"[inference] Polling for {duration_s:.1f} s at ~{poll_hz:.0f} Hz...")
    try:
        while time.time() < t_end:
            t_poll = time.time()
            obs = env.get_observation_from_buffer()
            obs_final = obs
            for stream_name, ts_key in LIVE_STREAM_MAP.items():
                if ts_key in obs:
                    ts = as_1d_time(obs[ts_key])
                    if len(ts) > 0:
                        poll_series[stream_name].append(float(ts[-1]))
            poll_wall.append(t_poll)
            n_polls += 1
            if period > 0:
                elapsed = time.time() - t_poll
                if elapsed < period:
                    time.sleep(period - elapsed)
    finally:
        buffer_stats: Dict[str, StreamStats] = {}
        for stream_name, ts_key in LIVE_STREAM_MAP.items():
            if ts_key in obs_final:
                ts = as_1d_time(obs_final[ts_key])
                if len(ts) >= 2:
                    buffer_stats[stream_name] = profile_buffer_intra(stream_name, ts)
        env.cleanup()

    poll_streams = {
        k: np.asarray(v, dtype=np.float64) for k, v in poll_series.items() if len(v) >= 2
    }
    poll_stats = profile_all_streams(poll_streams)

    print(f"[inference] Completed {n_polls} polls.")
    if buffer_stats:
        print("\n--- Intra-buffer spacing (final snapshot) ---")
        for name, s in buffer_stats.items():
            if s.valid:
                print(f"  {name}: {s.median_hz:.2f} Hz (n={s.n_samples})")

    return poll_streams, poll_stats, buffer_stats


def maybe_plot(
    collection_stats: Optional[Dict[str, StreamStats]],
    inference_stats: Optional[Dict[str, StreamStats]],
    out_path: Optional[Path],
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2 if collection_stats and inference_stats else 1, figsize=(10, 4))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    def _bar(ax, stats: Dict[str, StreamStats], title: str) -> None:
        names = [k for k, s in stats.items() if s.valid]
        hz = [stats[k].median_hz for k in names]
        ax.bar(names, hz)
        ax.set_ylabel("median Hz")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)

    idx = 0
    if collection_stats:
        _bar(axes[idx], collection_stats, "Collection")
        idx += 1
    if inference_stats:
        _bar(axes[idx if len(axes) > 1 else 0], inference_stats, "Inference (live)")

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120)
        print(f"Wrote plot to {out_path}")
    else:
        plt.show()


def cmd_collection(args: argparse.Namespace) -> dict:
    all_stats: Dict[str, List[StreamStats]] = {k: [] for k in UNIFIED_STREAMS}

    if args.unified_zarr:
        paths = [Path(args.unified_zarr)]
    elif args.unified_zarr_dir:
        paths = list_unified_zarrs(Path(args.unified_zarr_dir))
        if not paths:
            raise SystemExit(f"No *_unified.zarr in {args.unified_zarr_dir}")
    else:
        raise SystemExit("Provide --unified-zarr or --unified-zarr-dir")

    last_stats: Dict[str, StreamStats] = {}
    for path in paths:
        streams = load_unified_zarr(path)
        stats = profile_all_streams(streams)
        print_summary_table(stats, title=f"Unified zarr: {path.name}")
        print_policy_windows(stats, args.wrench_horizon, args.wrench_downsample_steps)
        for k, s in stats.items():
            all_stats[k].append(s)
        last_stats = stats

    if len(paths) > 1:
        print("\n=== Aggregate over unified zarr episodes ===")
        for name, lst in all_stats.items():
            valid = [s for s in lst if s.valid]
            if not valid:
                continue
            hz = [s.median_hz for s in valid]
            gaps = [s.end_gap_s for s in valid if np.isfinite(s.end_gap_s)]
            print(
                f"  {name}: hz min/mean/max = "
                f"{min(hz):.2f} / {np.mean(hz):.2f} / {max(hz):.2f}"
                + (
                    f"  end_gap mean = {np.mean(gaps):.3f} s"
                    if gaps
                    else ""
                )
            )

    result = {"unified": stats_to_dict(last_stats)}

    if args.also_replay_buffer:
        rb_path = Path(args.also_replay_buffer)
        episodes = list_replay_episodes(rb_path)
        if args.replay_episode:
            episodes = [args.replay_episode]
        gap_rgb_wrench = []
        for ep in episodes[: args.max_replay_episodes]:
            streams = load_replay_episode(rb_path, ep)
            if not streams:
                continue
            stats = profile_all_streams(streams)
            if "rgb" in stats and "wrench" in stats:
                gap = stats["rgb"].duration_s - stats["wrench"].duration_s
                if stats["rgb"].n_samples >= 2 and stats["wrench"].n_samples >= 2:
                    gap_rgb_wrench.append(
                        float(
                            as_1d_time(streams["rgb"])[-1]
                            - as_1d_time(streams["wrench"])[-1]
                        )
                    )
        if args.replay_episode or len(episodes) == 1:
            ep = episodes[0]
            streams = load_replay_episode(rb_path, ep)
            stats = profile_all_streams(streams)
            print_summary_table(stats, title=f"Replay buffer: {ep}")
            print_policy_windows(stats, args.wrench_horizon, args.wrench_downsample_steps)
            result["replay"] = stats_to_dict(stats)
        if gap_rgb_wrench:
            print(
                f"\nReplay buffer ({len(gap_rgb_wrench)} eps): "
                f"rgb_end - wrench_end: mean={np.mean(gap_rgb_wrench):.3f}s "
                f"max={np.max(gap_rgb_wrench):.3f}s"
            )

    return result


def cmd_inference(args: argparse.Namespace) -> dict:
    hw = Path(args.hardware_config)
    if not hw.is_file():
        raise SystemExit(f"Hardware config not found: {hw}")
    poll_streams, stats, buffer_stats = sample_live_layup(
        hardware_config=hw,
        duration_s=args.duration_s,
        depth_clip=args.depth_clip,
        poll_hz=args.poll_hz,
        rgb_query=args.rgb_query,
        depth_query=args.depth_query,
        pose_query=args.pose_query,
        wrench_query=args.wrench_query,
        realsense_serial=getattr(args, "realsense_serial", None),
    )
    print_summary_table(stats, title="Live inference (poll series)")
    print_policy_windows(
        stats,
        args.wrench_horizon,
        args.wrench_downsample_steps,
        compare_train_deploy=True,
    )
    return {
        "inference": stats_to_dict(stats),
        "inference_buffer": stats_to_dict(buffer_stats),
        "n_polls": {k: len(v) for k, v in poll_streams.items()},
    }


def cmd_compare(args: argparse.Namespace) -> dict:
    coll_args = argparse.Namespace(
        unified_zarr=args.unified_zarr,
        unified_zarr_dir=args.unified_zarr_dir,
        also_replay_buffer=None,
        replay_episode=None,
        max_replay_episodes=1,
        wrench_horizon=args.wrench_horizon,
        wrench_downsample_steps=args.wrench_downsample_steps,
    )
    coll = cmd_collection(coll_args)
    inf = cmd_inference(args)
    if args.plot or args.plot_out:
        coll_stats = {
            k: StreamStats(**v)
            for k, v in coll.get("unified", {}).items()
        }
        inf_stats = {
            k: StreamStats(**v)
            for k, v in inf.get("inference", {}).items()
        }
        maybe_plot(coll_stats, inf_stats, Path(args.plot_out) if args.plot_out else None)
    return {"collection": coll, "inference": inf}


def _default_hardware_config() -> str:
    env = os.environ.get("UMIFT_HW_CONFIG")
    if env:
        return env
    folder = os.environ.get("PYRITE_HARDWARE_CONFIG_FOLDERS")
    if folder:
        return str(Path(folder) / "ur5e_wrist_ft.yaml")
    return str(
        UMI_FT_ROOT.parent
        / "hardware_interfaces/workcell/table_top_manip/config/ur5e_wrist_ft.yaml"
    )


INFERENCE_EPILOG = """
Hardware requirements (inference / compare):
  - UR5e + wrist FT (RTDE ~500 Hz wrench/pose)
  - Intel RealSense D405 (RGB + depth via pyrealsense2, ~15 Hz)
  - Built manip_server pybind module
  - Do not run realsense-viewer or ffplay on the D405 immediately before profiling.

Example:
  python UMIFT_Data/scripts/profile_data_frequencies.py inference \\
    --hardware-config $PYRITE_HARDWARE_CONFIG_FOLDERS/ur5e_wrist_ft.yaml \\
    --realsense-serial 344123071493 \\
    --duration-s 10 --depth-clip 0.5 --poll-hz 50
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile UMIFT data stream frequencies (collection zarr vs live layup inference)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_coll = sub.add_parser("collection", help="Offline: unified zarr and optional replay buffer")
    p_coll.add_argument("--unified-zarr", type=str, help="Path to one *_unified.zarr")
    p_coll.add_argument("--unified-zarr-dir", type=str, help="Directory of unified zarrs")
    p_coll.add_argument("--also-replay-buffer", type=str, help="UMIFT replay buffer .zarr")
    p_coll.add_argument("--replay-episode", type=str, help="e.g. episode_47")
    p_coll.add_argument("--max-replay-episodes", type=int, default=64)
    _add_common_flags(p_coll)

    p_inf = sub.add_parser(
        "inference",
        help="Live: ManipServerLayupEnv (needs robot + D405; includes depth)",
        epilog=INFERENCE_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_inf.add_argument(
        "--hardware-config",
        type=str,
        default=_default_hardware_config(),
        help="ManipServer yaml",
    )
    p_inf.add_argument("--duration-s", type=float, default=10.0)
    p_inf.add_argument(
        "--realsense-serial",
        type=str,
        default=os.environ.get("UMIFT_REALSENSE_SERIAL"),
        help="D405 serial for pyrealsense2 (default: $UMIFT_REALSENSE_SERIAL)",
    )
    p_inf.add_argument("--depth-clip", type=float, default=0.5)
    p_inf.add_argument("--poll-hz", type=float, default=50.0)
    p_inf.add_argument("--rgb-query", type=int, default=30)
    p_inf.add_argument("--depth-query", type=int, default=30)
    p_inf.add_argument("--pose-query", type=int, default=200)
    p_inf.add_argument("--wrench-query", type=int, default=200)
    _add_common_flags(p_inf)

    p_cmp = sub.add_parser(
        "compare",
        help="Collection zarr stats vs live inference run",
        epilog=INFERENCE_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_cmp.add_argument("--unified-zarr", type=str)
    p_cmp.add_argument("--unified-zarr-dir", type=str)
    p_cmp.add_argument(
        "--hardware-config",
        type=str,
        default=_default_hardware_config(),
        help="ManipServer yaml",
    )
    p_cmp.add_argument("--duration-s", type=float, default=10.0)
    p_cmp.add_argument(
        "--realsense-serial",
        type=str,
        default=os.environ.get("UMIFT_REALSENSE_SERIAL"),
        help="D405 serial for pyrealsense2",
    )
    p_cmp.add_argument("--depth-clip", type=float, default=0.5)
    p_cmp.add_argument("--poll-hz", type=float, default=50.0)
    p_cmp.add_argument("--rgb-query", type=int, default=30)
    p_cmp.add_argument("--depth-query", type=int, default=30)
    p_cmp.add_argument("--pose-query", type=int, default=200)
    p_cmp.add_argument("--wrench-query", type=int, default=200)
    _add_common_flags(p_cmp)

    return parser


def _add_common_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--wrench-horizon", type=int, default=DEFAULT_WRENCH_HORIZON)
    p.add_argument("--wrench-downsample-steps", type=int, default=DEFAULT_WRENCH_DS)
    p.add_argument("--json-out", type=str, help="Write summary JSON")
    p.add_argument("--plot", action="store_true", help="Show Hz bar chart")
    p.add_argument("--plot-out", type=str, help="Save Hz bar chart to file")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "collection":
        result = cmd_collection(args)
        if args.plot or args.plot_out:
            coll_stats = {k: StreamStats(**v) for k, v in result.get("unified", {}).items()}
            maybe_plot(coll_stats, None, Path(args.plot_out) if args.plot_out else None)
    elif args.command == "inference":
        result = cmd_inference(args)
        if args.plot or args.plot_out:
            inf_stats = {k: StreamStats(**v) for k, v in result.get("inference", {}).items()}
            maybe_plot(None, inf_stats, Path(args.plot_out) if args.plot_out else None)
    elif args.command == "compare":
        result = cmd_compare(args)
    else:
        raise SystemExit(f"Unknown command: {args.command}")

    if args.json_out:
        out = Path(args.json_out)

        def _default(o):
            if isinstance(o, (np.floating, np.integer)):
                return float(o)
            raise TypeError

        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=_default)
        print(f"Wrote JSON to {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

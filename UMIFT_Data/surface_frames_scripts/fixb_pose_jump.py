"""Flag tracking jumps in the October stringer_v3.5 tool pose.

A step is the interval between consecutive robot-rate TCP samples. Speed uses
dt floored at one control period so a sub-millisecond timestamp glitch cannot
look like a multi-meter-per-second move. Thresholds sit in the sparse tail of
that histogram: 0.3 m/s is about the 94th percentile of ordinary layup motion
and hits every episode, so the flag is 1.0 m/s or 150 deg/s.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SPEED_M_S = 1.0
ANG_DEG_S = 150.0
DT_FLOOR_S = 0.004
N_EPISODES = 50

FLAG_NPZ = (
    Path(__file__).resolve().parent / "assets" / "oct_v35_pose_jump_flags.npz"
)
ZARR = (
    "/mnt/data/umidata/ft/mold_aware_datasets/"
    "stringer_v3.5_fixb/r1_world_world/acp_replay_buffer.zarr"
)


def speeds_for_episode(pose7: np.ndarray, times: np.ndarray):
    """Return (speed_m_s, ang_deg_s) for each interval pose[i] -> pose[i+1]."""
    from scipy.spatial.transform import Rotation

    pose7 = np.asarray(pose7, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    dt = np.maximum(np.diff(times), DT_FLOOR_S)
    speed = np.linalg.norm(np.diff(pose7[:, :3], axis=0), axis=1) / dt
    quat_xyzw = pose7[:, [4, 5, 6, 3]]
    rot = Rotation.from_quat(quat_xyzw)
    ang = np.degrees((rot[:-1].inv() * rot[1:]).magnitude() / dt)
    return speed.astype(np.float64), ang.astype(np.float64)


def flag_from_speeds(speed: np.ndarray, ang: np.ndarray) -> np.ndarray:
    """Length n_pose. Index 0 is never a jump; index i flags the arrival sample."""
    hit = (speed > SPEED_M_S) | (ang > ANG_DEG_S)
    flag = np.zeros(hit.size + 1, dtype=bool)
    flag[1:] = hit
    return flag


def build_flag_table(zarr_path: str = ZARR) -> dict[str, np.ndarray]:
    import zarr

    root = zarr.open(zarr_path, "r")
    table = {}
    for i in range(N_EPISODES):
        ep = root["data"][f"episode_{i}"]
        speed, ang = speeds_for_episode(
            np.asarray(ep["ts_pose_fb_0"]),
            np.asarray(ep["robot_time_stamps_0"]),
        )
        table[f"episode_{i}"] = flag_from_speeds(speed, ang)
    return table


def histogram(zarr_path: str = ZARR) -> dict:
    import zarr

    root = zarr.open(zarr_path, "r")
    speeds = []
    angs = []
    for i in range(N_EPISODES):
        ep = root["data"][f"episode_{i}"]
        speed, ang = speeds_for_episode(
            np.asarray(ep["ts_pose_fb_0"]),
            np.asarray(ep["robot_time_stamps_0"]),
        )
        speeds.append(speed)
        angs.append(ang)
    speed = np.concatenate(speeds)
    ang = np.concatenate(angs)
    bins = np.array(
        [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 6.0]
    )
    counts, edges = np.histogram(speed, bins=bins)
    return {
        "n_steps": int(speed.size),
        "speed_bins": edges.tolist(),
        "speed_counts": counts.astype(int).tolist(),
        "speed_percentiles": {
            str(p): float(np.percentile(speed, p))
            for p in (50, 90, 95, 99, 99.5, 99.9)
        },
        "ang_percentiles_deg_s": {
            str(p): float(np.percentile(ang, p))
            for p in (50, 90, 95, 99, 99.5, 99.9)
        },
    }


def save_flags(path: Path = FLAG_NPZ, zarr_path: str = ZARR) -> dict:
    table = build_flag_table(zarr_path)
    hist = histogram(zarr_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "speed_m_s": np.array([SPEED_M_S]),
        "ang_deg_s": np.array([ANG_DEG_S]),
        "dt_floor_s": np.array([DT_FLOOR_S]),
        **table,
    }
    np.savez_compressed(path, **payload)
    meta = {
        "speed_m_s": SPEED_M_S,
        "ang_deg_s": ANG_DEG_S,
        "dt_floor_s": DT_FLOOR_S,
        "n_flagged_steps": int(sum(int(v.sum()) for v in table.values())),
        "n_flagged_episodes": int(sum(bool(v.any()) for v in table.values())),
        "histogram": hist,
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return meta


def load_flags(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"pose jump flags not found: {path}")
    with np.load(path) as data:
        return {
            key: np.asarray(data[key], dtype=bool)
            for key in data.files
            if key.startswith("episode_")
        }


def action_indices_for_query(dataset, epi_id: int, rgb_id: int) -> np.ndarray:
    from diffusion_policy.common.sampler_umift import (
        sparse_action_indices_from_rgb_timestamps,
    )

    ep = dataset.replay_buffer["data"][f"episode_{int(epi_id)}"]
    action_horizon = int(dataset.shape_meta["sample"]["action"]["sparse"]["horizon"])
    rgb_frame_stride = int(
        dataset.shape_meta["sample"]["action"]["sparse"]["down_sample_steps"]
    )
    action_idx, _ = sparse_action_indices_from_rgb_timestamps(
        action_times=np.squeeze(np.asarray(ep["action_time_stamps"], dtype=np.float64)),
        rgb_times=np.squeeze(
            np.asarray(ep["obs"]["rgb_time_stamps_0"], dtype=np.float64)
        ),
        rgb_id=int(rgb_id),
        action_horizon=action_horizon,
        rgb_frame_stride=rgb_frame_stride,
    )
    return np.asarray(action_idx, dtype=np.int64)


def keep_weight(flag: np.ndarray, action_idx: np.ndarray, horizon: int) -> np.ndarray:
    """1 on action steps whose pose has not jumped since the chunk start.

    Step 0 stays supervised. A later step is zero once any flagged arrival
    sample lies in [start, this step]. A jump on the chunk's first sample
    still zeros the rest of the horizon.
    """
    keep = np.ones(horizon, dtype=np.float32)
    if flag is None or len(action_idx) == 0:
        return keep
    a0 = int(action_idx[0])
    for t in range(1, horizon):
        a_t = int(action_idx[min(t, len(action_idx) - 1)])
        lo = a0
        hi = a_t + 1
        if lo < hi and np.any(flag[lo:hi]):
            keep[t] = 0.0
    return keep


def loss_weight_for_query(dataset, flags: dict[str, np.ndarray], epi_id: int, rgb_id: int, horizon: int, dim: int) -> np.ndarray:
    flag = flags.get(f"episode_{int(epi_id)}")
    if flag is None:
        return np.ones((horizon, dim), dtype=np.float32)
    try:
        action_idx = action_indices_for_query(dataset, epi_id, rgb_id)
    except (ValueError, IndexError):
        return np.ones((horizon, dim), dtype=np.float32)
    if int(flag.shape[0]) < int(action_idx.max()) + 1 if len(action_idx) else 0:
        raise RuntimeError(
            f"episode_{epi_id} pose-jump flag length {flag.shape[0]} "
            f"is shorter than action index {int(action_idx.max())}"
        )
    return np.repeat(keep_weight(flag, action_idx, horizon)[:, None], dim, axis=1)


def log_pose_jump_mask(dataset, flags: dict[str, np.ndarray], split: str) -> dict:
    horizon = int(dataset.shape_meta["sample"]["action"]["sparse"]["horizon"])
    n_steps = 0
    n_masked = 0
    for epi_id, rgb_id in dataset.sampler.indices:
        keep = loss_weight_for_query(
            dataset, flags, int(epi_id), int(rgb_id), horizon, 1
        )[:, 0]
        n_steps += horizon
        n_masked += int(np.sum(keep == 0.0))
    frac = float(n_masked / max(n_steps, 1))
    stats = {
        "split": split,
        "n_chunks": len(dataset.sampler.indices),
        "n_action_steps": n_steps,
        "n_steps_masked": n_masked,
        "frac_steps_masked": frac,
        "speed_m_s": SPEED_M_S,
        "ang_deg_s": ANG_DEG_S,
    }
    print(
        f"[pose-jump] {split} action-loss mask: "
        f"{n_masked}/{n_steps} steps masked ({frac:.1%}) "
        f"over {len(dataset.sampler.indices)} chunks "
        f"(>{SPEED_M_S:g} m/s or >{ANG_DEG_S:g} deg/s)"
    )
    return stats

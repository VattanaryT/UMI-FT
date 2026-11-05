"""Pick one demonstrated episode-start pose and express it in the robot base.

Dataset ``ts_pose_fb_0`` for the tag-tracked layup sets is fixture-frame pose7
``[x, y, z, qw, qx, qy, qz]``. The runner commands the UR in right_base. The
chosen start is the real episode whose first position is closest to the median
start, so the target is inside the demonstrated cluster and is a pose a person
actually used.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import zarr
from scipy.spatial.transform import Rotation

# Local copies of the Fix B sets. Checkpoints trained on Hyak store the
# container path ``/data/acp_replay_buffer.zarr``, which is not on this PC.
_LOCAL_DATASET_BY_TASK = {
    "OCT-V35-FIXB-R1-WORLD-WORLD": (
        "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb/"
        "r1_world_world/acp_replay_buffer.zarr"
    ),
    "OCT-V35-FIXB-R1-H4S": (
        "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb/"
        "r1_world_world/acp_replay_buffer.zarr"
    ),
    # r1rel is the same October zarr as r1, with relative obs and actions.
    "OCT-V35-FIXB-R1REL": (
        "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb/"
        "r1_world_world/acp_replay_buffer.zarr"
    ),
    "OCT-V35-FIXB-R1REL-H4S": (
        "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb/"
        "r1_world_world/acp_replay_buffer.zarr"
    ),
    "OCT-V35-FIXB-R2-WORLD-FIXB": (
        "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb/"
        "r2_world_fixb/acp_replay_buffer.zarr"
    ),
    "OCT-V35-FIXB-R2-H4S": (
        "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb/"
        "r2_world_fixb/acp_replay_buffer.zarr"
    ),
    "OCT-V35-FIXB-R4-WORLD-FIXB-TILT": (
        "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb/"
        "r4_world_fixb_tilt/acp_replay_buffer.zarr"
    ),
    "OCT-V35-FIXB-R4-H4S": (
        "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb/"
        "r4_world_fixb_tilt/acp_replay_buffer.zarr"
    ),
    "SCANNER-FIXB-R1-WORLD-WORLD": (
        "/mnt/data/umidata/ft/mold_aware_datasets/stringer_scanner_v2_fixb/"
        "r1_world_world/acp_replay_buffer.zarr"
    ),
    "SCANNER-FIXB-R2-WORLD-FIXB": (
        "/mnt/data/umidata/ft/mold_aware_datasets/stringer_scanner_v2_fixb/"
        "r2_world_fixb/acp_replay_buffer.zarr"
    ),
    "SCANNER-FIXB-R4-WORLD-FIXB-TILT": (
        "/mnt/data/umidata/ft/mold_aware_datasets/stringer_scanner_v2_fixb/"
        "r4_world_fixb_tilt/acp_replay_buffer.zarr"
    ),
}


def resolve_dataset_zarr(cfg_dataset_path: str, task_name: str) -> Optional[Path]:
    """Return a zarr that exists on this machine, or None."""
    raw = str(cfg_dataset_path or "").strip()
    if raw:
        path = Path(raw)
        if path.is_dir():
            return path
    fallback = _LOCAL_DATASET_BY_TASK.get(str(task_name or "").strip())
    if fallback and Path(fallback).is_dir():
        return Path(fallback)
    return None


def _episode_names(root) -> list:
    names = [k for k in root["data"].keys() if str(k).startswith("episode_")]
    return sorted(names, key=lambda s: int(str(s).split("_")[-1]))


# episode_30 is the median start. On the robot its tool +z was 22 deg from
# straight down and the TCP sat ~48 mm off the mesh. episode_39 is 27 mm from
# that median, 30 mm lower (+Y), and 7 deg from straight down.
DATASET_START_EPISODE = "episode_39"


def central_episode_start(
    zarr_path: Path, episode_name: Optional[str] = None
) -> Tuple[str, np.ndarray, np.ndarray]:
    """Episode whose first TCP position is closest to the median start.

    ``episode_name`` picks one demonstrated start instead of the median.
    Returns ``(episode_name, pose7, median_xyz)``. ``pose7`` is in the dataset
    frame (fixture for tag-tracked layup sets).
    """
    root = zarr.open(str(zarr_path), mode="r")
    names = _episode_names(root)
    if not names:
        raise ValueError(f"No episodes in {zarr_path}")
    starts = np.stack(
        [
            np.asarray(root["data"][name]["ts_pose_fb_0"][0], dtype=np.float64).reshape(7)
            for name in names
        ]
    )
    median_xyz = np.median(starts[:, :3], axis=0)
    if episode_name:
        if episode_name not in names:
            raise ValueError(f"{episode_name} not in {zarr_path}")
        index = names.index(episode_name)
    else:
        distances = np.linalg.norm(starts[:, :3] - median_xyz, axis=1)
        index = int(np.argmin(distances))
    return names[index], starts[index].copy(), median_xyz


def dataset_poses_are_fixture(zarr_path: Path) -> bool:
    """True when conversion_info says the poses are tag / fixture frame."""
    root = zarr.open(str(zarr_path), mode="r")
    info = root.attrs.get("conversion_info", "")
    if isinstance(info, (bytes, bytearray)):
        info = info.decode("utf-8", errors="replace")
    if not isinstance(info, str):
        info = json.dumps(info)
    try:
        parsed = json.loads(info)
    except json.JSONDecodeError:
        parsed = {}
    source = str(parsed.get("pose_source", "") if isinstance(parsed, dict) else "")
    text = (source + " " + info).lower()
    return "fixture" in text or "tag" in text


def pose7_to_Rp(pose7: np.ndarray):
    flat = np.asarray(pose7, dtype=np.float64).reshape(-1, 7)
    rot = Rotation.from_quat(flat[:, [4, 5, 6, 3]]).as_matrix()
    return rot, flat[:, :3].copy()


def Rp_to_pose7(rot: np.ndarray, position: np.ndarray) -> np.ndarray:
    quat_xyzw = Rotation.from_matrix(rot).as_quat()
    return np.concatenate([position, quat_xyzw[:, 3:4], quat_xyzw[:, :3]], axis=1)


def pose7_change_frame(pose7: np.ndarray, T_dst_from_src: np.ndarray) -> np.ndarray:
    """Apply a 4x4 SE3 to pose7. ``v_dst = T_dst_from_src @ v_src``."""
    rot, position = pose7_to_Rp(pose7)
    transform = np.asarray(T_dst_from_src, dtype=np.float64)
    rot_dst = np.einsum("ij,njk->nik", transform[:3, :3], rot)
    pos_dst = (transform[:3, :3] @ position.T).T + transform[:3, 3]
    return Rp_to_pose7(rot_dst, pos_dst).reshape(np.asarray(pose7).shape)


def tcp_inside_safe_zone(xyz: np.ndarray, safe_zone) -> bool:
    """``safe_zone`` is ``[xmin, xmax, ymin, ymax, zmin, zmax]`` in meters."""
    x, y, z = np.asarray(xyz, dtype=np.float64).reshape(3)
    xmin, xmax, ymin, ymax, zmin, zmax = [float(v) for v in safe_zone]
    return (
        xmin <= x <= xmax
        and ymin <= y <= ymax
        and zmin <= z <= zmax
    )


def read_cartesian_safe_zone(hardware_config_path: str):
    """Return the UR cartesian safe zone from the hardware yaml, or None."""
    import yaml

    with open(hardware_config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    for key, node in cfg.items():
        if not str(key).startswith("ur_rtde"):
            continue
        if not isinstance(node, dict):
            continue
        interface = node.get("robot_interface_config") or {}
        zone = interface.get("safe_zone")
        if zone is not None and len(zone) == 6:
            return [float(v) for v in zone]
    return None

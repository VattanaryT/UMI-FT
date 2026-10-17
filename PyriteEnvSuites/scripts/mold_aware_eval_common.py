"""Shared decode + mold-field helpers for the mold-aware open-loop evaluation.

Action layout (per robot, action21):
  [cmd pos(3), cmd rot6d(6), vt pos(3), vt rot6d(6), stiffness, gripper, force]

world_world labels are absolute fixture-frame poses. Surface labels at robot
sample i are
  p = R_field(i)^T (p_target(i) - p_fb(i)) / surface_delta_scale
  R = R_field(i)^T R_target(i)
so decoding needs a reference feedback pose and surface frame:
  "oracle"   uses the expert fb / R_field at each horizon step's own sample
             (what the label means; not available on hardware),
  "anchored" holds fb / R_field fixed at the query sample (what an open-loop
             executor could use).
For world_world both modes are identical.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import open3d as o3d
import zarr
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from PyriteUtility.spatial_math import spatial_utilities as su

RUNS = ("world_world", "world_surface", "surface_surface")
DECODE_MODES = ("oracle", "anchored")
TARGETS = ("vt", "cmd")
TOOL_NORMAL_AXIS = 2  # TCP +z (forward); the axis that faces the mold


def frames_from_normals(normals: np.ndarray) -> np.ndarray:
    """Same construction as build_mold_aware_datasets._frames_from_normals."""
    n = np.asarray(normals, dtype=float)
    n = n / np.clip(np.linalg.norm(n, axis=1, keepdims=True), 1e-9, None)
    helper = np.zeros_like(n)
    helper[:, 0] = 1.0
    swap = np.abs(n[:, 0]) > 0.9
    helper[swap] = (0.0, 1.0, 0.0)
    t1 = np.cross(helper, n)
    t1 = t1 / np.clip(np.linalg.norm(t1, axis=1, keepdims=True), 1e-9, None)
    t2 = np.cross(n, t1)
    return np.stack([t1, t2, n], axis=-1)


class MoldField:
    """Point-to-plane signed distance against the fixture-frame mold cloud.

    Normals are re-oriented consistently (the stored training field has
    per-point sign flips); global sign follows the builder (mean n_y > 0).
    """

    def __init__(self, ply_path: Path, k: int = 30):
        pcd = o3d.io.read_point_cloud(str(ply_path))
        pcd.orient_normals_consistent_tangent_plane(k)
        n = np.asarray(pcd.normals, dtype=float)
        if float(np.mean(n[:, 1])) < 0:
            n = -n
        self.points = np.asarray(pcd.points, dtype=float)
        self.normals = n / np.linalg.norm(n, axis=1, keepdims=True)
        self.tree = cKDTree(self.points)

    def query(self, p: np.ndarray):
        shape = p.shape[:-1]
        _, j = self.tree.query(p.reshape(-1, 3))
        n = self.normals[j]
        sd = np.sum((p.reshape(-1, 3) - self.points[j]) * n, axis=1)
        return sd.reshape(shape), n.reshape(shape + (3,))


def load_expert_episode(dataset_root: Path, episode: str) -> Dict[str, np.ndarray]:
    """Fixture-frame expert streams (world_world zarr) + stored training field."""
    ep = zarr.open(str(dataset_root / "world_world" / "acp_replay_buffer.zarr"), "r")[
        "data"
    ][episode]
    fb = np.asarray(ep["ts_pose_fb_0"], dtype=float)
    cmd = np.asarray(ep["ts_pose_command_0"], dtype=float)
    vt = np.asarray(ep["ts_pose_virtual_target_0"], dtype=float)
    return {
        "fb": su.pose7_to_SE3(fb),
        "cmd": su.pose7_to_SE3(cmd),
        "vt": su.pose7_to_SE3(vt),
        "stiffness": np.asarray(ep["stiffness_0"], dtype=float).reshape(-1),
        "R_field": frames_from_normals(np.asarray(ep["mold_normal_0"], dtype=float)),
        "t": np.asarray(ep["robot_time_stamps_0"], dtype=float).reshape(-1),
    }


def load_surface_scale(dataset_root: Path) -> np.ndarray:
    meta = json.loads((dataset_root / "shared" / "meta.json").read_text())
    return np.asarray(meta["surface_delta_scale"], dtype=float)


def decode_actions(
    action: np.ndarray,
    run: str,
    expert: Dict[str, np.ndarray],
    action_robot_idx: np.ndarray,
    query_robot_idx: np.ndarray,
    scale: np.ndarray,
    mode: str,
) -> Dict[str, np.ndarray]:
    """action (Q, H, 21) -> fixture SE3 (Q, H, 4, 4) for cmd / vt, plus stiffness."""
    a = np.asarray(action, dtype=np.float64)
    out = {"stiffness": a[..., 18]}
    for target, sl in (("cmd", slice(0, 9)), ("vt", slice(9, 18))):
        T = su.pose9_to_SE3(np.ascontiguousarray(a[..., sl]))
        if run != "world_world":
            if mode == "oracle":
                ref = action_robot_idx
            elif mode == "anchored":
                ref = np.broadcast_to(query_robot_idx[:, None], action_robot_idx.shape)
            else:
                raise ValueError(mode)
            R_f = expert["R_field"][ref]
            p_fb = expert["fb"][ref][..., :3, 3]
            p = p_fb + np.einsum("...ij,...j->...i", R_f, T[..., :3, 3] * scale)
            R = np.einsum("...ij,...jk->...ik", R_f, T[..., :3, :3])
            T = T.copy()
            T[..., :3, :3] = R
            T[..., :3, 3] = p
        out[target] = T
    return out


def rot_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    rel = np.einsum("...ji,...jk->...ik", Ra, Rb)
    shape = rel.shape[:-2]
    rv = Rotation.from_matrix(rel.reshape(-1, 3, 3)).as_rotvec()
    return np.degrees(np.linalg.norm(rv, axis=-1)).reshape(shape)


def load_export(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as d:
        return {k: d[k] for k in d.files}

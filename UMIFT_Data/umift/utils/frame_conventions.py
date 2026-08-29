"""
UMIFT frame helpers (umift_std_v2).

Constants and core SE(3)/sensor levers are owned by:
  Robotiq-portable-real2sim/python/frame_conventions.py

This module re-exports that SSOT and adds offline-convert helpers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

_ROBOTIQ_PYTHON = (
    Path(__file__).resolve().parents[4] / "Robotiq-portable-real2sim" / "python"
)
if _ROBOTIQ_PYTHON.is_dir() and str(_ROBOTIQ_PYTHON) not in sys.path:
    sys.path.insert(0, str(_ROBOTIQ_PYTHON))

from frame_conventions import (  # noqa: E402
    DEFAULT_P_D405_IN_TCP_M,
    DEFAULT_P_FT_IN_TCP_M,
    DEFAULT_P_T265_IN_TCP_M,
    DEFAULT_R_D405_TCP_EULER_XYZ_DEG,
    DEFAULT_R_FT_IN_TCP_M,
    DEFAULT_R_TCP_D405_EULER_XYZ_DEG,
    DEFAULT_T_CAM_TCP_M,
    DEFAULT_T_D405_TCP_M,
    DEFAULT_T_T265_TCP_M,
    FRAME_CONVENTION_VERSION,
    GRAVITY_WORLD_MPS2,
    R_CAM_TCP,
    R_FIXTURE_MASTER_TAG,
    R_FT_TCP,
    R_STD_FT,
    R_STD_T265,
    R_from_euler_xyz_deg,
    T_from_Rt,
    default_T_d405_tcp,
    default_T_fixture_master_tag,
    default_T_t265_tcp,
    default_T_tcp_d405,
    default_T_tcp_t265,
    gravity_in_tcp_frame,
    inv_T,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion_wxyz,
)
from frame_conventions import ft_wrench_raw_to_tcp as ft_wrench_raw_to_tcp_wrench  # noqa: E402

# Legacy aliases still imported by older callers
R_TCPWORLD_FTWORLD = R_FT_TCP.T
R_FTBODY_TCPBODY = R_FT_TCP
R_CAM_FT = np.array(
    [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64
)
_FT_TO_T265_IN_FT_M = np.array([0.155, -0.10, -0.35], dtype=np.float64)
GRAVITY_WORLD_Y_UP_MPS2 = np.array([0.0, -9.8, 0.0], dtype=np.float64)


def apply_similarity_std_t265(
    positions: np.ndarray, orientations_wxyz: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """T265 raw → std world/body via similarity with R_STD_T265."""
    positions = np.asarray(positions, dtype=np.float64)
    orientations = np.asarray(orientations_wxyz, dtype=np.float64)
    M = R_STD_T265
    out_pos = (M @ positions.T).T
    out_quat = np.zeros_like(orientations)
    for i in range(positions.shape[0]):
        qw, qx, qy, qz = orientations[i]
        R = quaternion_to_rotation_matrix(qx, qy, qz, qw)
        Rp = M @ R @ M.T
        out_quat[i] = rotation_matrix_to_quaternion_wxyz(Rp)
    return out_pos, out_quat


def t265_std_poses_to_tcp_poses(
    positions_odom_t265: np.ndarray,
    orientations_wxyz: np.ndarray,
    t_t265_tcp_m: np.ndarray | None = None,
) -> np.ndarray:
    """T_odom←tcp = T_odom←t265_std · T_t265←tcp (identity R by default)."""
    positions = np.asarray(positions_odom_t265, dtype=np.float64)
    orientations = np.asarray(orientations_wxyz, dtype=np.float64)
    t = (
        DEFAULT_T_T265_TCP_M
        if t_t265_tcp_m is None
        else np.asarray(t_t265_tcp_m, dtype=np.float64).reshape(3)
    )
    tcp_poses = np.zeros((positions.shape[0], 7), dtype=np.float64)
    for idx in range(positions.shape[0]):
        qw, qx, qy, qz = orientations[idx]
        R = quaternion_to_rotation_matrix(qx, qy, qz, qw)
        tcp_poses[idx, :3] = positions[idx] + R @ t
        tcp_poses[idx, 3:] = orientations[idx]
    return tcp_poses


def ft_poses_to_tcp_poses(
    positions_ft: np.ndarray,
    orientations_wxyz: np.ndarray,
    sensor_offset_xyz_m: np.ndarray | None = None,
) -> np.ndarray:
    """LEGACY: publisher FT-frame poses → TCP."""
    positions = np.asarray(positions_ft, dtype=np.float64)
    orientations = np.asarray(orientations_wxyz, dtype=np.float64)
    sensor_offset = (
        np.zeros(3, dtype=np.float64)
        if sensor_offset_xyz_m is None
        else np.asarray(sensor_offset_xyz_m, dtype=np.float64).reshape(3)
    )
    R_ft_tcp_legacy = np.array(
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    R_tcpworld_ftworld = R_ft_tcp_legacy.T
    tcp_poses = np.zeros((positions.shape[0], 7), dtype=np.float64)
    for idx in range(positions.shape[0]):
        qw, qx, qy, qz = orientations[idx]
        r_ftworld_ftbody = quaternion_to_rotation_matrix(qx, qy, qz, qw)
        r_tcpworld_tcpbody = (
            R_tcpworld_ftworld @ r_ftworld_ftbody @ R_ft_tcp_legacy
        )
        p_tcpworld_sensor = R_tcpworld_ftworld @ positions[idx]
        tcp_poses[idx, :3] = p_tcpworld_sensor - r_tcpworld_tcpbody @ sensor_offset
        tcp_poses[idx, 3:] = rotation_matrix_to_quaternion_wxyz(r_tcpworld_tcpbody)
    return tcp_poses


def ft_wrench_to_tcp_wrench(wrench_ft: np.ndarray) -> np.ndarray:
    """LEGACY: rotate FT wrench with legacy R_FT_TCP (row-vector @ R)."""
    R_ft_tcp_legacy = np.array(
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    wrench_ft = np.asarray(wrench_ft, dtype=np.float64)
    wrench_tcp = np.empty_like(wrench_ft)
    wrench_tcp[:, 0:3] = wrench_ft[:, 0:3] @ R_ft_tcp_legacy
    wrench_tcp[:, 3:6] = wrench_ft[:, 3:6] @ R_ft_tcp_legacy
    return wrench_tcp


def camera_poses_to_tcp_poses(
    positions_world_cam: np.ndarray,
    orientations_wxyz: np.ndarray,
    t_cam_tcp_m: np.ndarray = DEFAULT_T_CAM_TCP_M,
) -> np.ndarray:
    """LEGACY: camera poses → TCP using R_CAM_TCP."""
    positions = np.asarray(positions_world_cam, dtype=np.float64)
    orientations = np.asarray(orientations_wxyz, dtype=np.float64)
    t_cam_tcp = np.asarray(t_cam_tcp_m, dtype=np.float64).reshape(3)
    tcp_poses = np.zeros((positions.shape[0], 7), dtype=np.float64)
    for idx in range(positions.shape[0]):
        qw, qx, qy, qz = orientations[idx]
        r_world_cam = quaternion_to_rotation_matrix(qx, qy, qz, qw)
        r_world_tcp = r_world_cam @ R_CAM_TCP
        tcp_poses[idx, :3] = positions[idx] + r_world_cam @ t_cam_tcp
        tcp_poses[idx, 3:] = rotation_matrix_to_quaternion_wxyz(r_world_tcp)
    return tcp_poses


def check_tcp_pose_consistency(ts_pose_fb: np.ndarray) -> Dict[str, float]:
    """Sanity check: position deltas vs TCP +Z when moving forward."""
    ts_pose_fb = np.asarray(ts_pose_fb, dtype=np.float64)
    if ts_pose_fb.shape[0] < 3:
        return {"num_samples": float(ts_pose_fb.shape[0]), "forward_alignment": 0.0}

    positions = ts_pose_fb[:, :3]
    deltas = positions[1:] - positions[:-1]
    delta_norms = np.linalg.norm(deltas, axis=1)
    valid = delta_norms > 1e-6
    if not np.any(valid):
        return {"num_samples": float(ts_pose_fb.shape[0]), "forward_alignment": 0.0}

    qw, qx, qy, qz = (
        ts_pose_fb[:-1, 3],
        ts_pose_fb[:-1, 4],
        ts_pose_fb[:-1, 5],
        ts_pose_fb[:-1, 6],
    )
    z_tcp = np.stack(
        [
            2 * (qx * qz + qw * qy),
            2 * (qy * qz - qw * qx),
            1 - 2 * (qx * qx + qy * qy),
        ],
        axis=1,
    )
    cos_angles = np.sum(deltas[valid] * z_tcp[valid], axis=1) / delta_norms[valid]
    return {
        "num_samples": float(ts_pose_fb.shape[0]),
        "forward_alignment": float(np.mean(cos_angles)),
    }


def conversion_info_dict(
    t_t265_tcp_m: np.ndarray | None = None,
    r_ft_in_tcp_m: np.ndarray | None = None,
    extra: Dict[str, Any] | None = None,
    *,
    t_cam_tcp_m: np.ndarray | None = None,
) -> Dict[str, Any]:
    if t_t265_tcp_m is None:
        t_t265_tcp_m = DEFAULT_T_CAM_TCP_M if t_cam_tcp_m is None else t_cam_tcp_m
    if r_ft_in_tcp_m is None:
        r_ft_in_tcp_m = DEFAULT_R_FT_IN_TCP_M
    info: Dict[str, Any] = {
        "frame_convention_version": FRAME_CONVENTION_VERSION,
        "world_frame": "+Y down, +X right, +Z forward (umift_std_v2)",
        "tcp_frame": "+x right, +y down, +z forward",
        "pose_and_wrench_body_frame": "tcp",
        "sensor_levers_measured_in": "tcp",
        "R_std_t265": R_STD_T265.tolist(),
        "R_std_ft": R_STD_FT.tolist(),
        "R_fixture_master_tag": R_FIXTURE_MASTER_TAG.tolist(),
        "p_ft_in_tcp_m": np.asarray(DEFAULT_P_FT_IN_TCP_M, dtype=np.float64).tolist(),
        "p_t265_in_tcp_m": np.asarray(DEFAULT_P_T265_IN_TCP_M, dtype=np.float64).tolist(),
        "p_d405_in_tcp_m": np.asarray(DEFAULT_P_D405_IN_TCP_M, dtype=np.float64).tolist(),
        "r_tcp_d405_euler_xyz_deg": np.asarray(
            DEFAULT_R_TCP_D405_EULER_XYZ_DEG, dtype=np.float64
        ).tolist(),
        # Derived (T_sensor←tcp translations / legacy names)
        "t_t265_tcp_m": np.asarray(t_t265_tcp_m, dtype=np.float64).tolist(),
        "r_ft_in_tcp_m": np.asarray(r_ft_in_tcp_m, dtype=np.float64).tolist(),
        "t_d405_tcp_m": np.asarray(DEFAULT_T_D405_TCP_M, dtype=np.float64).tolist(),
        "r_d405_tcp_euler_xyz_deg": np.asarray(
            DEFAULT_R_D405_TCP_EULER_XYZ_DEG, dtype=np.float64
        ).tolist(),
        "gravity_world_mps2": GRAVITY_WORLD_MPS2.tolist(),
        "ssot": "Robotiq-portable-real2sim/python/frame_conventions.py",
    }
    if extra:
        info.update(extra)
    return info


def conversion_info_json(
    t_t265_tcp_m: np.ndarray | None = None,
    r_ft_in_tcp_m: np.ndarray | None = None,
    extra: Dict[str, Any] | None = None,
    *,
    t_cam_tcp_m: np.ndarray | None = None,
) -> str:
    return json.dumps(
        conversion_info_dict(
            t_t265_tcp_m=t_t265_tcp_m,
            r_ft_in_tcp_m=r_ft_in_tcp_m,
            extra=extra,
            t_cam_tcp_m=t_cam_tcp_m,
        )
    )

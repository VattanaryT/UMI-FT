"""
Canonical frame conventions for the portable gripper -> UMIFT pipeline.

Single source of truth for axis labels and fixed rig rotations.

References:
  - UMIFT_Data/scripts/new_process_demos.py (GoPro/TCP and world +Y up)
  - UMIFT_Data/umift/utils/rotation_utils.py (gopro2tcp)
  - Robotiq-portable-real2sim t265_ros_publisher (T265 / FT sensor axis labels)
"""

from __future__ import annotations

import json
from typing import Any, Dict

import numpy as np

# World frame (T265 / ARKit native): +Y up, gravity opposite +Y.
GRAVITY_WORLD_MPS2 = np.array([0.0, -9.8, 0.0], dtype=np.float64)

# Body frames (axis labels):
#   TCP / GoPro: +x right, +y down, +z forward (arm outward from wall at inference)
#   T265 camera pose frame: +X right, +Y up, +Z backward
#   FT300 sensor frame: +X down, +Y left, +Z forward

# x=-y, y=x, z=z

# Fixed rotations between body frames (column-vector convention: v_parent = R @ v_child).
R_CAM_TCP = np.diag([1.0, -1.0, -1.0])  # 180 deg about camera +X (GoPro/TCP convention)
R_FT_TCP = np.array(
    [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
)
# Column-vector convention:
#   R_TCPWORLD_FTWORLD maps coordinates expressed in FT-style world axes into
#   TCP/UR5-style world axes.
#   R_FTBODY_TCPBODY maps TCP body axes into FT body coordinates.
R_TCPWORLD_FTWORLD = R_FT_TCP.T
R_FTBODY_TCPBODY = R_FT_TCP
R_CAM_FT = np.array(
    [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64
)

# Pure translation from GoPro frame to TCP (matches rotation_utils.CONST_TRANSFORM).
GOPRO_TO_TCP_TRANSLATION_M = np.array([0.0, 0.086, 0.14465], dtype=np.float64)

# Default camera->TCP lever arm (meters): TCP origin expressed in camera frame.
# Rig geometry: FT sensor is [0.155, -0.10, -0.35] m from T265 in FT frame;
# TCP origin defaults to the FT sensor origin for the portable gripper rig.
_FT_TO_T265_IN_FT_M = np.array([0.155, -0.10, -0.35], dtype=np.float64)
DEFAULT_T_CAM_TCP_M = R_CAM_FT @ (-_FT_TO_T265_IN_FT_M)

FRAME_CONVENTION_VERSION = "umift_tcp_v1"


def rotation_matrix_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [qw, qx, qy, qz]."""
    rotation = np.asarray(rotation, dtype=np.float64)
    trace = np.trace(rotation)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rotation[2, 1] - rotation[1, 2]) / s
        qy = (rotation[0, 2] - rotation[2, 0]) / s
        qz = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        qw = (rotation[2, 1] - rotation[1, 2]) / s
        qx = 0.25 * s
        qy = (rotation[0, 1] + rotation[1, 0]) / s
        qz = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        qw = (rotation[0, 2] - rotation[2, 0]) / s
        qx = (rotation[0, 1] + rotation[1, 0]) / s
        qy = 0.25 * s
        qz = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        qw = (rotation[1, 0] - rotation[0, 1]) / s
        qx = (rotation[0, 2] + rotation[2, 0]) / s
        qy = (rotation[1, 2] + rotation[2, 1]) / s
        qz = 0.25 * s
    quat = np.array([qw, qx, qy, qz], dtype=np.float64)
    return quat / np.linalg.norm(quat)


def quaternion_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert quaternion (x,y,z,w) to 3x3 rotation matrix."""
    norm = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm < 1e-8:
        return np.eye(3)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def gravity_in_ft_sensor_frame_from_camera_orientation(
    rotation_world_cam: np.ndarray,
) -> np.ndarray:
    """Project world gravity into the FT sensor frame using camera orientation."""
    rotation_world_cam = np.asarray(rotation_world_cam, dtype=np.float64)
    if rotation_world_cam.shape != (3, 3):
        raise ValueError(f"rotation must be 3x3, got shape {rotation_world_cam.shape}")
    g_cam = rotation_world_cam.T @ GRAVITY_WORLD_MPS2
    return R_CAM_FT @ g_cam


def camera_poses_to_tcp_poses(
    positions_world_cam: np.ndarray,
    orientations_wxyz: np.ndarray,
    t_cam_tcp_m: np.ndarray = DEFAULT_T_CAM_TCP_M,
) -> np.ndarray:
    """
    Convert camera poses to TCP poses in UMIFT format [x, y, z, qw, qx, qy, qz].
    """
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


def ft_poses_to_tcp_poses(
    positions_ft: np.ndarray,
    orientations_wxyz: np.ndarray,
    sensor_offset_xyz_m: np.ndarray | None = None,
) -> np.ndarray:
    """
    Convert publisher-corrected FT-frame poses to UMIFT TCP poses.

    The collection publisher maps T265 camera poses into the FT-style world and
    body convention. This step expresses the absolute pose in TCP/UR5-style
    world axes and TCP body axes.
    sensor_offset_xyz_m uses the same convention as wrench conversion: target
    point -> sensor point, expressed in the TCP frame. For a TCP 2 cm forward
    of the FT sensor, pass [0, 0, -0.02].
    """
    positions = np.asarray(positions_ft, dtype=np.float64)
    orientations = np.asarray(orientations_wxyz, dtype=np.float64)
    sensor_offset = (
        np.zeros(3, dtype=np.float64)
        if sensor_offset_xyz_m is None
        else np.asarray(sensor_offset_xyz_m, dtype=np.float64).reshape(3)
    )
    tcp_poses = np.zeros((positions.shape[0], 7), dtype=np.float64)

    for idx in range(positions.shape[0]):
        qw, qx, qy, qz = orientations[idx]
        r_ftworld_ftbody = quaternion_to_rotation_matrix(qx, qy, qz, qw)
        r_tcpworld_tcpbody = (
            R_TCPWORLD_FTWORLD @ r_ftworld_ftbody @ R_FTBODY_TCPBODY
        )
        p_tcpworld_sensor = R_TCPWORLD_FTWORLD @ positions[idx]
        tcp_poses[idx, :3] = p_tcpworld_sensor - r_tcpworld_tcpbody @ sensor_offset
        tcp_poses[idx, 3:] = rotation_matrix_to_quaternion_wxyz(r_tcpworld_tcpbody)

    return tcp_poses


def ft_wrench_to_tcp_wrench(wrench_ft: np.ndarray) -> np.ndarray:
    """Rotate FT-sensor-frame wrench rows into the TCP frame."""
    wrench_ft = np.asarray(wrench_ft, dtype=np.float64)
    wrench_tcp = np.empty_like(wrench_ft)
    wrench_tcp[:, 0:3] = wrench_ft[:, 0:3] @ R_FT_TCP
    wrench_tcp[:, 3:6] = wrench_ft[:, 3:6] @ R_FT_TCP
    return wrench_tcp


def check_tcp_pose_consistency(ts_pose_fb: np.ndarray) -> Dict[str, float]:
    """
    Sanity check: position deltas should align with TCP +Z when moving forward.
    """
    ts_pose_fb = np.asarray(ts_pose_fb, dtype=np.float64)
    if ts_pose_fb.shape[0] < 3:
        return {"num_samples": float(ts_pose_fb.shape[0]), "forward_alignment": 0.0}

    positions = ts_pose_fb[:, :3]
    deltas = positions[1:] - positions[:-1]
    delta_norms = np.linalg.norm(deltas, axis=1)
    valid = delta_norms > 1e-6
    if not np.any(valid):
        return {"num_samples": float(ts_pose_fb.shape[0]), "forward_alignment": 0.0}

    qw, qx, qy, qz = ts_pose_fb[:-1, 3], ts_pose_fb[:-1, 4], ts_pose_fb[:-1, 5], ts_pose_fb[:-1, 6]
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
    t_cam_tcp_m: np.ndarray = DEFAULT_T_CAM_TCP_M,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Metadata blob stored in zarr attrs."""
    info: Dict[str, Any] = {
        "frame_convention_version": FRAME_CONVENTION_VERSION,
        "world_frame": "+Y up (T265/ARKit native)",
        "tcp_frame": "+x right, +y down, +z forward (GoPro/TCP UMIFT convention)",
        "pose_and_wrench_body_frame": "tcp",
        "R_cam_tcp": R_CAM_TCP.tolist(),
        "R_ft_tcp": R_FT_TCP.tolist(),
        "R_tcpworld_ftworld": R_TCPWORLD_FTWORLD.tolist(),
        "R_ftbody_tcpbody": R_FTBODY_TCPBODY.tolist(),
        "R_cam_ft": R_CAM_FT.tolist(),
        "t_cam_tcp_m": np.asarray(t_cam_tcp_m, dtype=np.float64).tolist(),
        "gopro_to_tcp_translation_m": GOPRO_TO_TCP_TRANSLATION_M.tolist(),
    }
    if extra:
        info.update(extra)
    return info


def conversion_info_json(
    t_cam_tcp_m: np.ndarray = DEFAULT_T_CAM_TCP_M,
    extra: Dict[str, Any] | None = None,
) -> str:
    return json.dumps(conversion_info_dict(t_cam_tcp_m=t_cam_tcp_m, extra=extra))

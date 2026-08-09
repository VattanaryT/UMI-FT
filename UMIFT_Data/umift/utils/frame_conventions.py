"""
Canonical frame conventions for the portable gripper -> UMIFT pipeline.

Contract version: umift_std_v2

Column-vector math everywhere:
  v_parent = R_parent←child @ v_child
  T_a←c = T_a←b · T_b←c

Body and world (when we own the frame): +X right, +Y down, +Z forward.
Gravity in world: [0, +9.8, 0].
"""

from __future__ import annotations

import json
from typing import Any, Dict

import numpy as np

# ---------------------------------------------------------------------------
# umift_std_v2
# ---------------------------------------------------------------------------

FRAME_CONVENTION_VERSION = "umift_std_v2"

# World / body std: +X right, +Y down, +Z forward.
GRAVITY_WORLD_MPS2 = np.array([0.0, 9.8, 0.0], dtype=np.float64)

# Raw T265 pose body: +X right, +Y up, +Z back → std via 180° about X.
R_STD_T265 = np.diag([1.0, -1.0, -1.0]).astype(np.float64)

# Raw FT300: +X down, +Y left, +Z forward → std.
R_STD_FT = np.array(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
)

# Flat master AprilTag → fixture std (+X right, +Y down, +Z forward_from_base).
# Base look-out map is Rx(-90°)=[[1,0,0],[0,0,-1],[0,1,0]] (tag Y=forward, Z=up).
# Looking *at* the tag (facing the robot) is 180° yaw about vertical vs that view;
# in Y-down fixture that is Ry(180), so:
#   R = Ry(180) @ Rx(-90) = [[-1,0,0],[0,0,-1],[0,-1,0]]  (det=+1)
# Maps: tag+X→−X, tag+Y→−Z, tag+Z(up)→−Y (still up). Verify on table.
R_FIXTURE_MASTER_TAG = np.array(
    [[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]], dtype=np.float64
)

# Default: FT origin relative to TCP (same as old convert sensor_offset / target→sensor).
# TCP 2 cm forward of FT ⇒ FT is 2 cm behind TCP along +Z.
DEFAULT_R_FT_IN_TCP_M = np.array([0.0, 0.0, -0.02], dtype=np.float64)

# Default TCP origin in T265 body after std similarity (meters).
# Historical: FT ≈ [0.10, -0.155, 0.35] m in raw T265; TCP ≈ FT then +Z tip offset.
_FT_IN_T265_RAW_M = np.array([0.10, -0.155, 0.35], dtype=np.float64)
_FT_IN_T265_STD_M = R_STD_T265 @ _FT_IN_T265_RAW_M
# TCP is tip; with r_ft_in_tcp = FT in TCP, t_t265←tcp = p_ft_in_t265 - R @ r_ft
# For identity R_t265_tcp (axes already match after std): t = p_ft - r_ft_in_tcp
DEFAULT_T_T265_TCP_M = _FT_IN_T265_STD_M - DEFAULT_R_FT_IN_TCP_M

# D405 optical is already OpenCV/std; translation must be measured (placeholder zeros).
DEFAULT_T_D405_TCP_M = np.zeros(3, dtype=np.float64)

# Legacy GoPro lever (runtime UR TCP should absorb this; kept for old demos only).
GOPRO_TO_TCP_TRANSLATION_M = np.array([0.0, 0.086, 0.14465], dtype=np.float64)

# ---------------------------------------------------------------------------
# Legacy aliases (umift_tcp_v1 / old convert) — do not use for new v2 data
# ---------------------------------------------------------------------------

R_CAM_TCP = R_STD_T265.copy()  # historical name; T265-only, NOT for D405
R_FT_TCP = R_STD_FT.copy()  # column-vector: v_std = R_STD_FT @ v_ft
R_TCPWORLD_FTWORLD = R_FT_TCP.T
R_FTBODY_TCPBODY = R_FT_TCP
# Old gravity sandwich matrix (deleted from tare paths; kept only for legacy readers).
R_CAM_FT = np.array(
    [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64
)
_FT_TO_T265_IN_FT_M = np.array([0.155, -0.10, -0.35], dtype=np.float64)
DEFAULT_T_CAM_TCP_M = DEFAULT_T_T265_TCP_M.copy()
GRAVITY_WORLD_Y_UP_MPS2 = np.array([0.0, -9.8, 0.0], dtype=np.float64)


def T_from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build 4x4 SE(3) from rotation and translation."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def default_T_t265_tcp() -> np.ndarray:
    """Identity rotation (after std) + DEFAULT_T_T265_TCP_M."""
    return T_from_Rt(np.eye(3), DEFAULT_T_T265_TCP_M)


def default_T_d405_tcp() -> np.ndarray:
    return T_from_Rt(np.eye(3), DEFAULT_T_D405_TCP_M)


def default_T_fixture_master_tag() -> np.ndarray:
    return T_from_Rt(R_FIXTURE_MASTER_TAG, np.zeros(3))


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


def apply_similarity_std_t265(
    positions: np.ndarray, orientations_wxyz: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    T265 raw → std world/body via similarity with R_STD_T265.
    p' = M @ p,  R' = M @ R @ M.T
    """
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


def gravity_in_tcp_frame(rotation_odom_tcp: np.ndarray) -> np.ndarray:
    """g_tcp = R_odom←tcp.T @ g_world  (umift_std_v2; no sandwich)."""
    rotation_odom_tcp = np.asarray(rotation_odom_tcp, dtype=np.float64)
    if rotation_odom_tcp.shape != (3, 3):
        raise ValueError(f"rotation must be 3x3, got shape {rotation_odom_tcp.shape}")
    return rotation_odom_tcp.T @ GRAVITY_WORLD_MPS2


def ft_wrench_raw_to_tcp_wrench(
    wrench_ft: np.ndarray,
    r_ft_in_tcp_m: np.ndarray | None = None,
) -> np.ndarray:
    """
    Raw FT wrench rows → TCP body: rotate by R_STD_FT, then point adjoint.
    F_tcp = R @ F_ft;  τ_tcp = R @ τ_ft + r_ft_in_tcp × F_tcp
    """
    wrench_ft = np.asarray(wrench_ft, dtype=np.float64)
    r = (
        DEFAULT_R_FT_IN_TCP_M
        if r_ft_in_tcp_m is None
        else np.asarray(r_ft_in_tcp_m, dtype=np.float64).reshape(3)
    )
    out = np.empty_like(wrench_ft)
    # column-vector: v_std = R @ v_ft  ⇒ row form uses v @ R.T
    F = wrench_ft[:, 0:3] @ R_STD_FT.T
    Tau = wrench_ft[:, 3:6] @ R_STD_FT.T
    out[:, 0:3] = F
    out[:, 3:6] = Tau + np.cross(np.broadcast_to(r, F.shape), F)
    return out


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
        # Identity R_t265_tcp: R_odom_tcp = R_odom_t265, p = p_t265 + R @ t
        tcp_poses[idx, :3] = positions[idx] + R @ t
        tcp_poses[idx, 3:] = orientations[idx]
    return tcp_poses


# --- Legacy helpers for old bags (umift_tcp_v1 / FT-frame publisher) ---


def ft_poses_to_tcp_poses(
    positions_ft: np.ndarray,
    orientations_wxyz: np.ndarray,
    sensor_offset_xyz_m: np.ndarray | None = None,
) -> np.ndarray:
    """
    LEGACY: publisher FT-frame poses (+Y-up world style) → TCP.
    Not used for umift_std_v2 bags.
    """
    positions = np.asarray(positions_ft, dtype=np.float64)
    orientations = np.asarray(orientations_wxyz, dtype=np.float64)
    sensor_offset = (
        np.zeros(3, dtype=np.float64)
        if sensor_offset_xyz_m is None
        else np.asarray(sensor_offset_xyz_m, dtype=np.float64).reshape(3)
    )
    # Legacy used inverted R_FT_TCP relative to R_STD_FT in some trees; keep
    # historical UMIFT convert math: R_FTBODY_TCPBODY = [[0,1,0],[-1,0,0],[0,0,1]]
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
    t_cam_tcp_m: np.ndarray | None = None,  # legacy kw
) -> Dict[str, Any]:
    """Metadata blob stored in zarr attrs."""
    if t_t265_tcp_m is None:
        t_t265_tcp_m = DEFAULT_T_CAM_TCP_M if t_cam_tcp_m is None else t_cam_tcp_m
    if r_ft_in_tcp_m is None:
        r_ft_in_tcp_m = DEFAULT_R_FT_IN_TCP_M
    info: Dict[str, Any] = {
        "frame_convention_version": FRAME_CONVENTION_VERSION,
        "world_frame": "+Y down, +X right, +Z forward (umift_std_v2)",
        "tcp_frame": "+x right, +y down, +z forward",
        "pose_and_wrench_body_frame": "tcp",
        "R_std_t265": R_STD_T265.tolist(),
        "R_std_ft": R_STD_FT.tolist(),
        "R_fixture_master_tag": R_FIXTURE_MASTER_TAG.tolist(),
        "t_t265_tcp_m": np.asarray(t_t265_tcp_m, dtype=np.float64).tolist(),
        "r_ft_in_tcp_m": np.asarray(r_ft_in_tcp_m, dtype=np.float64).tolist(),
        "gravity_world_mps2": GRAVITY_WORLD_MPS2.tolist(),
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

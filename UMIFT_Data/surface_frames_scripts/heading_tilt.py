"""Heading and tilt for the causal-foot action.

Swing-twist split, same order in the labels and the decode:

R_cmd = R_zero(e_long, n) · Rz(ψ) · exp([a, b, 0])

R_zero has tool z along the inward normal. Rz is about that axis, so +ψ moves
tool x toward n × e_long. (a, b) is the full swing rotvec in the frame after
Rz(ψ), including the along-heading lean. ψ is the twist about the inward
normal, read from the swing-removed rotation. v1, v2, and δn stay in the
carried tangent frame.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

UP = np.array([0.0, -1.0, 0.0])
DEGEN_DEG = 20.0
COLUMNS = ("v1", "v2", "delta_n", "stiffness", "sin_psi", "cos_psi", "tilt_a", "tilt_b")


def long_axis(vertices: np.ndarray) -> np.ndarray:
    centered = np.asarray(vertices, dtype=np.float64) - np.mean(vertices, axis=0)
    _values, vectors = np.linalg.eigh(np.cov(centered.T))
    axis = vectors[:, -1]
    axis = axis / np.linalg.norm(axis)
    if axis[0] < 0.0:
        axis = -axis
    return axis


def _tangent(vector: np.ndarray, normal: np.ndarray):
    normal = np.asarray(normal, dtype=np.float64)
    vector = np.asarray(vector, dtype=np.float64)
    flat = vector.reshape(normal.shape)
    dropped = flat - np.sum(flat * normal, axis=-1, keepdims=True) * normal
    length = np.linalg.norm(dropped, axis=-1, keepdims=True)
    return dropped, length


def e_long_sequence(normal_out: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project the long axis onto each tangent plane. Carry the last good one.

    A sample is good when the axis is at least 20° from the normal. The first
    samples, before any good one, use the long axis projected onto the horizontal
    and then onto the tangent plane.
    """
    normal_out = np.asarray(normal_out, dtype=np.float64).reshape(-1, 3)
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    dropped, length = _tangent(np.broadcast_to(axis, normal_out.shape), normal_out)
    angle = np.degrees(np.arcsin(np.clip(length[:, 0], 0.0, 1.0)))
    good = angle >= DEGEN_DEG
    direction = np.zeros_like(normal_out)
    usable = length[:, 0] > 1e-8
    direction[usable] = dropped[usable] / length[usable]
    horizontal = axis - np.dot(axis, UP) * UP
    h_norm = float(np.linalg.norm(horizontal))
    horizontal = horizontal / h_norm if h_norm > 1e-8 else np.array([0.0, 0.0, 1.0])
    last = None
    for i in range(len(direction)):
        if good[i]:
            last = direction[i].copy()
            continue
        seed = last if last is not None else horizontal
        tangent, size = _tangent(seed, normal_out[i])
        if float(size) < 1e-8:
            tangent, size = _tangent(np.array([0.0, 0.0, 1.0]), normal_out[i])
        direction[i] = tangent / max(float(size), 1e-12)
        last = direction[i].copy()
    return direction, angle


def frame_zero(e_long: np.ndarray, normal_out: np.ndarray) -> np.ndarray:
    """Tool frame at ψ = 0 and zero tilt. Columns are tool x, tool y, tool z = −n."""
    e_long = np.asarray(e_long, dtype=np.float64)
    normal_out = np.asarray(normal_out, dtype=np.float64)
    side = np.cross(normal_out, e_long)
    side = side / np.clip(np.linalg.norm(side, axis=-1, keepdims=True), 1e-12, None)
    inward = -normal_out
    return np.stack([e_long, -side, inward], axis=-1)


def _Rz(psi: np.ndarray) -> np.ndarray:
    """About inward tool z, so +ψ moves tool x toward n × e_long."""
    psi = np.asarray(psi, dtype=np.float64)
    c = np.cos(psi)
    s = np.sin(psi)
    out = np.zeros(psi.shape + (3, 3), dtype=np.float64)
    out[..., 0, 0] = c
    out[..., 0, 1] = s
    out[..., 1, 0] = -s
    out[..., 1, 1] = c
    out[..., 2, 2] = 1.0
    return out


def psi_of(e_long: np.ndarray, tool_x: np.ndarray, normal_out: np.ndarray) -> np.ndarray:
    tangent = tool_x - np.sum(tool_x * normal_out, axis=-1, keepdims=True) * normal_out
    tangent = tangent / np.clip(np.linalg.norm(tangent, axis=-1, keepdims=True), 1e-12, None)
    sin_p = np.sum(np.cross(e_long, tangent) * normal_out, axis=-1)
    cos_p = np.sum(e_long * tangent, axis=-1)
    return np.arctan2(sin_p, cos_p)


def _psi_from_swing_removed(relative: np.ndarray) -> np.ndarray:
    """Twist of R = Rz(ψ) @ Exp([a, b, 0]).

    ψ = atan2(R_01 − R_10, R_00 + R_11). That is the angle of R @ Exp([a, b, 0]).T,
    the rotation left after the swing is removed. Exp then has no about-z part,
    so (a, b) is stored whole.
    """
    relative = np.asarray(relative, dtype=np.float64).reshape(-1, 3, 3)
    return np.arctan2(
        relative[:, 0, 1] - relative[:, 1, 0],
        relative[:, 0, 0] + relative[:, 1, 1],
    )


def pack_tilt(rotation: np.ndarray, normal_out: np.ndarray, e_long: np.ndarray) -> np.ndarray:
    """(N, 4) sin ψ, cos ψ, a, b. a and b are radians in the frame after Rz(ψ)."""
    rotation = np.asarray(rotation, dtype=np.float64).reshape(-1, 3, 3)
    relative = np.einsum("nji,njk->nik", frame_zero(e_long, normal_out), rotation)
    psi = _psi_from_swing_removed(relative)
    swing = np.einsum("nji,njk->nik", _Rz(psi), relative)
    rotvec = Rotation.from_matrix(swing).as_rotvec()
    return np.column_stack([np.sin(psi), np.cos(psi), rotvec[:, 0], rotvec[:, 1]]).astype(np.float64)


def clip_tilt(tilt: np.ndarray, max_rad: float) -> np.ndarray:
    """Scale (a, b) so the tilt magnitude stays within the training range."""
    out = np.array(tilt, dtype=np.float64, copy=True)
    magnitude = np.linalg.norm(out[..., 2:4], axis=-1, keepdims=True)
    scale = np.minimum(1.0, float(max_rad) / np.maximum(magnitude, 1e-12))
    out[..., 2:4] *= scale
    return out


def rotation_from_tilt(normal_out, e_long, tilt: np.ndarray) -> np.ndarray:
    tilt = np.asarray(tilt, dtype=np.float64).reshape(-1, 4)
    psi = np.arctan2(tilt[:, 0], tilt[:, 1])
    head = np.einsum("nij,njk->nik", frame_zero(e_long, normal_out), _Rz(psi))
    rotvec = np.column_stack([tilt[:, 2], tilt[:, 3], np.zeros(len(tilt))])
    residual = Rotation.from_rotvec(rotvec).as_matrix()
    return np.einsum("nij,njk->nik", head, residual)


def slew_orientations(rotation: np.ndarray, previous, max_deg: float = 3.0):
    """The causal-foot 3° command slew. Not used when checking label round-trip."""
    from mold_aware_runtime import _slew_orientations

    return _slew_orientations(rotation, previous, max_deg)


def action_heading8_from_raw(raw) -> tuple[np.ndarray, np.ndarray]:
    """Training action and its timestamps. Does not go through the 27-d pack."""
    command = np.asarray(raw["ts_pose_command_0"])
    stiffness = np.asarray(raw["stiffness_0"]).reshape(-1)
    tilt = np.asarray(raw["fixb_heading_tilt_0"]).reshape(-1, 4)
    times = np.asarray(raw["robot_time_stamps_0"]).reshape(-1)
    n = min(len(command), len(stiffness), len(tilt), len(times))
    action = np.column_stack(
        [command[:n, 0], command[:n, 1], command[:n, 2], stiffness[:n], tilt[:n]]
    ).astype(np.float32)
    return action, np.asarray(times[:n])


def action_heading8_to_action_sample(action_sparse, action_dense, id_list, base_pose):
    """Pass the 8-d heading-tilt vector through. It is not an SE3 pose."""
    del id_list, base_pose
    if np.asarray(action_sparse).shape[-1] != 8:
        raise ValueError(f"heading action dim {np.asarray(action_sparse).shape[-1]}, expected 8")
    if len(action_dense) > 0:
        raise NotImplementedError
    return {"sparse": action_sparse, "dense": {}}


def command_rotation(normal_out, e_long, tilt, clip_rad=None, prev=None, slew_deg=None):
    """Decode R_cmd. clip_rad is the training p99.5. slew_deg is the 3° step cap.

    Label checks pass clip_rad=None and slew_deg=None.
    """
    packed = np.asarray(tilt, dtype=np.float64).reshape(-1, 4)
    if clip_rad is not None:
        packed = clip_tilt(packed, float(clip_rad))
    rotation = rotation_from_tilt(normal_out, e_long, packed)
    if slew_deg is not None:
        rotation, _hit = slew_orientations(rotation, prev, float(slew_deg))
    return rotation


def _self_check() -> None:
    """Rz sign and a pure-tilt round trip, before any zarr write."""
    normal = np.array([0.0, -1.0, 0.0])
    axis = np.array([1.0, 0.0, 0.0])
    zero = frame_zero(axis, normal)
    if abs(float(np.linalg.det(zero)) - 1.0) > 1e-8:
        raise RuntimeError(f"R_zero det {np.linalg.det(zero)}")
    psi = 0.3
    side = np.cross(normal, axis)
    got = (zero @ _Rz(np.array([psi]))[0])[:, 0]
    expect = np.cos(psi) * axis + np.sin(psi) * side
    if not np.allclose(got, expect, atol=1e-8):
        raise RuntimeError("Rz(ψ) does not move tool x toward n × e_long")
    # Along-heading lean is b. It must round-trip whole, not be dropped or rescaled.
    swing = np.array([0.4, -0.55])
    tool = (zero @ _Rz(np.array([psi]))[0]) @ Rotation.from_rotvec([swing[0], swing[1], 0.0]).as_matrix()
    packed = pack_tilt(tool[None], normal[None], axis[None])
    back = rotation_from_tilt(normal[None], axis[None], packed)
    err = float(geodesic_deg(tool[None], back)[0])
    if err > 1e-8 or not np.allclose(packed[0, 2:4], swing, atol=1e-8):
        raise RuntimeError(f"swing-twist round trip {err} deg, ab={packed[0, 2:4]}")


def action_from_episode(group) -> np.ndarray:
    """8-vector from the stored causal-foot command plus the heading-tilt array."""
    command = np.asarray(group["ts_pose_command_0"], dtype=np.float64)
    stiffness = np.asarray(group["stiffness_0"], dtype=np.float64).reshape(-1)
    tilt = np.asarray(group["fixb_heading_tilt_0"], dtype=np.float64).reshape(-1, 4)
    n = min(len(command), len(stiffness), len(tilt))
    return np.column_stack(
        [command[:n, 0], command[:n, 1], command[:n, 2], stiffness[:n], tilt[:n]]
    ).astype(np.float32)


def geodesic_deg(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    relative = np.einsum("nji,njk->nik", np.asarray(left, dtype=np.float64), np.asarray(right))
    return np.degrees(np.linalg.norm(Rotation.from_matrix(relative).as_rotvec(), axis=1))

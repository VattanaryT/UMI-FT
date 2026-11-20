"""Live mold-aware obs + Fix B decode for umift_env_runner.

Reuses FixBMoldField / heading_branch / tracked_project from the
training/eval scripts. Training stringer assets default to
``stringer_scanner_v2_fixb/shared/fixb_field.npz``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from fixb_field import FixBMoldField
from fixb_labels import QUERY_SMOOTH_S, heading_branch, pose7_to_Rp, smooth_positions, so3_to_rot6d_rows

DEFAULT_FIELD = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_scanner_v2_fixb/shared/fixb_field.npz"
)
DEFAULT_META = DEFAULT_FIELD.with_name("fixb_meta.json")


def rot6d_rows_to_R(r6: np.ndarray) -> np.ndarray:
    """Gram-Schmidt. Columns of the result are the rotation's axes."""
    a1 = r6[..., 0:3]
    a2 = r6[..., 3:6]
    b1 = a1 / np.clip(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-9, None)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.clip(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-9, None)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2)


# Identity rot6d: the first two columns of I. The 10-d action drops these slots.
_IDENTITY_ROT6 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)


def expand_r4_action(action: np.ndarray) -> np.ndarray:
    """10-d ``[v1, v2, δn, stiffness, ΔR]`` → the 27-d layout.

    Dropped slots are the constant identity rotations, the unused virtual
    target, and the gripper and grasp force. A 27-d action is returned as-is.
    """
    action = np.asarray(action, dtype=np.float64)
    if action.shape[-1] != 10:
        return action
    out = np.zeros(action.shape[:-1] + (27,), dtype=np.float64)
    out[..., 0:3] = action[..., 0:3]
    out[..., 3:9] = _IDENTITY_ROT6
    out[..., 12:18] = _IDENTITY_ROT6
    out[..., 18] = action[..., 3]
    out[..., 21:27] = action[..., 4:10]
    return out


def Rp_to_SE3(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    if R.ndim == 2:
        T[:3, :3] = R
        T[:3, 3] = p
        return T
    out = np.zeros(R.shape[:-2] + (4, 4), dtype=np.float64)
    out[..., :3, :3] = R
    out[..., :3, 3] = p
    out[..., 3, 3] = 1.0
    return out


def infer_run_type(task_name: str, shape_meta) -> str:
    name = (task_name or "").upper()
    has_geom = "robot0_mold_geom" in shape_meta["obs"]
    has_curv = "robot0_mold_curv" in shape_meta["obs"]
    action_dim = int(shape_meta.get("action", {}).get("shape", [21])[0])
    if not has_geom:
        return "none"
    if "R4" in name or "FIXB-TILT" in name or "WORLD_FIXB_TILT" in name or action_dim == 27:
        return "r4_world_fixb_tilt"
    if "R3" in name or "SURFACE_FIXB" in name or has_curv:
        return "r3_surface_fixb"
    if "R2" in name or "WORLD_FIXB" in name:
        return "r2_world_fixb"
    if "R1" in name or "WORLD_WORLD" in name:
        return "r1_world_world"
    # fallback: curv → r3, else r1 (absolute poses)
    return "r3_surface_fixb" if has_curv else "r1_world_world"


@dataclass
class SafetyLimits:
    max_press_depth_m: float = 0.005
    enable_delta_n_floor: bool = True
    # Measured TCP only: (p_meas - x_s) · n. Negative is into the mold.
    max_tcp_penetration_m: float = 0.020
    enable_tcp_penetration_abort: bool = False
    # First replan at or above this holds; the next one still there aborts.
    wrench_hold_n: float = 90.0
    enable_wrench_hold: bool = False
    dry_run_hover_m: float = 0.015
    max_force_n: float = 80.0
    # Live policy peaks ~0.19 m/s on val demos; 0.25 leaves headroom without aborting.
    max_tangential_speed_mps: float = 0.25
    max_step_m: float = 0.025
    # 0.030 aborted the 2026-10-29 contact runs (commanded δn already ~27 mm).
    # 0.10 lets those chunks run; still stops a large fly-off. Demos peak ~0.32 m.
    max_distance_from_mesh_m: float = 0.10
    # Off: abort when commanded ||x_d - x_s|| exceeds max_distance_from_mesh_m.
    # On: skip that check and abort when the measured TCP is that far from its foot.
    abort_on_measured_mesh_distance: bool = False
    max_foot_hop_m: float = 0.020
    stop_on_normal_flip: bool = True
    max_tilt_from_normal_deg: float = 45.0
    max_rot_rate_deg_per_step: float = 20.0
    # Orientation rewrites added 2026-10-26. Off matches the Oct 22 contact decode
    # (speed, step, and δn clamps stay on).
    enable_tilt_clamp: bool = False
    enable_rot_rate_clamp: bool = False
    enable_mesh_jump_hold: bool = False
    # Cross-chunk cap on commanded R_d. On for the r4 runtime.
    enable_rd_slew: bool = False
    max_rd_slew_deg: float = 3.0
    # None: the mold's lowest vertex (fixture +Y down). A point past that plus
    # the margin is through the table.
    table_y_m: Optional[float] = None


# Soft clamps (applied / flagged) vs hard stops (abort-worthy).
SOFT_CLAMP_KEYS = (
    "max_tangential_speed",
    "max_step",
    "delta_n",
    "max_tilt",
    "max_rot_rate",
    "mesh_jump_hold",
    "rd_slew",
    "face_hold",
)
HARD_STOP_KEYS = (
    "normal_flip",
    "max_distance_from_mesh",
    "tcp_penetration",
    "wrench_abort",
    "table_plane",
    "mesh_far",
    "camera_table",
)


def fixb_decode_dt(
    n_steps: int, down_sample_steps: int, raw_time_step_s: float
) -> np.ndarray:
    """Demo-time step for Fix B integration. Slow-down stays on the waypoint clock."""
    step_s = float(down_sample_steps) * float(raw_time_step_s)
    return np.full(int(n_steps), step_s, dtype=np.float64)


def read_bench_dn_floor_m(environ=None) -> float:
    """Signed δn floor in meters. ``UMIFT_BENCH_DN_FLOOR_MM`` defaults to −5.

    Negative is into the mold. The bench flag decides whether the floor is applied.
    """
    source = os.environ if environ is None else environ
    text = str(source.get("UMIFT_BENCH_DN_FLOOR_MM", "")).strip()
    mm = -5.0 if text == "" else float(text)
    if not np.isfinite(mm) or mm > 0.0:
        shown = text or "-5"
        raise ValueError(
            f"UMIFT_BENCH_DN_FLOOR_MM={shown} must be a finite floor <= 0"
        )
    return float(mm) / 1000.0


def r4_safety_limits(
    max_tcp_penetration_m: float = 0.020,
    wrench_hold_n: float = 90.0,
    abort_on_measured_mesh_distance: bool = False,
    enable_delta_n_floor: bool = False,
    max_press_depth_m: float = 0.005,
) -> SafetyLimits:
    """r4vt limits. The δn floor stays off unless a bench replay asks for it.

    Commanded orientation slews at 3° per step, including across replans.

    Abort on measured TCP depth. Hold, then abort, on force. The measured-mesh
    flag skips the commanded-depth abort and checks the TCP instead.
    ``max_press_depth_m`` is how far the bench floor may press, default 5 mm.
    """
    return SafetyLimits(
        max_press_depth_m=float(max_press_depth_m),
        enable_delta_n_floor=bool(enable_delta_n_floor),
        enable_tcp_penetration_abort=True,
        max_tcp_penetration_m=float(max_tcp_penetration_m),
        enable_wrench_hold=True,
        wrench_hold_n=float(wrench_hold_n),
        abort_on_measured_mesh_distance=bool(abort_on_measured_mesh_distance),
        enable_rd_slew=True,
    )


# One replan, and one step inside a replan. A larger hop keeps the previous foot.
FOOT_CAP_M = 0.010
# Rejected closest-point jumps. More than this many in a row re-anchors to the TCP.
HOP_HOLD_LIMIT = 5
# Same-face hops are accepted up to the commanded step. A larger turn is a new face.
HOP_FACE_DEG = 25.0
# A new face whose normal has turned further than this keeps the held frame.
FACE_HOLD_DEG = 5.0
# After the first second, the orientation cap sits above the label's 6.2° step.
RD_SLEW_AFTER_RAMP_DEG = 8.0
# Virtual-target offset grows from the measured TCP to the command over this long.
VT_RAMP_S = 1.0
# One chunk of travel on this layup. A same-sheet v_par slide may lead the TCP
# by this much. Only the extra is walked back, at most one 6 mm step.
LEAD_M = 0.020
# Fixture +Y is down. Bench replay lifts along −Y, off the table on every face.
FIXTURE_UP = np.array([0.0, -1.0, 0.0])
# How far a D405 housing corner must stay above the flange table.
# Episode 40's label brings a corner to 0.1 mm above that plane, so a
# positive margin would stop a faithful replay. Through the plane still holds.
CAM_TABLE_MARGIN_M = 0.0
# Replay only. A commanded tool this far from the label orientation is held.
ORI_LABEL_HOLD_DEG = 25.0
# Off unless UMIFT_RESET_TCP_TO_MEASURED is set. A carried TCP farther than
# this from the measured TCP is dropped and the tracker starts at the measurement.
TCP_RESET_GAP_M = 0.015


def tcp_reset_enabled() -> bool:
    return os.environ.get("UMIFT_RESET_TCP_TO_MEASURED", "").strip().lower() in (
        "1",
        "true",
        "on",
        "yes",
    )


def reset_carried_tcp(tcp_state: np.ndarray, measured: np.ndarray) -> np.ndarray:
    """Keep the carried TCP. With the flag on, a gap over 15 mm uses the measurement."""
    carried = np.asarray(tcp_state, dtype=np.float64).reshape(3)
    measured = np.asarray(measured, dtype=np.float64).reshape(3)
    if not tcp_reset_enabled():
        return carried
    gap = float(np.linalg.norm(carried - measured))
    if gap <= TCP_RESET_GAP_M:
        return carried
    print(f"[mold][tcp] reset internal TCP to measured, gap {gap * 1e3:.1f} mm")
    return measured.copy()


def _import_frame_conventions():
    import sys

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "Robotiq-portable-real2sim" / "python"
        if candidate.is_dir():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            break
    else:
        raise RuntimeError("Robotiq-portable-real2sim/python is not above this file")
    import frame_conventions

    return frame_conventions


def d405_housing_corners_in_tcp() -> np.ndarray:
    """Eight corners of the D405 case, in the TCP frame.

    Intel ``realsense2_description`` ``_d405.urdf.xacro`` collision box:
    23 × 42 × 42 mm, centered in the depth frame at (3.7 − 11.5, −9, 0) mm.
    That frame shares the optical origin. Optical +Z is depth +X, optical +X
    is depth −Y, optical +Y is depth −Z. The mount is ``T_tcp←d405``.
    """
    center = np.array([0.0037 - 0.0115, -0.009, 0.0])
    half = np.array([0.0115, 0.021, 0.021])
    signs = np.array(
        [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)]
    )
    corners_depth = center + signs * half
    # p_depth = R @ p_optical. Columns are the optical axes written in depth.
    depth_from_opt = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    corners_opt = (depth_from_opt.T @ corners_depth.T).T
    mount = _import_frame_conventions().default_T_tcp_d405()
    return (mount[:3, :3] @ corners_opt.T).T + mount[:3, 3]


def camera_table_from_flange(field_path: Path, T_fixture_from_base: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Level plane through the lowest edge of the scanned flange, in the fixture frame.

    Clearance of a fixture point ``p`` is ``offset - normal @ p`` (meters, positive
    above the plane). The flange is the flat scan vertices within 15 mm of the
    lowest flat band, apron excluded. Gravity is base +Y. The mold's lowest
    vertex is below this edge and is not the table.
    """
    field = np.load(field_path)
    apron = field["is_apron"].astype(bool) if "is_apron" in field.files else np.zeros(len(field["vertices"]), dtype=bool)
    scan = np.asarray(field["vertices"], dtype=np.float64)[~apron]
    normals = np.asarray(field["normals"], dtype=np.float64)[~apron]
    flat = np.abs(normals[:, 1]) > 0.98
    if not np.any(flat):
        raise RuntimeError(f"no flat flange vertices in {field_path}")
    flange = flat & (scan[:, 1] > float(scan[flat, 1].max()) - 0.015)
    rotation = np.asarray(T_fixture_from_base, dtype=np.float64).reshape(4, 4)[:3, :3]
    base_from_fixture = rotation.T
    base_down = base_from_fixture[1]
    origin_down = float((-base_from_fixture @ T_fixture_from_base[:3, 3])[1])
    edge = float((scan[flange] @ base_down).max() + origin_down)
    return base_down.copy(), edge - origin_down, edge


def _geodesic_deg(a: np.ndarray, b: np.ndarray) -> float:
    rel = np.asarray(a, dtype=np.float64).reshape(3, 3).T @ np.asarray(b, dtype=np.float64).reshape(3, 3)
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(rel).as_rotvec())))


def hold_orientation_near_label(
    R_cmd: np.ndarray,
    R_label: np.ndarray,
    R_prev: np.ndarray | None,
    max_deg: float = ORI_LABEL_HOLD_DEG,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the previous tool orientation when a step is more than ``max_deg`` from the label."""
    cmd = np.asarray(R_cmd, dtype=np.float64).reshape(-1, 3, 3).copy()
    label = np.asarray(R_label, dtype=np.float64).reshape(-1, 3, 3)
    prev = None if R_prev is None else np.asarray(R_prev, dtype=np.float64).reshape(3, 3).copy()
    held = np.zeros(len(cmd), dtype=bool)
    for i in range(len(cmd)):
        if _geodesic_deg(cmd[i], label[i]) <= float(max_deg):
            prev = cmd[i].copy()
            continue
        if prev is None:
            prev = cmd[i].copy()
            continue
        cmd[i] = prev
        held[i] = True
    return cmd, held


def shift_fixture_up(points: np.ndarray, lift_m: float) -> np.ndarray:
    """Move positions by ``lift_m`` along fixture up (−Y). Pose7 uses [:, :3]."""
    out = np.array(points, dtype=np.float64, copy=True)
    if float(lift_m) == 0.0:
        return out
    out[..., 1] = out[..., 1] - float(lift_m)
    return out


def _hold_if_far(x, R, prev_x, prev_R, cap: float = FOOT_CAP_M):
    if prev_x is None:
        return x, R
    if float(np.linalg.norm(x - np.asarray(prev_x, dtype=np.float64).reshape(3))) > cap:
        return (
            np.asarray(prev_x, dtype=np.float64).reshape(3).copy(),
            np.asarray(prev_R, dtype=np.float64).reshape(3, 3).copy(),
        )
    return x, R


def _step_toward_cap(x, R, origin, cap: float = FOOT_CAP_M):
    """Move ``x`` toward ``origin`` until it is within ``cap``. Keep ``R``."""
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    point = np.asarray(x, dtype=np.float64).reshape(3)
    delta = point - origin
    dist = float(np.linalg.norm(delta))
    if dist <= cap or dist < 1e-12:
        return point, R
    return origin + delta * (cap / dist), R


def _cap_onto_mesh(field, x, R, origin, cap: float = FOOT_CAP_M):
    """Shorten ``x`` toward ``origin``, then put that point back on the mesh.

    The tangent is carried onto the projected frame. Returns ``(x, R, limited)``.
    """
    point = np.asarray(x, dtype=np.float64).reshape(3)
    capped, capped_R = _step_toward_cap(point, R, origin, cap)
    limited = float(np.linalg.norm(capped - point)) > 1e-9
    if not limited:
        return capped, np.asarray(R, dtype=np.float64).reshape(3, 3), False
    sx, sR = _project_foot(field, capped, None, None, seed_R=capped_R)
    return sx, sR, True


def _hold_long_steps(pos: np.ndarray, prev, max_step: float):
    """Copy the last safe position when a step is longer than ``max_step``."""
    out = np.asarray(pos, dtype=np.float64).reshape(-1, 3).copy()
    hit = np.zeros(len(out), dtype=bool)
    origin = None if prev is None else np.asarray(prev, dtype=np.float64).reshape(3).copy()
    for i in range(len(out)):
        if origin is not None and float(np.linalg.norm(out[i] - origin)) > max_step:
            out[i] = origin
            hit[i] = True
        else:
            origin = out[i].copy()
    return out, hit


def _reject_new_face(x, R, prev_x, prev_R, max_deg: float = FACE_HOLD_DEG):
    """Keep the previous foot and frame when the normal turns more than max_deg."""
    if prev_R is None or prev_x is None:
        return x, R, False
    n_new = np.asarray(R, dtype=np.float64).reshape(3, 3)[:, 2]
    n_prev = np.asarray(prev_R, dtype=np.float64).reshape(3, 3)[:, 2]
    cosine = float(np.clip(np.dot(n_new, n_prev), -1.0, 1.0))
    if float(np.degrees(np.arccos(cosine))) > max_deg:
        return (
            np.asarray(prev_x, dtype=np.float64).reshape(3).copy(),
            np.asarray(prev_R, dtype=np.float64).reshape(3, 3).copy(),
            True,
        )
    return x, R, False


def _label_query(field, p):
    """The query the TCP-track labels use. Same smoothing, no tangent hold."""
    from fixb_tcp_velocity_labels import _query

    return _query(field, np.asarray(p, dtype=np.float64).reshape(-1, 3))


def track_feet(field, points):
    """Label foot tracker. Same function, 6 mm gate, and tangent rule as the zarr."""
    from fixb_tcp_track import track_local_feet

    return track_local_feet(lambda pts: _label_query(field, pts), points)


def _project_foot(field, p, prev_x, prev_R, seed_R=None):
    """First sample queries the field. Later samples use the label's ``step_foot``.

    ``seed_R`` is used only when there is no previous frame. Replay passes the
    label frame. With no seed, the frame is the field lookup at that foot.
    """
    from fixb_tcp_track import step_foot

    p = np.asarray(p, dtype=np.float64).reshape(3)
    if prev_x is None or prev_R is None:
        x, R = _label_query(field, p)
        x = np.asarray(x, dtype=np.float64).reshape(3).copy()
        R = np.asarray(R, dtype=np.float64).reshape(3, 3).copy()
        if seed_R is not None:
            from fixb_smooth_field import tangent_on_normal

            R = tangent_on_normal(seed_R, R)
        return x, R
    free_x, free_R = _label_query(field, p)
    x, R = step_foot(
        lambda pts: _label_query(field, pts),
        p,
        prev_x,
        prev_R,
        np.asarray(free_x, dtype=np.float64).reshape(3),
        np.asarray(free_R, dtype=np.float64).reshape(3, 3),
    )
    return (
        np.asarray(x, dtype=np.float64).reshape(3).copy(),
        np.asarray(R, dtype=np.float64).reshape(3, 3).copy(),
    )


def _same_face(R, prev_R, max_deg: float = HOP_FACE_DEG) -> bool:
    """True when column 2 has turned no more than ``max_deg``."""
    if prev_R is None:
        return True
    n_new = np.asarray(R, dtype=np.float64).reshape(3, 3)[:, 2]
    n_prev = np.asarray(prev_R, dtype=np.float64).reshape(3, 3)[:, 2]
    cosine = float(np.clip(np.dot(n_new, n_prev), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine))) <= max_deg


def _commit_foot(
    field,
    candidate_x,
    candidate_R,
    prev_x,
    prev_R,
    tcp,
    cap_origin,
    streak: int,
    episode_start: bool,
    step_allow_m: float = 0.0,
):
    """Accept a foot, cap it onto the mesh, hold a hop, or re-anchor.

    Episode start keeps the queried closest point. No 10 mm cap.
    A requested cap shortens toward ``cap_origin`` and projects back onto the
    mesh, carrying the tangent. A hop whose normal still agrees is accepted
    up to this step's own length (at least 10 mm). A larger turn of the
    normal keeps the previous foot. More than ``HOP_HOLD_LIMIT`` rejected
    hops re-anchor to the TCP's closest point and carry the tangent forward.
    """
    x = np.asarray(candidate_x, dtype=np.float64).reshape(3).copy()
    R = np.asarray(candidate_R, dtype=np.float64).reshape(3, 3).copy()
    if episode_start:
        return x, R, False, False, False, 0
    capped = False
    if cap_origin is not None:
        x, R, capped = _cap_onto_mesh(field, x, R, cap_origin)
    if prev_x is None or prev_R is None:
        return x, R, False, False, capped, int(streak)
    prev_x = np.asarray(prev_x, dtype=np.float64).reshape(3)
    prev_R = np.asarray(prev_R, dtype=np.float64).reshape(3, 3)
    dist = float(np.linalg.norm(x - prev_x))
    # A few percent covers a closest-point landing just past the tangent step.
    allow = float(step_allow_m) * 1.05 + 1e-4
    if _same_face(R, prev_R) and dist <= allow:
        return x, R, False, False, capped, 0
    streak = int(streak) + 1
    if streak > HOP_HOLD_LIMIT:
        anchored, anchored_R = _project_foot(field, tcp, None, None, seed_R=prev_R)
        print(
            f"[mold] foot re-anchor after {streak} rejected closest-point jumps"
        )
        return anchored, anchored_R, False, True, capped, 0
    # A held point left off the mesh by a cap goes back on the surface.
    sx, sR = _project_foot(field, prev_x, None, None, seed_R=prev_R)
    if float(np.linalg.norm(sx - prev_x)) > 1e-4:
        return sx, sR, True, False, True, streak
    return prev_x.copy(), prev_R.copy(), True, False, capped, streak


def press_ray_hit(field, origin: np.ndarray, direction: np.ndarray):
    """Nearest mesh hit of the tool press axis. None when the line misses.

    A field may implement ``ray_hit(origin, direction)``. Otherwise the mold
    scene is cast in both directions along the axis.
    """
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    length = float(np.linalg.norm(direction))
    if length < 1e-9:
        return None
    direction = direction / length
    if hasattr(field, "ray_hit"):
        hit = field.ray_hit(origin, direction)
        if hit is None:
            return None
        return np.asarray(hit, dtype=np.float64).reshape(3)
    scene = getattr(field, "scene", None)
    if scene is None:
        return None
    import open3d as o3d

    hits = []
    for sign in (1.0, -1.0):
        ray = np.concatenate([origin, sign * direction]).astype(np.float32)
        ans = scene.cast_rays(o3d.core.Tensor(ray.reshape(1, 6)))
        t_hit = float(np.asarray(ans["t_hit"].numpy()).reshape(-1)[0])
        if np.isfinite(t_hit) and t_hit > 1e-4:
            hits.append(origin + sign * direction * t_hit)
    if not hits:
        return None
    dist = [float(np.linalg.norm(hit - origin)) for hit in hits]
    return hits[int(np.argmin(dist))]


def _slide_on_mesh(query, x, R, v_par, dt):
    """Add ``v_par * dt`` in the tangent plane and project each piece onto the mesh.

    A piece that hops to another sheet is dropped. The foot stays where the
    last accepted piece landed.
    """
    from fixb_tcp_track import MAX_FOOT_STEP_M, WALK_STEP_M

    step = np.asarray(R, dtype=np.float64).reshape(3, 3)[:, :2] @ np.asarray(v_par, dtype=np.float64).reshape(2)
    step = step * float(dt)
    x = np.asarray(x, dtype=np.float64).reshape(3).copy()
    R = np.asarray(R, dtype=np.float64).reshape(3, 3).copy()
    length = float(np.linalg.norm(step))
    if length < 1e-6 or not np.isfinite(length):
        return x, R
    # One step is still 6 mm. A longer command is clipped, then walked in pieces.
    if length > MAX_FOOT_STEP_M:
        step = step * (MAX_FOOT_STEP_M / length)
        length = MAX_FOOT_STEP_M
    pieces = max(1, int(np.ceil(length / WALK_STEP_M)))
    delta = step / pieces
    outward = -R[:, 2]
    for _ in range(pieces):
        landed_x, landed_R = query((x + delta).reshape(1, 3))
        landed_x = np.asarray(landed_x, dtype=np.float64).reshape(3)
        landed_R = np.asarray(landed_R, dtype=np.float64).reshape(3, 3)
        landed_n = -landed_R[:, 2]
        if float(np.linalg.norm(landed_x - x)) > WALK_STEP_M + 0.002:
            break
        if float(np.dot(landed_n, outward)) < np.cos(np.deg2rad(25.0)):
            break
        x, R, outward = landed_x, landed_R, landed_n
    return x, R


def _same_sheet(frame, previous) -> bool:
    """True when column 2 has not turned far enough to be the other sheet."""
    new = np.asarray(frame, dtype=np.float64).reshape(3, 3)[:, 2]
    old = np.asarray(previous, dtype=np.float64).reshape(3, 3)[:, 2]
    return float(np.dot(new, old)) >= np.cos(np.deg2rad(25.0))


def _tangential_offset(foot, frame, tcp) -> np.ndarray:
    """Foot minus TCP, with the surface-normal part removed."""
    outward = -np.asarray(frame, dtype=np.float64).reshape(3, 3)[:, 2]
    length = float(np.linalg.norm(outward))
    if length < 1e-12:
        return np.asarray(foot, dtype=np.float64).reshape(3) - np.asarray(tcp, dtype=np.float64).reshape(3)
    outward = outward / length
    delta = np.asarray(foot, dtype=np.float64).reshape(3) - np.asarray(tcp, dtype=np.float64).reshape(3)
    return delta - float(np.dot(delta, outward)) * outward


def _lead_anchor(query, foot, frame, tcp):
    """Point the gate measures against.

    The TCP's closest point when it is still on this sheet. Otherwise the
    foot with its sideways offset from the TCP removed, so a closest point on
    the other sheet cannot pull the foot across.
    """
    hit_x, hit_R = query(np.asarray(tcp, dtype=np.float64).reshape(1, 3))
    hit_x = np.asarray(hit_x, dtype=np.float64).reshape(3)
    hit_R = np.asarray(hit_R, dtype=np.float64).reshape(3, 3)
    if _same_sheet(hit_R, frame):
        return hit_x
    tang = _tangential_offset(foot, frame, tcp)
    return np.asarray(foot, dtype=np.float64).reshape(3) - tang


def _clamp_lead(query, foot, frame, tcp, max_gap, max_step):
    """Walk the foot toward the TCP's anchor until the gap is ``max_gap``.

    Each piece is projected. A piece that leaves the sheet, or that does not
    get closer to the anchor, is refused.
    """
    from fixb_tcp_track import WALK_STEP_M

    foot = np.asarray(foot, dtype=np.float64).reshape(3).copy()
    frame = np.asarray(frame, dtype=np.float64).reshape(3, 3).copy()
    anchor = _lead_anchor(query, foot, frame, tcp)
    limited = False
    moved = 0.0
    while moved < max_step - 1e-6:
        delta = anchor - foot
        dist = float(np.linalg.norm(delta))
        if dist <= max_gap + 1e-6 or dist < 1e-12:
            break
        limited = True
        step = min(WALK_STEP_M, max_step - moved, dist - max_gap)
        guess = foot + delta * (step / dist)
        landed_x, landed_R = query(guess.reshape(1, 3))
        landed_x = np.asarray(landed_x, dtype=np.float64).reshape(3)
        landed_R = np.asarray(landed_R, dtype=np.float64).reshape(3, 3)
        if not _same_sheet(landed_R, frame):
            break
        if float(np.linalg.norm(anchor - landed_x)) >= dist - 1e-4:
            break
        moved += float(np.linalg.norm(landed_x - foot))
        foot, frame = landed_x, landed_R
    return foot, frame, limited


def _outward_angle_deg(frame) -> tuple[float, np.ndarray]:
    """Angle of the outward normal from fixture up, and that unit normal."""
    outward = -np.asarray(frame, dtype=np.float64).reshape(3, 3)[:, 2]
    length = float(np.linalg.norm(outward))
    if length < 1e-12:
        return 0.0, outward
    outward = outward / length
    up = np.array([0.0, -1.0, 0.0])
    angle = float(np.degrees(np.arccos(np.clip(float(np.dot(outward, up)), -1.0, 1.0))))
    return angle, outward


def _climb_up_fillet(query, foot, frame, v_par, dt):
    """One chord back up the fillet when short pieces snap onto the steep face.

    From the wall, a 2 mm step toward the fillet still closest-points onto the
    wall. Probe 8, 10, 12, then 15 mm along ``v_par`` and take the shortest
    landing that is at least 8° less steep and still on this face.
    """
    frame = np.asarray(frame, dtype=np.float64).reshape(3, 3)
    step = frame[:, :2] @ np.asarray(v_par, dtype=np.float64).reshape(2)
    step = step * float(dt)
    length = float(np.linalg.norm(step))
    if length < 1e-4 or not np.isfinite(length):
        return np.asarray(foot, dtype=np.float64).reshape(3), frame
    direction = step / length
    here, here_n = _outward_angle_deg(frame)
    foot = np.asarray(foot, dtype=np.float64).reshape(3)
    for dist in (0.008, 0.010, 0.012, 0.015):
        landed_x, landed_R = query((foot + direction * dist).reshape(1, 3))
        landed_x = np.asarray(landed_x, dtype=np.float64).reshape(3)
        landed_R = np.asarray(landed_R, dtype=np.float64).reshape(3, 3)
        angle, landed_n = _outward_angle_deg(landed_R)
        if float(np.dot(landed_n, here_n)) < np.cos(np.deg2rad(50.0)):
            continue
        if angle > here - 8.0:
            continue
        return landed_x, landed_R
    return foot, frame


def foot_walk_mode() -> str:
    """``legacy`` keeps the 20 mm lead. ``geodesic`` follows the mesh triangles.

    Set ``UMIFT_FOOT_WALK=geodesic`` to walk triangle to triangle. Anything else,
    including an unset variable, keeps the 20 mm lead.
    """
    mode = os.environ.get("UMIFT_FOOT_WALK", "legacy").strip().lower()
    if mode in ("geodesic", "surface"):
        return "geodesic"
    return "legacy"


def foot_climb_enabled() -> bool:
    """The fillet-climb chord is off unless ``UMIFT_FOOT_CLIMB`` is set.

    ``1``, ``true``, ``on``, or ``yes`` enable it. Unset is off. The 20 mm lead
    does not depend on this flag.
    """
    return os.environ.get("UMIFT_FOOT_CLIMB", "").strip().lower() in ("1", "true", "on", "yes")


def _face_neighbors(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Neighbor face and its edge index across each local edge. -1 is a boundary."""
    faces = np.asarray(faces, dtype=np.int64)
    buckets: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for face_id, (a, b, c) in enumerate(faces):
        for edge, pair in enumerate(((b, c), (c, a), (a, b))):
            buckets.setdefault(tuple(sorted((int(pair[0]), int(pair[1])))), []).append((face_id, edge))
    neighbor = np.full((len(faces), 3), -1, dtype=np.int64)
    neighbor_edge = np.full((len(faces), 3), -1, dtype=np.int64)
    for pairs in buckets.values():
        if len(pairs) != 2:
            continue
        (face_a, edge_a), (face_b, edge_b) = pairs
        neighbor[face_a, edge_a] = face_b
        neighbor_edge[face_a, edge_a] = edge_b
        neighbor[face_b, edge_b] = face_a
        neighbor_edge[face_b, edge_b] = edge_a
    return neighbor, neighbor_edge


def _mesh_neighbors(field) -> tuple[np.ndarray, np.ndarray]:
    cached = getattr(field, "_face_neighbor_cache", None)
    if cached is None:
        cached = _face_neighbors(field.faces)
        field._face_neighbor_cache = cached
    return cached


def _barycentric(verts: np.ndarray, point: np.ndarray) -> np.ndarray | None:
    e1 = verts[1] - verts[0]
    e2 = verts[2] - verts[0]
    gram = np.array([[np.dot(e1, e1), np.dot(e1, e2)], [np.dot(e2, e1), np.dot(e2, e2)]])
    if abs(float(np.linalg.det(gram))) < 1e-18:
        return None
    rel = np.asarray(point, dtype=np.float64).reshape(3) - verts[0]
    a, b = np.linalg.solve(gram, [np.dot(e1, rel), np.dot(e2, rel)])
    return np.array([1.0 - a - b, a, b], dtype=np.float64)


def _bary_velocity(verts: np.ndarray, direction: np.ndarray) -> np.ndarray | None:
    weights = _barycentric(verts, verts[0] + np.asarray(direction, dtype=np.float64).reshape(3))
    if weights is None:
        return None
    # _barycentric of V0 + d is the weight of d, since V0 has weight (1, 0, 0).
    return weights - np.array([1.0, 0.0, 0.0])


def _locate_face(field, point: np.ndarray) -> tuple[int, np.ndarray | None]:
    point = np.asarray(point, dtype=np.float64).reshape(3)
    scene = getattr(field, "scene", None)
    if scene is not None:
        import open3d as o3d

        query = o3d.core.Tensor(point.astype(np.float32).reshape(1, 3))
        answer = scene.compute_closest_points(query)
        tri = int(answer["primitive_ids"].numpy()[0])
        if 0 <= tri < len(field.faces):
            uv = answer["primitive_uvs"].numpy()[0]
            return tri, np.array([1.0 - uv[0] - uv[1], uv[0], uv[1]], dtype=np.float64)
    best_face, best_w, best_score = -1, None, -1e9
    for face_id, face in enumerate(np.asarray(field.faces)):
        weights = _barycentric(field.vertices[face], point)
        if weights is None:
            continue
        score = float(np.min(weights))
        if score > best_score:
            best_face, best_w, best_score = face_id, weights, score
    if best_w is None:
        return -1, None
    weights = np.clip(best_w, 0.0, None)
    total = float(weights.sum())
    if total < 1e-12:
        return -1, None
    return best_face, weights / total


def _frame_on_triangle(field, face: int, weights: np.ndarray, prev_R: np.ndarray, travel: np.ndarray) -> np.ndarray:
    from fixb_field import orthonormal_frames
    from fixb_smooth_field import hold_tangent_through_spin

    orientations = getattr(field, "_ori", None)
    if orientations is None:
        orientations = getattr(field, "frames", None)
    if orientations is not None:
        vid = np.asarray(field.faces[face], dtype=np.int64)
        samples = orientations[vid]
        t1 = weights @ samples[:, :, 0]
        normal = weights @ samples[:, :, 2]
        if float(np.linalg.norm(normal)) > 1e-8 and float(np.linalg.norm(t1)) > 1e-8:
            landed = orthonormal_frames(t1, normal)
            return hold_tangent_through_spin(prev_R, landed)
    verts = field.vertices[field.faces[face]]
    normal = np.cross(verts[1] - verts[0], verts[2] - verts[0])
    if getattr(field, "orientation_z_inward", False):
        normal = -normal
    if float(np.linalg.norm(normal)) < 1e-12 or float(np.linalg.norm(travel)) < 1e-12:
        return np.asarray(prev_R, dtype=np.float64).reshape(3, 3)
    return orthonormal_frames(travel, normal)


def _walk_on_surface(field, foot, frame, v_par, dt):
    """Move ``v_par * dt`` across triangles. The path does not leave the surface."""
    frame = np.asarray(frame, dtype=np.float64).reshape(3, 3)
    foot = np.asarray(foot, dtype=np.float64).reshape(3)
    if not hasattr(field, "faces") or not hasattr(field, "vertices"):
        return foot, frame
    step = frame[:, :2] @ np.asarray(v_par, dtype=np.float64).reshape(2)
    step = step * float(dt)
    if float(np.linalg.norm(step)) < 1e-6 or not np.isfinite(np.linalg.norm(step)):
        return foot, frame
    face, weights = _locate_face(field, foot)
    if face < 0 or weights is None:
        return foot, frame
    neighbor, _neighbor_edge = _mesh_neighbors(field)
    verts = field.vertices[field.faces[face]]
    normal = np.cross(verts[1] - verts[0], verts[2] - verts[0])
    norm = float(np.linalg.norm(normal))
    if norm < 1e-12:
        return foot, frame
    normal = normal / norm
    direction = step - normal * float(np.dot(step, normal))
    remaining = float(np.linalg.norm(direction))
    if remaining < 1e-6:
        return foot, frame
    direction = direction / remaining
    came_from = -1
    for _ in range(100):
        if remaining < 1e-6:
            break
        verts = field.vertices[field.faces[face]]
        velocity = _bary_velocity(verts, direction)
        if velocity is None:
            break
        hit = None
        for edge in range(3):
            if velocity[edge] >= -1e-12:
                continue
            distance = -weights[edge] / velocity[edge]
            if distance < -1e-8:
                continue
            if hit is None or distance < hit[0]:
                hit = (distance, edge)
        if hit is None or hit[0] >= remaining - 1e-9:
            weights = weights + remaining * velocity
            weights = np.clip(weights, 0.0, None)
            weights = weights / max(float(weights.sum()), 1e-12)
            remaining = 0.0
            break
        distance, edge = hit
        if distance > 1e-8:
            weights = weights + distance * velocity
            remaining -= distance
        weights = np.clip(weights, 0.0, None)
        weights[edge] = 0.0
        total = float(weights.sum())
        if total < 1e-12:
            break
        weights = weights / total
        nxt = int(neighbor[face, edge])
        if nxt < 0 or nxt == came_from:
            remaining = 0.0
            break
        point = weights @ verts
        other = field.vertices[field.faces[nxt]]
        edge_ids = [i for i in range(3) if i != edge]
        edge_vec = verts[edge_ids[1]] - verts[edge_ids[0]]
        edge_len = float(np.linalg.norm(edge_vec))
        if edge_len < 1e-12:
            break
        edge_hat = edge_vec / edge_len
        mid = 0.5 * (verts[edge_ids[0]] + verts[edge_ids[1]])
        inward_old = verts[edge] - mid
        inward_old = inward_old - edge_hat * float(np.dot(inward_old, edge_hat))
        inward_len = float(np.linalg.norm(inward_old))
        if inward_len < 1e-12:
            break
        outward = -inward_old / inward_len
        along = float(np.dot(direction, edge_hat))
        across = float(np.dot(direction, outward))
        new_weights = _barycentric(other, point)
        if new_weights is None:
            break
        new_weights = np.clip(new_weights, 0.0, None)
        new_weights = new_weights / max(float(new_weights.sum()), 1e-12)
        # The vertex that is not on this edge is the one the neighbor stores opposite the shared edge.
        shared = {int(field.faces[face][i]) for i in edge_ids}
        interior = 0
        for local, vid in enumerate(field.faces[nxt]):
            if int(vid) not in shared:
                interior = local
                break
        inward_new = other[interior] - mid
        inward_new = inward_new - edge_hat * float(np.dot(inward_new, edge_hat))
        inward_new_len = float(np.linalg.norm(inward_new))
        if inward_new_len < 1e-12:
            break
        inward_new = inward_new / inward_new_len
        direction = along * edge_hat + across * inward_new
        dir_len = float(np.linalg.norm(direction))
        if dir_len < 1e-8:
            break
        direction = direction / dir_len
        came_from = face
        face = nxt
        weights = new_weights
    point = weights @ field.vertices[field.faces[face]]
    landed = _frame_on_triangle(field, face, weights, frame, direction)
    return point, landed


def _command_tangential(foot, frame, v_par, delta_n, dt) -> np.ndarray:
    """foot + (v1·t1 + v2·t2)·dt + δn·n_out. Frame z is the inward normal."""
    frame = np.asarray(frame, dtype=np.float64).reshape(3, 3)
    tang = frame[:, :2] @ np.asarray(v_par, dtype=np.float64).reshape(2) * float(dt)
    outward = -frame[:, 2]
    return np.asarray(foot, dtype=np.float64).reshape(3) + tang + float(delta_n) * outward


def _causal_foot_for(field):
    foot = getattr(field, "_causal_foot", None)
    if foot is not None:
        return foot
    paths = getattr(field, "_causal_npz", None)
    if paths is None:
        return None
    from causal_foot import CausalFoot

    foot = CausalFoot(paths[0], paths[1])
    field._causal_foot = foot
    return foot


def _integrate_causal_tcp(
    field,
    p0,
    v_par,
    delta_n,
    dt,
    prev_x,
    prev_R,
    tcp_carry,
    tcp_step,
) -> Dict[str, np.ndarray]:
    """Foot from the causal tracker at an internal TCP.

    The TCP starts at the measurement. Later chunks keep it. Each step it
    advances by ``tcp_step`` when that is the known full displacement (tangent
    and normal), otherwise by the command ``x_d``. The tangent is added to
    ``x_d`` once. ``dt`` is the value the caller passes (1/15 on the runner).
    """
    tracker = _causal_foot_for(field)
    v_par = np.asarray(v_par, dtype=np.float64).reshape(-1, 2)
    delta_n = np.asarray(delta_n, dtype=np.float64).reshape(-1)
    dt = np.asarray(dt, dtype=np.float64).reshape(-1)
    count = len(delta_n)
    measured = np.asarray(p0, dtype=np.float64).reshape(3)
    prev = None
    if prev_x is not None and prev_R is not None:
        prev = (
            np.asarray(prev_x, dtype=np.float64).reshape(3),
            np.asarray(prev_R, dtype=np.float64).reshape(3, 3),
        )
    if prev is None or tcp_carry is None:
        tcp_state = measured.copy()
    else:
        tcp_state = reset_carried_tcp(tcp_carry, measured)
    step = None if tcp_step is None else np.asarray(tcp_step, dtype=np.float64).reshape(-1, 3)
    xs = np.zeros((count, 3), dtype=np.float64)
    frames = np.zeros((count, 3, 3), dtype=np.float64)
    commanded = np.zeros((count, 3), dtype=np.float64)
    quiet = np.zeros(count, dtype=bool)
    for k in range(count):
        x, rotation = tracker.step(tcp_state, prev)
        xs[k] = x
        frames[k] = rotation
        commanded[k] = _command_tangential(x, rotation, v_par[k], delta_n[k], dt[k])
        if step is not None and k < len(step):
            tcp_state = tcp_state + step[k]
        else:
            tcp_state = commanded[k]
        prev = (x, rotation)
    outward = -frames[:, :, 2]
    return {
        "x_s": xs,
        "x_d": commanded,
        "R": frames,
        "n_out": outward,
        "face_held": quiet,
        "hop_hold": quiet,
        "foot_reanchor": quiet,
        "foot_cap": quiet,
        "hop_streak": np.array(0),
        "reject_streak": np.array(0),
        "step_reject": quiet,
        "ray_recovery": quiet,
        "tcp_carry": tcp_state,
    }


def integrate_tcp_foot(
    field,
    p0: np.ndarray,
    v_par: np.ndarray,
    delta_n: np.ndarray,
    dt: np.ndarray,
    prev_x: np.ndarray | None = None,
    prev_R: np.ndarray | None = None,
    hold_face: bool = False,
    cap_origin: np.ndarray | None = None,
    seed_R: np.ndarray | None = None,
    hop_streak: int = 0,
    step_allow: np.ndarray | None = None,
    reject_streak: int = 0,
    R_tool: np.ndarray | None = None,
    tcp_carry: np.ndarray | None = None,
    tcp_step: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    """Track the measured TCP, and slide the foot by ``v_par``.

    Episode start queries the closest point, as the labels do. Every later
    step adds ``v_par * dt`` in the tangent plane and projects that point back
    onto the mesh, at most ``MAX_FOOT_STEP_M``. A same-sheet slide is kept.
    The foot may lead the measured TCP by ``LEAD_M``. Only the extra is walked
    back, at most ``MAX_FOOT_STEP_M`` per step. The fillet-climb chord stays
    off unless ``UMIFT_FOOT_CLIMB`` is set. A step onto another sheet is
    refused. The press offset along
    the normal is not a lead.
    ``cap_origin`` is unused. ``hop_streak`` and ``step_allow`` are kept so
    older callers do not break. A smooth r4vt field uses the causal tracker
    instead of this walk.
    """
    if _causal_foot_for(field) is not None:
        return _integrate_causal_tcp(
            field, p0, v_par, delta_n, dt, prev_x, prev_R, tcp_carry, tcp_step
        )
    v_par = np.asarray(v_par, dtype=np.float64).reshape(-1, 2)
    delta_n = np.asarray(delta_n, dtype=np.float64).reshape(-1)
    dt = np.asarray(dt, dtype=np.float64).reshape(-1)
    del cap_origin, hop_streak, step_allow
    h = len(delta_n)
    from fixb_tcp_track import MAX_FOOT_STEP_M

    tcp = np.asarray(p0, dtype=np.float64).reshape(3)
    del R_tool
    query = lambda pts: _label_query(field, pts)
    episode_start = prev_x is None or prev_R is None
    x_s = np.zeros((h, 3))
    frames = np.zeros((h, 3, 3))
    face_held = np.zeros(h, dtype=bool)
    hop_held = np.zeros(h, dtype=bool)
    reanchored = np.zeros(h, dtype=bool)
    foot_capped = np.zeros(h, dtype=bool)
    streak = 0 if episode_start else int(reject_streak)
    xs = rk = None
    if episode_start:
        xs, rk = _project_foot(field, tcp, None, None, seed_R=seed_R)
    else:
        xs = np.asarray(prev_x, dtype=np.float64).reshape(3).copy()
        rk = np.asarray(prev_R, dtype=np.float64).reshape(3, 3).copy()
    for k in range(h):
        if not (episode_start and k == 0):
            if foot_walk_mode() == "geodesic":
                # On the triangles. No chord snap, no 20 mm lead, no fillet climb.
                xs, rk = _walk_on_surface(field, xs, rk, v_par[k], dt[k])
            else:
                # The gate is the distance to the TCP's closest point on this sheet.
                anchor = _lead_anchor(query, xs, rk, tcp)
                lead = float(np.linalg.norm(xs - anchor))
                rejected = lead > LEAD_M
                if rejected:
                    streak += 1
                else:
                    streak = 0
                # A same-sheet v_par slide stands, including when it increases the
                # lead. Only the part past LEAD_M is walked back, one step at most.
                slid_x, slid_R = _slide_on_mesh(query, xs, rk, v_par[k], dt[k])
                xs, rk, limited = _clamp_lead(
                    query, slid_x, slid_R, tcp, LEAD_M, MAX_FOOT_STEP_M
                )
                if foot_climb_enabled():
                    xs, rk = _climb_up_fillet(query, xs, rk, v_par[k], dt[k])
                foot_capped[k] = limited
                hop_held[k] = rejected
            if hold_face:
                xs, rk, face_held[k] = _reject_new_face(xs, rk, prev_x if k == 0 else x_s[k - 1], prev_R if k == 0 else frames[k - 1])
                # A query whose normal has turned is not adopted above. Hold the
                # commanded orientation anyway, as the 5° face hold used to.
                if not face_held[k]:
                    _hit_x, hit_R = query(tcp.reshape(1, 3))
                    hit_R = np.asarray(hit_R, dtype=np.float64).reshape(3, 3)
                    if not _same_face(hit_R, rk, FACE_HOLD_DEG):
                        face_held[k] = True
        x_s[k] = xs
        frames[k] = rk
        prev_x, prev_R = xs, rk
    if getattr(field, "orientation_z_inward", False):
        n_out = -frames[:, :, 2]
    else:
        n_out = frames[:, :, 2]
    return {
        "x_s": x_s,
        "x_d": x_s + delta_n[:, None] * n_out,
        "R": frames,
        "n_out": n_out,
        "face_held": face_held,
        "hop_hold": hop_held,
        "foot_reanchor": reanchored,
        "foot_cap": foot_capped,
        "hop_streak": np.array(0),
        "reject_streak": np.array(streak),
        "step_reject": hop_held,
        "ray_recovery": reanchored,
    }


def mean_rotation(rotations: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted rotation mean. Quaternion signs follow the heaviest sample."""
    mats = np.asarray(rotations, dtype=np.float64).reshape(-1, 3, 3)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    quats = Rotation.from_matrix(mats).as_quat()
    reference = quats[int(np.argmax(w))]
    flip = np.sum(quats * reference, axis=1) < 0.0
    quats = np.where(flip[:, None], -quats, quats)
    total = np.sum(quats * w[:, None], axis=0)
    norm = float(np.linalg.norm(total))
    total = reference if norm == 0.0 else total / norm
    return Rotation.from_quat(total).as_matrix()


def blend_action_rows(actions, starts, steps, decay_m: float = 0.01) -> np.ndarray:
    """Average v, δn, and ΔR from every chunk that predicts each absolute step.

    ``actions[k][i]`` predicts absolute step ``starts[k] + i``. Other columns
    come from the heaviest contributor, newest chunk on a tie.
    """
    from fixb_labels import so3_to_rot6d_rows

    actions = [expand_r4_action(np.asarray(a, dtype=np.float64)) for a in actions]
    starts = [int(s) for s in starts]
    horizon = actions[0].shape[0]
    width = actions[0].shape[1]
    out = np.zeros((len(steps), width), dtype=np.float64)
    for row, step in enumerate(steps):
        chosen = []
        for action, start in zip(actions, starts):
            index = int(step) - start
            if 0 <= index < horizon:
                chosen.append((float(np.exp(-float(decay_m) * index)), action[index]))
        if not chosen:
            raise KeyError(f"no action chunk covers absolute step {step}")
        weights = np.array([item[0] for item in chosen], dtype=np.float64)
        rows = np.stack([item[1] for item in chosen], axis=0)
        total = float(np.sum(weights))
        best = int(np.argmax(weights))
        # argmax keeps the first max. Walk so a later equal weight wins.
        for i, weight in enumerate(weights):
            if weight >= weights[best]:
                best = i
        base = rows[best].copy()
        base[0:2] = np.sum(rows[:, 0:2] * weights[:, None], axis=0) / total
        base[2] = float(np.sum(rows[:, 2] * weights) / total)
        if width >= 27:
            base[21:27] = so3_to_rot6d_rows(
                mean_rotation(rot6d_rows_to_R(rows[:, 21:27]), weights)
            ).reshape(6)
        out[row] = base
    return out


class FixBActionEnsembler:
    """Overlap raw r4 chunks, then the runner integrates the blend once."""

    def __init__(self, decay_m: float = 0.01):
        self.decay_m = float(decay_m)
        self._chunks: list = []
        self._next = 0

    def reset(self) -> None:
        self._chunks.clear()
        self._next = 0

    def push(self, start_step: int, action: np.ndarray) -> None:
        self._chunks.append((int(start_step), np.asarray(action, dtype=np.float64)))

    def pop(self, n: int) -> np.ndarray:
        if n <= 0:
            raise ValueError(f"pop n must be > 0, got {n}")
        steps = [self._next + i for i in range(int(n))]
        blended = blend_action_rows(
            [action for _start, action in self._chunks],
            [start for start, _action in self._chunks],
            steps,
            decay_m=self.decay_m,
        )
        self._next += int(n)
        self._chunks = [
            (start, action)
            for start, action in self._chunks
            if start + len(action) > self._next
        ]
        return blended


def _clamp_tilt_to_normal(R_d: np.ndarray, n: np.ndarray, max_deg: float) -> tuple:
    """Keep tool-z within max_deg of −n (press into mold). Returns (R_out, clamp_mask)."""
    R = np.asarray(R_d, dtype=np.float64).reshape(-1, 3, 3).copy()
    n = np.asarray(n, dtype=np.float64).reshape(-1, 3)
    n = n / np.clip(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9, None)
    target = -n
    z = R[:, :, 2]
    cos = np.clip(np.sum(z * target, axis=-1), -1.0, 1.0)
    ang = np.degrees(np.arccos(cos))
    hit = ang > max_deg
    if not np.any(hit):
        return R, hit
    # Rotate z toward target about z×target to land exactly at max_deg.
    for i in np.flatnonzero(hit):
        axis = np.cross(z[i], target[i])
        an = np.linalg.norm(axis)
        if an < 1e-9:
            continue
        axis = axis / an
        excess = np.radians(ang[i] - max_deg)
        dR = Rotation.from_rotvec(axis * excess).as_matrix()
        R[i] = dR @ R[i]
    return R, hit


def _clamp_rot_rate(R_d: np.ndarray, max_deg: float) -> tuple:
    """Cap geodesic step between consecutive R_d. Returns (R_out, clamp_mask)."""
    return _slew_orientations(R_d, None, max_deg)


def _slew_orientations(
    R_d: np.ndarray,
    prev: np.ndarray | None,
    max_deg: float | np.ndarray,
    freeze: np.ndarray | None = None,
) -> tuple:
    """Walk R_d toward each target, at most max_deg from the previous command.

    ``max_deg`` is one cap or one cap per step.

    ``prev`` is the last commanded orientation of the previous chunk. The first
    step of this chunk slews from it, so a seam is capped the same way as a step.
    A frozen step keeps that previous orientation instead of walking toward
    the new face.
    """
    R = np.asarray(R_d, dtype=np.float64).reshape(-1, 3, 3).copy()
    hit = np.zeros(len(R), dtype=bool)
    if len(R) == 0:
        return R, hit
    origin = None if prev is None else np.asarray(prev, dtype=np.float64).reshape(3, 3)
    held = np.zeros(len(R), dtype=bool) if freeze is None else np.asarray(freeze, dtype=bool)
    limits = np.asarray(max_deg, dtype=np.float64).reshape(-1)
    if limits.size == 1:
        limits = np.full(len(R), float(limits[0]))
    for i in range(len(R)):
        if held[i] and origin is not None:
            R[i] = origin
            continue
        cap = float(limits[i])
        if origin is None or cap <= 0:
            origin = R[i]
            continue
        R[i], hit[i] = _slew_one(origin, R[i], cap)
        origin = R[i]
    return R, hit


def _slew_one(origin: np.ndarray, target: np.ndarray, max_deg: float) -> tuple:
    rel = origin.T @ target
    rv = Rotation.from_matrix(rel).as_rotvec()
    ang = float(np.degrees(np.linalg.norm(rv)))
    if ang > max_deg and ang > 1e-9:
        stepped = origin @ Rotation.from_rotvec(rv * (max_deg / ang)).as_matrix()
        return stepped, True
    return target, False


@dataclass
class MoldAwareRuntime:
    field: FixBMoldField
    R_off: np.ndarray
    run_type: str
    limits: SafetyLimits = field(default_factory=SafetyLimits)
    dry_run: bool = True
    _prev_foot: Optional[np.ndarray] = None
    _prev_n: Optional[np.ndarray] = None
    _prev_R: Optional[np.ndarray] = None
    _prev_R_out: Optional[np.ndarray] = None
    _prev_p: Optional[np.ndarray] = None
    _prev_R_tool: Optional[np.ndarray] = None
    _cache_t: Optional[np.ndarray] = None
    _cache_x: Optional[np.ndarray] = None
    _cache_R: Optional[np.ndarray] = None
    _cache_sd: Optional[np.ndarray] = None
    _cache_curv: Optional[np.ndarray] = None
    _cache_reason: Optional[np.ndarray] = None
    _raw_t: Optional[np.ndarray] = None
    _raw_p: Optional[np.ndarray] = None
    _raw_s: Optional[np.ndarray] = None
    _wrench_over: bool = False
    # Last commanded foot. The next replan holds this face and this point.
    _cmd_foot: Optional[np.ndarray] = None
    _cmd_R: Optional[np.ndarray] = None
    # Internal TCP for the causal tracker. Not reset from the measurement.
    _cmd_tcp: Optional[np.ndarray] = None
    _tcp_step: Optional[np.ndarray] = None
    _cmd_Rd: Optional[np.ndarray] = None
    _cmd_vt: Optional[np.ndarray] = None
    _vt_elapsed_s: float = 0.0
    _hop_hold_streak: int = 0
    _reject_streak: int = 0
    last_min_table_clearance_m: Optional[float] = None
    # Level flange plane in the fixture frame. Clearance = offset - normal @ p.
    # Shared by the TCP guard and the camera guard.
    _table_normal: Optional[np.ndarray] = None
    _table_offset: Optional[float] = None
    last_triggers: list = field(default_factory=list)
    # Per-step bool masks from the latest decode_chunk (length H).
    last_clamp_speed: Optional[np.ndarray] = None
    last_clamp_step: Optional[np.ndarray] = None
    last_clamp_dn: Optional[np.ndarray] = None
    last_clamp_tilt: Optional[np.ndarray] = None
    last_clamp_rot_rate: Optional[np.ndarray] = None
    last_mesh_jump_hold: Optional[np.ndarray] = None
    max_mesh_jump_deg: float = 30.0

    @classmethod
    def from_assets(
        cls,
        run_type: str,
        field_path: Path = DEFAULT_FIELD,
        meta_path: Path = DEFAULT_META,
        dry_run: bool = True,
        limits: Optional[SafetyLimits] = None,
    ) -> "MoldAwareRuntime":
        meta = json.loads(Path(meta_path).read_text())
        field_path = Path(field_path)
        mold = FixBMoldField(field_path)
        # r1rel keeps the closest-face field. r4vt uses the smooth sidecar when it exists.
        if run_type == "r4_world_fixb_tilt":
            side = field_path.with_name("fixb_smooth_frames.npz")
            if side.exists():
                from fixb_smooth_field import SmoothFrameField

                mold = SmoothFrameField(field_path, side)
                mold._causal_npz = (field_path, side)
        return cls(
            field=mold,
            R_off=np.asarray(meta["R_off"], dtype=np.float64),
            run_type=run_type,
            dry_run=dry_run,
            limits=limits or SafetyLimits(),
        )

    def reset(self) -> None:
        self._prev_foot = None
        self._prev_n = None
        self._prev_R = None
        self._prev_R_out = None
        self._prev_p = None
        self._prev_R_tool = None
        self._cache_t = None
        self._cache_x = None
        self._cache_R = None
        self._cache_sd = None
        self._cache_curv = None
        self._cache_reason = None
        self._raw_t = None
        self._raw_p = None
        self._raw_s = None
        self._wrench_over = False
        self._cmd_foot = None
        self._cmd_R = None
        self._cmd_Rd = None
        self._cmd_vt = None
        self._cmd_tcp = None
        self._tcp_step = None
        self._vt_elapsed_s = 0.0
        self._hop_hold_streak = 0
        self._reject_streak = 0
        self.last_min_table_clearance_m = None
        self.last_triggers = []

    def release_command_foot(self) -> None:
        """Next decode queries the TCP. Offline windows do this; the live runner does not."""
        self._cmd_foot = None
        self._cmd_R = None
        self._cmd_Rd = None
        self._cmd_vt = None
        self._cmd_tcp = None
        self._tcp_step = None
        self._vt_elapsed_s = 0.0
        self._hop_hold_streak = 0
        self._reject_streak = 0
        self.last_min_table_clearance_m = None
        self.last_clamp_speed = None
        self.last_clamp_step = None
        self.last_clamp_dn = None
        self.last_clamp_tilt = None
        self.last_clamp_rot_rate = None
        self.last_rd_slew = None
        self.last_face_hold = None
        self.last_mesh_jump_hold = None

    @property
    def last_hard_triggers(self) -> list:
        return [t for t in self.last_triggers if t in HARD_STOP_KEYS]

    @property
    def last_soft_triggers(self) -> list:
        return [t for t in self.last_triggers if t in SOFT_CLAMP_KEYS]

    def fill_obs_raw(self, obs_raw: dict, robot_id: int = 0) -> Dict[str, np.ndarray]:
        """Write mold_* streams into obs_raw from ts_pose_fb (fixture/tool tip)."""
        self.last_triggers = []
        pose7 = np.asarray(obs_raw[f"ts_pose_fb_{robot_id}"], dtype=np.float64)
        R_fb, p_fb = pose7_to_Rp(pose7)
        if len(p_fb) == 0:
            return {}
        p_meas = np.asarray(p_fb, dtype=np.float64).copy()

        # Same tracker as the labels: 0.1 s causal position smooth, 5 mm gate,
        # tool-rotation switch, 25° slew. The decision face stays unslewed.
        # The published frame is the slew. With robot timestamps, each sample
        # is smoothed and tracked once from episode reset.
        times = obs_raw.get(f"robot_time_stamps_{robot_id}")
        times = None if times is None else np.asarray(times, dtype=np.float64).reshape(-1)
        if times is not None and len(times) == len(p_fb):
            p_fb = self._smooth_query(times, p_fb)
            self._track_new_samples(times, p_fb, R_fb)
            idx = self._cache_index(times)
            x = self._cache_x[idx]
            R = self._cache_R[idx]
            sd = self._cache_sd[idx]
            curv = self._cache_curv[idx]
            reason = self._cache_reason[idx]
        else:
            q = self.field.query(
                p_fb,
                with_curvature=True,
                R_tool=R_fb,
                prev_x=self._prev_foot,
                prev_R=self._prev_R,
                prev_R_tool=self._prev_R_tool,
                prev_p=self._prev_p,
                prev_R_out=self._prev_R_out,
            )
            self._remember_tail(p_fb, R_fb, q)
            x = np.asarray(q["x_s"], dtype=np.float64)
            R = np.asarray(q["R"], dtype=np.float64)
            sd = np.asarray(q["sd"], dtype=np.float64).reshape(-1)
            curv = np.asarray(q["curv"], dtype=np.float64)
        n = self._outward_normal(R)
        self.note_tcp_penetration(sd)
        self.note_measured_mesh_distance(p_meas, x)
        if self._prev_n is not None and self.limits.stop_on_normal_flip:
            if float(np.dot(self._prev_n, n[-1])) < 0:
                self.last_triggers.append("normal_flip")
        self._prev_n = n[-1].copy()

        signed = sd.reshape(-1, 1).astype(np.float32)
        normal = n.astype(np.float32)
        rot6d = so3_to_rot6d_rows(R).astype(np.float32)
        curv = np.asarray(curv, dtype=np.float32)

        obs_raw[f"mold_signed_distance_{robot_id}"] = signed
        obs_raw[f"mold_normal_{robot_id}"] = normal
        obs_raw[f"mold_R_field_rot6d_{robot_id}"] = rot6d
        obs_raw[f"mold_curvature_{robot_id}"] = curv
        obs_raw[f"mold_x_s_{robot_id}"] = x.astype(np.float32)
        out = {
            "sd": sd,
            "normal": n,
            "x_s": x,
            "R": R,
            "curv": curv,
        }
        if times is not None and len(times) == len(pose7):
            out["reason"] = reason
        return out

    def _remember_tail(self, p, R_tool, q) -> None:
        """Face state for the next decision, and the published frame for the next slew."""
        face = q["R_face"] if "R_face" in q else np.asarray(q["R"])[-1]
        self._prev_foot = np.asarray(q["x_s"][-1], dtype=np.float64).copy()
        self._prev_R = np.asarray(face, dtype=np.float64).reshape(3, 3).copy()
        self._prev_R_out = np.asarray(q["R"][-1], dtype=np.float64).reshape(3, 3).copy()
        self._prev_p = np.asarray(p[-1], dtype=np.float64).reshape(3).copy()
        self._prev_R_tool = np.asarray(R_tool[-1], dtype=np.float64).reshape(3, 3).copy()

    def _append_cache(self, times, q) -> None:
        x = np.asarray(q["x_s"], dtype=np.float64)
        R = np.asarray(q["R"], dtype=np.float64)
        sd = np.asarray(q["sd"], dtype=np.float64).reshape(-1)
        curv = np.asarray(q["curv"], dtype=np.float64)
        reason = np.asarray(q.get("reason", np.zeros(len(sd))), dtype=np.int8).reshape(-1)
        times = np.asarray(times, dtype=np.float64).reshape(-1)
        if self._cache_t is None:
            self._cache_t = times.copy()
            self._cache_x = x.copy()
            self._cache_R = R.copy()
            self._cache_sd = sd.copy()
            self._cache_curv = curv.copy()
            self._cache_reason = reason.copy()
            return
        self._cache_t = np.concatenate([self._cache_t, times])
        self._cache_x = np.concatenate([self._cache_x, x])
        self._cache_R = np.concatenate([self._cache_R, R])
        self._cache_sd = np.concatenate([self._cache_sd, sd])
        self._cache_curv = np.concatenate([self._cache_curv, curv])
        self._cache_reason = np.concatenate([self._cache_reason, reason])

    def _smooth_query(self, times: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Same 0.1 s trailing smooth as the labels. Past samples only, kept across calls."""
        times = np.asarray(times, dtype=np.float64).reshape(-1)
        p = np.asarray(p, dtype=np.float64).reshape(-1, 3)
        if self._raw_t is None:
            self._raw_t = times.copy()
            self._raw_p = p.copy()
            self._raw_s = smooth_positions(self._raw_t, self._raw_p, QUERY_SMOOTH_S)
        else:
            new = times > float(self._raw_t[-1]) + 1e-9
            if np.any(new):
                if not bool(np.all(new[int(np.flatnonzero(new)[0]) :])):
                    raise RuntimeError("mold tracker expects new robot timestamps as a suffix")
                start = len(self._raw_t)
                self._raw_t = np.concatenate([self._raw_t, times[new]])
                self._raw_p = np.concatenate([self._raw_p, p[new]])
                extra = smooth_positions(self._raw_t, self._raw_p, QUERY_SMOOTH_S, start=start)
                self._raw_s = np.concatenate([self._raw_s, extra])
        idx = self._match_times(self._raw_t, times)
        return self._raw_s[idx]

    @staticmethod
    def _match_times(have: np.ndarray, want: np.ndarray) -> np.ndarray:
        pos = np.searchsorted(have, want)
        pos = np.clip(pos, 0, len(have) - 1)
        prev = np.clip(pos - 1, 0, len(have) - 1)
        use_prev = np.abs(have[prev] - want) < np.abs(have[pos] - want)
        idx = np.where(use_prev, prev, pos)
        if float(np.max(np.abs(have[idx] - want))) > 1e-6:
            raise RuntimeError("mold tracker is missing a robot timestamp")
        return idx

    def _track_new_samples(self, times, p, R_tool) -> None:
        if self._cache_t is None:
            q = self.field.query(p, with_curvature=True, R_tool=R_tool)
            self._append_cache(times, q)
            self._remember_tail(p, R_tool, q)
            return
        new = np.asarray(times) > float(self._cache_t[-1]) + 1e-9
        if not np.any(new):
            return
        if not bool(np.all(new[int(np.flatnonzero(new)[0]) :])):
            raise RuntimeError("mold tracker expects new robot timestamps as a suffix")
        q = self.field.query(
            p[new],
            with_curvature=True,
            R_tool=R_tool[new],
            prev_x=self._prev_foot,
            prev_R=self._prev_R,
            prev_R_tool=self._prev_R_tool,
            prev_p=self._prev_p,
            prev_R_out=self._prev_R_out,
        )
        self._append_cache(times[new], q)
        self._remember_tail(p[new], R_tool[new], q)

    def _cache_index(self, times) -> np.ndarray:
        pos = np.searchsorted(self._cache_t, times)
        pos = np.clip(pos, 0, len(self._cache_t) - 1)
        prev = np.clip(pos - 1, 0, len(self._cache_t) - 1)
        use_prev = np.abs(self._cache_t[prev] - times) < np.abs(self._cache_t[pos] - times)
        idx = np.where(use_prev, prev, pos)
        if float(np.max(np.abs(self._cache_t[idx] - times))) > 1e-6:
            raise RuntimeError("mold tracker cache is missing a robot timestamp")
        return idx

    def unpack_fixb_action(
        self,
        raw_action: np.ndarray,
        apply_safety: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """raw_action (H,21) → v_par, delta_n, stiff, eoat, raw, clamp_dn (H,)."""
        a = np.asarray(raw_action, dtype=np.float64)
        v_par = a[:, 0:2].copy()
        delta_n_raw = a[:, 2].copy()
        delta_n = delta_n_raw.copy()
        clamp_dn = np.zeros(len(delta_n), dtype=bool)
        if self.dry_run:
            delta_n[:] = self.limits.dry_run_hover_m
            # dry-run override is intentional hover, not a press-depth clamp
        elif apply_safety and self.limits.enable_delta_n_floor:
            floor = -self.limits.max_press_depth_m
            clamp_dn = delta_n_raw < floor
            delta_n = np.maximum(delta_n_raw, floor)
        return v_par, delta_n, a[:, 18], a[:, 19:21], a, clamp_dn

    def set_table_plane(self, normal_fixture: np.ndarray, offset_m: float) -> None:
        """Level table in the fixture frame. Clearance is ``offset - normal @ p``.

        Only the D405 corner guard uses this plane. The TCP guard does not.
        """
        self._table_normal = np.asarray(normal_fixture, dtype=np.float64).reshape(3).copy()
        self._table_offset = float(offset_m)

    def _clearance(self, points: np.ndarray) -> Optional[np.ndarray]:
        """Meters above the table. Positive is above. None when no plane is known."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if self._table_normal is not None:
            return float(self._table_offset) - points @ self._table_normal
        table_y = self._table_plane_y()
        if table_y is None:
            return None
        return float(table_y) - points[:, 1]

    def _table_plane_y(self) -> Optional[float]:
        if self.limits.table_y_m is not None:
            return float(self.limits.table_y_m)
        verts = getattr(self.field, "vertices", None)
        if verts is None:
            return None
        return float(np.max(np.asarray(verts, dtype=np.float64)[:, 1]))

    def _hold_if_off_the_table(self, commanded: np.ndarray, measured: np.ndarray, held_vt: np.ndarray | None) -> np.ndarray:
        """Hold the previous target when the TCP is farther than the mesh limit.

        The flange plane is not applied here. A point farther than
        ``max_distance_from_mesh_m`` from the mesh is refused.
        """
        commanded = np.asarray(commanded, dtype=np.float64).reshape(-1, 3).copy()
        measured = np.asarray(measured, dtype=np.float64).reshape(3)
        points = np.vstack([measured.reshape(1, 3), commanded])
        limit = float(self.limits.max_distance_from_mesh_m)
        for point in points:
            closest = np.asarray(self.field.query(point.reshape(1, 3))["x_s"], dtype=np.float64).reshape(3)
            if float(np.linalg.norm(point - closest)) > limit:
                self.last_triggers.append("mesh_far")
                break
        if "mesh_far" not in self.last_triggers:
            return commanded
        hold = measured if held_vt is None else np.asarray(held_vt, dtype=np.float64).reshape(3)
        self._cmd_vt = hold.copy()
        return np.broadcast_to(hold, commanded.shape).copy()

    def _hold_if_camera_hits_table(
        self,
        R_cmd: np.ndarray,
        x_cmd: np.ndarray,
        R_meas: np.ndarray,
        p_meas: np.ndarray,
        held_vt: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Hold when a D405 housing corner meets the table.

        The box is the Intel case, checked against the flange plane. The TCP
        guard does not use that plane. Checked on the commanded pose and the
        measured pose.
        """
        R_cmd = np.asarray(R_cmd, dtype=np.float64).reshape(-1, 3, 3).copy()
        x_cmd = np.asarray(x_cmd, dtype=np.float64).reshape(-1, 3).copy()
        corners = d405_housing_corners_in_tcp()
        R_meas = np.asarray(R_meas, dtype=np.float64).reshape(3, 3)
        p_meas = np.asarray(p_meas, dtype=np.float64).reshape(3)
        commanded_cam = np.einsum("nij,kj->nki", R_cmd, corners) + x_cmd[:, None, :]
        measured_cam = (R_meas @ corners.T).T + p_meas
        points = np.vstack([measured_cam, commanded_cam.reshape(-1, 3)])
        clearance = self._clearance(points)
        if clearance is None:
            return R_cmd, x_cmd
        if np.all(clearance >= CAM_TABLE_MARGIN_M):
            return R_cmd, x_cmd
        self.last_triggers.append("camera_table")
        hold_p = p_meas if held_vt is None else np.asarray(held_vt, dtype=np.float64).reshape(3)
        hold_R = R_meas if self._cmd_Rd is None else np.asarray(self._cmd_Rd, dtype=np.float64).reshape(3, 3)
        self._cmd_vt = hold_p.copy()
        return np.broadcast_to(hold_R, R_cmd.shape).copy(), np.broadcast_to(hold_p, x_cmd.shape).copy()

    def note_tcp_penetration(self, sd: np.ndarray) -> None:
        """Abort when the measured TCP, not the virtual target, is too deep."""
        if not self.limits.enable_tcp_penetration_abort:
            return
        depth = np.asarray(sd, dtype=np.float64).reshape(-1)
        if np.any(depth < -float(self.limits.max_tcp_penetration_m)):
            self.last_triggers.append("tcp_penetration")

    def note_measured_mesh_distance(self, p_meas: np.ndarray, x_s: np.ndarray) -> None:
        """Abort when the newest measured TCP is farther than max_distance_from_mesh_m.

        The observation window still holds the approach and the homing lift.
        Those older poses are not the pose about to move. Off unless
        abort_on_measured_mesh_distance is set. On, decode_chunk does not
        abort on the commanded ||x_d - x_s||.
        """
        if not self.limits.abort_on_measured_mesh_distance:
            return
        p = np.asarray(p_meas, dtype=np.float64).reshape(-1, 3)
        x = np.asarray(x_s, dtype=np.float64).reshape(-1, 3)
        if len(p) == 0 or len(x) == 0:
            return
        dist = float(np.linalg.norm(p[-1] - x[-1]))
        if dist > float(self.limits.max_distance_from_mesh_m):
            self.last_triggers.append("max_distance_from_mesh")

    def wrench_hold_update(self, force_n: float) -> str:
        """'ok', 'hold', or 'abort'. Hold on the first sample at the cap."""
        if not self.limits.enable_wrench_hold:
            return "ok"
        if float(force_n) < float(self.limits.wrench_hold_n):
            self._wrench_over = False
            return "ok"
        if self._wrench_over:
            self._wrench_over = True
            self.last_triggers.append("wrench_abort")
            return "abort"
        self._wrench_over = True
        self.last_triggers.append("wrench_hold")
        return "hold"

    def _limit_commanded_vt(
        self,
        x_d: np.ndarray,
        x_s: np.ndarray,
        normal: np.ndarray,
        dt: np.ndarray,
        p0: np.ndarray,
    ):
        """Clip a step longer than 25 mm. Do not rebuild the point on the normal.

        ``x_d`` is already ``foot + v·dt + δn·n``. The first ``VT_RAMP_S``
        seconds may move at most that fraction of ``max_step_m``. A longer
        step is shortened along the same direction. It is not held in place.
        """
        desired = np.asarray(x_d, dtype=np.float64).reshape(-1, 3)
        x_s = np.asarray(x_s, dtype=np.float64).reshape(-1, 3)
        normal = np.asarray(normal, dtype=np.float64).reshape(-1, 3)
        normal = normal / np.clip(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-9, None)
        dt = np.asarray(dt, dtype=np.float64).reshape(-1)
        steps = len(desired)
        out = np.zeros((steps, 3), dtype=np.float64)
        delta_n = np.zeros(steps, dtype=np.float64)
        hit = np.zeros(steps, dtype=bool)
        if self._cmd_vt is None:
            prev = np.asarray(p0, dtype=np.float64).reshape(3).copy()
        else:
            prev = np.asarray(self._cmd_vt, dtype=np.float64).reshape(3).copy()
        full = float(self.limits.max_step_m)
        for i in range(steps):
            self._vt_elapsed_s += float(dt[i])
            scale = 1.0 if VT_RAMP_S <= 0 else min(1.0, self._vt_elapsed_s / VT_RAMP_S)
            delta = desired[i] - prev
            dist = float(np.linalg.norm(delta))
            allow = full * scale
            if dist > allow and dist > 1e-12:
                point = prev + delta * (allow / dist)
                hit[i] = True
            else:
                point = desired[i]
            out[i] = point
            delta_n[i] = float(np.dot(point - x_s[i], normal[i]))
            prev = point.copy()
        self._cmd_vt = prev.copy()
        return out, delta_n, hit

    def decode_chunk(
        self,
        raw_action: np.ndarray,
        p0: np.ndarray,
        R0: np.ndarray,
        dt: np.ndarray,
        apply_safety: bool = True,
        frame_seed: np.ndarray | None = None,
        label_R: np.ndarray | None = None,
    ) -> Dict[str, np.ndarray]:
        """Anchored Fix B decode for one chunk. Returns fixture-frame poses.

        apply_safety=False skips speed/depth clamps (offline parity vs eval).
        Live robot path keeps the default True. ``label_R`` is the replay-only
        label tool orientation: a step more than 25° from it is held before
        the flange-plane check. Rollouts leave it unset.
        """
        self.last_triggers = []
        H = raw_action.shape[0]
        self.last_clamp_speed = np.zeros(H, dtype=bool)
        self.last_clamp_step = np.zeros(H, dtype=bool)
        self.last_clamp_dn = np.zeros(H, dtype=bool)
        self.last_clamp_tilt = np.zeros(H, dtype=bool)
        self.last_clamp_rot_rate = np.zeros(H, dtype=bool)
        self.last_rd_slew = np.zeros(H, dtype=bool)
        self.last_face_hold = np.zeros(H, dtype=bool)
        self.last_mesh_jump_hold = np.zeros(H, dtype=bool)
        self.last_hop_hold = np.zeros(H, dtype=bool)
        self.last_foot_reanchor = np.zeros(H, dtype=bool)
        self.last_foot_cap = np.zeros(H, dtype=bool)
        self.last_ori_hold = np.zeros(H, dtype=bool)
        raw_action = expand_r4_action(np.asarray(raw_action, dtype=np.float64))
        H = raw_action.shape[0]

        if self.run_type == "r1_world_world":
            pos = raw_action[:, 0:3].astype(np.float64)
            R = rot6d_rows_to_R(raw_action[:, 3:9])
            q = self.field.query(pos)
            if apply_safety:
                seed = p0 if self._cmd_vt is None else self._cmd_vt
                pos, step_hit = _hold_long_steps(pos, seed, self.limits.max_step_m)
                self.last_clamp_step = step_hit
                if np.any(step_hit):
                    self.last_triggers.append("max_step")
                if len(pos):
                    self._cmd_vt = np.asarray(pos[-1], dtype=np.float64).copy()
            return {
                "x_d": pos,
                "R_d": R,
                "x_s": q["x_s"],
                "R_field": q["R"],
                "sd": q["sd"],
                "stiffness": raw_action[:, 18],
                "eoat": raw_action[:, 19:21],
                "raw": raw_action,
                "clamp_speed": self.last_clamp_speed,
                "clamp_step": self.last_clamp_step,
                "clamp_dn": self.last_clamp_dn,
                "clamp_tilt": self.last_clamp_tilt,
                "clamp_rot_rate": self.last_clamp_rot_rate,
                "mesh_jump_hold": self.last_mesh_jump_hold,
            }

        v_par, delta_n, stiff, eoat, raw, clamp_dn = self.unpack_fixb_action(
            raw_action, apply_safety=apply_safety
        )
        self.last_clamp_dn = clamp_dn
        if np.any(clamp_dn):
            self.last_triggers.append("delta_n")

        speed = np.linalg.norm(v_par, axis=-1)
        # Hop allowance is this step's own length, before the speed clamp.
        label_step = speed * np.asarray(dt, dtype=np.float64).reshape(H)
        clamp_speed = np.zeros(H, dtype=bool)
        if apply_safety:
            clamp_speed = speed > self.limits.max_tangential_speed_mps
            if np.any(clamp_speed):
                self.last_triggers.append("max_tangential_speed")
                scale = np.minimum(
                    1.0, self.limits.max_tangential_speed_mps / np.clip(speed, 1e-9, None)
                )
                v_par = v_par * scale[:, None]
        self.last_clamp_speed = clamp_speed

        dt = np.asarray(dt, dtype=np.float64).reshape(H)
        # The 5° face hold freezes a smooth fillet, and the labels do not use it.
        # Episode start uses the mesh closest point and does not apply the 10 mm cap.
        # No carried frame yet. Replay passes the label frame. Rollout keeps
        # the field lookup at the foot; later steps and later chunks carry it.
        seed_R = None
        if self._cmd_R is None and frame_seed is not None:
            seed_R = np.asarray(frame_seed, dtype=np.float64).reshape(3, 3)
        dec = integrate_tcp_foot(
            self.field,
            p0,
            v_par,
            delta_n,
            dt,
            prev_x=self._cmd_foot,
            prev_R=self._cmd_R,
            hold_face=apply_safety and not getattr(self.field, "orientation_z_inward", False),
            cap_origin=None,
            seed_R=seed_R,
            hop_streak=self._hop_hold_streak,
            step_allow=label_step,
            reject_streak=self._reject_streak,
            R_tool=np.asarray(R0, dtype=np.float64).reshape(3, 3),
            tcp_carry=self._cmd_tcp,
            tcp_step=self._tcp_step,
        )
        self._tcp_step = None
        if "tcp_carry" in dec:
            self._cmd_tcp = np.asarray(dec["tcp_carry"], dtype=np.float64).reshape(3).copy()
        self._hop_hold_streak = int(np.asarray(dec["hop_streak"]).reshape(-1)[0])
        self._reject_streak = int(np.asarray(dec["reject_streak"]).reshape(-1)[0])
        self.last_step_reject = np.asarray(dec["step_reject"], dtype=bool).reshape(H)
        self.last_ray_recovery = np.asarray(dec["ray_recovery"], dtype=bool).reshape(H)
        self.last_hop_hold = np.asarray(dec["hop_hold"], dtype=bool).reshape(H)
        self.last_foot_reanchor = np.asarray(dec["foot_reanchor"], dtype=bool).reshape(H)
        self.last_foot_cap = np.asarray(dec["foot_cap"], dtype=bool).reshape(H)
        if np.any(self.last_hop_hold):
            self.last_triggers.append("hop_hold")
        if np.any(self.last_foot_reanchor):
            self.last_triggers.append("foot_reanchor")
        if np.any(self.last_foot_cap):
            self.last_triggers.append("foot_cap")
        R_field = dec["R"]
        x_d = dec["x_d"]
        x_s = dec["x_s"]
        n_out = dec["n_out"]
        face_held = np.asarray(dec["face_held"], dtype=bool).reshape(H)
        self.last_face_hold = face_held
        if np.any(face_held):
            self.last_triggers.append("face_hold")

        if self.run_type == "r4_world_fixb_tilt":
            # Learned ΔR in action[..., 21:27]; project rot6d → SO(3).
            # Same mesh R_f as the re-projected foot — no twist re-resolution
            # (flipping R_f here without flipping v_par would reverse the slide).
            # Hover keeps the predicted ΔR (only δn is overridden in unpack).
            if raw_action.shape[-1] < 27:
                raise ValueError(
                    f"r4 decode needs action dim >= 27, got {raw_action.shape[-1]}"
                )
            dR = rot6d_rows_to_R(raw_action[:, 21:27])
            R_d = np.einsum("hij,hjk->hik", R_field, dR)
            # If mesh R_f jumps mid-chunk, hold previous R_d (then rate-clamp).
            if self.limits.enable_mesh_jump_hold and H > 1:
                from scipy.spatial.transform import Rotation as _Rot

                hold = np.zeros(H, dtype=bool)
                for t in range(1, H):
                    rel = R_field[t - 1].T @ R_field[t]
                    ang = float(
                        np.degrees(np.linalg.norm(_Rot.from_matrix(rel).as_rotvec()))
                    )
                    if ang > self.max_mesh_jump_deg:
                        R_d[t] = R_d[t - 1]
                        hold[t] = True
                self.last_mesh_jump_hold = hold
                if np.any(hold):
                    self.last_triggers.append("mesh_jump_hold")
        else:
            B = heading_branch(
                R_field[0:1],
                np.asarray(R0, dtype=np.float64).reshape(1, 3, 3),
                self.R_off,
            )
            R_d = np.einsum("hij,jk->hik", R_field, B[0])

        if apply_safety and self.limits.enable_tilt_clamp:
            R_d, tilt_hit = _clamp_tilt_to_normal(
                R_d, n_out, self.limits.max_tilt_from_normal_deg
            )
            self.last_clamp_tilt = tilt_hit
            if np.any(tilt_hit):
                self.last_triggers.append("max_tilt")
        if apply_safety and self.limits.enable_rot_rate_clamp:
            R_d, rate_hit = _clamp_rot_rate(R_d, self.limits.max_rot_rate_deg_per_step)
            self.last_clamp_rot_rate = rate_hit
            if np.any(rate_hit):
                self.last_triggers.append("max_rot_rate")
        if (apply_safety and self.limits.enable_rd_slew) or np.any(face_held):
            if apply_safety and self.limits.enable_rd_slew:
                # 3° during the virtual-target ramp, then 8° so a label step fits.
                elapsed = float(self._vt_elapsed_s)
                max_deg = np.empty(H, dtype=np.float64)
                for i in range(H):
                    max_deg[i] = (
                        float(self.limits.max_rd_slew_deg)
                        if elapsed < VT_RAMP_S
                        else RD_SLEW_AFTER_RAMP_DEG
                    )
                    elapsed += float(dt[i])
            else:
                max_deg = 180.0
            # No previous command yet: slew the first step from the measured tool.
            slew_prev = self._cmd_Rd
            if (
                slew_prev is None
                and apply_safety
                and self.limits.enable_rd_slew
            ):
                slew_prev = np.asarray(R0, dtype=np.float64).reshape(3, 3)
            R_d, slew_hit = _slew_orientations(
                R_d, slew_prev, max_deg, freeze=face_held
            )
            self.last_rd_slew = slew_hit
            if np.any(slew_hit):
                self.last_triggers.append("rd_slew")

        if apply_safety:
            held_vt = None if self._cmd_vt is None else np.asarray(self._cmd_vt, dtype=np.float64).reshape(3).copy()
            x_d, delta_n, step_hit = self._limit_commanded_vt(
                x_d, x_s, n_out, dt, p0
            )
            self.last_clamp_step = step_hit
            if np.any(step_hit):
                self.last_triggers.append("max_step")
        else:
            held_vt = None
        if (
            apply_safety
            and not self.dry_run
            and not self.limits.abort_on_measured_mesh_distance
        ):
            dist = np.linalg.norm(x_d - x_s, axis=1)
            if np.any(dist > self.limits.max_distance_from_mesh_m):
                self.last_triggers.append("max_distance_from_mesh")
        if label_R is not None:
            # Replay only. The flange check below must see this held orientation,
            # or a step the 25° rule would reject aborts the episode first.
            held_R = (
                np.asarray(R0, dtype=np.float64).reshape(3, 3)
                if self._cmd_Rd is None
                else np.asarray(self._cmd_Rd, dtype=np.float64).reshape(3, 3)
            )
            R_d, self.last_ori_hold = hold_orientation_near_label(
                R_d,
                np.asarray(label_R, dtype=np.float64).reshape(-1, 3, 3),
                held_R,
                ORI_LABEL_HOLD_DEG,
            )
        if apply_safety:
            x_d = self._hold_if_off_the_table(x_d, p0, held_vt)
            R_d, x_d = self._hold_if_camera_hits_table(R_d, x_d, R0, p0, held_vt)

        self._cmd_foot = np.asarray(x_s[-1], dtype=np.float64).copy()
        self._cmd_R = np.asarray(R_field[-1], dtype=np.float64).reshape(3, 3).copy()
        self._cmd_Rd = np.asarray(R_d[-1], dtype=np.float64).reshape(3, 3).copy()

        return {
            "x_d": x_d,
            "R_d": R_d,
            "x_s": x_s,
            "R_field": R_field,
            "n_out": n_out,
            "sd": np.sum((x_d - x_s) * n_out, axis=1),
            "stiffness": stiff,
            "eoat": eoat,
            "raw": raw,
            "v_par": v_par,
            "delta_n": delta_n,
            "clamp_speed": self.last_clamp_speed,
            "clamp_step": self.last_clamp_step,
            "clamp_dn": self.last_clamp_dn,
            "clamp_tilt": self.last_clamp_tilt,
            "clamp_rot_rate": self.last_clamp_rot_rate,
            "rd_slew": self.last_rd_slew,
            "face_hold": self.last_face_hold,
            "mesh_jump_hold": self.last_mesh_jump_hold,
            "hop_hold": self.last_hop_hold,
            "foot_reanchor": self.last_foot_reanchor,
            "foot_cap": self.last_foot_cap,
            "ori_hold": self.last_ori_hold,
        }

    def _outward_normal(self, rotation: np.ndarray) -> np.ndarray:
        """Column 2 is outward, except on the smooth field where it points into the mold."""
        n = np.asarray(rotation, dtype=np.float64)[..., 2]
        if getattr(self.field, "orientation_z_inward", False):
            return -n
        return n

    def note_commanded_rd(self, rotation: np.ndarray) -> None:
        """Last R_d that was actually scheduled, in the fixture frame."""
        self._cmd_Rd = np.asarray(rotation, dtype=np.float64).reshape(3, 3).copy()

    def decoded_to_action_mats(
        self, decoded: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reference pose is the foot. Virtual target is foot + delta_n along n.

        The runner already schedules the virtual target and builds stiffness
        from VT − nominal (label stiffness on that axis, 5000 on the other two),
        the same way as standard_bc. The wrench command stays zero.
        """
        x_s = np.asarray(decoded["x_s"], dtype=np.float64)
        r_d = decoded["R_d"]
        if "n_out" in decoded:
            n = np.asarray(decoded["n_out"], dtype=np.float64)
        else:
            n = np.asarray(decoded["R_field"][..., 2], dtype=np.float64)
        delta_n = np.asarray(decoded["delta_n"], dtype=np.float64).reshape(-1)
        target = Rp_to_SE3(r_d, x_s)
        vt = Rp_to_SE3(r_d, x_s + delta_n[:, None] * n)
        return target, vt

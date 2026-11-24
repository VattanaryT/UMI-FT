"""Opt-in policy start: episode N's first TCP pose, with a safe wrist-3 winding.

Policy rollouts only. GT replay keeps its own start. The Cartesian approach
still hovers 5 cm back along the tool axis, but rotation is split into steps
of at most 5 deg so a tool-roll match cannot trip the 0.2 rad servo limit.

Wrist 3 is the UR5e factory range, ±360 deg. The return stroke adds about
+174 deg. A start near +π finishes that flip a few degrees short of +360, so
the selected winding is the 2π-equivalent that keeps both the start and the
flipped angle at least 30 deg inside the stops.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import zarr
from scipy.spatial.transform import Rotation

from PyriteUtility.spatial_math import spatial_utilities as su

POLICY_START_ENV = "UMIFT_POLICY_START_EPISODE"
POLICY_START_INDEX_ENV = "UMIFT_POLICY_START_INDEX"
POLICY_LOOKBACK_S = 8.0
MAX_ROT_STEP_DEG = 5.0
POS_TOL_M = 0.005
ANG_TOL_DEG = 3.0
WRIST3_INDEX = 5
# UR5e factory wrist 3. ±180 is only the description URDF.
WRIST3_LIMIT_RAD = 2.0 * np.pi
RETURN_FLIP_DEG = 174.0
# Clearance that still counts as leaving room after the flip.
WRIST3_MARGIN_DEG = 30.0


def read_policy_start_episode(environ=None) -> Optional[int]:
    """Episode number from the env, or None when unset.

    A GT replay must not use this. Both variables set is an error.
    """
    environ = os.environ if environ is None else environ
    raw = str(environ.get(POLICY_START_ENV, "")).strip()
    if raw == "":
        return None
    if str(environ.get("UMIFT_REPLAY_EPISODE", "")).strip():
        raise RuntimeError(
            f"{POLICY_START_ENV} is for policy rollouts. "
            "Unset it for a GT replay."
        )
    try:
        episode = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{POLICY_START_ENV}={raw!r} is not an episode number."
        ) from exc
    if episode < 0:
        raise ValueError(f"{POLICY_START_ENV}={episode} is negative.")
    return episode


def read_policy_start_index(environ=None) -> int:
    """Row inside the start episode. Unset means the first frame."""
    environ = os.environ if environ is None else environ
    raw = str(environ.get(POLICY_START_INDEX_ENV, "")).strip()
    if raw == "":
        return 0
    try:
        index = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{POLICY_START_INDEX_ENV}={raw!r} is not a frame index."
        ) from exc
    if index < 0:
        raise ValueError(f"{POLICY_START_INDEX_ENV}={index} is negative.")
    return index


def frame_pose7(zarr_path, episode: int, index: int = 0) -> np.ndarray:
    """``ts_pose_fb_0[index]`` of ``episode_<n>``, dataset frame."""
    root = zarr.open(str(zarr_path), mode="r")
    name = f"episode_{int(episode)}"
    if name not in root["data"]:
        raise ValueError(f"{name} is not in {zarr_path}")
    pose = np.asarray(root["data"][name]["ts_pose_fb_0"], dtype=np.float64)
    index = int(index)
    if index >= len(pose):
        raise ValueError(f"{name} has {len(pose)} poses, index {index}")
    return pose[index].reshape(7).copy()


def first_frame_pose7(zarr_path, episode: int) -> np.ndarray:
    """First ``ts_pose_fb_0`` row of ``episode_<n>``, dataset frame."""
    return frame_pose7(zarr_path, episode, 0)


def pose_rotation_deg(pose_a, pose_b) -> float:
    """Geodesic angle between two pose7 orientations, in degrees."""
    a = np.asarray(pose_a, dtype=np.float64).reshape(7)
    b = np.asarray(pose_b, dtype=np.float64).reshape(7)
    ra = Rotation.from_quat([a[4], a[5], a[6], a[3]])
    rb = Rotation.from_quat([b[4], b[5], b[6], b[3]])
    return float(np.degrees((ra.inv() * rb).magnitude()))


def pose_position_m(pose_a, pose_b) -> float:
    a = np.asarray(pose_a, dtype=np.float64).reshape(7)
    b = np.asarray(pose_b, dtype=np.float64).reshape(7)
    return float(np.linalg.norm(a[:3] - b[:3]))


def rotation_capped_waypoints(
    pose_a, pose_b, max_step_deg: float = MAX_ROT_STEP_DEG, min_steps: int = 1
) -> np.ndarray:
    """Slerp from ``pose_a`` to ``pose_b`` with each step at most ``max_step_deg``.

    ``min_steps`` keeps a long translation from being one big position jump.
    The returned array includes both endpoints.
    """
    a = np.asarray(pose_a, dtype=np.float64).reshape(7)
    b = np.asarray(pose_b, dtype=np.float64).reshape(7)
    ang = pose_rotation_deg(a, b)
    n_ang = int(np.ceil(ang / float(max_step_deg) - 1e-9)) if ang > 1e-9 else 1
    n = max(n_ang, int(min_steps), 1)
    alpha = np.linspace(0.0, 1.0, n + 1)
    return su.pose7_interp(a, b, alpha)


def consecutive_rotation_deg(waypoints) -> np.ndarray:
    pts = np.asarray(waypoints, dtype=np.float64).reshape(-1, 7)
    if len(pts) < 2:
        return np.zeros(0)
    return np.array(
        [pose_rotation_deg(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    )


def wrist3_has_room(
    q6: float,
    *,
    limit_rad: float = WRIST3_LIMIT_RAD,
    flip_deg: float = RETURN_FLIP_DEG,
    margin_deg: float = WRIST3_MARGIN_DEG,
) -> bool:
    """True when q6 and q6 plus the return flip sit ``margin_deg`` inside ±limit."""
    margin = np.deg2rad(margin_deg)
    end = float(q6) + np.deg2rad(flip_deg)
    lo = -float(limit_rad) + margin
    hi = float(limit_rad) - margin
    return lo <= float(q6) <= hi and lo <= end <= hi


def wrist3_seeds(joints, limit_rad: float = WRIST3_LIMIT_RAD) -> list:
    """Current joints plus the wrist-3 windings that are still inside ±limit."""
    q = np.asarray(joints, dtype=np.float64).reshape(6).copy()
    seeds = []
    for k in (-1, 0, 1):
        seed = q.copy()
        seed[WRIST3_INDEX] = q[WRIST3_INDEX] + k * 2.0 * np.pi
        if abs(seed[WRIST3_INDEX]) <= float(limit_rad) + 1e-9:
            seeds.append(seed)
    return seeds


def select_wrist3(
    current_q6: float,
    candidate_q6,
    *,
    limit_rad: float = WRIST3_LIMIT_RAD,
    flip_deg: float = RETURN_FLIP_DEG,
    margin_deg: float = WRIST3_MARGIN_DEG,
) -> float:
    """Wrist 3 closest to ``current_q6`` that still has room for the flip."""
    passing = [
        float(q)
        for q in candidate_q6
        if wrist3_has_room(
            q, limit_rad=limit_rad, flip_deg=flip_deg, margin_deg=margin_deg
        )
    ]
    if not passing:
        shown = ", ".join(f"{float(q):.3f}" for q in candidate_q6)
        raise ValueError(
            f"No wrist-3 winding leaves room for +{flip_deg:.0f} deg "
            f"inside ±{np.degrees(limit_rad):.0f} deg. Candidates: [{shown}]"
        )
    return min(passing, key=lambda q: abs(q - float(current_q6)))


def plan_wrist_unwind(current_q, ik_solutions: Sequence[Optional[Sequence[float]]]):
    """Joint path that shifts wrist 3 onto the winding with room for the flip.

    ``ik_solutions`` aligns with ``wrist3_seeds(current_q)``. A None entry is
    a failed IK. The unshifted seed is the continuous Cartesian solution.
    Returns ``(path, chosen_q6)``. ``path`` is empty when no joint move is needed.
    """
    current = np.asarray(current_q, dtype=np.float64).reshape(6)
    seeds = wrist3_seeds(current)
    if len(ik_solutions) != len(seeds):
        raise ValueError(
            f"Expected {len(seeds)} IK solutions, got {len(ik_solutions)}."
        )
    here = None
    candidates = []
    for seed, sol in zip(seeds, ik_solutions):
        if sol is None:
            continue
        q6 = float(np.asarray(sol, dtype=np.float64).reshape(6)[WRIST3_INDEX])
        candidates.append(q6)
        if abs(float(seed[WRIST3_INDEX]) - float(current[WRIST3_INDEX])) < 1e-6:
            here = q6
    if here is None or not candidates:
        raise RuntimeError(
            "Inverse kinematics failed for the episode start pose."
        )
    chosen = select_wrist3(float(current[WRIST3_INDEX]), candidates)
    target_q6 = float(current[WRIST3_INDEX] + (chosen - here))
    return wrist3_joint_path(current, target_q6), chosen


def wrist3_joint_path(current_q, target_q6: float, step_deg: float = MAX_ROT_STEP_DEG):
    """Other joints fixed. Empty when wrist 3 is already within one step."""
    current = np.asarray(current_q, dtype=np.float64).reshape(6).copy()
    delta = float(target_q6) - float(current[WRIST3_INDEX])
    step = np.deg2rad(step_deg)
    if abs(delta) <= step + 1e-9:
        return []
    n = int(np.ceil(abs(delta) / step - 1e-9))
    path = []
    for i in range(1, n + 1):
        q = current.copy()
        q[WRIST3_INDEX] = current[WRIST3_INDEX] + delta * i / n
        path.append(q)
    return path


def pose7_to_ur_rotvec(pose7) -> list:
    """pose7 ``[x,y,z,qw,qx,qy,qz]`` to a UR TCP pose ``[x,y,z,rx,ry,rz]``."""
    p = np.asarray(pose7, dtype=np.float64).reshape(7)
    rotvec = Rotation.from_quat([p[4], p[5], p[6], p[3]]).as_rotvec()
    return [float(v) for v in (*p[:3], *rotvec)]


def descent_alpha_at_surface(alphas, signed_distance) -> float:
    """Fraction from the hover to the episode TCP at which to stop.

    Signed distance is positive outside the mold. An episode TCP that is
    already outside is left alone (alpha 1). One that is inside is cut at the
    surface crossing so the descent is not commanded into the mesh.
    """
    a = np.asarray(alphas, dtype=np.float64).reshape(-1)
    sd = np.asarray(signed_distance, dtype=np.float64).reshape(-1)
    if a.shape != sd.shape or len(a) < 2:
        raise ValueError("Need matching alpha and signed-distance samples.")
    if float(sd[-1]) >= -1e-9:
        return 1.0
    inside = np.flatnonzero(sd < 0.0)
    i = int(inside[0])
    if i == 0:
        raise ValueError("The hover is already inside the mold.")
    s0 = float(sd[i - 1])
    s1 = float(sd[i])
    frac = s0 / (s0 - s1)
    return float(a[i - 1] + frac * (a[i] - a[i - 1]))


def position_parts(reached_xyz, target_xyz, normal):
    """Tangential distance and signed normal gap, both in metres.

    The normal gap is positive when the reached point is outward of the target.
    """
    err = np.asarray(reached_xyz, dtype=np.float64).reshape(3) - np.asarray(
        target_xyz, dtype=np.float64
    ).reshape(3)
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    n = n / np.linalg.norm(n)
    normal_m = float(np.dot(err, n))
    tangential = err - normal_m * n
    return float(np.linalg.norm(tangential)), normal_m


def allow_normal_gap(target_sd_m: float, reached_sd_m: float, normal_m: float) -> bool:
    """True when the episode TCP is inside the mesh and the arm is at the surface.

    The normal gap is then the distance from that buried TCP out to contact.
    A point still inside the mesh, or past the episode TCP, is not accepted.
    """
    # Fixed contact band. The reach tolerance is separate.
    band_m = 0.002
    return (
        float(target_sd_m) < -band_m
        and float(reached_sd_m) >= -band_m
        and float(normal_m) >= -band_m
    )


def position_accepted(
    tangential_m: float, normal_m: float, ang_deg: float, *, allow_normal: bool
) -> bool:
    if float(ang_deg) > ANG_TOL_DEG:
        return False
    if allow_normal:
        return float(tangential_m) <= POS_TOL_M
    gap = float(np.hypot(tangential_m, normal_m))
    return gap <= POS_TOL_M


def arrival_error(
    thread_alive: bool,
    tangential_m: float,
    normal_m: float,
    ang_deg: float,
    q6: float,
    *,
    allow_normal: bool = False,
) -> Optional[str]:
    """None when the arm may continue. Otherwise the abort sentence."""
    if not thread_alive:
        return (
            "Robot thread is not running. "
            "The approach tripped a safety stop. Aborting."
        )
    if not position_accepted(
        tangential_m, normal_m, ang_deg, allow_normal=allow_normal
    ):
        gap = float(np.hypot(tangential_m, normal_m))
        normal_note = (
            " Normal gap allowed: the episode TCP is inside the mold."
            if allow_normal
            else ""
        )
        return (
            f"TCP is {gap * 1e3:.1f} mm from the episode start "
            f"({tangential_m * 1e3:.1f} mm tangential, {normal_m * 1e3:+.1f} mm normal) "
            f"and {ang_deg:.1f} deg "
            f"(limit {POS_TOL_M * 1e3:.0f} mm / {ANG_TOL_DEG:.0f} deg). "
            f"Aborting.{normal_note}"
        )
    if not wrist3_has_room(q6):
        end = float(q6) + np.deg2rad(RETURN_FLIP_DEG)
        return (
            f"Wrist 3 is {q6:.3f} rad. +{RETURN_FLIP_DEG:.0f} deg would finish at "
            f"{end:.3f} rad, outside ±{np.degrees(WRIST3_LIMIT_RAD):.0f} deg "
            f"with {WRIST3_MARGIN_DEG:.0f} deg of room. Aborting."
        )
    return None


def _rtde_python(python: str, ip: str, body: str) -> str:
    script = (
        "import json\n"
        f"ip = {ip!r}\n"
        + body
    )
    proc = subprocess.run(
        [str(python), "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"RTDE call failed: {detail}")
    return proc.stdout


def query_wrist_ik(python: str, ip: str, rotvec, seeds) -> list:
    """One IK solution per seed. None where the solver has no solution."""
    payload = {
        "x": [float(v) for v in rotvec],
        "seeds": [[float(v) for v in seed] for seed in seeds],
    }
    body = f"""
payload = json.loads({json.dumps(json.dumps(payload))})
from rtde_control import RTDEControlInterface
ctrl = RTDEControlInterface(ip)
out = []
for seed in payload["seeds"]:
    try:
        sol = ctrl.getInverseKinematics(payload["x"], seed, 1e-4, 1e-3)
        out.append(list(sol))
    except Exception as exc:
        out.append(None)
        print("IK_FAIL", exc, file=__import__("sys").stderr)
ctrl.stopScript()
ctrl.disconnect()
print("UMIFT_IK " + json.dumps(out))
"""
    for line in _rtde_python(python, ip, body).splitlines():
        if line.startswith("UMIFT_IK "):
            return json.loads(line[len("UMIFT_IK ") :])
    raise RuntimeError("RTDE inverse kinematics returned no UMIFT_IK line.")


def movej_path(python: str, ip: str, path) -> None:
    """Blocking moveJ through ``path``. Speed is low; each step is ≤5 deg."""
    if not path:
        return
    joints = [[float(v) for v in q] for q in path]
    body = f"""
joints = json.loads({json.dumps(json.dumps(joints))})
from rtde_control import RTDEControlInterface
ctrl = RTDEControlInterface(ip)
for q in joints:
    if not ctrl.moveJ(q, 0.2, 0.5):
        ctrl.stopScript()
        ctrl.disconnect()
        raise SystemExit("moveJ failed")
ctrl.stopScript()
ctrl.disconnect()
print("ok")
"""
    _rtde_python(python, ip, body)


def zarr_for_policy_start(environ=None, fallback: Optional[Path] = None) -> Optional[Path]:
    """``UMIFT_R4VT_ZARR`` when it exists, otherwise ``fallback``."""
    environ = os.environ if environ is None else environ
    raw = str(environ.get("UMIFT_R4VT_ZARR", "")).strip()
    if raw and Path(raw).is_dir():
        return Path(raw)
    if fallback is not None and Path(fallback).is_dir():
        return Path(fallback)
    return None

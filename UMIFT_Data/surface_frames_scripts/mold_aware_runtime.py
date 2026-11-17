"""Live mold-aware obs + Fix B decode for umift_env_runner.

Reuses FixBMoldField / heading_branch / tracked_project from the
training/eval scripts. Training stringer assets default to
``stringer_scanner_v2_fixb/shared/fixb_field.npz``.
"""

from __future__ import annotations

import json
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
    a1 = r6[..., 0:3]
    a2 = r6[..., 3:6]
    b1 = a1 / np.clip(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-9, None)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.clip(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-9, None)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2)


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


# Soft clamps (applied / flagged) vs hard stops (abort-worthy).
SOFT_CLAMP_KEYS = (
    "max_tangential_speed",
    "max_step",
    "delta_n",
    "max_tilt",
    "max_rot_rate",
    "mesh_jump_hold",
)
HARD_STOP_KEYS = (
    "normal_flip",
    "max_distance_from_mesh",
    "tcp_penetration",
    "wrench_abort",
)


def fixb_decode_dt(
    n_steps: int, down_sample_steps: int, raw_time_step_s: float
) -> np.ndarray:
    """Demo-time step for Fix B integration. Slow-down stays on the waypoint clock."""
    step_s = float(down_sample_steps) * float(raw_time_step_s)
    return np.full(int(n_steps), step_s, dtype=np.float64)


def r4_safety_limits(
    max_tcp_penetration_m: float = 0.020,
    wrench_hold_n: float = 90.0,
    abort_on_measured_mesh_distance: bool = False,
    enable_delta_n_floor: bool = False,
) -> SafetyLimits:
    """r4vt limits. The δn floor stays off unless a bench replay asks for it.

    Abort on measured TCP depth. Hold, then abort, on force. The measured-mesh
    flag skips the commanded-depth abort and checks the TCP instead.
    """
    return SafetyLimits(
        enable_delta_n_floor=bool(enable_delta_n_floor),
        enable_tcp_penetration_abort=True,
        max_tcp_penetration_m=float(max_tcp_penetration_m),
        enable_wrench_hold=True,
        wrench_hold_n=float(wrench_hold_n),
        abort_on_measured_mesh_distance=bool(abort_on_measured_mesh_distance),
    )


# One replan, and one step inside a replan. A larger hop keeps the previous foot.
FOOT_CAP_M = 0.010
# Fixture +Y is down. Bench replay lifts along −Y, off the table on every face.
FIXTURE_UP = np.array([0.0, -1.0, 0.0])


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


def _project_foot(field, p, prev_x, prev_R):
    """First replan queries the TCP. Later calls hold the previous face."""
    p = np.asarray(p, dtype=np.float64).reshape(3)
    if prev_x is None or prev_R is None:
        cur = field.query(p.reshape(1, 3))
        return (
            np.asarray(cur["x_s"], dtype=np.float64).reshape(3).copy(),
            np.asarray(cur["R"], dtype=np.float64).reshape(3, 3).copy(),
        )
    x, R = field.track_step(
        p.reshape(1, 3),
        np.asarray(prev_x, dtype=np.float64).reshape(1, 3),
        np.asarray(prev_x, dtype=np.float64).reshape(1, 3),
        np.asarray(prev_R, dtype=np.float64).reshape(1, 3, 3),
    )
    return (
        np.asarray(x, dtype=np.float64).reshape(3).copy(),
        np.asarray(R, dtype=np.float64).reshape(3, 3).copy(),
    )


def integrate_tcp_foot(
    field,
    p0: np.ndarray,
    v_par: np.ndarray,
    delta_n: np.ndarray,
    dt: np.ndarray,
    prev_x: np.ndarray | None = None,
    prev_R: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    """Face-held foot of the measured TCP, then tangent steps on that face.

    The first replan queries the TCP. Later replans call ``track_step`` with
    the measured TCP and the previous foot. A hop over 10 mm, at the replan
    or at a step, keeps the previous foot and frame.
    """
    v_par = np.asarray(v_par, dtype=np.float64).reshape(-1, 2)
    delta_n = np.asarray(delta_n, dtype=np.float64).reshape(-1)
    dt = np.asarray(dt, dtype=np.float64).reshape(-1)
    h = len(delta_n)
    xs, rk = _project_foot(field, p0, prev_x, prev_R)
    xs, rk = _hold_if_far(xs, rk, prev_x, prev_R)
    x_s = np.zeros((h, 3))
    frames = np.zeros((h, 3, 3))
    for k in range(h):
        x_s[k] = xs
        frames[k] = rk
        if k + 1 < h:
            step = rk[:, :2] @ v_par[k] * float(dt[k])
            nxt_x, nxt_R = _project_foot(field, xs + step, xs, rk)
            xs, rk = _hold_if_far(nxt_x, nxt_R, xs, rk)
    normal = frames[:, :, 2]
    return {
        "x_s": x_s,
        "x_d": x_s + delta_n[:, None] * normal,
        "R": frames,
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

    actions = [np.asarray(a, dtype=np.float64) for a in actions]
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
    R = np.asarray(R_d, dtype=np.float64).reshape(-1, 3, 3).copy()
    hit = np.zeros(len(R), dtype=bool)
    if len(R) < 2 or max_deg <= 0:
        return R, hit
    for i in range(1, len(R)):
        rel = R[i - 1].T @ R[i]
        rv = Rotation.from_matrix(rel).as_rotvec()
        ang = float(np.degrees(np.linalg.norm(rv)))
        if ang > max_deg and ang > 1e-9:
            hit[i] = True
            R[i] = R[i - 1] @ Rotation.from_rotvec(rv * (max_deg / ang)).as_matrix()
    return R, hit


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
        return cls(
            field=FixBMoldField(Path(field_path)),
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
        self.last_triggers = []

    def release_command_foot(self) -> None:
        """Next decode queries the TCP. Offline windows do this; the live runner does not."""
        self._cmd_foot = None
        self._cmd_R = None
        self.last_clamp_speed = None
        self.last_clamp_step = None
        self.last_clamp_dn = None
        self.last_clamp_tilt = None
        self.last_clamp_rot_rate = None
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
        n = R[..., 2]
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

    def note_tcp_penetration(self, sd: np.ndarray) -> None:
        """Abort when the measured TCP, not the virtual target, is too deep."""
        if not self.limits.enable_tcp_penetration_abort:
            return
        depth = np.asarray(sd, dtype=np.float64).reshape(-1)
        if np.any(depth < -float(self.limits.max_tcp_penetration_m)):
            self.last_triggers.append("tcp_penetration")

    def note_measured_mesh_distance(self, p_meas: np.ndarray, x_s: np.ndarray) -> None:
        """Abort when the measured TCP is farther than max_distance_from_mesh_m.

        Off unless abort_on_measured_mesh_distance is set. On, decode_chunk
        does not abort on the commanded ||x_d - x_s||.
        """
        if not self.limits.abort_on_measured_mesh_distance:
            return
        dist = np.linalg.norm(
            np.asarray(p_meas, dtype=np.float64).reshape(-1, 3)
            - np.asarray(x_s, dtype=np.float64).reshape(-1, 3),
            axis=1,
        )
        if np.any(dist > float(self.limits.max_distance_from_mesh_m)):
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

    def decode_chunk(
        self,
        raw_action: np.ndarray,
        p0: np.ndarray,
        R0: np.ndarray,
        dt: np.ndarray,
        apply_safety: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Anchored Fix B decode for one chunk. Returns fixture-frame poses.

        apply_safety=False skips speed/depth clamps (offline parity vs eval).
        Live robot path keeps the default True.
        """
        self.last_triggers = []
        H = raw_action.shape[0]
        self.last_clamp_speed = np.zeros(H, dtype=bool)
        self.last_clamp_step = np.zeros(H, dtype=bool)
        self.last_clamp_dn = np.zeros(H, dtype=bool)
        self.last_clamp_tilt = np.zeros(H, dtype=bool)
        self.last_clamp_rot_rate = np.zeros(H, dtype=bool)
        self.last_mesh_jump_hold = np.zeros(H, dtype=bool)

        if self.run_type == "r1_world_world":
            pos = raw_action[:, 0:3].astype(np.float64)
            R = rot6d_rows_to_R(raw_action[:, 3:9])
            q = self.field.query(pos)
            if apply_safety and H > 1:
                steps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
                step_hit = steps > self.limits.max_step_m
                self.last_clamp_step[1:] = step_hit
                if np.any(step_hit):
                    self.last_triggers.append("max_step")
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
        # First replan queries the TCP. Later replans hold the previous face.
        dec = integrate_tcp_foot(
            self.field,
            p0,
            v_par,
            delta_n,
            dt,
            prev_x=self._cmd_foot,
            prev_R=self._cmd_R,
        )
        R_field = dec["R"]
        x_d = dec["x_d"]
        x_s = dec["x_s"]

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
                R_d, R_field[:, :, 2], self.limits.max_tilt_from_normal_deg
            )
            self.last_clamp_tilt = tilt_hit
            if np.any(tilt_hit):
                self.last_triggers.append("max_tilt")
        if apply_safety and self.limits.enable_rot_rate_clamp:
            R_d, rate_hit = _clamp_rot_rate(R_d, self.limits.max_rot_rate_deg_per_step)
            self.last_clamp_rot_rate = rate_hit
            if np.any(rate_hit):
                self.last_triggers.append("max_rot_rate")

        if apply_safety and H > 1:
            steps = np.linalg.norm(np.diff(x_d, axis=0), axis=1)
            step_hit = steps > self.limits.max_step_m
            self.last_clamp_step[1:] = step_hit
            if np.any(step_hit):
                self.last_triggers.append("max_step")
        if (
            apply_safety
            and not self.dry_run
            and not self.limits.abort_on_measured_mesh_distance
        ):
            dist = np.linalg.norm(x_d - x_s, axis=1)
            if np.any(dist > self.limits.max_distance_from_mesh_m):
                self.last_triggers.append("max_distance_from_mesh")

        self._cmd_foot = np.asarray(x_s[-1], dtype=np.float64).copy()
        self._cmd_R = np.asarray(R_field[-1], dtype=np.float64).reshape(3, 3).copy()

        return {
            "x_d": x_d,
            "R_d": R_d,
            "x_s": x_s,
            "R_field": R_field,
            "sd": np.sum((x_d - x_s) * R_field[:, :, 2], axis=1),
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
            "mesh_jump_hold": self.last_mesh_jump_hold,
        }

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
        n = np.asarray(decoded["R_field"][..., 2], dtype=np.float64)
        delta_n = np.asarray(decoded["delta_n"], dtype=np.float64).reshape(-1)
        target = Rp_to_SE3(r_d, x_s)
        vt = Rp_to_SE3(r_d, x_s + delta_n[:, None] * n)
        return target, vt

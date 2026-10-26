"""Live mold-aware obs + Fix B decode for umift_env_runner.

Reuses FixBMoldField / integrate_fixb / heading_branch / tracked_project from the
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

from fixb_field import FixBMoldField, integrate_fixb
from fixb_labels import heading_branch, pose7_to_Rp, so3_to_rot6d_rows
from fixb_labels_tracked import HOP_EXCESS_M, tracked_project

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
    dry_run_hover_m: float = 0.015
    max_force_n: float = 80.0
    # Live policy peaks ~0.19 m/s on val demos; 0.25 leaves headroom without aborting.
    max_tangential_speed_mps: float = 0.25
    max_step_m: float = 0.025
    max_distance_from_mesh_m: float = 0.030
    max_foot_hop_m: float = 0.020
    stop_on_normal_flip: bool = True
    max_tilt_from_normal_deg: float = 45.0
    max_rot_rate_deg_per_step: float = 20.0


# Soft clamps (applied / flagged) vs hard stops (abort-worthy).
SOFT_CLAMP_KEYS = (
    "max_tangential_speed",
    "max_step",
    "delta_n",
    "max_tilt",
    "max_rot_rate",
)
HARD_STOP_KEYS = ("normal_flip", "max_distance_from_mesh")


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
    last_triggers: list = field(default_factory=list)
    # Per-step bool masks from the latest decode_chunk (length H).
    last_clamp_speed: Optional[np.ndarray] = None
    last_clamp_step: Optional[np.ndarray] = None
    last_clamp_dn: Optional[np.ndarray] = None
    last_clamp_tilt: Optional[np.ndarray] = None
    last_clamp_rot_rate: Optional[np.ndarray] = None

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
        self.last_triggers = []
        self.last_clamp_speed = None
        self.last_clamp_step = None
        self.last_clamp_dn = None
        self.last_clamp_tilt = None
        self.last_clamp_rot_rate = None

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

        # Online tracked foot: batch project with hop repair, seeded by previous foot.
        x = tracked_project(self.field, p_fb)
        if self._prev_foot is not None and len(x) > 0:
            hop = np.linalg.norm(x[0] - self._prev_foot)
            if hop > HOP_EXCESS_M:
                x[0] = self.field.project((self._prev_foot + (p_fb[0] - self._prev_foot))[None])[0]
                for i in range(1, len(x)):
                    step = p_fb[i] - p_fb[i - 1]
                    cand = self.field.project((x[i - 1] + step)[None])[0]
                    if np.linalg.norm(x[i] - x[i - 1]) - np.linalg.norm(step) > HOP_EXCESS_M:
                        x[i] = cand
        self._prev_foot = x[-1].copy()

        q = self.field.query(x, with_curvature=True)
        # sd of the *tool* vs foot (same as training labels)
        n = q["R"][..., 2]
        sd = np.sum((p_fb - x) * n, axis=1)
        if self._prev_n is not None and self.limits.stop_on_normal_flip:
            if float(np.dot(self._prev_n, n[-1])) < 0:
                self.last_triggers.append("normal_flip")
        self._prev_n = n[-1].copy()

        signed = sd.reshape(-1, 1).astype(np.float32)
        normal = n.astype(np.float32)
        rot6d = so3_to_rot6d_rows(q["R"]).astype(np.float32)
        curv = q["curv"].astype(np.float32)

        obs_raw[f"mold_signed_distance_{robot_id}"] = signed
        obs_raw[f"mold_normal_{robot_id}"] = normal
        obs_raw[f"mold_R_field_rot6d_{robot_id}"] = rot6d
        obs_raw[f"mold_curvature_{robot_id}"] = curv
        obs_raw[f"mold_x_s_{robot_id}"] = x.astype(np.float32)
        return {
            "sd": sd,
            "normal": n,
            "x_s": x,
            "R": q["R"],
            "curv": q["curv"],
        }

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
        elif apply_safety:
            floor = -self.limits.max_press_depth_m
            clamp_dn = delta_n_raw < floor
            delta_n = np.maximum(delta_n_raw, floor)
        return v_par, delta_n, a[:, 18], a[:, 19:21], a, clamp_dn

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
        dec = integrate_fixb(
            self.field,
            np.asarray(p0, dtype=np.float64).reshape(1, 3),
            v_par[None],
            delta_n[None],
            dt[None],
        )
        R_field = dec["R"][0]
        x_d = dec["x_d"][0]
        x_s = dec["x_s"][0]

        if self.run_type == "r4_world_fixb_tilt":
            # Learned ΔR in action[..., 21:27]; project rot6d → SO(3).
            # Hover keeps the predicted ΔR (only δn is overridden in unpack).
            if raw_action.shape[-1] < 27:
                raise ValueError(
                    f"r4 decode needs action dim >= 27, got {raw_action.shape[-1]}"
                )
            dR = rot6d_rows_to_R(raw_action[:, 21:27])
            R_d = np.einsum("hij,hjk->hik", R_field, dR)
        else:
            B = heading_branch(
                R_field[0:1],
                np.asarray(R0, dtype=np.float64).reshape(1, 3, 3),
                self.R_off,
            )
            R_d = np.einsum("hij,jk->hik", R_field, B[0])

        if apply_safety:
            R_d, tilt_hit = _clamp_tilt_to_normal(
                R_d, R_field[:, :, 2], self.limits.max_tilt_from_normal_deg
            )
            self.last_clamp_tilt = tilt_hit
            if np.any(tilt_hit):
                self.last_triggers.append("max_tilt")
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
        if apply_safety and not self.dry_run:
            dist = np.linalg.norm(x_d - x_s, axis=1)
            if np.any(dist > self.limits.max_distance_from_mesh_m):
                self.last_triggers.append("max_distance_from_mesh")

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
        }

    def decoded_to_action_mats(
        self, decoded: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Nominal target = x_d; VT slightly deeper along -n for compliance direction."""
        x_d, R_d = decoded["x_d"], decoded["R_d"]
        n = decoded["R_field"][..., 2]
        target = Rp_to_SE3(R_d, x_d)
        vt = Rp_to_SE3(R_d, x_d - 0.002 * n)
        return target, vt

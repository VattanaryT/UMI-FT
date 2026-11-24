"""Cartesian admittance plus an SDF contact spring. No mesh collision geom."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import yaml
from fixb_field import FixBMoldField

_DEFAULT_HARDWARE_CONFIG = Path(
    "/home/boeing/umi_ft_ws/hardware_interfaces/workcell/table_top_manip/config/ur5e_wrist_ft.yaml"
)

# Environment stand-in for the wrist FT. Not a controller gain.
CONTACT_N_PER_M = 3000.0
# C++ admittance has no tangential contact damper. The 5 N·s/m term is gone.
TANGENT_DAMPING_NS_PER_M = 0.0
CONTACT_SPLIT_N = 2.0


def hardware_config_path(config_path: Path | None = None) -> Path:
    if config_path is None:
        folder = os.environ.get("PYRITE_HARDWARE_CONFIG_FOLDERS", "").strip()
        config_path = (
            Path(folder) / "ur5e_wrist_ft.yaml" if folder else _DEFAULT_HARDWARE_CONFIG
        )
    return Path(config_path)


def controller_gains(config_path: Path | None = None) -> dict:
    """Translational admittance numbers from admittance_controller0.

    Stiffness is not here. The policy overwrites it at run time. Rotational
    inertia and damping stay in the yaml; this plant commands orientation.
    """
    config_path = hardware_config_path(config_path)
    block = yaml.safe_load(config_path.read_text())["admittance_controller0"]
    compliance = block["compliance6d"]
    return {
        "mass_kg": float(compliance["inertia"][0]),
        "damping_ns_per_m": float(compliance["damping"][0]),
        "dt_s": float(block["dt"]),
        "spring_force_limit_n": float(block["max_spring_force_magnitude"]),
        "config": config_path,
    }


_GAINS = controller_gains()
MASS_KG = _GAINS["mass_kg"]
DAMPING_NS_PER_M = _GAINS["damping_ns_per_m"]
INTEGRATE_DT_S = _GAINS["dt_s"]


def spring_force_limit_n(config_path: Path | None = None) -> tuple[float, Path]:
    """Return ``max_spring_force_magnitude`` and the yaml it came from.

    The admittance controller scales the whole translational spring wrench so
    its norm stays under this value. It does not clip each axis separately.
    """
    gains = controller_gains(config_path)
    return gains["spring_force_limit_n"], gains["config"]


def contact_masks(force_world: np.ndarray, threshold_n: float = CONTACT_SPLIT_N):
    force = np.linalg.norm(np.asarray(force_world, dtype=np.float64).reshape(-1, 3), axis=1)
    return force > threshold_n, force <= threshold_n, force


class CartesianSpringPlant:
    """Tool translation under K (x_cmd - x) and F_n = 3000 * max(-sd, 0) * n_out.

    Orientation is the commanded rotation. Contact is FixBMoldField.query with
    no R_tool, so the signed distance is not the convex hull. Positive sd is
    outside the mold. The contact force is along the outward normal.
    """

    def __init__(self, field: FixBMoldField, position: np.ndarray, rotation: np.ndarray):
        self.field = field
        self.x = np.asarray(position, dtype=np.float64).reshape(3).copy()
        self.R = np.asarray(rotation, dtype=np.float64).reshape(3, 3).copy()
        self.v = np.zeros(3, dtype=np.float64)
        self.spring_force_limit_n, self.spring_force_config = spring_force_limit_n()

    def step(self, x_cmd: np.ndarray, rotation: np.ndarray, stiffness_6: np.ndarray, dt: float) -> dict:
        self.R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        command = np.asarray(x_cmd, dtype=np.float64).reshape(3)
        gain = np.asarray(stiffness_6, dtype=np.float64)[:3, :3]
        steps = max(1, int(np.ceil(float(dt) / INTEGRATE_DT_S)))
        sub_dt = float(dt) / steps
        for _ in range(steps):
            state = self._forces(command, gain)
            self.v = self.v + state["accel"] * sub_dt
            self.x = self.x + self.v * sub_dt
        landed = self._forces(command, gain)
        return {key: value for key, value in landed.items() if key != "accel"}

    def _forces(self, command: np.ndarray, gain: np.ndarray) -> dict:
        query = self.field.query(self.x.reshape(1, 3))
        normal = np.asarray(query["R"][0, :, 2], dtype=np.float64)
        normal = normal / np.linalg.norm(normal)
        sd = float(query["sd"][0])
        penetration = max(-sd, 0.0)
        tangent_v = self.v - normal * float(np.dot(normal, self.v))
        contact = normal * (CONTACT_N_PER_M * penetration) - TANGENT_DAMPING_NS_PER_M * tangent_v
        error_tool = self.R.T @ (command - self.x)
        spring_tool = gain @ error_tool
        limit = self.spring_force_limit_n
        magnitude = float(np.linalg.norm(spring_tool))
        if limit > 0.0 and magnitude > limit:
            spring_tool = spring_tool * (limit / magnitude)
        spring = self.R @ spring_tool
        accel = (spring + contact - DAMPING_NS_PER_M * self.v) / MASS_KG
        return {
            "sd_m": sd,
            "penetration_m": penetration,
            "normal": normal,
            "contact_n": float(np.dot(contact, normal)),
            "contact_world": contact.copy(),
            "spring_n": float(np.dot(spring, normal)),
            "speed_mps": float(np.linalg.norm(self.v)),
            "accel": accel,
        }


def _check() -> None:
    gains = controller_gains()
    assert gains["mass_kg"] == 0.2
    assert gains["damping_ns_per_m"] == 15.0
    assert gains["dt_s"] == 0.002
    assert gains["spring_force_limit_n"] == 40.0
    assert TANGENT_DAMPING_NS_PER_M == 0.0
    on, off, force = contact_masks([[0.0, 0.0, 1.0], [0.0, 3.0, 0.0]])
    assert force.tolist() == [1.0, 3.0]
    assert on.tolist() == [False, True]
    assert off.tolist() == [True, False]


if __name__ == "__main__":
    _check()

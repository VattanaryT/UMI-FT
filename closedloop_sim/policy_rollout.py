#!/usr/bin/env python3
"""Closed-loop policy rollouts on the clipped SDF plant. No robot.

Each policy is loaded with load_policy and queried through
ModelPredictiveController. r4vt chunks are action-blended, then
decode_chunk / decoded_to_action_mats / compliance_command. Pose policies
(fk60, r1rel) go through action_postprocess, the first-pose gate, and the
pose ensembler. The plant is CartesianSpringPlant.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "UMIFT_Data" / "surface_frames_scripts"
# ~/.bashrc pins PYTHONPATH at the main checkout. The package dirs have to
# come first or load_policy and the runner import the other tree.
for _entry in (
    ROOT,
    ROOT / "PyriteUtility",
    ROOT / "PyriteML" / "multimodal_representation" / "multimodal",
    ROOT / "PyriteML",
    SCRIPTS,
):
    sys.path.insert(0, str(_entry))

# Drop the main checkout after the insert. ~/.bashrc puts it on PYTHONPATH, and
# a later import would still find it if this list only prepends.
_MAIN = Path("/home/boeing/umi_ft_ws/UMI-FT")
_kept_path = []
for _entry in sys.path:
    try:
        _resolved = Path(_entry).resolve()
    except OSError:
        _kept_path.append(_entry)
        continue
    if _resolved == _MAIN or _MAIN in _resolved.parents:
        continue
    _kept_path.append(_entry)
sys.path[:] = _kept_path

# Runner defaults. A shell export of these flags must not change this run.
for _flag in (
    "UMIFT_LEAD_IN_S",
    "UMIFT_FOOT_WALK",
    "UMIFT_ALIGN_OBS_TO_IMAGE",
    "UMIFT_ENABLE_DELTA_N_FLOOR",
    "UMIFT_BENCH_LIFT_M",
    "UMIFT_ABORT_ON_MEASURED_MESH_DISTANCE",
):
    os.environ.pop(_flag, None)
os.environ["UMIFT_STIFFNESS_FLOOR"] = "3000"
os.environ.setdefault(
    "PYRITE_CHECKPOINT_FOLDERS", "/mnt/data/umidata/ft/training_outputs"
)
os.environ.setdefault(
    "PYRITE_HARDWARE_CONFIG_FOLDERS",
    "/home/boeing/umi_ft_ws/hardware_interfaces/workcell/table_top_manip/config",
)
os.environ.setdefault("PYRITE_CONTROL_LOG_FOLDERS", "/home/boeing/data/control_log")

from closedloop_sim.api import (  # noqa: E402
    ModelPredictiveController,
    compliance_command,
    load_policy,
)
from closedloop_sim.plant import (  # noqa: E402
    CONTACT_SPLIT_N,
    INTEGRATE_DT_S,
    CartesianSpringPlant,
    contact_masks,
)
from closedloop_sim.rollout_view import publish  # noqa: E402
from closedloop_sim.segments import events, label_segments, run_span  # noqa: E402
from r4vt_horizon import decode_horizon  # noqa: E402
from fixb_field import FixBMoldField  # noqa: E402
from fixb_labels import Rp_to_pose7, pose7_to_Rp  # noqa: E402
from mold_aware_runtime import (  # noqa: E402
    FixBActionEnsembler,
    MoldAwareRuntime,
    camera_table_from_flange,
    fixb_decode_dt,
    infer_run_type,
    r4_safety_limits,
)
from openloop_lib import REGION_NAMES, region_code  # noqa: E402
from PyriteConfig.tasks.umift.umift_type_conversions import (  # noqa: E402
    raw_to_obs_inference,
)
from PyriteConfig.tasks.umift.workcell_frames import (  # noqa: E402
    load_workcell_extrinsics,
)
from PyriteEnvSuites.env_runners.stiffness_floor import (  # noqa: E402
    read_press_stiffness_floor,
)
from PyriteEnvSuites.env_runners.umift_env_runner import (  # noqa: E402
    _stiffness_6x6n_to_per_step,
    first_pose_ok,
)
from PyriteUtility.planning_control.policy_clock import (  # noqa: E402
    scale_obs_windows_by_slow_down,
)
from PyriteUtility.planning_control.temporal_ensemble import (  # noqa: E402
    DEFAULT_EXECUTION_HORIZON,
    blend_pose_chunk,
    make_pose_ensembler,
    resolve_execution_horizon,
)
from scipy.spatial.transform import Rotation, Slerp  # noqa: E402

FIELD = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz"
)
ZARR = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "r4vt_causal_foot/acp_replay_buffer.zarr"
)
OUT = Path(__file__).resolve().parent / "out"

_TRAIN = "/mnt/data/umidata/ft/training_outputs"
CHECKPOINTS = {
    "fk60": f"{_TRAIN}/fk3000-c/epoch=0060-val_loss=0.071.ckpt",
    "e40": (
        f"{_TRAIN}/oct_v35_h4s_r4vt_causal_foot/"
        "oct-v35-h4s-r4vt-causal-foot_e200_bs64_valtopk_43390127/checkpoints/"
        "epoch=0040-val_loss=0.063.ckpt"
    ),
    "r1rel": f"{_TRAIN}/oct_v35_h4s_odom_r1rel/checkpoints/latest.ckpt",
    "r1rel_e195": f"{_TRAIN}/oct_v35_h4s_odom_r1rel/checkpoints/latest.ckpt",
    "field_v2_e25": (
        f"{_TRAIN}/oct_v35_h4s_r4vt_field_v2_k3000/checkpoints/epoch=0025-val_loss=0.048.ckpt"
    ),
    "field_v2_e140": f"{_TRAIN}/oct_v35_h4s_r4vt_field_v2_k3000/checkpoints/latest_1005.ckpt",
    "k3000_e199": (
        f"{_TRAIN}/oct-v35-h4s-r4vt-causal-foot-carry-k3000_e200_bs64_valtopk_43472139/"
        "checkpoints/e199_final.ckpt"
    ),
}
# k3000 e199 was scored with the thin tangent flip t1, t2 = -t1, -t2.
SIGN_FLIP = {"k3000_e199": np.diag([-1.0, -1.0, 1.0])}

RAW_DT_S = 1.0 / 15.0
SLOW_DOWN = 2.0
EXEC_H_CFG = DEFAULT_EXECUTION_HORIZON
DECAY_M = 0.01
STIFFNESS_FLOOR = 3000.0
POSE_HZ = 500.0
SPEED_RUNAWAY_MPS = 20.0
PENETRATION_ABORT_M = 0.15
ROBOT_DS_CORR = 500.0 / 200.0
WRENCH_DS_CORR = 500.0 / 28.0
DEPTH_DS_CORR = 1


def require_worktree() -> None:
    """Raise if this process imported the main checkout instead of this worktree."""
    import mold_aware_runtime as runtime
    import PyriteConfig
    import PyriteEnvSuites.env_runners.umift_env_runner as runner

    for module in (runtime, PyriteConfig, runner):
        path = Path(module.__file__).resolve()
        if ROOT not in path.parents:
            raise RuntimeError(f"{module.__name__} imported from {path}, expected {ROOT}")
    print(f"[sim] mold_aware_runtime {runtime.__file__}", flush=True)
    print(f"[sim] umift_env_runner {runner.__file__}", flush=True)


def bypass_d405_table(runtime) -> None:
    """Sim has no D405 housing. Hardware still holds on camera_table.

    The instance method is replaced only in this process. The runner never
    calls this, so a housing corner that meets the flange plane still holds.
    """

    def _keep(R_cmd, x_cmd, _R_meas, _p_meas, _held_vt):
        return (
            np.asarray(R_cmd, dtype=np.float64).reshape(-1, 3, 3),
            np.asarray(x_cmd, dtype=np.float64).reshape(-1, 3),
        )

    runtime._hold_if_camera_hits_table = _keep


class Waypoint:
    def __init__(self, position, rotation, stiffness, k_npm, delta_n_m):
        self.position = np.asarray(position, dtype=np.float64).reshape(3)
        self.rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        self.stiffness = np.asarray(stiffness, dtype=np.float64).reshape(6, 6)
        self.k_npm = float(k_npm)
        self.delta_n_m = float(delta_n_m)


class Schedule:
    def __init__(self, t0, t1, waypoints, step):
        self.t0 = float(t0)
        self.t1 = float(t1)
        self.waypoints = waypoints
        self.step = float(step)
        self._seg = None
        self._slerp = None

    def sample(self, t):
        wps = self.waypoints
        if len(wps) == 1 or self.step <= 0.0:
            return wps[0]
        frac = (float(t) - self.t0) / self.step
        if frac >= len(wps) - 1:
            return wps[-1]
        i = min(int(np.floor(frac)), len(wps) - 2)
        alpha = float(np.clip(frac - i, 0.0, 1.0))
        left, right = wps[i], wps[i + 1]
        if self._seg != i:
            self._slerp = Slerp(
                [0.0, 1.0],
                Rotation.from_matrix(np.stack([left.rotation, right.rotation])),
            )
            self._seg = i
        rotation = self._slerp([alpha]).as_matrix()[0]
        position = (1.0 - alpha) * left.position + alpha * right.position
        return Waypoint(position, rotation, left.stiffness, left.k_npm, left.delta_n_m)


def _need(sparse, key) -> int:
    if key not in sparse:
        return 0
    horizon = int(sparse[key]["horizon"])
    down = int(sparse[key]["down_sample_steps"])
    return (horizon - 1) * down + 1


def _apply_runner_windows(shape_meta) -> None:
    """Same down_sample_steps edits as umift_env_runner.main."""
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(shape_meta):
            OmegaConf.set_struct(shape_meta, False)
            OmegaConf.set_struct(shape_meta.sample.obs.sparse, False)
    except Exception:
        pass
    sparse = shape_meta["sample"]["obs"]["sparse"]
    scale_obs_windows_by_slow_down(sparse, SLOW_DOWN)
    if "depth_0" in sparse:
        sparse["depth_0"]["down_sample_steps"] = int(
            sparse["depth_0"]["down_sample_steps"]
        ) // DEPTH_DS_CORR
    for key in (
        "robot0_eef_pos",
        "robot0_eef_rot_axis_angle",
        "robot0_abs_eef_pos",
        "robot0_abs_eef_rot_axis_angle",
        "robot0_mold_geom",
        "robot0_mold_curv",
    ):
        if key in sparse:
            ds = int(sparse[key]["down_sample_steps"])
            sparse[key]["down_sample_steps"] = max(1, int(round(ds * ROBOT_DS_CORR)))
    for key in ("robot0_eef_wrench_left", "robot0_eef_wrench_right"):
        if key in sparse:
            ds = int(sparse[key]["down_sample_steps"])
            sparse[key]["down_sample_steps"] = max(1, int(round(ds * WRENCH_DS_CORR)))


def _press_gain(stiffness, measured_R, nominal_p, virtual_p) -> float:
    delta = measured_R.T @ (virtual_p - nominal_p)
    norm = float(np.linalg.norm(delta))
    axis = np.array([1.0, 0.0, 0.0]) if norm < 0.001 else delta / norm
    gain = np.asarray(stiffness, dtype=np.float64)[:3, :3]
    return float(axis @ gain @ axis)


def _image_zeros(attr, count: int, depth: bool) -> np.ndarray:
    channels, height, width = [int(v) for v in attr["shape"]]
    shape = (count, height, width, channels)
    if depth:
        return np.zeros(shape, dtype=np.float16)
    return np.zeros(shape, dtype=np.uint8)


def _unstable(row, fast_s, dt):
    if not np.isfinite(row["speed_mps"]) or not np.isfinite(row["penetration_m"]):
        return "non_finite", fast_s
    if row["penetration_m"] > PENETRATION_ABORT_M:
        return "penetration", fast_s
    if row["speed_mps"] > SPEED_RUNAWAY_MPS:
        return "speed_runaway", fast_s
    if row["t_s"] > 0.5 and row["speed_mps"] > 1.0:
        fast_s += float(dt)
        if fast_s > 0.15:
            return "speed_runaway", fast_s
    else:
        fast_s = 0.0
    return None, fast_s


def _stats(values) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _med_p90(values) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
    }


def _window(trace, mask) -> dict:
    if not np.any(mask):
        return {"n": 0}
    part = {key: np.asarray(value)[mask] for key, value in trace.items()}
    contact = part["penetration_m"] > 0.0
    commanded_inside = part["delta_n_m"] < -0.001
    lost = commanded_inside & (part["penetration_m"] <= 0.0)
    pressing = contact & (part["delta_n_m"] < 0.0)
    out = {
        "n": int(mask.sum()),
        "duration_s": float(part["t_s"][-1] - part["t_s"][0]),
        "penetration_mm": _stats(part["penetration_m"] * 1e3),
        "contact_loss_pct": float(100.0 * np.mean(lost)),
        "contact_force_n": _stats(part["contact_n"][contact]) if np.any(contact) else None,
        "k_delta_n_n": _stats(part["k_delta_n_n"][pressing]) if np.any(pressing) else None,
        "force_over_minus_k_dn": None,
        "speed_mps": _stats(part["speed_mps"]),
        "n_speed_over_1": int(np.sum(part["speed_mps"] > 1.0)),
    }
    if np.any(pressing):
        force = part["contact_n"][pressing]
        scale = -part["k_delta_n_n"][pressing]
        ok = scale > 1.0
        if np.any(ok):
            out["force_over_minus_k_dn"] = _stats(force[ok] / scale[ok])
    return out


def _segments(trace, demo) -> dict:
    labels = label_segments(trace["force_n"], trace["clearance_m"])
    out = {}
    for name in ("contact", "approach", "lift-off", "air-travel"):
        mask = labels == name
        if not np.any(mask):
            out[name] = {"n": 0}
            continue
        part = {key: np.asarray(value)[mask] for key, value in trace.items()}
        block = _window(trace, mask)
        block["path_vs_demo"] = _path_vs_demo(part, demo)
        block["clearance_mm"] = _stats(part["clearance_m"] * 1e3)
        out[name] = block
    out["events"] = events(
        demo["t_s"],
        demo["segment"],
        demo["clearance_m"],
        demo["position"],
        trace["t_s"],
        trace["clearance_m"],
        trace["position"],
    )
    return out


def _by_force(trace, demo, force_world) -> dict:
    on, off, _force = contact_masks(force_world)

    def one(mask):
        if not np.any(mask):
            return {"n": 0}
        part = {key: np.asarray(value)[mask] for key, value in trace.items()}
        block = _window(trace, mask)
        block["path_vs_demo"] = _path_vs_demo(part, demo)
        return block

    return {
        "threshold_n": CONTACT_SPLIT_N,
        "contact": one(on),
        "non_contact": one(off),
    }


def _tilt_deg(rotation, normal) -> np.ndarray:
    z_axis = np.asarray(rotation, dtype=np.float64)[:, :, 2]
    target = -np.asarray(normal, dtype=np.float64)
    target = target / np.clip(np.linalg.norm(target, axis=1, keepdims=True), 1e-12, None)
    cosine = np.clip(np.sum(z_axis * target, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def _yaw_deg(rotation_a, rotation_b, normal) -> np.ndarray:
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.clip(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12, None)

    def _project(rotation):
        axis = np.asarray(rotation, dtype=np.float64)[:, :, 0]
        axis = axis - normal * np.sum(axis * normal, axis=1, keepdims=True)
        length = np.linalg.norm(axis, axis=1, keepdims=True)
        return axis / np.clip(length, 1e-12, None), length[:, 0]

    a_axis, a_len = _project(rotation_a)
    b_axis, b_len = _project(rotation_b)
    sine = np.sum(normal * np.cross(b_axis, a_axis), axis=1)
    cosine = np.sum(a_axis * b_axis, axis=1)
    angle = np.abs(np.degrees(np.arctan2(sine, cosine)))
    angle[(a_len < 1e-8) | (b_len < 1e-8)] = np.nan
    return angle


def _path_vs_demo(trace, demo) -> dict:
    t = trace["t_s"]
    demo_t = demo["t_s"]
    index = np.searchsorted(demo_t, t, side="left")
    index = np.clip(index, 0, len(demo_t) - 1)
    prev = np.clip(index - 1, 0, len(demo_t) - 1)
    use_prev = np.abs(demo_t[prev] - t) < np.abs(demo_t[index] - t)
    index = np.where(use_prev, prev, index)
    position = np.linalg.norm(trace["position"] - demo["position"][index], axis=1) * 1e3
    tilt = np.abs(_tilt_deg(trace["rotation"], trace["normal"]) - demo["tilt"][index])
    yaw = _yaw_deg(trace["rotation"], demo["rotation"][index], trace["normal"])
    region = region_code(trace["normal"])
    out = {}
    for mask_name, mask in (
        ("all", np.ones(len(t), dtype=bool)),
        ("after_start_pad", t > 5.0),
    ):
        rows = {}
        groups = [("all", mask)] + [
            (name, mask & (region == code)) for code, name in enumerate(REGION_NAMES)
        ]
        for name, group in groups:
            rows[name] = {
                "position_mm": _med_p90(position[group]),
                "tilt_deg": _med_p90(tilt[group]),
                "yaw_deg": _med_p90(yaw[group]),
            }
        out[mask_name] = rows
    return out


def _demo_episode(episode: int) -> dict:
    import zarr

    group = zarr.open_group(str(ZARR), mode="r")["data"][f"episode_{episode}"]
    pose = np.asarray(group["ts_pose_fb_0"], dtype=np.float64)
    times = np.asarray(group["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
    rotation, position = pose7_to_Rp(pose)
    field = FixBMoldField(FIELD)
    query = field.query(position)
    normal = np.asarray(query["R"][:, :, 2], dtype=np.float64)
    normal = normal / np.clip(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12, None)
    clearance = np.maximum(np.asarray(query["sd"], dtype=np.float64).reshape(-1), 0.0)
    wrench_t = np.asarray(group["wrench_time_stamps_0"], dtype=np.float64).reshape(-1)
    wrench = np.asarray(group["wrench_left_0"], dtype=np.float64)
    nearest = np.clip(np.searchsorted(wrench_t, times), 0, len(wrench) - 1)
    prev = np.clip(nearest - 1, 0, len(wrench) - 1)
    use_prev = np.abs(wrench_t[prev] - times) < np.abs(wrench_t[nearest] - times)
    force = np.linalg.norm(wrench[np.where(use_prev, prev, nearest), :3], axis=1)
    dt = np.diff(times, prepend=times[0])
    dt[0] = np.median(dt[1:]) if dt.size > 1 else 1.0
    step = np.linalg.norm(np.diff(position, axis=0, prepend=position[:1]), axis=1)
    segment = label_segments(force, clearance)
    return {
        "t_s": times - times[0],
        "position": position,
        "rotation": rotation,
        "normal": normal,
        "tilt": _tilt_deg(rotation, normal),
        "duration_s": float(times[-1] - times[0]),
        "pose0": pose[0].copy(),
        "pose7": pose,
        "force_n": force,
        "clearance_m": clearance,
        "speed_mps": step / np.maximum(dt, 1e-6),
        "segment": segment,
    }


def _obs_raw(shape_meta, pose, wrench, times, count, now_s):
    sparse = shape_meta["sample"]["obs"]["sparse"]
    obs_raw = {
        "ts_pose_fb_0": pose[:count],
        "robot_time_stamps_0": times[:count],
        "wrench_left_0": wrench[:count],
        "wrench_right_0": np.zeros_like(wrench[:count]),
        "wrench_time_stamps_0": times[:count],
    }
    raw_meta = shape_meta["raw"]
    for key, attr in raw_meta.items():
        if str(attr.get("type", "low_dim")) != "rgb":
            continue
        if key in obs_raw:
            continue
        depth = "depth" in key
        obs_raw[key] = _image_zeros(shape_meta["obs"][key], max(_need(sparse, key), 1), depth)
    if "gripper_0" in raw_meta or "robot0_gripper" in shape_meta["obs"]:
        obs_raw["gripper_0"] = np.zeros((count, 1), dtype=np.float64)
        obs_raw["gripper_time_stamps_0"] = times[:count]
    n_rgb = max(_need(sparse, "rgb_0"), 1)
    obs_raw["rgb_time_stamps_0"] = now_s - np.arange(n_rgb)[::-1] / 15.0
    return obs_raw


def _waypoints_from_mats(
    nominal_pose, scheduled_pose, stiffness_per_step, measured_R, delta_n
):
    rotation, position = pose7_to_Rp(np.asarray(scheduled_pose, dtype=np.float64))
    _nominal_R, nominal_p = pose7_to_Rp(np.asarray(nominal_pose, dtype=np.float64))
    stiffness_per_step = np.asarray(stiffness_per_step, dtype=np.float64)
    delta_n = None if delta_n is None else np.asarray(delta_n, dtype=np.float64).reshape(-1)
    waypoints = []
    for i in range(position.shape[0]):
        gain = _press_gain(stiffness_per_step[i], measured_R, nominal_p[i], position[i])
        signed = float(delta_n[i]) if delta_n is not None else float(
            np.dot(position[i] - nominal_p[i], measured_R[:, 2])
        )
        # Pose policies: δn is the virtual-target offset along the mold normal,
        # filled in by the caller when it has that normal. The fallback above
        # is replaced below when delta_n is None and a normal was stored on the
        # waypoint by the caller. This branch only runs for an explicit array.
        waypoints.append(
            Waypoint(position[i], rotation[i], stiffness_per_step[i], gain, signed)
        )
    return waypoints


def _pose_waypoints(nominal_pose, scheduled_pose, stiffness_per_step, measured_R, normal):
    rotation, position = pose7_to_Rp(np.asarray(scheduled_pose, dtype=np.float64))
    _nominal_R, nominal_p = pose7_to_Rp(np.asarray(nominal_pose, dtype=np.float64))
    stiffness_per_step = np.asarray(stiffness_per_step, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64).reshape(3)
    waypoints = []
    for i in range(position.shape[0]):
        gain = _press_gain(stiffness_per_step[i], measured_R, nominal_p[i], position[i])
        signed = float(np.dot(position[i] - nominal_p[i], normal))
        waypoints.append(
            Waypoint(position[i], rotation[i], stiffness_per_step[i], gain, signed)
        )
    return waypoints


def _checkpoint_task_name(path: str) -> str:
    """cfg.task.name. load_policy does not return it, and infer_run_type needs it."""
    import dill
    import torch

    payload = torch.load(open(path, "rb"), map_location="cpu", pickle_module=dill)
    task_name = str(payload["cfg"].task.name)
    del payload
    return task_name


def rollout(name: str, demo: dict, episode: int = 40, mode: str = "full") -> dict:
    import torch

    torch.manual_seed(0)
    np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = CHECKPOINTS[name]
    task_name = _checkpoint_task_name(ckpt)
    print(f"[{name}] loading {ckpt} on {device} task={task_name}", flush=True)
    policy, shape_meta = load_policy(ckpt, device)
    run_type = infer_run_type(task_name, shape_meta)
    r4vt = run_type == "r4_world_fixb_tilt"
    action_dim = int(shape_meta["action"]["shape"][0])
    if name in ("fk60", "e40") and not r4vt:
        raise RuntimeError(
            f"{name} inferred as {run_type} (action {action_dim}), expected r4vt"
        )
    _apply_runner_windows(shape_meta)
    sparse = shape_meta["sample"]["obs"]["sparse"]
    action_ds = int(shape_meta["sample"]["action"]["sparse"]["down_sample_steps"])
    action_h = int(shape_meta["sample"]["action"]["sparse"]["horizon"])
    exec_h = resolve_execution_horizon(EXEC_H_CFG, action_h, False)
    step_wall = action_ds * RAW_DT_S * SLOW_DOWN
    period = max(exec_h - 1, 1) * step_wall
    floor = read_press_stiffness_floor()
    if floor != STIFFNESS_FLOOR:
        raise RuntimeError(f"stiffness floor is {floor}, expected {STIFFNESS_FLOOR}")
    print(
        f"[{name}] run_type={run_type} action={action_dim} "
        f"exec_h={exec_h} action_h={action_h} step_wall={step_wall:.4f}s "
        f"pose_frames={dict(shape_meta.get('pose_frames', {}))}",
        flush=True,
    )

    robot_need = 1
    for key in sparse:
        if key.startswith("robot0_"):
            robot_need = max(robot_need, _need(sparse, key))
    cap = robot_need + int(demo["duration_s"] / INTEGRATE_DT_S) + 8
    pose = np.zeros((cap, 7), dtype=np.float64)
    wrench = np.zeros((cap, 6), dtype=np.float64)
    times = np.zeros(cap, dtype=np.float64)
    t_start, t_end, start = run_span(demo, mode)
    start_R, start_p = pose7_to_Rp(start.reshape(1, 7))
    plant = CartesianSpringPlant(FixBMoldField(FIELD), start_p[0], start_R[0])
    snap = plant._forces(plant.x.copy(), np.zeros((3, 3), dtype=np.float64))
    contact = np.asarray(snap["contact_world"], dtype=np.float64)
    hold_wrench = np.zeros(6, dtype=np.float64)
    hold_wrench[:3] = plant.R.T @ contact
    pre_dt = 1.0 / POSE_HZ
    for i in range(robot_need):
        times[i] = t_start - (robot_need - 1 - i) * pre_dt
        pose[i] = start
        wrench[i] = hold_wrench
    count = robot_need

    runtime = None
    if run_type != "none":
        limits = (
            r4_safety_limits(
                max_tcp_penetration_m=0.020,
                wrench_hold_n=90.0,
                abort_on_measured_mesh_distance=False,
                enable_delta_n_floor=False,
                max_press_depth_m=0.005,
            )
            if r4vt
            else None
        )
        runtime = MoldAwareRuntime.from_assets(
            run_type=run_type,
            field_path=FIELD,
            dry_run=False,
            limits=limits,
        )
        extrinsics = load_workcell_extrinsics()
        table_n, table_off, _table_y = camera_table_from_flange(
            FIELD, extrinsics.T_fixture_from_right_base
        )
        runtime.set_table_plane(table_n, table_off)
        bypass_d405_table(runtime)
    controller = ModelPredictiveController(
        shape_meta=shape_meta,
        id_list=[0],
        policy=policy,
        action_to_trajectory=lambda _action: None,
        execution_horizon=action_ds * exec_h,
        fix_orientation=False,
    )
    pose_ensemble = None if r4vt else make_pose_ensembler(DECAY_M)
    action_ensemble = FixBActionEnsembler(DECAY_M) if r4vt else None
    jump_guard = None
    if r4vt:
        from jump_guard import TargetJumpGuard

        jump_guard = TargetJumpGuard()

    rows = []
    stop = None
    fast_s = 0.0
    horizon_count = 0
    n_holds = 0
    n_gate = 0
    next_replan = 0.0
    t = t_start
    schedule = None
    mold_normal = np.asarray(snap["normal"], dtype=np.float64)

    def _append(state, k_npm, delta_n_m, virtual_target):
        pose7 = Rp_to_pose7(plant.R.reshape(1, 3, 3), plant.x.reshape(1, 3))[0]
        rows.append(
            {
                "t_s": float(t),
                "penetration_m": float(state["penetration_m"]),
                "contact_n": float(state["contact_n"]),
                "spring_n": float(state["spring_n"]),
                "speed_mps": float(state["speed_mps"]),
                "delta_n_m": float(delta_n_m),
                "k_npm": float(k_npm),
                "k_delta_n_n": float(k_npm) * float(delta_n_m),
                "position": plant.x.copy(),
                "normal": np.asarray(state["normal"], dtype=np.float64).copy(),
                "rotation": plant.R.copy(),
                "pose7": np.asarray(pose7, dtype=np.float64).copy(),
                "virtual_target": np.asarray(virtual_target, dtype=np.float64).reshape(3).copy(),
                "force": np.asarray(state["contact_world"], dtype=np.float64).reshape(3).copy(),
                "sd_m": float(state["sd_m"]),
                "clearance_m": float(max(state["sd_m"], 0.0)),
            }
        )

    _append(snap, 0.0, 0.0, plant.x)

    def _record_plant(state):
        nonlocal count
        if count >= cap:
            raise RuntimeError("observation buffer filled before the episode ended")
        pose7 = Rp_to_pose7(plant.R.reshape(1, 3, 3), plant.x.reshape(1, 3))[0]
        force = plant.R.T @ np.asarray(state["contact_world"], dtype=np.float64)
        times[count] = t
        pose[count] = pose7
        wrench[count, :3] = force
        wrench[count, 3:] = 0.0
        count += 1

    last_command = Waypoint(start_p[0], start_R[0], np.zeros((6, 6)), 0.0, 0.0)

    def _hold_waypoint():
        return last_command

    def _plan():
        nonlocal n_holds, n_gate, horizon_count, mold_normal
        obs_raw = _obs_raw(shape_meta, pose, wrench, times, count, t)
        if runtime is not None:
            runtime.fill_obs_raw(obs_raw, robot_id=0)
            hard = list(runtime.last_hard_triggers)
            if hard:
                return None, "safety:" + ",".join(hard)
            if r4vt:
                force_n = float(np.linalg.norm(obs_raw["wrench_left_0"][-1, :3]))
                decision = runtime.wrench_hold_update(force_n)
                if decision == "abort":
                    return None, "wrench_abort"
                if decision == "hold":
                    n_holds += 1
                    return Schedule(t, t + period, [_hold_waypoint()], step_wall), None
        episode_data = {}
        raw_to_obs_inference(obs_raw, episode_data, shape_meta)
        obs = episode_data["obs"]
        if "depth_0" in obs:
            obs["depth_0"] = np.zeros_like(obs["depth_0"])
        if "elapsed_time_0" in sparse and "elapsed_time_0" not in obs:
            horizon = int(sparse["elapsed_time_0"]["horizon"])
            obs["elapsed_time_0"] = np.full((horizon, 1), (t - t_start) / SLOW_DOWN, np.float32)
        missing = [
            key
            for key in sparse
            if key not in obs and not str(key).endswith("time_stamps_0")
        ]
        if missing:
            raise RuntimeError(f"observation is missing {missing}")
        result = controller.compute_one_horizon_action(obs, device)
        measured = pose[count - 1]
        measured_R = pose7_to_Rp(measured.reshape(1, 7))[0][0]
        query = plant.field.query(plant.x.reshape(1, 3))
        mold_normal = np.asarray(query["R"][0, :, 2], dtype=np.float64)
        mold_normal = mold_normal / np.linalg.norm(mold_normal)

        if r4vt:
            raw = np.asarray(controller.last_raw_action, dtype=np.float64)
            action_ensemble.push(horizon_count * exec_h, raw)
            raw = action_ensemble.pop(exec_h)
            dt = fixb_decode_dt(raw.shape[0], action_ds, RAW_DT_S)
            p0_R, p0 = pose7_to_Rp(measured.reshape(1, 7))
            decoded = runtime.decode_chunk(raw, p0[0], p0_R[0], dt)
            wide = raw.reshape(-1, raw.shape[-1])
            if wide.shape[1] == 27:
                action10 = np.zeros((wide.shape[0], 10), dtype=np.float64)
                action10[:, 0:3] = wide[:, 0:3]
                action10[:, 3] = wide[:, 18]
                action10[:, 4:10] = wide[:, 21:27]
            elif wide.shape[1] == 10:
                action10 = wide
            else:
                raise RuntimeError(f"{name} r4vt action {wide.shape}")
            frame = np.asarray(query["R"][0], dtype=np.float64)
            flip = SIGN_FLIP.get(name)
            if flip is not None:
                frame = frame @ flip
            frames = np.repeat(frame.reshape(1, 3, 3), action10.shape[0], axis=0)
            xyz, tool_R = decode_horizon(action10[:, :10], p0[0], frames, dt)
            decoded["x_d"] = xyz
            decoded["R_d"] = tool_R
            hard = list(runtime.last_hard_triggers)
            if hard:
                return None, "safety:" + ",".join(hard)
            guarded_x, guarded_R, guarded_field, aborted = jump_guard.apply(
                decoded["x_d"],
                decoded["R_d"],
                R_field=decoded["R_field"],
            )
            n_keep = int(guarded_x.shape[0])
            if n_keep == 0:
                return None, "jump_abort"
            full = int(np.asarray(decoded["x_d"]).shape[0])
            for key, value in list(decoded.items()):
                array = np.asarray(value)
                if array.shape[:1] == (full,):
                    decoded[key] = array[:n_keep]
            decoded["x_d"] = guarded_x
            decoded["R_d"] = guarded_R
            decoded["R_field"] = guarded_field
            runtime.note_commanded_rd(np.asarray(guarded_R)[n_keep - 1])
            nominal, virtual = runtime.decoded_to_action_mats(decoded)
            nominal_pose, scheduled, packed, _raw_k, _floored = compliance_command(
                nominal,
                virtual,
                np.asarray(decoded["stiffness"]).reshape(-1),
                measured,
                stiffness_floor=floor,
                test_nominal_target=False,
                test_nominal_target_stiffness=-1.0,
            )
            per_step = _stiffness_6x6n_to_per_step(packed)
            waypoints = _waypoints_from_mats(
                nominal_pose,
                scheduled,
                per_step,
                measured_R,
                np.asarray(decoded["delta_n"]).reshape(-1),
            )
            horizon_count += 1
            schedule_now = Schedule(t, t + period, waypoints, step_wall)
            if aborted:
                return schedule_now, "jump_abort"
            return schedule_now, None

        target, virtual, stiffness, _eoat = result
        ok, dist_m, ang_deg = first_pose_ok(target[0], measured)
        if not ok:
            n_gate += 1
            n_holds += 1
            print(
                f"[{name}] t={t:.2f}s first-pose gate hold "
                f"{dist_m * 1e3:.1f} mm {ang_deg:.1f} deg",
                flush=True,
            )
            return Schedule(t, t + period, [_hold_waypoint()], step_wall), None
        nominal_pose, scheduled, packed, _raw_k, _floored = compliance_command(
            target[0],
            virtual[0],
            None if stiffness is None else stiffness[0],
            measured,
            stiffness_floor=floor,
            test_nominal_target=False,
            test_nominal_target_stiffness=-1.0,
        )
        per_step = _stiffness_6x6n_to_per_step(packed)
        if exec_h > per_step.shape[0]:
            raise RuntimeError(
                f"sparse_execution_horizon={exec_h} exceeds chunk {per_step.shape[0]}"
            )
        blended = blend_pose_chunk(
            pose_ensemble,
            horizon_count * exec_h,
            {
                "pose7": scheduled,
                "nominal_pose7": nominal_pose,
                "stiffness": per_step,
                "eoat": None,
            },
            exec_h,
        )
        waypoints = _pose_waypoints(
            blended["nominal_pose7"],
            blended["pose7"],
            blended["stiffness"],
            measured_R,
            mold_normal,
        )
        horizon_count += 1
        return Schedule(t, t + period, waypoints, step_wall), None

    last_print = -5.0
    pending_stop = None
    while t < t_end - 1e-9:
        if t + 1e-9 >= next_replan:
            if pending_stop is not None:
                stop = pending_stop
                break
            schedule, pending_stop = _plan()
            next_replan = t + period
            if schedule is None:
                stop = pending_stop
                break
        command = schedule.sample(t)
        last_command = command
        state = plant.step(command.position, command.rotation, command.stiffness, INTEGRATE_DT_S)
        t += INTEGRATE_DT_S
        state["t_s"] = t
        _record_plant(state)
        _append(state, command.k_npm, command.delta_n_m, command.position)
        reason, fast_s = _unstable(rows[-1], fast_s, INTEGRATE_DT_S)
        if reason is not None:
            stop = reason
            break
        if t - last_print >= 5.0:
            last_print = t
            print(
                f"[{name}] t={t:.1f}s pen={rows[-1]['penetration_m'] * 1e3:.1f} mm "
                f"speed={rows[-1]['speed_mps']:.3f} k={rows[-1]['k_npm']:.0f} "
                f"dn={rows[-1]['delta_n_m'] * 1e3:.1f} mm",
                flush=True,
            )

    scalar = [
        "t_s",
        "penetration_m",
        "contact_n",
        "spring_n",
        "speed_mps",
        "delta_n_m",
        "k_npm",
        "k_delta_n_n",
    ]
    trace = {key: np.asarray([row[key] for row in rows]) for key in scalar}
    trace["position"] = np.asarray([row["position"] for row in rows])
    trace["normal"] = np.asarray([row["normal"] for row in rows])
    trace["rotation"] = np.asarray([row["rotation"] for row in rows])
    trace["pose7"] = np.asarray([row["pose7"] for row in rows])
    commanded_inside = trace["delta_n_m"] < -0.001
    contact = trace["penetration_m"] > 0.0
    pressing = contact & (trace["delta_n_m"] < 0.0)
    lost = commanded_inside & ~contact
    report = {
        "policy": name,
        "checkpoint": ckpt,
        "episode": int(episode),
        "mold": str(FIELD),
        "run_type": run_type,
        "task_name_for_infer": task_name,
        "action_dim": action_dim,
        "device": device,
        "seed": 0,
        "stopped": stop,
        "diverged": stop in ("non_finite", "penetration", "speed_runaway"),
        "aborted": bool(stop) and stop not in ("non_finite", "penetration", "speed_runaway"),
        "n_steps": int(trace["t_s"].size),
        "duration_s": float(trace["t_s"][-1] - trace["t_s"][0]) if trace["t_s"].size else 0.0,
        "episode_duration_s": float(demo["duration_s"]),
        "n_replans": int(horizon_count),
        "n_holds": int(n_holds),
        "n_first_pose_holds": int(n_gate),
        "execution_horizon": int(exec_h),
        "action_down_sample_steps": int(action_ds),
        "waypoint_s": float(step_wall),
        "replan_period_s": float(period),
        "stiffness_floor_n_per_m": float(floor),
        "penetration_mm": _stats(trace["penetration_m"] * 1e3),
        "contact_loss_pct": float(100.0 * np.mean(lost)) if lost.size else 0.0,
        "contact_force_n": _stats(trace["contact_n"][contact]) if np.any(contact) else None,
        "k_delta_n_n": _stats(trace["k_delta_n_n"][pressing]) if np.any(pressing) else None,
        "force_over_minus_k_dn": _window(trace, np.ones(len(trace["t_s"]), dtype=bool))[
            "force_over_minus_k_dn"
        ],
        "speed_mps": _stats(trace["speed_mps"]),
        "n_speed_over_1": int(np.sum(trace["speed_mps"] > 1.0)),
        "after_start_pad": _window(trace, trace["t_s"] > 5.0),
        "mode": mode,
        "t_start_s": float(t_start),
        "t_end_s": float(t_end),
        "path_vs_demo": _path_vs_demo(trace, demo),
        "by_force": _by_force(trace, demo, np.asarray([row["force"] for row in rows])),
        "plant": {
            "spring_force_limit_n": float(plant.spring_force_limit_n),
            "spring_force_config": str(plant.spring_force_config),
            "rgb": "zeros",
            "depth": "zeros",
            "wrench": "plant contact, tool frame",
            "start_pose": "episode ts_pose_fb_0[0]",
            "camera_table": "bypassed",
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"policy_{name}_{mode}_fixb_ep{episode}"
    trace["virtual_target"] = np.asarray([row["virtual_target"] for row in rows])
    trace["force"] = np.asarray([row["force"] for row in rows])
    trace["force_n"] = np.linalg.norm(trace["force"], axis=1)
    trace["sd_m"] = np.asarray([row["sd_m"] for row in rows])
    trace["clearance_m"] = np.asarray([row["clearance_m"] for row in rows])
    report["segment_force"] = {
        "demo": "wrist wrench",
        "rollout": "plant contact",
        "contact_above_n": 5.0,
        "air_below_n": 2.0,
        "air_clearance_mm": 10.0,
    }
    report["segments"] = _segments(trace, demo)
    report["demo_segments"] = {
        name: int(np.sum(demo["segment"] == name)) for name in ("contact", "approach", "lift-off", "air-travel")
    }
    np.savez_compressed(
        OUT / f"{tag}.npz",
        **{
            key: trace[key]
            for key in list(scalar) + ["position", "normal", "pose7", "virtual_target", "force"]
        },
    )
    view = publish(
        OUT,
        tag,
        plant.field,
        trace["position"],
        demo["position"],
        trace["normal"],
        trace["force"],
    )
    report["view"] = view
    (OUT / f"{tag}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"[{name}] stopped={stop} duration={report['duration_s']:.2f}s", flush=True)
    del policy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def main() -> None:
    require_worktree()
    names = [arg for arg in sys.argv[1:] if not arg.startswith("-")] or ["fk60", "e40", "r1rel"]
    demo = _demo_episode(40)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "demo_ep40.npz",
        t_s=demo["t_s"],
        position=demo["position"],
        pose7=demo["pose7"],
        segment=np.asarray(demo["segment"]),
    )
    print(f"episode 40 length {demo['duration_s']:.3f}s", flush=True)
    print(
        "demo segments "
        + " ".join(
            f"{name}={int(np.sum(demo['segment'] == name))}"
            for name in ("contact", "approach", "lift-off", "air-travel")
        ),
        flush=True,
    )
    reports = []
    for name in names:
        reports.append(rollout(name, demo, mode="full"))
        reports.append(rollout(name, demo, mode="contact"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "policy_agreed_ep40.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()

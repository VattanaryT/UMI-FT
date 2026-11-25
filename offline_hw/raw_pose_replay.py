#!/usr/bin/env python3
"""Command the recorded TCP through the runner's schedule and clamp functions.

No robot. The pose is ts_pose_fb converted fixture to right_base. Labels, the
foot, and ΔR are not used. slow_down_factor is 2, the same value main() uses.
"""

from __future__ import annotations

import builtins
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "UMIFT_Data" / "surface_frames_scripts"
for entry in (ROOT, ROOT / "PyriteUtility", SCRIPTS, ROOT / "offline_hw"):
    sys.path.insert(0, str(entry))
main_tree = Path("/home/boeing/umi_ft_ws/UMI-FT")
sys.path[:] = [
    entry
    for entry in sys.path
    if main_tree not in Path(entry).resolve().parents and Path(entry).resolve() != main_tree
]

import os

os.environ.setdefault("PYRITE_CHECKPOINT_FOLDERS", "/mnt/data/umidata/ft/training_outputs")
os.environ.setdefault(
    "PYRITE_HARDWARE_CONFIG_FOLDERS",
    "/home/boeing/umi_ft_ws/hardware_interfaces/workcell/table_top_manip/config",
)
os.environ.setdefault("PYRITE_CONTROL_LOG_FOLDERS", str(ROOT / "offline_hw" / "out"))
os.environ["UMIFT_WORKCELL_EXTRINSICS"] = str(
    ROOT / "PyriteConfig/tasks/umift/layup_workcell_extrinsics_20261116_orange.yaml"
)

from fixb_labels import Rp_to_pose7, pose7_to_Rp
from mold_aware_runtime import (
    RD_SLEW_AFTER_RAMP_DEG,
    VT_RAMP_S,
    _slew_orientations,
    compliance_command,
)
from offline_hw.mock_env import MockLayupEnv, save_schedules
from PyriteConfig.tasks.umift.workcell_frames import load_workcell_extrinsics
from PyriteEnvSuites.env_runners.dataset_start_pose import pose7_change_frame
from PyriteEnvSuites.env_runners.umift_env_runner import (
    APPROACH_HOVER_M,
    approach_start_pose,
)
from PyriteUtility.spatial_math import spatial_utilities as su
import zarr

ZARR = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "r4vt_causal_foot/acp_replay_buffer.zarr"
)
FIELD = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz"
)
HW = (
    "/home/boeing/umi_ft_ws/hardware_interfaces/workcell/table_top_manip/config/ur5e_wrist_ft.yaml"
)
OUT = ROOT / "offline_hw" / "out" / "raw_ep9"
START = 1220
DT_DEMO = 1.0 / 15.0
SLOW = 2.0
MAX_STEP_M = 0.025
SPRING_CAP_N = 40.0
PROPOSED_CAP_N = 25.0
PROPOSED_K = 3000.0
FORCE_ON_N = 5.0


def _rot_deg(a, b) -> float:
    rel = np.asarray(a).T @ np.asarray(b)
    return float(np.degrees(np.linalg.norm(__import__("scipy").spatial.transform.Rotation.from_matrix(rel).as_rotvec())))


def main() -> None:
    from fixb_smooth_field import SmoothFrameField

    group = zarr.open_group(str(ZARR), mode="r")["data"]["episode_9"]
    pose_fix = np.asarray(group["ts_pose_fb_0"], dtype=np.float64)
    pose_t = np.asarray(group["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
    wrench = np.asarray(group["wrench_left_0"], dtype=np.float64)
    wrench_t = np.asarray(group["wrench_time_stamps_0"], dtype=np.float64).reshape(-1)
    T = load_workcell_extrinsics().T_fixture_from_right_base
    pose_base = pose7_change_frame(pose_fix, np.linalg.inv(T))
    field = SmoothFrameField(FIELD, FIELD.with_name("fixb_smooth_frames.npz"))

    times = np.arange(pose_t[START], pose_t[-1] + 1e-9, DT_DEMO)
    idx = np.clip(np.searchsorted(pose_t, times), 0, len(pose_t) - 1)
    demo = pose_base[idx]
    demo_fix = pose_fix[idx]
    rotation, position = pose7_to_Rp(demo)
    elapsed = 0.0
    caps = np.empty(len(demo))
    for i in range(len(demo)):
        caps[i] = 3.0 if elapsed < VT_RAMP_S else RD_SLEW_AFTER_RAMP_DEG
        elapsed += DT_DEMO
    rotation_cmd, slew_hit = _slew_orientations(rotation, None, caps)
    # Same shorten-along-the-step rule as MoldAwareRuntime._limit_commanded_vt.
    # The first second may use only that fraction of the 25 mm cap.
    position_cmd = np.empty_like(position)
    step_hit = np.zeros(len(position), dtype=bool)
    prev = position[0].copy()
    position_cmd[0] = prev
    elapsed_vt = 0.0
    for i in range(len(position)):
        elapsed_vt += DT_DEMO
        allow = MAX_STEP_M * (1.0 if i == 0 else min(1.0, elapsed_vt / VT_RAMP_S))
        if i == 0:
            continue
        delta = position[i] - prev
        dist = float(np.linalg.norm(delta))
        if dist > allow and dist > 1e-12:
            prev = prev + delta * (allow / dist)
            step_hit[i] = True
        else:
            prev = position[i].copy()
        position_cmd[i] = prev
    commanded = Rp_to_pose7(rotation_cmd, position_cmd)

    step_mm = np.linalg.norm(np.diff(position_cmd, axis=0), axis=1) * 1e3
    step_deg = np.array([_rot_deg(rotation_cmd[i], rotation_cmd[i + 1]) for i in range(len(demo) - 1)])
    err_mm = np.linalg.norm(position_cmd - position, axis=1) * 1e3
    err_deg = np.array([_rot_deg(rotation_cmd[i], rotation[i]) for i in range(len(demo))])

    lifted = demo_fix[0].copy()
    lifted[1] -= 0.12
    MockLayupEnv.initial_pose = pose7_change_frame(lifted, np.linalg.inv(T))
    MockLayupEnv.follow = "command"
    MockLayupEnv.project_pose = None
    MockLayupEnv.schedules = []
    MockLayupEnv.episode_folder = str(OUT / "robot_dump")
    env = MockLayupEnv(query_sizes={"sparse": {}})
    builtins.input = lambda prompt="": ""
    real_sleep = time.sleep
    time.sleep = lambda dt: env.advance(float(dt)) if dt else None
    approached = approach_start_pose(
        env,
        pose_base[START],
        field,
        T,
        HW,
        max_rot_step_deg=5.0,
        stop_at_contact=True,
    )
    time.sleep = real_sleep
    approach_n = len(MockLayupEnv.schedules)

    t = env.current_hardware_time_s
    press_k = []
    for i, pose in enumerate(commanded):
        se3 = su.pose7_to_SE3(pose.reshape(1, 7))
        nominal, virtual, stiff, raw_k, floored_k = compliance_command(
            se3, se3, None, pose
        )
        stamps = np.array([t, t + DT_DEMO * SLOW])
        env.schedule_controls(
            pose7_cmd=virtual.T,
            timestamps=stamps * 1000.0,
            stiffness_matrices_6x6=stiff,
        )
        press_k.append(float(floored_k[0]))
        t = float(stamps[-1])

    widx = np.clip(np.searchsorted(wrench_t, pose_t[idx]), 0, len(wrench_t) - 1)
    force = np.linalg.norm(wrench[widx, :3], axis=1)
    queried = field.query(demo_fix[:, :3])
    outward = -np.asarray(queried["R"], dtype=np.float64)[:, :, 2]
    depth = np.where(force > FORCE_ON_N, -np.minimum(force, PROPOSED_CAP_N) / PROPOSED_K, 0.0)
    proposed_vt = position + depth[:, None] * outward
    proposed_n = np.minimum(PROPOSED_K * np.abs(depth), PROPOSED_CAP_N)

    approach_pose = np.concatenate(
        [item["pose"] for item in MockLayupEnv.schedules[:approach_n]], axis=0
    )
    approach_fix = pose7_change_frame(approach_pose, T)
    approach_sd = np.asarray(field.query(approach_fix[:, :3])["sd"], dtype=np.float64).reshape(-1)
    free = ~(slew_hit | step_hit)
    OUT.mkdir(parents=True, exist_ok=True)
    save_schedules(OUT / "schedules.npz")
    np.savez_compressed(
        OUT / "raw_pose.npz",
        demo_base=demo,
        commanded_base=commanded,
        proposed_vt_base=proposed_vt,
        err_mm=err_mm,
        err_deg=err_deg,
        step_mm=step_mm,
        step_deg=step_deg,
        slew_hit=slew_hit,
        step_hit=step_hit,
        press_k=np.asarray(press_k),
        demo_force_n=force,
        proposed_force_n=proposed_n,
        times_demo=times,
    )
    report = {
        "start_index": START,
        "n_steps": int(len(demo)),
        "dt_demo_s": DT_DEMO,
        "slow_down": SLOW,
        "wall_step_s": DT_DEMO * SLOW,
        "approach_returned": bool(approached),
        "approach_schedules": int(approach_n),
        "approach_clearance_mm": {
            "min": float(np.min(approach_sd) * 1e3),
            "median": float(np.median(approach_sd) * 1e3),
        },
        "err_mm_where_no_clamp": {
            "n": int(np.sum(free)),
            "max": float(np.max(err_mm[free])) if np.any(free) else None,
        },
        "err_deg_where_no_clamp": {
            "n": int(np.sum(free)),
            "max": float(np.max(err_deg[free])) if np.any(free) else None,
        },
        "err_mm": {
            "median": float(np.median(err_mm)),
            "max": float(np.max(err_mm)),
        },
        "err_deg": {
            "median": float(np.median(err_deg)),
            "max": float(np.max(err_deg)),
        },
        "step_mm_max": float(np.max(step_mm)) if len(step_mm) else 0.0,
        "step_deg_p95": float(np.percentile(step_deg, 95)) if len(step_deg) else 0.0,
        "step_deg_max": float(np.max(step_deg)) if len(step_deg) else 0.0,
        "slew_hits": int(np.sum(slew_hit)),
        "step_hits": int(np.sum(step_hit)),
        "slew_cap_deg": f"3 for {VT_RAMP_S:.0f}s then {RD_SLEW_AFTER_RAMP_DEG:.0f}",
        "hover_m": APPROACH_HOVER_M,
        "baseline_stiffness": (
            "nominal = virtual = demo TCP, so the press axis is undefined. "
            "compliance_command then uses tool x at 5000 N/m. The two poses coincide, so the spring force is 0."
        ),
        "press_axis_k_median": float(np.median(press_k)),
        "proposed": {
            "rule": "when demo |F| > 5 N, δn = -min(|F|, 25) / 3000 along the outward normal",
            "steps_pressing": int(np.sum(force > FORCE_ON_N)),
            "max_force_n": float(np.max(proposed_n)),
            "max_depth_mm": float(np.max(np.abs(depth)) * 1e3),
        },
    }
    (OUT / "raw_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

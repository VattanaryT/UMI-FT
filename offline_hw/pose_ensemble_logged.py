#!/usr/bin/env python3
"""Compare pose-ensemble and newest-chunk on liftos_hw_fix logged windows.

Re-queries the policy once per logged observation (seed 0), then builds the
command each flag would schedule. Does not move the robot and does not write
the rollout log.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
from pathlib import Path

import numpy as np

os.environ.setdefault("UMIFT_MAX_STEP_MM", "4")
os.environ.setdefault("UMIFT_SCHEDULE_LIMITED_VT", "1")
os.environ.setdefault("UMIFT_FOOT_FACE_HYST", "1")
os.environ.setdefault(
    "UMIFT_WORKCELL_EXTRINSICS",
    "/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames/"
    "PyriteConfig/tasks/umift/layup_workcell_extrinsics_20261116_orange.yaml",
)

ROLLOUT = Path("/mnt/data/umidata/ft/vtevy/rollouts/2026-11-24_liftos_hw_fix")
CKPT = Path(
    "/mnt/data/umidata/ft/training_outputs/"
    "oct_v35_h4s_r4vt_field_v2n_v3r_liftos/latest.ckpt"
)
DECAY = 0.005


def _log_root(path: Path) -> Path:
    if (path / "horizon_0").exists():
        return path
    nested = sorted(path.glob("*/horizon_0"))
    return nested[0].parent if nested else path


def _load_robot(folder: str):
    path = Path(folder) / "robot_data_0.json"
    text = re.sub(r",(\s*[}\]])", r"\1", path.read_text())
    rows = json.loads(text)
    time_s = np.array([row["robot_time_stamps"] for row in rows], dtype=np.float64) / 1000.0
    pose = np.array([row["ts_pose_fb"] for row in rows], dtype=np.float64)
    return time_s, pose


def _nearest(times: np.ndarray, stamp: float) -> int:
    index = int(np.clip(np.searchsorted(times, stamp), 1, len(times) - 1))
    if abs(times[index - 1] - stamp) < abs(times[index] - stamp):
        index -= 1
    return index


def _to_base(rotation, position, T_fixture_from_base):
    world_from_fixture = np.linalg.inv(np.asarray(T_fixture_from_base, dtype=np.float64))
    rotation_base = np.einsum("ij,njk->nik", world_from_fixture[:3, :3], rotation)
    position_base = (world_from_fixture[:3, :3] @ position.T).T + world_from_fixture[:3, 3]
    return rotation_base, position_base


def _sign_flips(values: np.ndarray, tol: float = 1e-4) -> int:
    previous = 0.0
    flips = 0
    for value in np.asarray(values, dtype=np.float64).reshape(-1):
        if abs(value) < tol:
            continue
        sign = 1.0 if value > 0.0 else -1.0
        if previous != 0.0 and sign != previous:
            flips += 1
        previous = sign
    return flips


def _ori_steps(rotation: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    steps = []
    for index in range(len(rotation) - 1):
        relative = rotation[index].T @ rotation[index + 1]
        steps.append(np.degrees(np.linalg.norm(Rotation.from_matrix(relative).as_rotvec())))
    return np.asarray(steps, dtype=np.float64)


def _tangential(position: np.ndarray, field: np.ndarray, dt: float) -> np.ndarray:
    delta = position[1:] - position[:-1]
    return np.einsum("nji,nj->ni", field[:-1], delta) / float(dt)


def _load_policy():
    import torch

    script = (
        Path(__file__).resolve().parents[1]
        / "PyriteEnvSuites/scripts/run_policy_open_loop_val_export.py"
    )
    spec = importlib.util.spec_from_file_location("ol", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, _ = module.load_policy_and_cfg(CKPT, device)
    return policy, device


def _predict(policy, device, obs: dict):
    import torch

    batched = {
        "sparse": {
            key: torch.as_tensor(np.asarray(value), dtype=torch.float32, device=device).unsqueeze(0)
            for key, value in obs.items()
        }
    }
    torch.manual_seed(0)
    with torch.no_grad():
        return policy.predict_action(batched)["sparse"][0].detach().cpu().numpy()


def _scheduled(runtime, decoded):
    """Nominal and virtual-target pose7, in the same frame as ``decoded``."""
    from PyriteUtility.spatial_math import spatial_utilities as su

    nominal, virtual = runtime.decoded_to_action_mats(decoded)
    return su.SE3_to_pose7(nominal), su.SE3_to_pose7(virtual)


def _new_runtime():
    from mold_aware_runtime import MoldAwareRuntime, r4_safety_limits

    runtime = MoldAwareRuntime.from_assets(
        "r4_world_fixb_tilt",
        dry_run=False,
        limits=r4_safety_limits(),
    )
    runtime.tcp_relative_rotation = True
    runtime.reset()
    return runtime


def _decode(runtime, action, position, rotation, dt):
    return runtime.decode_chunk(
        np.asarray(action, dtype=np.float64),
        np.asarray(position, dtype=np.float64).reshape(3),
        np.asarray(rotation, dtype=np.float64).reshape(3, 3),
        np.full(len(action), float(dt)),
        apply_safety=True,
    )


def main() -> None:
    import zarr
    from PyriteConfig.tasks.umift.workcell_frames import load_workcell_extrinsics
    from fixb_labels import pose7_to_Rp
    from pose_ensemble import (
        apply_blended_carry,
        pose7_base_to_fixture,
        restore_carry,
        snapshot_carry,
    )
    from PyriteUtility.planning_control.temporal_ensemble import (
        blend_pose_chunk,
        make_pose_ensembler,
    )

    group = zarr.open_group(str(_log_root(ROLLOUT)), mode="r")
    names = sorted(
        (key for key in group.keys() if str(key).startswith("horizon_")),
        key=lambda name: int(str(name).split("_")[1]),
    )
    exec_h = int(group.attrs.get("sparse_execution_horizon", 6))
    dt = float(group.attrs.get("decode_dt_s", 1.0 / 15.0))
    robot_t, robot_pose = _load_robot(str(group.attrs["episode_data_folder"]))
    transform = load_workcell_extrinsics().T_fixture_from_right_base
    policy, device = _load_policy()
    print(
        f"windows {len(names)} exec_h {exec_h} dt {dt:.4f}s device {device} decay {DECAY}",
        flush=True,
    )

    windows = []
    for name in names:
        obs = {
            key: np.asarray(group[name]["obs_policy"][key])
            for key in group[name]["obs_policy"].keys()
        }
        stamp = float(np.asarray(group[name]["action"]["timestamps_s"]).reshape(-1)[0])
        pose = robot_pose[_nearest(robot_t, stamp)]
        action = _predict(policy, device, obs)
        windows.append((pose, action))
        print(f"predicted {name} action {action.shape}", flush=True)

    newest_R = []
    newest_p = []
    newest_field = []
    newest_v = []
    pose_R = []
    pose_p = []
    pose_field = []

    newest = _new_runtime()
    posed = _new_runtime()
    ensembler = make_pose_ensembler(DECAY)
    for index, (pose_base, action) in enumerate(windows):
        rotation_base, position_base = pose7_to_Rp(np.asarray(pose_base, dtype=np.float64).reshape(1, 7))
        position_fix = position_base @ transform[:3, :3].T + transform[:3, 3]
        rotation_fix = np.einsum("ij,njk->nik", transform[:3, :3], rotation_base)

        short = _decode(newest, action[:exec_h], position_fix[0], rotation_fix[0], dt)
        nominal, virtual = _scheduled(newest, short)
        rotation, position = pose7_to_Rp(virtual)
        newest_R.append(rotation)
        newest_p.append(position)
        newest_field.append(np.asarray(short["R_field"], dtype=np.float64))
        newest_v.append(np.asarray(short["v_par"], dtype=np.float64)[:, :2])

        snap = snapshot_carry(posed)
        full = _decode(posed, action, position_fix[0], rotation_fix[0], dt)
        restore_carry(posed, snap)
        _decode(posed, action[:exec_h], position_fix[0], rotation_fix[0], dt)
        field = np.asarray(full["R_field"], dtype=np.float64)
        rotation_base_cmd, position_base_cmd = _to_base(full["R_d"], full["x_d"], transform)
        field_base, foot_base = _to_base(full["R_field"], full["x_s"], transform)
        moved = dict(full)
        moved["R_d"] = rotation_base_cmd
        moved["x_d"] = position_base_cmd
        moved["R_field"] = field_base
        moved["x_s"] = foot_base
        nominal_b, virtual_b = _scheduled(posed, moved)
        blended = blend_pose_chunk(
            ensembler,
            index * exec_h,
            {"pose7": virtual_b, "nominal_pose7": nominal_b},
            exec_h,
        )
        apply_blended_carry(posed, blended["pose7"], blended["nominal_pose7"], transform)
        rotation, position = pose7_to_Rp(
            pose7_base_to_fixture(blended["pose7"], transform)
        )
        pose_R.append(rotation)
        pose_p.append(position)
        pose_field.append(field[:exec_h])

    def _report(title, rotation, position, field, action_v=None):
        rotation = np.concatenate(rotation, axis=0)
        position = np.concatenate(position, axis=0)
        field = np.concatenate(field, axis=0)
        steps = _ori_steps(rotation)
        tangent = _tangential(position, field, dt)
        moved = np.linalg.norm(position[1:] - position[:-1], axis=1) * 1e3
        print(
            f"{title}: ori step median {np.median(steps):.2f} deg, "
            f"p95 {np.percentile(steps, 95):.2f} deg, "
            f"max {np.max(steps):.2f} deg, "
            f"under 0.2 deg {int(np.sum(steps < 0.2))}/{len(steps)}"
        )
        print(
            f"{title}: position step median {np.median(moved):.2f} mm, "
            f"p95 {np.percentile(moved, 95):.2f} mm"
        )
        print(
            f"{title}: command v sign flips v1 {_sign_flips(tangent[:, 0])} "
            f"v2 {_sign_flips(tangent[:, 1])} "
            f"(steps faster than 0.1 mm/s); "
            f"v1 {_sign_flips(tangent[:, 0], 0.005)} "
            f"v2 {_sign_flips(tangent[:, 1], 0.005)} "
            f"(faster than 5 mm/s)"
        )
        if action_v is not None:
            action_v = np.concatenate(action_v, axis=0)
            print(
                f"{title}: action v sign flips v1 {_sign_flips(action_v[:, 0])} "
                f"v2 {_sign_flips(action_v[:, 1])}"
            )

    _report("newest-chunk", newest_R, newest_p, newest_field, newest_v)
    _report("pose-ensemble", pose_R, pose_p, pose_field)


if __name__ == "__main__":
    main()

"""Replay pose-ensemble logs and compare the jump guard windows.

The live runner checked every integrated step. This checks only the blended
steps that would be sent, plus the next blended target. Seed 0. No robot.
"""

from __future__ import annotations

import os
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

RUNS = [
    Path("/mnt/data/umidata/ft/vtevy/rollouts/2026-11-24_liftos_hw_pose_ensemble"),
    Path("/mnt/data/umidata/ft/vtevy/rollouts/2026-11-24_liftos_hw_pe_sd6"),
]
CKPT = Path(
    "/mnt/data/umidata/ft/training_outputs/"
    "oct_v35_h4s_r4vt_field_v2n_v3r_liftos/latest.ckpt"
)


def _load_policy():
    import importlib.util
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


def _predict(policy, device, obs):
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


def _max_step(guard, position, rotation):
    from jump_guard import TargetJumpGuard

    probe = TargetJumpGuard()
    probe.prev_x = None if guard.prev_x is None else guard.prev_x.copy()
    probe.prev_R = None if guard.prev_R is None else guard.prev_R.copy()
    probe.run = int(guard.run)
    probe.apply(position, rotation)
    if not probe.holds:
        return 0.0, 0.0, False, None
    worst = max(probe.holds, key=lambda row: row["pos_mm"])
    return worst["pos_mm"], worst["rot_deg"], bool(probe.aborted), int(worst["step"])


def main() -> None:
    import json
    import re

    import zarr
    from PyriteConfig.tasks.umift.workcell_frames import load_workcell_extrinsics
    from fixb_labels import pose7_to_Rp
    from jump_guard import MAX_STEP_DEG, MAX_STEP_MM, TargetJumpGuard
    from mold_aware_runtime import MoldAwareRuntime, r4_safety_limits
    from pose_ensemble import restore_carry, snapshot_carry
    from PyriteUtility.planning_control.temporal_ensemble import make_pose_ensembler
    from PyriteUtility.spatial_math import spatial_utilities as su

    transform = load_workcell_extrinsics().T_fixture_from_right_base
    policy, device = _load_policy()
    print(f"device {device} limits {MAX_STEP_MM:.1f} mm {MAX_STEP_DEG:.1f} deg", flush=True)

    for rollout in RUNS:
        group = zarr.open_group(str(rollout), mode="r")
        names = sorted(
            (key for key in group.keys() if str(key).startswith("horizon_")),
            key=lambda name: int(str(name).split("_")[1]),
        )
        exec_h = int(group.attrs.get("sparse_execution_horizon", 6))
        dt = float(group.attrs.get("decode_dt_s", 1.0 / 15.0))
        robot_path = Path(str(group.attrs["episode_data_folder"])) / "robot_data_0.json"
        text = re.sub(r",(\s*[}\]])", r"\1", robot_path.read_text())
        rows = json.loads(text)
        robot_t = np.array([row["robot_time_stamps"] for row in rows], dtype=np.float64) / 1000.0
        robot_pose = np.array([row["ts_pose_fb"] for row in rows], dtype=np.float64)

        runtime = MoldAwareRuntime.from_assets(
            "r4_world_fixb_tilt", dry_run=False, limits=r4_safety_limits()
        )
        runtime.tcp_relative_rotation = True
        runtime.reset()
        ensembler = make_pose_ensembler(0.005)
        old_prev = (None, None)
        new_prev = (None, None)
        print(f"\n{rollout.name} windows {len(names)} exec {exec_h}", flush=True)
        for index, name in enumerate(names):
            obs = {
                key: np.asarray(group[name]["obs_policy"][key])
                for key in group[name]["obs_policy"].keys()
            }
            stamp = float(np.asarray(group[name]["action"]["timestamps_s"]).reshape(-1)[0])
            nearest = int(np.clip(np.searchsorted(robot_t, stamp), 1, len(robot_t) - 1))
            if abs(robot_t[nearest - 1] - stamp) < abs(robot_t[nearest] - stamp):
                nearest -= 1
            rotation_base, position_base = pose7_to_Rp(robot_pose[nearest].reshape(1, 7))
            position_fix = position_base @ transform[:3, :3].T + transform[:3, 3]
            rotation_fix = np.einsum("ij,njk->nik", transform[:3, :3], rotation_base)
            action = _predict(policy, device, obs)
            snap = snapshot_carry(runtime)
            full = runtime.decode_chunk(
                action,
                position_fix[0],
                rotation_fix[0],
                np.full(len(action), dt),
                apply_safety=True,
            )
            restore_carry(runtime, snap)
            runtime.decode_chunk(
                action[:exec_h],
                position_fix[0],
                rotation_fix[0],
                np.full(exec_h, dt),
                apply_safety=True,
            )
            world_from_fixture = np.linalg.inv(transform)
            rotation_cmd = np.einsum(
                "ij,njk->nik", world_from_fixture[:3, :3], full["R_d"]
            )
            position_cmd = (
                world_from_fixture[:3, :3] @ full["x_d"].T
            ).T + world_from_fixture[:3, 3]
            old_guard = TargetJumpGuard()
            old_guard.prev_x, old_guard.prev_R = old_prev
            old_mm, old_deg, old_abort, old_step = _max_step(
                old_guard, position_cmd, rotation_cmd
            )
            old_prev = (position_cmd[-1].copy(), rotation_cmd[-1].copy())

            moved = dict(full)
            moved["R_d"] = rotation_cmd
            moved["x_d"] = position_cmd
            field_base = np.einsum("ij,njk->nik", world_from_fixture[:3, :3], full["R_field"])
            foot_base = (world_from_fixture[:3, :3] @ full["x_s"].T).T + world_from_fixture[:3, 3]
            moved["R_field"] = field_base
            moved["x_s"] = foot_base
            nominal, virtual = runtime.decoded_to_action_mats(moved)
            nominal_p = su.SE3_to_pose7(nominal)
            virtual_p = su.SE3_to_pose7(virtual)
            ensembler.push(
                index * exec_h,
                {"pose7": virtual_p, "nominal_pose7": nominal_p},
            )
            sent = np.stack(
                [ensembler.pose_at(index * exec_h + step) for step in range(exec_h)]
            )
            nxt = ensembler.pose_at(index * exec_h + exec_h)
            rows_p = sent if nxt is None else np.vstack([sent, nxt])
            rot, pos = pose7_to_Rp(rows_p)
            new_guard = TargetJumpGuard()
            new_guard.prev_x, new_guard.prev_R = new_prev
            new_mm, new_deg, new_abort, new_step = _max_step(new_guard, pos, rot)
            kept = pose7_to_Rp(sent[-1])
            new_prev = (kept[1][0].copy(), kept[0][0].copy())
            ensembler.pop(exec_h)
            last = index == len(names) - 1
            if old_abort or new_abort or last or index < 2:
                print(
                    f"  h{index:02d} full worst {old_mm:.1f} mm / {old_deg:.1f} deg "
                    f"step {old_step} abort {old_abort} | "
                    f"executed+next worst {new_mm:.1f} mm / {new_deg:.1f} deg "
                    f"step {new_step} abort {new_abort}",
                    flush=True,
                )
        print(f"{rollout.name} done", flush=True)


if __name__ == "__main__":
    main()

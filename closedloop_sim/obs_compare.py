#!/usr/bin/env python3
"""fk60 observation at one time, against episode 40.

The sim pose is the start pose held still. The demo pose is the recorded TCP.
Images are either the black frames the rollout feeds or the recorded RGB.
Wrench is the plant contact at the start in every case, so it is not the
difference. No robot.
"""

from __future__ import annotations

import json

import numpy as np
import torch
import zarr
from scipy.spatial.transform import Rotation, Slerp

import policy_rollout as sim
from mold_aware_runtime import MoldAwareRuntime, r4_safety_limits
from PyriteConfig.tasks.umift.umift_type_conversions import raw_to_obs_inference

ZARR = sim.ZARR
TIMES = (0.0, 8.0)


def _interp_pose(src_t, src_pose, query_t):
    src_t = np.asarray(src_t, dtype=np.float64).reshape(-1)
    src_pose = np.asarray(src_pose, dtype=np.float64)
    query_t = np.asarray(query_t, dtype=np.float64).reshape(-1)
    out = np.empty((len(query_t), 7), dtype=np.float64)
    before = query_t <= src_t[0]
    after = query_t >= src_t[-1]
    out[before] = src_pose[0]
    out[after] = src_pose[-1]
    mid = ~before & ~after
    if not np.any(mid):
        return out
    right = np.searchsorted(src_t, query_t[mid])
    left = right - 1
    span = src_t[right] - src_t[left]
    alpha = np.clip((query_t[mid] - src_t[left]) / np.clip(span, 1e-9, None), 0.0, 1.0)
    out[mid, :3] = (1.0 - alpha)[:, None] * src_pose[left, :3] + alpha[:, None] * src_pose[right, :3]
    # One slerp per queried segment. The window is a few seconds.
    groups = {}
    for row, a, b, mix in zip(np.flatnonzero(mid), left, right, alpha):
        groups.setdefault((int(a), int(b)), []).append((int(row), float(mix)))
    for (a, b), samples in groups.items():
        slerp = Slerp([0.0, 1.0], Rotation.from_quat(src_pose[[a, b]][:, [4, 5, 6, 3]]))
        mixes = np.array([mix for _row, mix in samples])
        quat = slerp(mixes).as_quat()
        for (row, _mix), q in zip(samples, quat):
            out[row, 3] = q[3]
            out[row, 4:7] = q[:3]
    return out


def _nearest(src_t, rows, query_t):
    src_t = np.asarray(src_t, dtype=np.float64).reshape(-1)
    query_t = np.asarray(query_t, dtype=np.float64).reshape(-1)
    right = np.clip(np.searchsorted(src_t, query_t), 0, len(src_t) - 1)
    left = np.clip(right - 1, 0, len(src_t) - 1)
    use_left = np.abs(src_t[left] - query_t) <= np.abs(src_t[right] - query_t)
    index = np.where(use_left, left, right)
    return np.asarray(rows)[index]


def _window_times(t_end, count):
    step = 1.0 / sim.POSE_HZ
    return t_end - (count - 1 - np.arange(count)) * step


def _assemble(shape_meta, runtime, pose, wrench, times, t_end, rgb):
    count = len(pose)
    obs_raw = sim._obs_raw(shape_meta, pose, wrench, times, count, t_end)
    if rgb is not None:
        obs_raw["rgb_0"] = rgb
    runtime.reset()
    runtime.fill_obs_raw(obs_raw, robot_id=0)
    episode = {}
    raw_to_obs_inference(obs_raw, episode, shape_meta)
    obs = episode["obs"]
    if "depth_0" in obs:
        obs["depth_0"] = np.zeros_like(obs["depth_0"])
    sparse = shape_meta["sample"]["obs"]["sparse"]
    if "elapsed_time_0" in sparse and "elapsed_time_0" not in obs:
        horizon = int(sparse["elapsed_time_0"]["horizon"])
        obs["elapsed_time_0"] = np.full((horizon, 1), t_end / sim.SLOW_DOWN, np.float32)
    return obs, obs_raw


def _policy_view(controller, obs):
    sampled = controller.sample_sparse_obs(obs)
    processed, _base = controller.preprocess_policy_obs(sampled, ignore_rgb=False)
    return sampled, processed


def _predict(controller, obs, device):
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    controller.compute_one_horizon_action(obs, device)
    raw = np.asarray(controller.last_raw_action, dtype=np.float64)
    step = raw[0]
    return {
        "v_par_mps": [float(step[0]), float(step[1])],
        "v_par_norm_mps": float(np.linalg.norm(step[:2])),
        "delta_n_mm": float(step[2] * 1e3),
        "stiffness_n_per_m": float(step[18]),
    }


def _max_abs(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def main() -> None:
    sim.require_worktree()
    torch.manual_seed(0)
    np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = sim.CHECKPOINTS["fk60"]
    policy, shape_meta = sim.load_policy(ckpt, device)
    sim._apply_runner_windows(shape_meta)
    sparse = shape_meta["sample"]["obs"]["sparse"]
    robot_need = 1
    for key in sparse:
        if str(key).startswith("robot0_"):
            robot_need = max(robot_need, sim._need(sparse, key))
    group = zarr.open_group(str(ZARR), mode="r")["data"]["episode_40"]
    pose = np.asarray(group["ts_pose_fb_0"], dtype=np.float64)
    robot_t = np.asarray(group["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
    robot_t = robot_t - robot_t[0]
    rgb = np.asarray(group["rgb_0"])
    rgb_t = np.asarray(group["rgb_time_stamps_0"], dtype=np.float64).reshape(-1)
    rgb_t = rgb_t - rgb_t[0]
    mold_sd = np.asarray(group["mold_signed_distance_0"], dtype=np.float64).reshape(-1)
    mold_n = np.asarray(group["mold_normal_0"], dtype=np.float64)
    mold_r6 = np.asarray(group["mold_R_field_rot6d_0"], dtype=np.float64)

    from closedloop_sim.plant import CartesianSpringPlant
    from fixb_field import FixBMoldField
    from fixb_labels import pose7_to_Rp

    trace_path = sim.OUT / "policy_fk60_fixb_ep40.npz"
    trace = np.load(trace_path)
    trace_t = np.asarray(trace["t_s"], dtype=np.float64).reshape(-1)
    trace_pose = np.asarray(trace["pose7"], dtype=np.float64)

    start_R, start_p = pose7_to_Rp(pose[0].reshape(1, 7))
    plant = CartesianSpringPlant(FixBMoldField(sim.FIELD), start_p[0], start_R[0])
    snap = plant._forces(plant.x.copy(), np.zeros((3, 3)))
    hold = np.zeros((robot_need, 6), dtype=np.float64)
    hold[:, :3] = plant.R.T @ np.asarray(snap["contact_world"], dtype=np.float64)

    runtime = MoldAwareRuntime.from_assets(
        run_type="r4_world_fixb_tilt",
        field_path=sim.FIELD,
        dry_run=False,
        limits=r4_safety_limits(
            max_tcp_penetration_m=0.020,
            wrench_hold_n=90.0,
            abort_on_measured_mesh_distance=False,
            enable_delta_n_floor=False,
            max_press_depth_m=0.005,
        ),
    )
    from closedloop_sim.api import ModelPredictiveController

    controller = ModelPredictiveController(
        shape_meta=shape_meta,
        id_list=[0],
        policy=policy,
        action_to_trajectory=lambda _action: None,
        execution_horizon=1,
        fix_orientation=False,
    )
    n_rgb = max(sim._need(sparse, "rgb_0"), 1)
    report = {
        "checkpoint": ckpt,
        "device": device,
        "pose_frames": dict(shape_meta.get("pose_frames", {})),
        "elapsed_time": "elapsed_time_0" in sparse,
        "robot_window_samples": int(robot_need),
        "robot_window_s": float((robot_need - 1) / sim.POSE_HZ),
        "rgb_window_samples": int(n_rgb),
        "sparse_pose": {
            "horizon": int(sparse["robot0_eef_pos"]["horizon"]),
            "down_sample_steps": int(sparse["robot0_eef_pos"]["down_sample_steps"]),
        },
        "times": {},
    }
    print(
        f"window {report['robot_window_s']:.2f}s "
        f"pose horizon {report['sparse_pose']['horizon']} "
        f"ds {report['sparse_pose']['down_sample_steps']} "
        f"elapsed_time {report['elapsed_time']} "
        f"frames {report['pose_frames']}",
        flush=True,
    )

    for t_end in TIMES:
        grid = _window_times(t_end, robot_need)
        demo_pose = _interp_pose(robot_t, pose, grid)
        demo_index = int(np.argmin(np.abs(robot_t - t_end)))
        label_geom = np.concatenate(
            [
                mold_sd[demo_index].reshape(1),
                mold_n[demo_index],
                mold_r6[demo_index],
            ]
        )
        rgb_stamps = t_end - np.arange(n_rgb)[::-1] / 15.0
        real_rgb = _nearest(rgb_t, rgb, rgb_stamps).astype(np.uint8)
        rollout_pose = _interp_pose(trace_t, trace_pose, grid)
        cases = {}
        views = {}
        for name, pose_now, images in (
            ("rollout_blank", rollout_pose, None),
            ("demo_pose_blank", demo_pose, None),
            ("demo", demo_pose, real_rgb),
        ):
            obs, obs_raw = _assemble(
                shape_meta, runtime, pose_now, hold, grid, t_end, images
            )
            sampled, processed = _policy_view(controller, obs)
            action = _predict(controller, obs, device)
            cases[name] = action
            views[name] = (obs_raw, sampled, processed)
            print(
                f"t={t_end:.1f}s {name} v={action['v_par_norm_mps']:.4f} m/s "
                f"dn={action['delta_n_mm']:.2f} mm",
                flush=True,
            )
        sim_raw, sim_sampled, sim_proc = views["rollout_blank"]
        demo_raw, demo_sampled, demo_proc = views["demo"]
        blank_raw, _blank_sampled, blank_proc = views["demo_pose_blank"]
        pos = np.asarray(sim_sampled["robot0_eef_pos"])
        demo_pos = np.asarray(demo_sampled["robot0_eef_pos"])
        filled = np.concatenate(
            [
                np.asarray(demo_raw["mold_signed_distance_0"])[-1].reshape(-1),
                np.asarray(demo_raw["mold_normal_0"])[-1].reshape(-1),
                np.asarray(demo_raw["mold_R_field_rot6d_0"])[-1].reshape(-1),
            ]
        )
        rgb_sim = np.asarray(sim_proc["rgb_0"], dtype=np.float64)
        rgb_demo = np.asarray(demo_proc["rgb_0"], dtype=np.float64)
        row = {
            "pose_abs_max_mm": _max_abs(pos, demo_pos) * 1e3,
            "pose_relative_max_mm": _max_abs(
                sim_proc["robot0_eef_pos"], demo_proc["robot0_eef_pos"]
            )
            * 1e3,
            "pose_relative_rot_max": _max_abs(
                sim_proc["robot0_eef_rot_axis_angle"],
                demo_proc["robot0_eef_rot_axis_angle"],
            ),
            "demo_window_travel_mm": float(
                np.max(np.linalg.norm(demo_pos - demo_pos[-1], axis=1)) * 1e3
            ),
            "sim_window_travel_mm": float(
                np.max(np.linalg.norm(pos - pos[-1], axis=1)) * 1e3
            ),
            "mold_geom_max_abs": _max_abs(sim_proc["robot0_mold_geom"], demo_proc["robot0_mold_geom"]),
            "mold_vs_label_max_abs": _max_abs(filled, label_geom),
            "blank_vs_demo_pose_mold_max_abs": _max_abs(
                blank_proc["robot0_mold_geom"], demo_proc["robot0_mold_geom"]
            ),
            "rgb_sim_mean": float(np.mean(rgb_sim)),
            "rgb_demo_mean": float(np.mean(rgb_demo)),
            "rgb_mean_abs": float(np.mean(np.abs(rgb_sim - rgb_demo))),
            "rgb_nonzero_frac": float(np.mean(np.abs(rgb_demo) > 1e-6)),
            "depth_in_policy": "depth_0" in sim_proc,
            "action": cases,
        }
        report["times"][f"{t_end:.0f}s"] = row
        print(json.dumps({f"{t_end:.0f}s": row}, indent=2), flush=True)

    sim.OUT.mkdir(parents=True, exist_ok=True)
    path = sim.OUT / "fk60_obs_compare.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()

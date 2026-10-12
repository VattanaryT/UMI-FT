#!/usr/bin/env python3
"""
Open-loop policy evaluation on validation episodes.

Runs MPC-style open-loop inference on recorded val observations, decodes
actions into virtual-target trajectories, and writes a replay-compatible zarr
that can be consumed by run_virtual_target_dataset_playback.py.

Also writes ``policy_open_loop_val_queries.json`` with per-query base poses /
expert times / open-loop index slices for plot_open_loop_vs_expert_3d.py.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import dill
import hydra
import numpy as np
import torch
import zarr
from einops import rearrange
from omegaconf import OmegaConf

from PyriteConfig.tasks.umift import umift_type_conversions as umift_task
from PyriteConfig.tasks.umift.workcell_frames import pose_frames_cfg
from PyriteML.diffusion_policy.common.pytorch_util import dict_apply
from PyriteML.diffusion_policy.workspace.base_workspace import BaseWorkspace
from PyriteUtility.planning_control.temporal_ensemble import PoseChunkEnsembler
from PyriteUtility.spatial_math import spatial_utilities as su

REQUIRED_KEYS = (
    "ts_pose_command_0",
    "ts_pose_virtual_target_0",
    "stiffness_0",
    "robot_time_stamps_0",
)

PLAYBACK_SCRIPT = (
    Path(__file__).resolve().parent / "run_virtual_target_dataset_playback.py"
)


def compute_sampler_stride(
    sparse_execution_horizon: int,
    sparse_action_down_sample_steps: int,
    sparse_query_frequency_down_sample_steps: int,
) -> int:
    """Map executed sparse-action steps to val-sampler index stride."""
    if sparse_query_frequency_down_sample_steps <= 0:
        raise ValueError("sparse_query_frequency_down_sample_steps must be > 0")
    stride = (
        sparse_execution_horizon
        * sparse_action_down_sample_steps
        / float(sparse_query_frequency_down_sample_steps)
    )
    stride_i = int(round(stride))
    if stride_i < 1:
        raise ValueError(
            f"Computed sampler stride {stride} rounds to {stride_i}; must be >= 1"
        )
    return stride_i


def compute_sparse_action_timesteps_s(
    sparse_action_horizon: int,
    sparse_action_down_sample_steps: int,
    raw_time_step_s: float,
    slow_down_factor: float = 1.0,
) -> np.ndarray:
    return (
        np.arange(sparse_action_horizon, dtype=np.float64)
        * sparse_action_down_sample_steps
        * raw_time_step_s
        * slow_down_factor
    )


def validate_export_episode_keys(episode_data: Dict[str, np.ndarray]) -> None:
    missing = [k for k in REQUIRED_KEYS if k not in episode_data]
    if missing:
        raise ValueError(f"Episode is missing required keys: {missing}")
    n = episode_data["ts_pose_command_0"].shape[0]
    if episode_data["ts_pose_virtual_target_0"].shape[0] != n:
        raise ValueError("ts_pose_virtual_target_0 length mismatch")
    if np.asarray(episode_data["stiffness_0"]).reshape(-1).shape[0] != n:
        raise ValueError("stiffness_0 length mismatch")
    if np.asarray(episode_data["robot_time_stamps_0"]).reshape(-1).shape[0] != n:
        raise ValueError("robot_time_stamps_0 length mismatch")
    if episode_data["ts_pose_command_0"].shape[1] != 7:
        raise ValueError("ts_pose_command_0 must be (N, 7)")
    if episode_data["ts_pose_virtual_target_0"].shape[1] != 7:
        raise ValueError("ts_pose_virtual_target_0 must be (N, 7)")


def decode_horizon_actions(
    raw_action: np.ndarray,
    base_se3: List[np.ndarray],
    id_list: Sequence[int],
    action_dim: int,
    fix_orientation: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Decode a sparse action horizon into world-frame pose7 trajectories.

    Returns:
        ts_pose_command (N, 7),
        ts_pose_virtual_target (N, 7),
        stiffness (N,),
        gripper (N,) or None
    """
    if action_dim == 21 or action_dim == 42:
        nominal_mats, vt_mats, stiffness_list, eoat_list = umift_task.action21_postprocess(
            raw_action, base_se3, list(id_list), fix_orientation
        )
        robot_id = id_list[0]
        ts_pose_command = su.SE3_to_pose7(nominal_mats[robot_id].reshape([-1, 4, 4]))
        ts_pose_virtual_target = su.SE3_to_pose7(vt_mats[robot_id].reshape([-1, 4, 4]))
        stiffness = np.asarray(stiffness_list[robot_id], dtype=np.float64).reshape(-1)
        gripper = np.asarray(eoat_list[robot_id][:, 0], dtype=np.float64).reshape(-1)
        return ts_pose_command, ts_pose_virtual_target, stiffness, gripper

    if action_dim == 19 or action_dim == 38:
        nominal_mats, vt_mats, stiffness_list = umift_task.action19_postprocess(
            raw_action, base_se3, list(id_list), fix_orientation
        )
        robot_id = id_list[0]
        ts_pose_command = su.SE3_to_pose7(nominal_mats[robot_id].reshape([-1, 4, 4]))
        ts_pose_virtual_target = su.SE3_to_pose7(vt_mats[robot_id].reshape([-1, 4, 4]))
        stiffness = np.asarray(stiffness_list[robot_id], dtype=np.float64).reshape(-1)
        return ts_pose_command, ts_pose_virtual_target, stiffness, None

    if action_dim == 9 or action_dim == 18:
        nominal_mats = umift_task.action9_postprocess(
            raw_action, base_se3, list(id_list), fix_orientation
        )
        robot_id = id_list[0]
        ts_pose_command = su.SE3_to_pose7(nominal_mats[robot_id].reshape([-1, 4, 4]))
        stiffness = np.full((ts_pose_command.shape[0],), 5000.0, dtype=np.float64)
        return ts_pose_command, ts_pose_command.copy(), stiffness, None

    raise RuntimeError(f"Unsupported action dim for decode: {action_dim}")


def get_query_base_info(
    sampler,
    sample_idx: int,
    id_list: Sequence[int],
) -> Dict[str, Any]:
    """Absolute SE3 / timing at the query pose (same convention as obs sampling)."""
    epi_id, rgb_id = sampler.indices[sample_idx]
    episode = f"episode_{epi_id}"
    data_episode = sampler.replay_buffer["data"][episode]
    query_time = float(np.squeeze(data_episode["obs"]["rgb_time_stamps_0"][rgb_id]))

    base_se3: List[np.ndarray] = []
    expert_pose_index = 0
    expert_robot_time_s = query_time
    for robot_id in id_list:
        robot_times = np.squeeze(data_episode["obs"][f"robot_time_stamps_{robot_id}"])
        query_id = int(np.searchsorted(robot_times, query_time))
        query_id = min(max(query_id, 0), len(robot_times) - 1)
        if robot_id == id_list[0]:
            expert_pose_index = query_id
            expert_robot_time_s = float(robot_times[query_id])
        pos = np.asarray(data_episode["obs"][f"robot{robot_id}_eef_pos"][query_id])
        rot = np.asarray(
            data_episode["obs"][f"robot{robot_id}_eef_rot_axis_angle"][query_id]
        )
        pose9 = np.concatenate([pos, rot], axis=-1)
        base_se3.append(su.pose9_to_SE3(pose9))

    # Absolute-action policies output poses in the dataset (fixture) frame, whose
    # "base" is the identity (see sparse_obs_to_obs_sample, pose_frames.action).
    if pose_frames_cfg(sampler.shape_meta)["action"] == "absolute":
        base_se3 = [np.eye(4, dtype=base_se3[0].dtype) for _ in base_se3]

    base_pose7 = su.SE3_to_pose7(base_se3[0].reshape(1, 4, 4))[0]
    return {
        "epi_id": int(epi_id),
        "rgb_id": int(rgb_id),
        "base_se3": base_se3,
        "expert_pose_index": int(expert_pose_index),
        "expert_robot_time_s": float(expert_robot_time_s),
        "rgb_query_time_s": float(query_time),
        "base_pose7": np.asarray(base_pose7, dtype=np.float64).reshape(-1).tolist(),
    }


def get_base_se3_for_sample(
    sampler,
    sample_idx: int,
    id_list: Sequence[int],
) -> List[np.ndarray]:
    """Absolute SE3 at the query pose (same convention as sparse_obs_to_obs_sample)."""
    return get_query_base_info(sampler, sample_idx, id_list)["base_se3"]


def group_sampler_indices_by_episode(
    indices: Sequence[Tuple[int, int]],
) -> Dict[int, List[Tuple[int, int]]]:
    """
    Group (global_sample_idx, rgb_id) by episode id.

    Returns dict: epi_id -> list of (global_sample_idx, rgb_id) in sampler order.
    """
    grouped: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for global_idx, (epi_id, rgb_id) in enumerate(indices):
        grouped[int(epi_id)].append((global_idx, int(rgb_id)))
    return dict(grouped)


def list_val_episode_names(dataset) -> List[str]:
    """Episode names selected by the training val_mask (same seed/split)."""
    names: List[str] = []
    for i, key in enumerate(dataset.replay_buffer["data"].keys()):
        if dataset.val_mask is not None and bool(dataset.val_mask[i]):
            names.append(key)
    return names


def resolve_checkpoint_path(checkpoint: str) -> str:
    ckpt_path = checkpoint
    if not ckpt_path.endswith(".ckpt"):
        ckpt_path = os.path.join(ckpt_path, "checkpoints", "latest.ckpt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return ckpt_path


def load_policy_and_cfg(ckpt_path: str, device: torch.device):
    payload = torch.load(open(ckpt_path, "rb"), map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=["optimizer"], include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    if hasattr(cfg, "policy") and hasattr(cfg.policy, "num_inference_steps"):
        policy.num_inference_steps = cfg.policy.num_inference_steps
    policy.eval().to(device)
    policy.reset()
    return policy, cfg


def build_val_dataset(cfg, dataset_path: Optional[str] = None):
    if dataset_path is not None:
        OmegaConf.update(cfg, "task.dataset.dataset_path", dataset_path, merge=False)
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    val_dataset = dataset.get_validation_dataset()
    return dataset, val_dataset


# Ablation: zero depth before policy. Flip to False (or comment out the block below) to restore.
ZERO_DEPTH = True


def predict_sparse_action(policy, obs_sample: Dict[str, Any], device: torch.device) -> np.ndarray:
    def _to_batched_tensor(x):
        t = torch.as_tensor(np.asarray(x)).to(device)
        return rearrange(t, "... -> 1 ...")

    if ZERO_DEPTH:
        sparse = obs_sample.get("sparse", obs_sample)
        if "depth_0" in sparse:
            sparse["depth_0"] = np.zeros_like(np.asarray(sparse["depth_0"]))

    obs_batched = dict_apply(obs_sample, _to_batched_tensor)
    with torch.no_grad():
        result = policy.predict_action(obs_batched)
    if "sparse" in result and result["sparse"] is not None:
        raw_action = result["sparse"][0].detach().to("cpu").numpy()
    else:
        raw_action = result["action"][0].detach().to("cpu").numpy()
    return raw_action


def get_action_normalizer(policy):
    """Return SingleFieldLinearNormalizer for sparse/action outputs."""
    if hasattr(policy, "sparse_normalizer"):
        return policy.sparse_normalizer["action"]
    if hasattr(policy, "normalizer") and "action" in policy.normalizer.params_dict:
        return policy.normalizer["action"]
    raise RuntimeError(
        "Policy has no sparse_normalizer['action'] or normalizer['action']"
    )


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def compute_action_field_errors(
    pred_sparse: np.ndarray,
    gt_sparse: np.ndarray,
    action_normalizer,
) -> Dict[str, float]:
    """
    Per-query action errors over horizon h and all robot blocks.

    pose9 layout per robot: [pos(3), rot6d(6), vt_pos(3), vt_rot6d(6),
    stiffness(1), gripper(1), force(1)] for action21; shorter prefixes for
    action19 / action9.
    """
    pred = np.asarray(pred_sparse, dtype=np.float64)
    gt = np.asarray(gt_sparse, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch: {pred.shape} vs {gt.shape}")
    if pred.ndim != 2:
        raise ValueError(f"Expected (H, D) actions, got {pred.shape}")

    h, action_dim = pred.shape
    dims_per_robot = 21 if action_dim % 21 == 0 else (19 if action_dim % 19 == 0 else 9)
    if action_dim % dims_per_robot != 0:
        raise ValueError(
            f"Unsupported action_dim={action_dim}; expected multiple of 9/19/21"
        )
    n_robots = action_dim // dims_per_robot

    pred_n = action_normalizer.normalize(pred).detach().cpu().numpy()
    gt_n = action_normalizer.normalize(gt).detach().cpu().numpy()
    metrics: Dict[str, float] = {
        "action_mse": float(np.mean((pred - gt) ** 2)),
        "naction_mse": float(np.mean((pred_n - gt_n) ** 2)),
    }

    pos_err: List[float] = []
    rot_err: List[float] = []
    vt_pos_err: List[float] = []
    vt_rot_err: List[float] = []
    stiff_err: List[float] = []
    grip_err: List[float] = []
    force_err: List[float] = []

    for r in range(n_robots):
        o = r * dims_per_robot
        pos_err.append(_rmse(pred[:, o : o + 3], gt[:, o : o + 3]))
        rot_err.append(_rmse(pred[:, o + 3 : o + 9], gt[:, o + 3 : o + 9]))
        if dims_per_robot >= 19:
            vt_pos_err.append(_rmse(pred[:, o + 9 : o + 12], gt[:, o + 9 : o + 12]))
            vt_rot_err.append(_rmse(pred[:, o + 12 : o + 18], gt[:, o + 12 : o + 18]))
            stiff_err.append(_mae(pred[:, o + 18], gt[:, o + 18]))
        if dims_per_robot >= 21:
            grip_err.append(_mae(pred[:, o + 19], gt[:, o + 19]))
            force_err.append(_mae(pred[:, o + 20], gt[:, o + 20]))

    # print(pred[:, :3])
    # print(gt[:, :3])
    # print(np.linalg.norm(pred[:, :3] - gt[:, :3], axis=-1))
    metrics["pos_rmse"] = float(np.mean(pos_err))
    metrics["rot_rmse"] = float(np.mean(rot_err))
    if vt_pos_err:
        metrics["vt_pos_rmse"] = float(np.mean(vt_pos_err))
        metrics["vt_rot_rmse"] = float(np.mean(vt_rot_err))
        metrics["stiffness_mae"] = float(np.mean(stiff_err))
    if grip_err:
        metrics["gripper_mae"] = float(np.mean(grip_err))
        metrics["force_mae"] = float(np.mean(force_err))
    return metrics


def _mean_metric_lists(metric_lists: Dict[str, List[float]]) -> Dict[str, float]:
    return {
        key: float(np.mean(vals)) if vals else float("nan")
        for key, vals in metric_lists.items()
    }


def format_field_metrics(metrics: Dict[str, float]) -> str:
    parts = [
        f"naction_mse={metrics.get('naction_mse', float('nan')):.6g}",
        f"pos_rmse={metrics.get('pos_rmse', float('nan')):.6g}",
        f"rot_rmse={metrics.get('rot_rmse', float('nan')):.6g}",
    ]
    if "vt_pos_rmse" in metrics:
        parts.extend(
            [
                f"vt_pos_rmse={metrics['vt_pos_rmse']:.6g}",
                f"vt_rot_rmse={metrics['vt_rot_rmse']:.6g}",
                f"stiffness_mae={metrics['stiffness_mae']:.6g}",
            ]
        )
    if "gripper_mae" in metrics:
        parts.extend(
            [
                f"gripper_mae={metrics['gripper_mae']:.6g}",
                f"force_mae={metrics['force_mae']:.6g}",
            ]
        )
    return ", ".join(parts)


def rollout_episode_open_loop(
    policy,
    val_dataset,
    episode_sample_entries: List[Tuple[int, int]],
    shape_meta: dict,
    id_list: Sequence[int],
    sparse_execution_horizon: int,
    sampler_stride: int,
    raw_time_step_s: float,
    device: torch.device,
    enable_ensemble_chunking: bool = True,
    ensemble_decay_m: float = 0.0,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float], List[Dict[str, Any]]]:
    """
    MPC-style open-loop rollout over one val episode's sampler indices.

    episode_sample_entries: list of (global_sample_idx, rgb_id)

    When ``enable_ensemble_chunking`` is True, each full predicted horizon is
    pushed into a PoseChunkEnsembler and only the blended first
    ``sparse_execution_horizon`` steps are retained (zero-added-latency
    temporal ensembling, matching live umift_env_runner).

    Returns episode_data, aggregate metrics, and per-query metadata records for
    expert/open-loop alignment plots.
    """
    action_dim = int(shape_meta["action"]["shape"][0])
    sparse_action_horizon = int(shape_meta["sample"]["action"]["sparse"]["horizon"])
    sparse_action_down_sample_steps = int(
        shape_meta["sample"]["action"]["sparse"]["down_sample_steps"]
    )
    action_timesteps_s = compute_sparse_action_timesteps_s(
        sparse_action_horizon=sparse_action_horizon,
        sparse_action_down_sample_steps=sparse_action_down_sample_steps,
        raw_time_step_s=raw_time_step_s,
    )
    action_normalizer = get_action_normalizer(policy)

    cmd_chunks: List[np.ndarray] = []
    vt_chunks: List[np.ndarray] = []
    stiff_chunks: List[np.ndarray] = []
    grip_chunks: List[np.ndarray] = []
    time_chunks: List[np.ndarray] = []
    # Raw (pre-ensemble) decoded poses — what the policy actually predicted.
    raw_cmd_chunks: List[np.ndarray] = []
    raw_vt_chunks: List[np.ndarray] = []
    raw_stiff_chunks: List[np.ndarray] = []
    query_records: List[Dict[str, Any]] = []
    field_metric_lists: Dict[str, List[float]] = {
        "action_mse": [],
        "naction_mse": [],
        "pos_rmse": [],
        "rot_rmse": [],
        "vt_pos_rmse": [],
        "vt_rot_rmse": [],
        "stiffness_mae": [],
        "gripper_mae": [],
        "force_mae": [],
    }

    ensembler: Optional[PoseChunkEnsembler] = None
    if enable_ensemble_chunking:
        ensembler = PoseChunkEnsembler(
            pose_keys=("cmd", "vt"),
            linear_keys=("stiffness", "gripper"),
            weight_mode="exp",
            decay_m=float(ensemble_decay_m),
        )

    t0 = 0.0
    pos = 0
    absolute_step = 0
    query_idx = 0
    while pos < len(episode_sample_entries):
        global_idx, _rgb_id = episode_sample_entries[pos]
        obs_sample, action_sample = val_dataset.sampler.sample_sequence(global_idx)
        query_info = get_query_base_info(val_dataset.sampler, global_idx, id_list)
        base_se3 = query_info["base_se3"]

        raw_action = predict_sparse_action(policy, obs_sample, device)
        exec_h = min(sparse_execution_horizon, raw_action.shape[0])

        ts_cmd, ts_vt, ts_stiff, ts_grip = decode_horizon_actions(
            raw_action=raw_action,
            base_se3=base_se3,
            id_list=id_list,
            action_dim=action_dim,
        )
        # Always keep the raw decoded executed slice (pre-ensemble).
        raw_cmd_chunks.append(ts_cmd[:exec_h].copy())
        raw_vt_chunks.append(ts_vt[:exec_h].copy())
        raw_stiff_chunks.append(ts_stiff[:exec_h].copy())

        open_loop_start_idx = absolute_step
        if ensembler is not None:
            fields = {
                "cmd": ts_cmd,
                "vt": ts_vt,
                "stiffness": ts_stiff,
                "gripper": ts_grip,
            }
            ensembler.push(start_step=absolute_step, fields=fields)
            blended = ensembler.pop(exec_h)
            cmd_chunks.append(blended["cmd"])
            vt_chunks.append(blended["vt"])
            stiff_chunks.append(blended["stiffness"].reshape(-1))
            if "gripper" in blended:
                grip_chunks.append(blended["gripper"].reshape(-1))
        else:
            cmd_chunks.append(ts_cmd[:exec_h])
            vt_chunks.append(ts_vt[:exec_h])
            stiff_chunks.append(ts_stiff[:exec_h])
            if ts_grip is not None:
                grip_chunks.append(ts_grip[:exec_h])

        chunk_times = t0 + action_timesteps_s[:exec_h]
        time_chunks.append(chunk_times)
        action_offsets_s = action_timesteps_s[:exec_h].astype(np.float64)
        if exec_h > 0:
            # Next chunk starts after the last executed sparse step interval.
            dt = (
                sparse_action_down_sample_steps * raw_time_step_s
                if exec_h == 1
                else float(action_timesteps_s[1] - action_timesteps_s[0])
            )
            t0 = float(chunk_times[-1] + dt)

        open_loop_end_idx = absolute_step + exec_h
        query_records.append(
            {
                "query_idx": int(query_idx),
                "episode_id": int(query_info["epi_id"]),
                "global_sample_idx": int(global_idx),
                "rgb_id": int(query_info["rgb_id"]),
                "expert_pose_index": int(query_info["expert_pose_index"]),
                "expert_robot_time_s": float(query_info["expert_robot_time_s"]),
                "rgb_query_time_s": float(query_info["rgb_query_time_s"]),
                "base_pose7": query_info["base_pose7"],
                "open_loop_start_idx": int(open_loop_start_idx),
                "open_loop_end_idx": int(open_loop_end_idx),
                "exec_horizon": int(exec_h),
                "action_offsets_s": action_offsets_s.tolist(),
                "open_loop_times_s": np.asarray(chunk_times, dtype=np.float64).tolist(),
                # Raw policy outputs (pre-ensemble): relative action + decoded world poses.
                "raw_action": np.asarray(raw_action[:exec_h], dtype=np.float64).tolist(),
                "raw_action_full_horizon": np.asarray(
                    raw_action, dtype=np.float64
                ).tolist(),
                "policy_pose_command_raw": np.asarray(
                    ts_cmd[:exec_h], dtype=np.float64
                ).tolist(),
                "policy_pose_virtual_target_raw": np.asarray(
                    ts_vt[:exec_h], dtype=np.float64
                ).tolist(),
                "policy_pose_command_raw_full_horizon": np.asarray(
                    ts_cmd, dtype=np.float64
                ).tolist(),
                "policy_pose_virtual_target_raw_full_horizon": np.asarray(
                    ts_vt, dtype=np.float64
                ).tolist(),
                "policy_stiffness_raw": np.asarray(
                    ts_stiff[:exec_h], dtype=np.float64
                ).reshape(-1).tolist(),
            }
        )
        query_idx += 1

        gt_sparse = np.asarray(action_sample["sparse"], dtype=np.float64)
        pred_sparse = np.asarray(raw_action, dtype=np.float64)
        h = min(exec_h, gt_sparse.shape[0], pred_sparse.shape[0])
        if h > 0:
            field_errs = compute_action_field_errors(
                pred_sparse=pred_sparse[:h],
                gt_sparse=gt_sparse[:h],
                action_normalizer=action_normalizer,
            )
            for key, value in field_errs.items():
                field_metric_lists[key].append(value)

        absolute_step += exec_h
        pos += sampler_stride

    if not cmd_chunks:
        raise RuntimeError("No actions generated for episode (empty sampler entries).")

    episode_data: Dict[str, np.ndarray] = {
        # Playback / live-matched stream (may be temporally ensembled).
        "ts_pose_command_0": np.concatenate(cmd_chunks, axis=0).astype(np.float64),
        "ts_pose_virtual_target_0": np.concatenate(vt_chunks, axis=0).astype(np.float64),
        "stiffness_0": np.concatenate(stiff_chunks, axis=0).astype(np.float64),
        "robot_time_stamps_0": np.concatenate(time_chunks, axis=0)
        .astype(np.float64)
        .reshape(-1, 1),
        # Raw policy decode (never ensembled) — preferred for open-loop vs expert plots.
        "ts_pose_command_raw_0": np.concatenate(raw_cmd_chunks, axis=0).astype(
            np.float64
        ),
        "ts_pose_virtual_target_raw_0": np.concatenate(raw_vt_chunks, axis=0).astype(
            np.float64
        ),
        "stiffness_raw_0": np.concatenate(raw_stiff_chunks, axis=0).astype(np.float64),
    }
    if grip_chunks:
        episode_data["gripper_0"] = np.concatenate(grip_chunks, axis=0).astype(
            np.float64
        ).reshape(-1, 1)

    validate_export_episode_keys(episode_data)
    field_means = _mean_metric_lists(
        {k: v for k, v in field_metric_lists.items() if v}
    )
    metrics = {
        "num_steps": float(episode_data["ts_pose_command_0"].shape[0]),
        "num_queries": float(len(field_metric_lists["action_mse"])),
        "action_mse_mean": field_means.get("action_mse", float("nan")),
        **{k: v for k, v in field_means.items() if k != "action_mse"},
    }
    return episode_data, metrics, query_records


def write_episode_to_zarr(
    output_path: str,
    episode_name: str,
    episode_data: Dict[str, np.ndarray],
) -> None:
    root = zarr.open(output_path, mode="a")
    data = root.require_group("data")
    if episode_name in data:
        del data[episode_name]
    ep = data.create_group(episode_name)
    for key, value in episode_data.items():
        ep.create_dataset(key, data=np.asarray(value), overwrite=True)


def run_playback_for_episode(
    output_path: str,
    episode_name: str,
    playback_dry_run: bool,
    hardware_config_path: str,
    execution_horizon: int,
) -> int:
    cmd = [
        sys.executable,
        str(PLAYBACK_SCRIPT),
        "--dataset_path",
        output_path,
        "--episode",
        episode_name,
        "--execution_horizon",
        str(execution_horizon),
        "--plot_trajectory",
    ]
    if playback_dry_run:
        cmd.append("--dry_run")
    elif hardware_config_path:
        cmd.extend(["--hardware_config_path", hardware_config_path])
    else:
        raise ValueError(
            "Live playback requires --hardware_config_path (or use --playback_dry_run)."
        )

    print(f"[export] Running playback: {' '.join(cmd)}")
    return subprocess.call(cmd)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open-loop val policy export to virtual-target replay zarr "
            "(MPC-style stepping on recorded observations)."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument(
        "--dataset_path",
        default="",
        type=str,
        help="Source zarr (default: checkpoint cfg task.dataset.dataset_path).",
    )
    parser.add_argument(
        "--output_path",
        default="",
        type=str,
        help="Output zarr root (default: <checkpoint_dir>/policy_open_loop_val.zarr).",
    )
    parser.add_argument(
        "--episodes",
        nargs="*",
        default=None,
        help="Optional subset of source episode names, e.g. episode_3 episode_7.",
    )
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument(
        "--sparse_execution_horizon",
        default=4,
        type=int,
        help="Sparse action steps executed per MPC query (matches umift_env_runner).",
    )
    parser.add_argument("--raw_time_step_s", default=0.0167, type=float)
    parser.add_argument(
        "--enable_ensemble_chunking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Temporally ensemble overlapping full-horizon predictions before "
            "keeping the executed slice (default: on). Use "
            "--no-enable_ensemble_chunking for the legacy hard-cut export."
        ),
    )
    parser.add_argument(
        "--ensemble_decay_m",
        default=0.0,
        type=float,
        help=(
            "Exponential decay for per-horizon-position ensemble weights "
            "(w[i]=exp(-m*i)). 0 = uniform averaging across overlapping queries."
        ),
    )
    parser.add_argument(
        "--dry_run_export",
        action="store_true",
        help="Validate export without writing zarr.",
    )
    parser.add_argument(
        "--run_playback",
        action="store_true",
        help="Invoke run_virtual_target_dataset_playback.py after export.",
    )
    parser.add_argument(
        "--playback_dry_run",
        action="store_true",
        help="Pass --dry_run to playback.",
    )
    parser.add_argument("--hardware_config_path", default="", type=str)
    parser.add_argument(
        "--playback_execution_horizon",
        default=12,
        type=int,
        help="execution_horizon passed to playback script.",
    )
    parser.add_argument(
        "--metrics_path",
        default="",
        type=str,
        help="Optional JSON path for per-episode metrics (default: next to output zarr).",
    )
    parser.add_argument(
        "--queries_path",
        default="",
        type=str,
        help=(
            "Optional JSON path for per-query alignment metadata "
            "(default: next to output zarr as policy_open_loop_val_queries.json)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    ckpt_path = resolve_checkpoint_path(args.checkpoint)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if str(device) != args.device:
        print(f"[export] Requested device {args.device} unavailable; using {device}")

    print(f"[export] Loading checkpoint: {ckpt_path}")
    policy, cfg = load_policy_and_cfg(ckpt_path, device)
    shape_meta = cfg.task.shape_meta
    id_list = list(shape_meta["id_list"])

    dataset_path = args.dataset_path or str(cfg.task.dataset.dataset_path)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    ckpt_dir = os.path.dirname(ckpt_path)
    if ckpt_dir.endswith("checkpoints"):
        ckpt_dir = os.path.dirname(ckpt_dir)
    output_path = args.output_path or os.path.join(ckpt_dir, "policy_open_loop_val.zarr")

    print(f"[export] Building val dataset from: {dataset_path}")
    print(
        f"[export] val_ratio={cfg.task.dataset.val_ratio}, "
        f"seed={cfg.task.dataset.seed}"
    )
    train_dataset, val_dataset = build_val_dataset(cfg, dataset_path=dataset_path)
    val_episode_names = list_val_episode_names(train_dataset)
    print(f"[export] Val episodes ({len(val_episode_names)}): {val_episode_names}")

    sparse_action_down_sample_steps = int(
        shape_meta["sample"]["action"]["sparse"]["down_sample_steps"]
    )
    sparse_query_frequency_down_sample_steps = int(
        cfg.task.dataset.sparse_query_frequency_down_sample_steps
    )
    sampler_stride = compute_sampler_stride(
        sparse_execution_horizon=args.sparse_execution_horizon,
        sparse_action_down_sample_steps=sparse_action_down_sample_steps,
        sparse_query_frequency_down_sample_steps=sparse_query_frequency_down_sample_steps,
    )
    print(
        f"[export] MPC sparse_execution_horizon={args.sparse_execution_horizon}, "
        f"sampler_stride={sampler_stride}, "
        f"enable_ensemble_chunking={args.enable_ensemble_chunking}, "
        f"ensemble_decay_m={args.ensemble_decay_m}"
    )

    grouped = group_sampler_indices_by_episode(val_dataset.sampler.indices)
    if not grouped:
        raise RuntimeError("Validation sampler has no indices. Check val_ratio/seed.")

    requested = set(args.episodes) if args.episodes else None
    exported_episode_names: List[str] = []
    all_metrics: Dict[str, Dict[str, float]] = {}
    all_queries: Dict[str, List[Dict[str, Any]]] = {}

    for epi_id in sorted(grouped.keys()):
        source_name = f"episode_{epi_id}"
        if requested is not None and source_name not in requested:
            continue
        if source_name not in val_episode_names:
            # Still allow if it appears in sampler (should match).
            print(
                f"[export] Warning: {source_name} in sampler but not in val_mask listing."
            )

        out_name = f"{source_name}_policy"
        print(f"[export] Rolling out {source_name} -> {out_name}")
        episode_data, metrics, query_records = rollout_episode_open_loop(
            policy=policy,
            val_dataset=val_dataset,
            episode_sample_entries=grouped[epi_id],
            shape_meta=shape_meta,
            id_list=id_list,
            sparse_execution_horizon=args.sparse_execution_horizon,
            sampler_stride=sampler_stride,
            raw_time_step_s=args.raw_time_step_s,
            device=device,
            enable_ensemble_chunking=bool(args.enable_ensemble_chunking),
            ensemble_decay_m=float(args.ensemble_decay_m),
        )
        all_metrics[out_name] = metrics
        all_queries[out_name] = query_records
        print(
            f"[export] {out_name}: steps={int(metrics['num_steps'])}, "
            f"queries={int(metrics['num_queries'])}, "
            f"action_mse_mean={metrics['action_mse_mean']:.6g}, "
            f"{format_field_metrics(metrics)}"
        )

        if not args.dry_run_export:
            write_episode_to_zarr(output_path, out_name, episode_data)
            exported_episode_names.append(out_name)
        else:
            print(f"[export] dry_run_export: skipped write for {out_name}")
            exported_episode_names.append(out_name)

    if not exported_episode_names:
        raise RuntimeError(
            f"No episodes exported. Requested={args.episodes}, "
            f"available_val={val_episode_names}"
        )

    metrics_path = args.metrics_path
    if not metrics_path:
        metrics_path = os.path.join(
            os.path.dirname(output_path.rstrip("/")),
            "policy_open_loop_val_metrics.json",
        )
    queries_path = args.queries_path
    if not queries_path:
        queries_path = os.path.join(
            os.path.dirname(output_path.rstrip("/")),
            "policy_open_loop_val_queries.json",
        )
    if not args.dry_run_export:
        with open(metrics_path, "w") as f:
            json.dump(
                {
                    "checkpoint": ckpt_path,
                    "dataset_path": dataset_path,
                    "output_path": output_path,
                    "val_ratio": float(cfg.task.dataset.val_ratio),
                    "seed": int(cfg.task.dataset.seed),
                    "val_episodes": val_episode_names,
                    "sparse_execution_horizon": args.sparse_execution_horizon,
                    "sampler_stride": sampler_stride,
                    "enable_ensemble_chunking": bool(args.enable_ensemble_chunking),
                    "ensemble_decay_m": float(args.ensemble_decay_m),
                    "raw_time_step_s": float(args.raw_time_step_s),
                    "queries_path": queries_path,
                    "episodes": all_metrics,
                },
                f,
                indent=2,
            )
        with open(queries_path, "w") as f:
            json.dump(
                {
                    "checkpoint": ckpt_path,
                    "dataset_path": dataset_path,
                    "output_path": output_path,
                    "sparse_execution_horizon": args.sparse_execution_horizon,
                    "sampler_stride": sampler_stride,
                    "raw_time_step_s": float(args.raw_time_step_s),
                    "enable_ensemble_chunking": bool(args.enable_ensemble_chunking),
                    "ensemble_decay_m": float(args.ensemble_decay_m),
                    "episodes": all_queries,
                },
                f,
                indent=2,
            )
        print(f"[export] Wrote metrics: {metrics_path}")
        print(f"[export] Wrote query metadata: {queries_path}")
        print(f"[export] Wrote zarr: {output_path}")
    else:
        print("[export] dry_run_export complete (no zarr/metrics written).")

    if args.run_playback:
        if args.dry_run_export:
            raise RuntimeError("Cannot --run_playback with --dry_run_export (no zarr).")
        for ep_name in exported_episode_names:
            rc = run_playback_for_episode(
                output_path=output_path,
                episode_name=ep_name,
                playback_dry_run=args.playback_dry_run,
                hardware_config_path=args.hardware_config_path,
                execution_horizon=args.playback_execution_horizon,
            )
            if rc != 0:
                print(f"[export] Playback failed for {ep_name} with code {rc}")
                return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

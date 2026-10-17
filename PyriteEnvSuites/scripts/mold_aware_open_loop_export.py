#!/usr/bin/env python3
"""Open-loop val export for the mold-aware ablation runs (raw chunks, no decode).

For every val query (MPC stride = sparse_execution_horizon) this stores the raw
full-horizon policy output, the training label, and the robot-sample index each
horizon step refers to. Decoding to fixture-frame poses happens downstream in
mold_aware_open_loop_metrics.py, because surface-frame actions need the
per-timestep feedback pose and surface frame to decode.

Needs the surface_frames code (robot0_mold_geom obs, absolute pose frames) on
PYTHONPATH. Depth is fed as recorded (no zero-depth ablation).

Usage:
  WT=/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames
  PYTHONPATH=$WT:$WT/PyriteUtility:$WT/PyriteML:$WT/PyriteConfig \
  python mold_aware_open_loop_export.py \
    --checkpoint latest=/mnt/.../mold_aware/world_world/latest.ckpt \
    --checkpoint ep40=/mnt/.../mold_aware/world_world/epoch=0040-train_loss=0.026.ckpt \
    --dataset_path /mnt/data/umidata/ft/mold_aware_datasets/stringer_data_v2_full/world_world/acp_replay_buffer.zarr \
    --out_dir /mnt/data/umidata/ft/training_outputs/mold_aware/eval/exports/world_world
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from PyriteML.diffusion_policy.common.sampler_umift import (
    sparse_action_indices_from_rgb_timestamps,
)
from PyriteML.diffusion_policy.workspace.base_workspace import BaseWorkspace


def parse_checkpoint_arg(value: str) -> Tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"expected LABEL=PATH, got {value!r}")
    label, path = value.split("=", 1)
    return label.strip(), path.strip()


def load_policy_and_cfg(ckpt_path: str, device: torch.device):
    payload = torch.load(open(ckpt_path, "rb"), map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    workspace: BaseWorkspace = hydra.utils.get_class(cfg._target_)(cfg)
    workspace.load_payload(payload, exclude_keys=["optimizer"], include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    if hasattr(cfg, "policy") and hasattr(cfg.policy, "num_inference_steps"):
        policy.num_inference_steps = cfg.policy.num_inference_steps
    policy.eval().to(device)
    policy.reset()
    return policy, cfg


def stack_obs(samples: List[Dict], device: torch.device) -> Dict:
    out = {}
    for key in samples[0]:
        if isinstance(samples[0][key], dict):
            out[key] = stack_obs([s[key] for s in samples], device)
        else:
            out[key] = torch.as_tensor(
                np.stack([np.asarray(s[key]) for s in samples])
            ).to(device)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", action="append", type=parse_checkpoint_arg, required=True)
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--episodes", nargs="*", default=None)
    ap.add_argument("--sparse_execution_horizon", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    dataset = val_dataset = None
    grouped: Dict[int, List[int]] = {}
    for label, ckpt_path in args.checkpoint:
        print(f"[export] {label}: loading {ckpt_path}")
        policy, cfg = load_policy_and_cfg(ckpt_path, device)
        shape_meta = cfg.task.shape_meta
        H = int(shape_meta["sample"]["action"]["sparse"]["horizon"])
        rgb_stride = int(shape_meta["sample"]["action"]["sparse"]["down_sample_steps"])
        query_ds = int(cfg.task.dataset.sparse_query_frequency_down_sample_steps)
        sampler_stride = max(
            1, int(round(args.sparse_execution_horizon * rgb_stride / float(query_ds)))
        )

        if dataset is None:
            OmegaConf.update(cfg, "task.dataset.dataset_path", args.dataset_path, merge=False)
            print(f"[export] building dataset from {args.dataset_path}")
            dataset = hydra.utils.instantiate(cfg.task.dataset)
            val_dataset = dataset.get_validation_dataset()
            by_ep = defaultdict(list)
            for gidx, (epi_id, _rgb_id) in enumerate(val_dataset.sampler.indices):
                by_ep[int(epi_id)].append(gidx)
            grouped = dict(by_ep)
            print(f"[export] val episodes: {sorted(grouped)}")

        ckpt_out = os.path.join(args.out_dir, label)
        os.makedirs(ckpt_out, exist_ok=True)
        sampler = val_dataset.sampler
        for epi_id in sorted(grouped):
            name = f"episode_{epi_id}"
            if args.episodes and name not in args.episodes:
                continue
            t0 = time.time()
            torch.manual_seed(args.seed + epi_id)
            entries = grouped[epi_id][::sampler_stride]
            ep = sampler.replay_buffer["data"][name]
            robot_t = np.squeeze(np.asarray(ep["obs"]["robot_time_stamps_0"]))
            action_t = np.squeeze(np.asarray(ep["action_time_stamps"]))
            rgb_t = np.squeeze(np.asarray(ep["obs"]["rgb_time_stamps_0"]))

            raw, gt, act_idx, n_valid, q_robot, q_time, rgb_ids, tgt_times = (
                [], [], [], [], [], [], [], []
            )
            for b in range(0, len(entries), args.batch_size):
                batch = entries[b : b + args.batch_size]
                obs_list = []
                for gidx in batch:
                    obs_sample, action_sample = sampler.sample_sequence(gidx)
                    obs_list.append(obs_sample)
                    gt.append(np.asarray(action_sample["sparse"], dtype=np.float32))
                    _e, rgb_id = sampler.indices[gidx]
                    qt = float(rgb_t[rgb_id])
                    idx, tt = sparse_action_indices_from_rgb_timestamps(
                        action_times=action_t,
                        rgb_times=rgb_t,
                        rgb_id=int(rgb_id),
                        action_horizon=H,
                        rgb_frame_stride=rgb_stride,
                    )
                    nv = len(idx)
                    pad = H - nv
                    if pad > 0:
                        idx = np.concatenate([idx, np.repeat(idx[-1:], pad)])
                        tt = np.concatenate([tt, np.repeat(tt[-1:], pad)])
                    act_idx.append(idx.astype(np.int64))
                    tgt_times.append(tt.astype(np.float64))
                    n_valid.append(nv)
                    q_robot.append(
                        int(np.clip(np.searchsorted(robot_t, qt), 0, len(robot_t) - 1))
                    )
                    q_time.append(qt)
                    rgb_ids.append(int(rgb_id))
                with torch.no_grad():
                    result = policy.predict_action(stack_obs(obs_list, device))
                raw.extend(result["sparse"].float().cpu().numpy())

            np.savez_compressed(
                os.path.join(ckpt_out, f"{name}.npz"),
                raw_action=np.stack(raw).astype(np.float32),
                gt_action=np.stack(gt).astype(np.float32),
                action_robot_idx=np.stack(act_idx),
                action_target_time_s=np.stack(tgt_times),
                n_valid=np.asarray(n_valid, dtype=np.int64),
                query_robot_idx=np.asarray(q_robot, dtype=np.int64),
                query_time_s=np.asarray(q_time, dtype=np.float64),
                rgb_id=np.asarray(rgb_ids, dtype=np.int64),
            )
            print(
                f"[export] {label} {name}: {len(entries)} queries in {time.time() - t0:.1f}s"
            )

        with open(os.path.join(ckpt_out, "meta.json"), "w") as f:
            json.dump(
                {
                    "checkpoint": ckpt_path,
                    "dataset_path": args.dataset_path,
                    "task_name": str(cfg.task.name),
                    "sparse_execution_horizon": args.sparse_execution_horizon,
                    "sampler_stride": sampler_stride,
                    "action_horizon": H,
                    "seed": args.seed,
                    "num_inference_steps": int(getattr(policy, "num_inference_steps", -1)),
                    "val_episodes": [f"episode_{e}" for e in sorted(grouped)],
                },
                f,
                indent=2,
            )
        del policy
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

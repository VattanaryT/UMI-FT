#!/usr/bin/env python3
"""Open-loop, recorded RGB, episodes 9, 12, 28, 37, 41.

v3r position is the field-v2 step. v3r rotation is R_tcp @ ΔR.
r1rel is its relative pose composed with the measured TCP.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import compare_r1rel_b as base

TRAIN = Path("/mnt/data/umidata/ft/training_outputs")
CKPT = {
    "v3r": TRAIN / "oct_v35_h4s_r4vt_field_v3r/latest_final.ckpt",
    "r1rel": TRAIN / "oct_v35_h4s_odom_r1rel/checkpoints/latest.ckpt",
}
V3R_SCRIPTS = Path(
    "/home/boeing/umi_ft_ws/worktrees/UMI-FT-field_v3r/UMIFT_Data/surface_frames_scripts"
)
OUT = base.ROOT / "closedloop_sim" / "out_v3r_r1rel_rgb"


def _check() -> None:
    tcp = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    frame = np.diag([1.0, 1.0, -1.0])
    action = np.zeros(10)
    action[4:10] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    _p, R, dn = _world_v3r(action, np.zeros(3), frame, tcp, 1.0 / 15.0)
    if dn != 0.0 or not np.allclose(R, tcp):
        raise RuntimeError((dn, R))


def _world_v3r(action, foot, frame, tcp_R, dt):
    position, _mold_R, dn = base._world_r4(action, foot, frame, dt)
    step = np.asarray(action, dtype=np.float64).reshape(-1)
    dR = base.rot6d_rows_to_R(step[4:10].reshape(1, 6))[0]
    return position, np.asarray(tcp_R, dtype=np.float64) @ dR, dn


def _load(ckpt, device):
    import dill
    import hydra
    import torch
    from diffusion_policy.workspace.base_workspace import BaseWorkspace

    payload = torch.load(open(ckpt, "rb"), map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    workspace: BaseWorkspace = hydra.utils.get_class(cfg._target_)(cfg)
    workspace.load_payload(payload, exclude_keys=["optimizer"], include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    if hasattr(cfg, "policy") and hasattr(cfg.policy, "num_inference_steps"):
        policy.num_inference_steps = cfg.policy.num_inference_steps
    policy.eval().to(device)
    policy.reset()
    epoch = dill.loads(payload["pickles"]["epoch"])
    return policy, cfg, int(epoch)


def _open_one(name, ckpt, device) -> dict:
    import hydra
    import torch
    import zarr
    from PyriteEnvSuites.scripts.run_policy_open_loop_val_export import predict_sparse_action

    policy, cfg, epoch = _load(ckpt, device)
    base._point_dataset(cfg)
    cfg.task.dataset.exclusion_stats_path = str(OUT / f"{name}_exclusion.json")
    target = str(cfg.task.dataset._target_)
    if "field_v3" in target:
        sys.path.insert(0, str(V3R_SCRIPTS))
    rgb_keys = list(getattr(policy.obs_encoder, "rgb_keys", []))
    print(
        f"[open] {name} {ckpt} epoch {epoch} planned {int(cfg.training.num_epochs)} "
        f"target {target} rgb {rgb_keys}",
        flush=True,
    )
    if "rgb_0" not in rgb_keys:
        raise SystemExit(f"{name} has no rgb encoder")
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    val = dataset.get_validation_dataset()
    root = zarr.open_group(str(cfg.task.dataset.dataset_path), mode="r")["data"]
    ds = int(cfg.task.shape_meta["sample"]["action"]["sparse"]["down_sample_steps"])
    dt = ds / 15.0
    v3r = "field_v3" in target
    keep = []
    last = {}
    rgb_t = {}
    for idx, (ep, rgb) in enumerate(val.sampler.indices):
        ep = int(ep)
        if ep not in base.EPISODES:
            continue
        if ep not in rgb_t:
            rgb_t[ep] = np.asarray(root[f"episode_{ep}"]["rgb_time_stamps_0"]).reshape(-1)
        t = float(rgb_t[ep][int(rgb)] - rgb_t[ep][0])
        if t - last.get(ep, -1e9) < 0.5:
            continue
        last[ep] = t
        keep.append((idx, ep, t))
    print(f"[open] {name} queries {len(keep)}", flush=True)
    buckets = {
        phase: {"position_mm": [], "rotation_deg": [], "delta_n_mm": []}
        for phase in ("approach", "contact", "lift-off")
    }
    phased = {}
    for n, (idx, ep, t) in enumerate(keep):
        group = root[f"episode_{ep}"]
        if ep not in phased:
            robot_t = np.asarray(group["robot_time_stamps_0"]).reshape(-1)
            robot_t = robot_t - robot_t[0]
            pose = np.asarray(group["ts_pose_fb_0"], dtype=np.float64)
            _rot, pos = base.pose7_to_Rp(pose)
            frame_all = np.asarray(group["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3)
            foot_all = np.asarray(group["mold_x_s_0"], dtype=np.float64)
            wrench = np.asarray(group["wrench_left_0"], dtype=np.float64)
            key = "wrench_time_stamps_left_0" if "wrench_time_stamps_left_0" in group else "wrench_time_stamps_0"
            wt = np.asarray(group[key]).reshape(-1) 
            wt = wt - wt[0]
            j = np.clip(np.searchsorted(wt, robot_t), 0, len(wt) - 1)
            force = np.linalg.norm(wrench[j, :3], axis=1)
            clearance = np.einsum("ij,ij->i", pos - foot_all, -frame_all[:, :, 2])
            phased[ep] = (robot_t, base._phases(force, clearance), frame_all, foot_all, pose)
        robot_t, labels, frame_all, foot_all, pose = phased[ep]
        sample = val[idx]
        sparse = {
            key: value.numpy()
            for key, value in sample["obs"]["sparse"].items()
            if not str(key).startswith("rgb") or key in rgb_keys
        }
        pred = np.asarray(predict_sparse_action(policy, {"sparse": sparse}, device), dtype=np.float64)
        gt = sample["action"]["sparse"].numpy()
        k = int(np.clip(np.searchsorted(robot_t, t), 1, len(robot_t) - 1))
        k = k - 1 if abs(robot_t[k - 1] - t) <= abs(robot_t[k] - t) else k
        rotation, position = base.pose7_to_Rp(pose[k : k + 1])
        frame = frame_all[k]
        foot = foot_all[k]
        if v3r:
            p_gt, R_gt, dn_gt = _world_v3r(gt[0], foot, frame, rotation[0], dt)
            p_hat, R_hat, dn_hat = _world_v3r(pred[0], foot, frame, rotation[0], dt)
            dn = abs(dn_hat - dn_gt) * 1e3
        else:
            p_gt, R_gt = base._world_r1(gt, position[0], rotation[0])
            p_hat, R_hat = base._world_r1(pred, position[0], rotation[0])
            dn = abs(float(np.dot(p_hat - p_gt, -frame[:, 2]))) * 1e3
        buckets[labels[k]]["position_mm"].append(float(np.linalg.norm(p_hat - p_gt) * 1e3))
        buckets[labels[k]]["rotation_deg"].append(base._angle(R_hat, R_gt))
        buckets[labels[k]]["delta_n_mm"].append(dn)
        if n and n % 40 == 0:
            print(f"[open] {name} {n}/{len(keep)}", flush=True)
    del policy, dataset, val
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    phases = {}
    for phase, cols in buckets.items():
        phases[phase] = {key: base._stat(values) for key, values in cols.items()}
        rot = np.asarray(cols["rotation_deg"], dtype=np.float64)
        phases[phase]["rotation_gt10"] = int(np.sum(rot > 10.0))
    return {
        "ckpt": str(ckpt),
        "epoch": epoch,
        "planned_epochs": int(cfg.training.num_epochs),
        "n": len(keep),
        "v3r": v3r,
        "dt_s": dt,
        "phases": phases,
    }


def main() -> None:
    import torch

    base._check()
    _check()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opened = {}
    for name, ckpt in CKPT.items():
        opened[name] = _open_one(name, ckpt, device)
        (OUT / "open_loop.json").write_text(json.dumps(opened, indent=2) + "\n")
    print(json.dumps(opened, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Open-loop chunk error for heading-tilt job 43448962.

Position comes from integrate_tcp_foot (v1, v2, δn). Rotation comes from
heading_tilt.rotation_from_tilt. The 27-d rot6d decode is not used.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

ROOT = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames")
SCRIPTS = ROOT / "UMIFT_Data/surface_frames_scripts"
sys.path[:0] = [
    str(SCRIPTS),
    str(ROOT / "PyriteEnvSuites/scripts"),
    str(ROOT / "PyriteML"),
    str(ROOT / "PyriteConfig"),
    str(ROOT / "PyriteUtility"),
    str(ROOT),
]

from diffusion_policy.common.drop_depth_inputs import apply_drop_depth_inputs
from diffusion_policy.common.drop_scan_inputs import apply_drop_scan_inputs
from fixb_field import FixBMoldField
from heading_tilt import e_long_sequence, geodesic_deg, long_axis, rotation_from_tilt
from mold_aware_runtime import integrate_tcp_foot, rot6d_rows_to_R
from run_policy_open_loop_val_export import load_policy_and_cfg, predict_sparse_action

EPISODES = (9, 12, 28, 37, 41)
STRIDE = 64
ZARR = (
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "r4vt_heading_tilt/acp_replay_buffer.zarr"
)
LONG_DT = (
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "shared/long_dt_gt_100ms.npz"
)
NORM = SCRIPTS / "assets/oct_v35_h4s_r4vt_action_normalizer_minmax_heading_tilt_swing_sc.pkl"
FIELD = (
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz"
)
OUT = SCRIPTS / "eval/heading_tilt_43448962_open_loop.json"
RUN_DIR = (
    "/mnt/data/umidata/ft/training_outputs/"
    "oct-v35-h4s-r4vt-heading-tilt_sc_e200_bs64_valtopk_43448962/checkpoints"
)
RUNS = {
    "e25": f"{RUN_DIR}/epoch=0025-val_loss=0.059.ckpt",
    "latest": f"{RUN_DIR}/latest.ckpt",
}


def hydra_instantiate(node):
    import hydra

    return hydra.utils.instantiate(node)


def build_dataset(cfg):
    ds_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    apply_drop_scan_inputs(ds_cfg)
    apply_drop_depth_inputs(ds_cfg)
    OmegaConf.update(ds_cfg, "task.dataset_path", ZARR, merge=False)
    OmegaConf.update(ds_cfg, "task.dataset.dataset_path", ZARR, merge=False)
    OmegaConf.update(ds_cfg, "task.dataset.action_normalizer_path", str(NORM), merge=False)
    OmegaConf.update(ds_cfg, "task.dataset.long_dt_mask_path", LONG_DT, merge=False)
    OmegaConf.update(ds_cfg, "task.dataset.exclusion_stats_path", None, merge=False)
    sha = str(ds_cfg.task.dataset.action_normalizer_sha256)
    if not sha.startswith("e93b8756"):
        raise SystemExit(f"normalizer sha is {sha}")
    if int(ds_cfg.task.shape_meta.action.shape[0]) != 8:
        raise SystemExit(f"action dim {ds_cfg.task.shape_meta.action.shape}")
    return hydra_instantiate(ds_cfg.task.dataset)


def query_indices(dataset) -> dict[int, list[int]]:
    chosen: dict[int, list[int]] = {ep: [] for ep in EPISODES}
    last: dict[int, int] = {}
    for idx, (epi, rgb) in enumerate(dataset.sampler.indices):
        epi = int(epi)
        if epi not in chosen:
            continue
        rgb = int(rgb)
        if epi not in last or rgb - last[epi] >= STRIDE:
            chosen[epi].append(idx)
            last[epi] = rgb
    missing = [ep for ep, ids in chosen.items() if not ids]
    if missing:
        raise SystemExit(f"no sampler queries for episodes {missing}")
    return chosen


def swing(ab: np.ndarray) -> np.ndarray:
    rotvec = np.column_stack([ab[:, 0], ab[:, 1], np.zeros(len(ab))])
    return Rotation.from_rotvec(rotvec).as_matrix()


def wrap_deg(delta: np.ndarray) -> np.ndarray:
    return (delta + 180.0) % 360.0 - 180.0


def decode_pose(field, axis, action, p0, R0):
    dt = np.full(action.shape[0], 1.0 / 15.0)
    foot = integrate_tcp_foot(field, p0, action[:, 0:2], action[:, 2], dt, R_tool=R0)
    e_long, _angle = e_long_sequence(foot["n_out"], axis)
    rotation = rotation_from_tilt(foot["n_out"], e_long, action[:, 4:8])
    return foot["x_d"], rotation


def score(field, axis, pred, gt, p0, R0):
    x_p, r_p = decode_pose(field, axis, pred, p0, R0)
    x_g, r_g = decode_pose(field, axis, gt, p0, R0)
    pos = np.linalg.norm(x_p - x_g, axis=1) * 1e3
    rot = geodesic_deg(r_p, r_g)
    psi_p = np.degrees(np.arctan2(pred[:, 4], pred[:, 5]))
    psi_g = np.degrees(np.arctan2(gt[:, 4], gt[:, 5]))
    head = np.abs(wrap_deg(psi_p - psi_g))
    tilt = geodesic_deg(swing(pred[:, 6:8]), swing(gt[:, 6:8]))
    stiff = np.abs(pred[:, 3] - gt[:, 3])
    norm = pred[:, 4] ** 2 + pred[:, 5] ** 2
    return {
        "pos0": float(pos[0]),
        "pos": float(pos.mean()),
        "rot0": float(rot[0]),
        "rot": float(rot.mean()),
        "head0": float(head[0]),
        "head": float(head.mean()),
        "tilt0": float(tilt[0]),
        "tilt": float(tilt.mean()),
        "stiff0": float(stiff[0]),
        "stiff": float(stiff.mean()),
        "sincos_mean": float(norm.mean()),
        "sincos_min": float(norm.min()),
    }


def mean_of(rows, key):
    return float(np.mean([row[key] for row in rows]))


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    field = FixBMoldField(Path(FIELD))
    axis = long_axis(field.vertices)
    first_name, first_path = next(iter(RUNS.items()))
    policy, cfg = load_policy_and_cfg(first_path, device)
    dataset = build_dataset(cfg).get_validation_dataset()
    queries = query_indices(dataset)
    for ep, ids in queries.items():
        print(f"episode {ep}: {len(ids)} windows", flush=True)
    report = {}
    held = {first_name: policy}
    for name, path in RUNS.items():
        policy = held.pop(name) if name in held else load_policy_and_cfg(path, device)[0]
        per_ep = {}
        norms = []
        for ep, ids in queries.items():
            rows = []
            for idx in ids:
                obs, action = dataset.sampler.sample_sequence(idx)
                pred = np.asarray(predict_sparse_action(policy, obs, device), dtype=np.float64)
                gt = np.asarray(action["sparse"], dtype=np.float64)
                if pred.shape[-1] != 8 or gt.shape[-1] != 8:
                    raise SystemExit(f"{name} action {pred.shape} vs {gt.shape}")
                pred = pred.reshape(-1, 8)
                gt = gt.reshape(-1, 8)
                n = min(len(pred), len(gt))
                sparse = obs["sparse"]
                p0 = np.asarray(sparse["robot0_eef_pos"][-1], dtype=np.float64).reshape(3)
                R0 = rot6d_rows_to_R(
                    np.asarray(sparse["robot0_eef_rot_axis_angle"][-1], dtype=np.float64)
                )
                row = score(field, axis, pred[:n], gt[:n], p0, R0)
                norms.append((row.pop("sincos_mean"), row.pop("sincos_min"), n))
                rows.append(row)
            per_ep[str(ep)] = {key: mean_of(rows, key) for key in rows[0]}
            per_ep[str(ep)]["n"] = len(rows)
            print(name, ep, per_ep[str(ep)], flush=True)
        keys = [k for k in per_ep["9"] if k != "n"]
        per_ep["mean"] = {key: float(np.mean([per_ep[str(ep)][key] for ep in EPISODES])) for key in keys}
        weight = np.array([item[2] for item in norms], dtype=np.float64)
        per_ep["sincos_mean"] = float(np.average([item[0] for item in norms], weights=weight))
        per_ep["sincos_min"] = float(min(item[1] for item in norms))
        report[name] = per_ep
        del policy
        torch.cuda.empty_cache()
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()

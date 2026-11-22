#!/usr/bin/env python3
"""Same open-loop protocol as heading_tilt_43448962_open_loop.py.

Causal-foot job 43390127 is the 10-d reduction of the 27-d layout
(v1, v2, δn, stiffness, ΔR rot6d). Its rotation is R_field @ ΔR, then
pack_tilt so ψ and tilt use the same columns as the 8-d model.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import zarr
from omegaconf import OmegaConf

ROOT = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames")
SCRIPTS = ROOT / "UMIFT_Data/surface_frames_scripts"
sys.path[:0] = [
    str(SCRIPTS),
    str(ROOT / "PyriteEnvSuites/scripts"),
    str(ROOT / "PyriteML"),
    str(ROOT / "PyriteConfig"),
    str(ROOT / "PyriteUtility"),
    str(ROOT),
    str(SCRIPTS / "eval"),
]

from diffusion_policy.common.drop_depth_inputs import apply_drop_depth_inputs
from diffusion_policy.common.drop_scan_inputs import apply_drop_scan_inputs
from fixb_field import FixBMoldField
from heading_tilt import e_long_sequence, geodesic_deg, long_axis, pack_tilt
from heading_tilt_43448962_open_loop import (
    EPISODES,
    FIELD,
    LONG_DT,
    STRIDE,
    mean_of,
    query_indices,
    swing,
    wrap_deg,
)
from mold_aware_runtime import integrate_tcp_foot, rot6d_rows_to_R
from run_policy_open_loop_val_export import load_policy_and_cfg, predict_sparse_action

CAUSAL_ZARR = (
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "r4vt_causal_foot/acp_replay_buffer.zarr"
)
CAUSAL_NORM = SCRIPTS / "assets/oct_v35_h4s_r4vt_action_normalizer_minmax_causal_foot.pkl"
CAUSAL_DIR = (
    "/mnt/data/umidata/ft/training_outputs/oct_v35_h4s_r4vt_causal_foot/"
    "oct-v35-h4s-r4vt-causal-foot_e200_bs64_valtopk_43390127/checkpoints"
)
HEAD_ZARR = (
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "r4vt_heading_tilt/acp_replay_buffer.zarr"
)
HEAD_NORM = SCRIPTS / "assets/oct_v35_h4s_r4vt_action_normalizer_minmax_heading_tilt_swing_sc.pkl"
HEAD_LATEST = (
    "/mnt/data/umidata/ft/training_outputs/"
    "oct-v35-h4s-r4vt-heading-tilt_sc_e200_bs64_valtopk_43448962/checkpoints/latest.ckpt"
)
OUT = SCRIPTS / "eval/heading_tilt_vs_causal_foot.json"


def hydra_instantiate(node):
    import hydra

    return hydra.utils.instantiate(node)


def build_dataset(cfg, zarr_path, norm, sha_prefix, action_dim):
    ds_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    apply_drop_scan_inputs(ds_cfg)
    apply_drop_depth_inputs(ds_cfg)
    OmegaConf.update(ds_cfg, "task.dataset_path", zarr_path, merge=False)
    OmegaConf.update(ds_cfg, "task.dataset.dataset_path", zarr_path, merge=False)
    OmegaConf.update(ds_cfg, "task.dataset.action_normalizer_path", str(norm), merge=False)
    OmegaConf.update(ds_cfg, "task.dataset.long_dt_mask_path", LONG_DT, merge=False)
    OmegaConf.update(ds_cfg, "task.dataset.exclusion_stats_path", None, merge=False)
    sha = str(ds_cfg.task.dataset.action_normalizer_sha256)
    dim = int(ds_cfg.task.shape_meta.action.shape[0])
    if not sha.startswith(sha_prefix) or dim != action_dim:
        raise SystemExit(f"sha {sha} dim {dim}, expected {sha_prefix} dim {action_dim}")
    return hydra_instantiate(ds_cfg.task.dataset)


def anchor(obs):
    sparse = obs["sparse"]
    p0 = np.asarray(sparse["robot0_eef_pos"][-1], dtype=np.float64).reshape(3)
    r0 = rot6d_rows_to_R(np.asarray(sparse["robot0_eef_rot_axis_angle"][-1], dtype=np.float64))
    return p0, r0


def foot_of(field, action, p0, r0):
    dt = np.full(action.shape[0], 1.0 / 15.0)
    return integrate_tcp_foot(field, p0, action[:, 0:2], action[:, 2], dt, R_tool=r0)


def rotation_of_rot6d(field, action, p0, r0):
    foot = foot_of(field, action, p0, r0)
    dR = rot6d_rows_to_R(action[:, 4:10])
    rotation = np.einsum("hij,hjk->hik", foot["R"], dR)
    return foot["x_d"], rotation, foot["n_out"]


def pair_metrics(x_p, r_p, pack_p, x_g, r_g, pack_g, stiff_p, stiff_g):
    pos = np.linalg.norm(x_p - x_g, axis=1) * 1e3
    rot = geodesic_deg(r_p, r_g)
    head = np.abs(wrap_deg(np.degrees(np.arctan2(pack_p[:, 0], pack_p[:, 1]) - np.arctan2(pack_g[:, 0], pack_g[:, 1]))))
    tilt = geodesic_deg(swing(pack_p[:, 2:4]), swing(pack_g[:, 2:4]))
    stiff = np.abs(stiff_p - stiff_g)
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
    }


def summarize(per_ep):
    keys = [k for k in per_ep["9"] if k != "n"]
    per_ep["mean"] = {key: float(np.mean([per_ep[str(ep)][key] for ep in EPISODES])) for key in keys}
    return per_ep


def run_causal(field, axis, device):
    runs = {
        "e25": f"{CAUSAL_DIR}/epoch=0025-val_loss=0.058.ckpt",
        "latest": f"{CAUSAL_DIR}/latest.ckpt",
    }
    first = next(iter(runs))
    policy, cfg = load_policy_and_cfg(runs[first], device)
    dataset = build_dataset(cfg, CAUSAL_ZARR, CAUSAL_NORM, "14308b72", 10).get_validation_dataset()
    queries = query_indices(dataset)
    held = {first: policy}
    report = {}
    for name, path in runs.items():
        policy = held.pop(name) if name in held else load_policy_and_cfg(path, device)[0]
        per_ep = {}
        for ep, ids in queries.items():
            rows = []
            for idx in ids:
                obs, action = dataset.sampler.sample_sequence(idx)
                pred = np.asarray(predict_sparse_action(policy, obs, device), dtype=np.float64).reshape(-1, 10)
                gt = np.asarray(action["sparse"], dtype=np.float64).reshape(-1, 10)
                if pred.shape[-1] != 10 or gt.shape[-1] != 10:
                    raise SystemExit(f"{name} action {pred.shape} vs {gt.shape}")
                n = min(len(pred), len(gt))
                p0, r0 = anchor(obs)
                xp, rp, _np = rotation_of_rot6d(field, pred[:n], p0, r0)
                xg, rg, ng = rotation_of_rot6d(field, gt[:n], p0, r0)
                e_long, _angle = e_long_sequence(ng, axis)
                pp = pack_tilt(rp, ng, e_long)
                pg = pack_tilt(rg, ng, e_long)
                rows.append(pair_metrics(xp, rp, pp, xg, rg, pg, pred[:n, 3], gt[:n, 3]))
            per_ep[str(ep)] = {key: mean_of(rows, key) for key in rows[0]}
            per_ep[str(ep)]["n"] = len(rows)
            print("causal", name, ep, per_ep[str(ep)], flush=True)
        report[name] = summarize(per_ep)
        del policy
        torch.cuda.empty_cache()
    return report


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def label_span():
    """a, b percentiles on the heading-tilt array, episodes 35 and 44 left out."""
    root = zarr.open(HEAD_ZARR, mode="r")
    data = root["data"] if "data" in root else root
    chunks = []
    for key in data.keys():
        epi = int(str(key).split("_")[-1])
        if epi in (35, 44):
            continue
        tilt = np.asarray(data[key]["fixb_heading_tilt_0"], dtype=np.float64).reshape(-1, 4)
        chunks.append(tilt[:, 2:4])
    ab = np.concatenate(chunks, axis=0)
    with HEAD_NORM.open("rb") as handle:
        saved = pickle.load(handle)
    stats = saved.params_dict["action"]["input_stats"]
    fit_min = stats["min"].detach().float().cpu().reshape(-1).numpy()
    fit_max = stats["max"].detach().float().cpu().reshape(-1).numpy()
    out = {}
    for index, name in ((6, "a"), (7, "b")):
        column = ab[:, index - 6]
        pct = np.percentile(column, [1, 5, 50, 95, 99])
        span = float(fit_max[index] - fit_min[index])
        out[name] = {
            "fit_min_deg": float(np.degrees(fit_min[index])),
            "fit_max_deg": float(np.degrees(fit_max[index])),
            "data_p01_deg": float(np.degrees(pct[0])),
            "data_p05_deg": float(np.degrees(pct[1])),
            "data_p50_deg": float(np.degrees(pct[2])),
            "data_p95_deg": float(np.degrees(pct[3])),
            "data_p99_deg": float(np.degrees(pct[4])),
            "mid90_over_fit_span": float(np.degrees(pct[3] - pct[1]) / np.degrees(span)),
            "n": int(column.size),
        }
    return out


def run_tilt(field, axis, device):
    policy, cfg = load_policy_and_cfg(HEAD_LATEST, device)
    dataset = build_dataset(cfg, HEAD_ZARR, HEAD_NORM, "e93b8756", 8).get_validation_dataset()
    queries = query_indices(dataset)
    per_ep = {}
    signed_a = []
    signed_b = []
    label_mag = []
    pred_mag = []
    for ep, ids in queries.items():
        err_a = []
        err_b = []
        tilt_rows = []
        for idx in ids:
            obs, action = dataset.sampler.sample_sequence(idx)
            pred = np.asarray(predict_sparse_action(policy, obs, device), dtype=np.float64).reshape(-1, 8)
            gt = np.asarray(action["sparse"], dtype=np.float64).reshape(-1, 8)
            n = min(len(pred), len(gt))
            pred, gt = pred[:n], gt[:n]
            da = pred[:, 6] - gt[:, 6]
            db = pred[:, 7] - gt[:, 7]
            err_a.append(da)
            err_b.append(db)
            signed_a.append(da)
            signed_b.append(db)
            label_mag.append(np.linalg.norm(gt[:, 6:8], axis=1))
            pred_mag.append(np.linalg.norm(pred[:, 6:8], axis=1))
            tilt_rows.append(float(geodesic_deg(swing(pred[:, 6:8]), swing(gt[:, 6:8])).mean()))
        a = np.concatenate(err_a)
        b = np.concatenate(err_b)
        per_ep[str(ep)] = {
            "tilt": float(np.mean(tilt_rows)),
            "mae_a_deg": float(np.degrees(np.mean(np.abs(a)))),
            "mae_b_deg": float(np.degrees(np.mean(np.abs(b)))),
            "signed_a_deg": float(np.degrees(a.mean())),
            "signed_b_deg": float(np.degrees(b.mean())),
            "n": len(ids),
        }
        print("tilt", ep, per_ep[str(ep)], flush=True)
    a = np.concatenate(signed_a)
    b = np.concatenate(signed_b)
    report = {
        "per_episode": per_ep,
        "label_mag_deg": _stats(np.degrees(np.concatenate(label_mag))),
        "pred_mag_deg": _stats(np.degrees(np.concatenate(pred_mag))),
        "signed_a_deg": float(np.degrees(a.mean())),
        "signed_b_deg": float(np.degrees(b.mean())),
        "std_a_deg": float(np.degrees(a.std())),
        "std_b_deg": float(np.degrees(b.std())),
        "mae_a_deg": float(np.degrees(np.mean(np.abs(a)))),
        "mae_b_deg": float(np.degrees(np.mean(np.abs(b)))),
    }
    del policy
    torch.cuda.empty_cache()
    return report


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    field = FixBMoldField(Path(FIELD))
    axis = long_axis(field.vertices)
    span = label_span()
    print("normalizer span", span, flush=True)
    report = {
        "causal_43390127": run_causal(field, axis, device),
        "tilt_e100": run_tilt(field, axis, device),
        "normalizer_ab": span,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()

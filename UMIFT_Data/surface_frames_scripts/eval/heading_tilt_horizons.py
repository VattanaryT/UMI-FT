#!/usr/bin/env python3
"""Errors at steps 1, 6 and 64 for heading-tilt 43448962 and causal-foot e199.

Same windows as heading_tilt_43448962_open_loop.py: val episodes 9, 12, 28,
37, 41, RGB stride 64, mean of episode means. Causal rotation is decomposed
with pack_tilt, as in heading_tilt_vs_causal_foot.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames")
SCRIPTS = ROOT / "UMIFT_Data/surface_frames_scripts"
sys.path[:0] = [str(SCRIPTS), str(SCRIPTS / "eval")]

from heading_tilt import e_long_sequence, geodesic_deg, long_axis, pack_tilt
from heading_tilt_43448962_open_loop import (
    EPISODES,
    FIELD,
    decode_pose,
    query_indices,
    wrap_deg,
)
from heading_tilt_vs_causal_foot import (
    CAUSAL_DIR,
    CAUSAL_NORM,
    CAUSAL_ZARR,
    HEAD_NORM,
    HEAD_ZARR,
    build_dataset,
    rotation_of_rot6d,
)
from fixb_field import FixBMoldField
from run_policy_open_loop_val_export import load_policy_and_cfg, predict_sparse_action

HEAD_DIR = (
    "/mnt/data/umidata/ft/training_outputs/"
    "oct-v35-h4s-r4vt-heading-tilt_sc_e200_bs64_valtopk_43448962/checkpoints"
)
STEPS = ((0, "step1"), (5, "step6"), (63, "step64"))
OUT = SCRIPTS / "eval/heading_tilt_43448962_horizons.json"
MD = SCRIPTS / "eval/heading_tilt_43448962_horizons.md"
RUNS = (
    ("heading_e25", f"{HEAD_DIR}/epoch=0025-val_loss=0.059.ckpt", "heading"),
    ("heading_e100", f"{HEAD_DIR}/latest.ckpt", "heading"),
    ("heading_e199", f"{HEAD_DIR}/epoch=0199-val_loss=0.237.ckpt", "heading"),
    ("causal_e199", f"{CAUSAL_DIR}/latest.ckpt", "causal"),
)


def _psi_deg(sin_cos_or_pack_sin, cos):
    return np.degrees(np.arctan2(sin_cos_or_pack_sin, cos))


def _at(pos, rot, psi, a_deg, b_deg, stiff):
    if len(pos) < 64:
        raise SystemExit(f"chunk length {len(pos)}, need 64")
    out = {}
    for index, name in STEPS:
        out[name] = {
            "pos_mm": float(pos[index]),
            "rot_deg": float(rot[index]),
            "psi_deg": float(psi[index]),
            "a_deg": float(a_deg[index]),
            "b_deg": float(b_deg[index]),
            "stiff": float(stiff[index]),
        }
    return out


def heading_window(field, axis, pred, gt, p0, r0):
    xp, rp = decode_pose(field, axis, pred, p0, r0)
    xg, rg = decode_pose(field, axis, gt, p0, r0)
    pos = np.linalg.norm(xp - xg, axis=1) * 1e3
    rot = geodesic_deg(rp, rg)
    psi = np.abs(wrap_deg(_psi_deg(pred[:, 4], pred[:, 5]) - _psi_deg(gt[:, 4], gt[:, 5])))
    a_deg = np.degrees(np.abs(pred[:, 6] - gt[:, 6]))
    b_deg = np.degrees(np.abs(pred[:, 7] - gt[:, 7]))
    return _at(pos, rot, psi, a_deg, b_deg, np.abs(pred[:, 3] - gt[:, 3]))


def causal_window(field, axis, pred, gt, p0, r0):
    xp, rp, _normal = rotation_of_rot6d(field, pred, p0, r0)
    xg, rg, normal = rotation_of_rot6d(field, gt, p0, r0)
    e_long, _angle = e_long_sequence(normal, axis)
    packed_p = pack_tilt(rp, normal, e_long)
    packed_g = pack_tilt(rg, normal, e_long)
    pos = np.linalg.norm(xp - xg, axis=1) * 1e3
    rot = geodesic_deg(rp, rg)
    psi = np.abs(wrap_deg(_psi_deg(packed_p[:, 0], packed_p[:, 1]) - _psi_deg(packed_g[:, 0], packed_g[:, 1])))
    a_deg = np.degrees(np.abs(packed_p[:, 2] - packed_g[:, 2]))
    b_deg = np.degrees(np.abs(packed_p[:, 3] - packed_g[:, 3]))
    return _at(pos, rot, psi, a_deg, b_deg, np.abs(pred[:, 3] - gt[:, 3]))


def mean_rows(rows):
    out = {}
    for _index, name in STEPS:
        out[name] = {
            key: float(np.mean([row[name][key] for row in rows]))
            for key in rows[0][name]
        }
    return out


def score_run(field, axis, policy, dataset, queries, kind, device):
    width = 8 if kind == "heading" else 10
    window = heading_window if kind == "heading" else causal_window
    per_ep = {}
    for episode, ids in queries.items():
        rows = []
        for idx in ids:
            obs, action = dataset.sampler.sample_sequence(idx)
            pred = np.asarray(predict_sparse_action(policy, obs, device), dtype=np.float64).reshape(-1, width)
            gt = np.asarray(action["sparse"], dtype=np.float64).reshape(-1, width)
            n = min(len(pred), len(gt))
            sparse = obs["sparse"]
            p0 = np.asarray(sparse["robot0_eef_pos"][-1], dtype=np.float64).reshape(3)
            from mold_aware_runtime import rot6d_rows_to_R

            r0 = rot6d_rows_to_R(np.asarray(sparse["robot0_eef_rot_axis_angle"][-1], dtype=np.float64))
            rows.append(window(field, axis, pred[:n], gt[:n], p0, r0))
        per_ep[str(episode)] = mean_rows(rows)
        per_ep[str(episode)]["n"] = len(rows)
        print(kind, episode, "n", len(rows), per_ep[str(episode)]["step6"], flush=True)
    mean = {}
    for _index, name in STEPS:
        mean[name] = {
            key: float(np.mean([per_ep[str(ep)][name][key] for ep in EPISODES]))
            for key in per_ep["9"][name]
        }
    return {"mean": mean, "episodes": per_ep}


def write_md(report):
    lines = [
        "# heading-tilt 43448962 vs causal-foot 43390127",
        "",
        "Val episodes 9, 12, 28, 37, 41. RGB stride 64. Mean of episode means.",
        "Step 1, step 6, and step 64 of each 64-step chunk. Not the chunk mean.",
        "",
        "| run | step | pos mm | rot deg | ψ deg | a deg | b deg | stiff |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {"step1": 1, "step6": 6, "step64": 64}
    for name, blob in report.items():
        if "error" in blob:
            lines.append(f"| {name} |  | unreadable checkpoint | | | | | |")
            continue
        for key, step in labels.items():
            row = blob["mean"][key]
            lines.append(
                f"| {name} | {step} | {row['pos_mm']:.1f} | {row['rot_deg']:.1f} | "
                f"{row['psi_deg']:.1f} | {row['a_deg']:.1f} | {row['b_deg']:.1f} | {row['stiff']:.0f} |"
            )
    MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    field = FixBMoldField(Path(FIELD))
    axis = long_axis(field.vertices)
    datasets = {}
    report = {}
    for name, path, kind in RUNS:
        try:
            policy, cfg = load_policy_and_cfg(path, device)
        except RuntimeError as exc:
            report[name] = {"error": str(exc), "path": path}
            print(name, "UNREADABLE", exc, flush=True)
            continue
        if kind not in datasets:
            if kind == "heading":
                datasets[kind] = (
                    build_dataset(cfg, HEAD_ZARR, HEAD_NORM, "e93b8756", 8).get_validation_dataset(),
                )
            else:
                datasets[kind] = (
                    build_dataset(cfg, CAUSAL_ZARR, CAUSAL_NORM, "14308b72", 10).get_validation_dataset(),
                )
            datasets[kind] = (datasets[kind][0], query_indices(datasets[kind][0]))
        dataset, queries = datasets[kind]
        print(name, path, flush=True)
        report[name] = score_run(field, axis, policy, dataset, queries, kind, device)
        del policy
        torch.cuda.empty_cache()
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    write_md(report)
    print("wrote", OUT, flush=True)
    print("wrote", MD, flush=True)


if __name__ == "__main__":
    main()

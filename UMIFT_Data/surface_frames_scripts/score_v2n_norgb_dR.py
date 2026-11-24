#!/usr/bin/env python3
"""ΔR of checkpoint B against the field_v2n label.

The 170° figure compared R_query @ ΔR to the TCP. R_query is the raw
closest-point frame. The label was built with mold_R_field_0, and
mold_R_field_0 @ ΔR is the TCP. This script reports both.

Does not write checkpoints, zarrs, or normalizers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "UMIFT_Data/surface_frames_scripts"),
    str(ROOT / "PyriteEnvSuites"),
    str(ROOT / "PyriteML"),
    str(ROOT / "PyriteConfig"),
    str(ROOT / "PyriteUtility"),
    str(ROOT),
]

from fixb_field import FixBMoldField  # noqa: E402
from fixb_labels import pose7_to_Rp  # noqa: E402
from mold_aware_runtime import rot6d_rows_to_R  # noqa: E402

CKPT = Path("/mnt/data/umidata/ft/training_outputs/oct_v35_h4s_r4vt_field_v2n_norgb/latest.ckpt")
ZARR = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "r4vt_field_v2n/acp_replay_buffer.zarr"
)
FIELD = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz"
)
NORM = Path(
    "/home/boeing/umi_ft_ws/worktrees/UMI-FT-field_v2n/UMIFT_Data/surface_frames_scripts/"
    "assets/oct_v35_h4s_r4vt_action_normalizer_minmax_field_v2n.pkl"
)
LONG = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/long_dt_gt_100ms.npz"
)
OUT = Path(__file__).resolve().parent / "eval/v2n_norgb_dR.json"
STRIDE_S = 0.5


def angle(a, b) -> float:
    c = float(np.clip((np.trace(np.asarray(a).T @ np.asarray(b)) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def phases(force, clearance) -> list[str]:
    phase = "approach"
    out = []
    for f, c in zip(force, clearance):
        if f > 5.0:
            phase = "contact"
        elif f < 2.0 and c > 0.010:
            phase = "approach"
        elif phase == "contact":
            phase = "lift-off"
        out.append(phase)
    return out


def stat(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": int(values.size),
        "median": round(float(np.median(values)), 3),
        "p90": round(float(np.percentile(values, 90)), 3),
    }


def main() -> None:
    import hydra
    import torch
    import zarr
    from omegaconf import OmegaConf, open_dict
    from PyriteEnvSuites.scripts.run_policy_open_loop_val_export import (
        load_policy_and_cfg,
        predict_sparse_action,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, cfg = load_policy_and_cfg(str(CKPT), device)
    shape = cfg.task.shape_meta
    for node in (shape, cfg.task.dataset.shape_meta):
        if "rgb_0" in node.obs:
            continue
        with open_dict(node):
            node.obs.rgb_0 = {"type": "rgb", "shape": [3, 224, 224]}
    OmegaConf.set_struct(cfg.task.dataset, False)
    cfg.task.dataset.dataset_path = str(ZARR)
    cfg.task.dataset.action_normalizer_path = str(NORM)
    cfg.task.dataset.long_dt_mask_path = str(LONG)
    cfg.task.dataset.exclusion_stats_path = "/tmp/v2n_norgb_dR_exclusion.json"
    cfg.task.dataset.pose_jump_flag_path = None
    val = hydra.utils.instantiate(cfg.task.dataset).get_validation_dataset()
    root = zarr.open_group(str(ZARR), mode="r")
    field = FixBMoldField(str(FIELD))
    by_ep: dict[int, list] = {}
    for idx, (ep, rgb) in enumerate(val.sampler.indices):
        ep, rgb = int(ep), int(rgb)
        group = root["data"][f"episode_{ep}"]
        t = float(np.asarray(group["rgb_time_stamps_0"]).reshape(-1)[rgb])
        by_ep.setdefault(ep, []).append((idx, rgb, t))

    buckets = {name: [] for name in ("approach", "contact", "lift-off")}
    bogus = []
    stored_tool = []
    for ep, rows in by_ep.items():
        rows.sort(key=lambda row: row[2])
        group = root["data"][f"episode_{ep}"]
        robot_t = np.asarray(group["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
        R_all, p_all = pose7_to_Rp(np.asarray(group["ts_pose_fb_0"], dtype=np.float64))
        wrench = np.asarray(group["wrench_left_0"], dtype=np.float64)
        wt = np.asarray(group["wrench_time_stamps_0"], dtype=np.float64).reshape(-1)
        stored = np.asarray(group["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3)
        kept = []
        last = -1e9
        for row in rows:
            if row[2] - last < STRIDE_S:
                continue
            last = row[2]
            kept.append(row)
        def robot_row(t):
            i = int(np.clip(np.searchsorted(robot_t, t), 1, len(robot_t) - 1))
            return i - 1 if abs(robot_t[i - 1] - t) <= abs(robot_t[i] - t) else i

        force, clearance, rows_at = [], [], []
        for _idx, _rgb, t in kept:
            k = robot_row(t)
            rows_at.append(k)
            j = int(np.clip(np.searchsorted(wt, t), 0, len(wt) - 1))
            force.append(float(np.linalg.norm(wrench[j, :3])))
            foot = np.asarray(field.query(p_all[k:k + 1])["x_s"]).reshape(3)
            outward = -stored[k, :, 2]
            clearance.append(float(np.dot(p_all[k] - foot, outward)))
        for (idx, _rgb, _t), name, k in zip(kept, phases(force, clearance), rows_at):
            sample = val[idx]
            obs = {
                key: value.numpy()
                for key, value in sample["obs"]["sparse"].items()
                if not str(key).startswith("rgb")
            }
            pred = np.asarray(predict_sparse_action(policy, {"sparse": obs}, device))
            gt = sample["action"]["sparse"].numpy()
            dR_pred = rot6d_rows_to_R(pred[0:1, 4:10])[0]
            dR_gt = rot6d_rows_to_R(gt[0:1, 4:10])[0]
            buckets[name].append(angle(dR_pred, dR_gt))
            frame_q = np.asarray(field.query(p_all[k:k + 1])["R"][0])
            bogus.append(angle(frame_q @ dR_gt, R_all[k]))
            stored_tool.append(angle(stored[k] @ dR_pred, R_all[k]))
            if sum(len(v) for v in buckets.values()) % 50 == 0:
                print(sum(len(v) for v in buckets.values()), flush=True)
    report = {
        "checkpoint": str(CKPT),
        "delta_R_vs_label_deg": {name: stat(values) for name, values in buckets.items() if values},
        "stored_frame_times_pred_vs_tcp_deg": stat(stored_tool),
        "raw_query_frame_times_label_vs_tcp_deg": stat(bogus),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Position from velocity only. δn scored on its own, at the runner's 3000 N/m."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import zarr
from scipy.spatial import cKDTree

ROOT = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames")
SCRIPTS = ROOT / "UMIFT_Data/surface_frames_scripts"
sys.path[:0] = [
    str(SCRIPTS / "eval" / "both_ckpt"),
    str(SCRIPTS / "eval"),
    str(SCRIPTS),
    str(ROOT / "PyriteEnvSuites/scripts"),
    str(ROOT / "PyriteML"),
    str(ROOT / "PyriteConfig"),
    str(ROOT / "PyriteUtility"),
    str(ROOT),
]

import compare_r1rel as fair
import score_both as both
from PyriteConfig.tasks.umift.umift_type_conversions import action21_postprocess
from PyriteUtility.data_pipeline.wrench_baseline_utils import time_based_moving_average
from PyriteUtility.spatial_math import spatial_utilities as su
from fixb_labels import pose7_to_Rp
from run_policy_open_loop_val_export import load_policy_and_cfg, predict_sparse_action

OUT = Path(__file__).resolve().parent
K_RUN = 3000.0
STEPS = (1, 6, 16, 64)
SKIP = {9, 12, 28, 37, 41, 35, 44}
ZARR = Path(fair.CAUSAL_ZARR)


def _check() -> None:
    force, normal = 12.0, np.array([0.0, 0.0, 1.0])
    dn = -float(np.dot(force * normal, normal)) / K_RUN
    if abs(-K_RUN * dn - force) > 1e-9:
        raise RuntimeError("δn at 3000 N/m did not return the normal force")


def tcp_from_velocity(field, action, row):
    """Foot walked by v, plus the query TCP's tangent offset. δn is not added."""
    stripped = np.array(action, dtype=np.float64, copy=True)
    stripped[:, 2] = 0.0
    return fair.rollout_tool(
        field, stripped, row["foot0"], row["frame0"], row["anchor_p"], row["dts"]
    )


def r1rel_parts(action, anchor_p, anchor_R, normal):
    base = np.eye(4, dtype=np.float64)
    base[:3, :3] = anchor_R
    base[:3, 3] = anchor_p
    command, virtual, _stiff, _eoat = action21_postprocess(
        np.asarray(action, dtype=np.float64), [base], [0]
    )
    ref = su.SE3_to_pose7(command[0])
    virt = su.SE3_to_pose7(virtual[0])
    ref_se3 = su.pose7_to_SE3(ref)
    virt_se3 = su.pose7_to_SE3(virt)
    dn = np.sum((virt_se3[:, :3, 3] - ref_se3[:, :3, 3]) * normal, axis=1)
    return ref_se3[:, :3, 3], ref_se3[:, :3, :3], dn


def mean_ep(items, key, step):
    by_ep = {}
    for item in items:
        by_ep.setdefault(item["episode"], []).append(float(item[key][step - 1]))
    return round(float(np.mean([np.mean(v) for v in by_ep.values()])), 2)


def pack(items):
    return {
        "n": len(items),
        "pos_mm": {str(s): mean_ep(items, "pos", s) for s in STEPS},
        "tangential_mm": {str(s): mean_ep(items, "tang", s) for s in STEPS},
        "dn_mm": {str(s): mean_ep(items, "dn", s) for s in STEPS},
        "force_n": {str(s): mean_ep(items, "force", s) for s in STEPS},
    }


def distribution(field, apron) -> dict:
    root = zarr.open(ZARR, "r")["data"]
    rows = {name: [] for name in both.REGIONS}
    match = []
    for key in sorted(root.keys(), key=lambda s: int(s.split("_")[-1])):
        ep = int(key.split("_")[-1])
        if ep in SKIP:
            continue
        group = root[key]
        robot_t = np.asarray(group["robot_time_stamps_0"]).reshape(-1)
        wrench_t = np.asarray(group["wrench_time_stamps_left_0"]).reshape(-1)
        wrench = np.asarray(group["wrench_left_0"], dtype=np.float64)
        smooth = time_based_moving_average(wrench, wrench_t, 1.0)
        j = np.searchsorted(wrench_t, robot_t)
        j = np.clip(j, 1, len(wrench_t) - 1)
        left = np.abs(robot_t - wrench_t[j - 1]) <= np.abs(robot_t - wrench_t[j])
        j = np.where(left, j - 1, j)
        force = smooth[j, :3]
        rot, tcp = pose7_to_Rp(np.asarray(group["ts_pose_fb_0"], dtype=np.float64))
        n_out = np.asarray(group["mold_normal_0"], dtype=np.float64).reshape(-1, 3)
        fn = np.einsum("ni,ni->n", np.einsum("nij,nj->ni", rot, force), n_out)
        mag = np.linalg.norm(force, axis=1)
        hit = mag >= 8.0
        if not np.any(hit):
            continue
        dn = -fn / K_RUN
        match.append(float(np.max(np.abs(-K_RUN * dn[hit] - fn[hit]))))
        xyz = tcp[hit]
        _h, vid, w = field.closest(xyz)
        raw = np.einsum("qk,qkd->qd", w, field.normals[vid])
        region = both.region_of(raw, xyz, apron)
        stored = np.asarray(group["fixb_delta_n_0"], dtype=np.float64).reshape(-1)[hit]
        for name in both.REGIONS:
            sel = region == name
            if np.any(sel):
                rows[name].append(np.column_stack([dn[hit][sel], stored[sel], fn[hit][sel]]))
    out = {}
    for name, parts in rows.items():
        if not parts:
            out[name] = {"n": 0}
            continue
        block = np.vstack(parts)
        dn_mm = block[:, 0] * 1e3
        out[name] = {
            "n": int(len(block)),
            "dn_mm_median": round(float(np.median(dn_mm)), 2),
            "dn_mm_p10": round(float(np.percentile(dn_mm, 10)), 2),
            "dn_mm_p90": round(float(np.percentile(dn_mm, 90)), 2),
            "force_n_median": round(float(np.median(-K_RUN * block[:, 0])), 2),
            "stored_dn_mm_median": round(float(np.median(block[:, 1]) * 1e3), 2),
        }
    out["force_identity_max_n"] = round(float(max(match) if match else 0.0), 6)
    return out


def score_one(name, policy, dataset, kind, field, rows, device, qmap):
    items = []
    for row in rows:
        key = (row["episode"], row["rgb"])
        if kind == "r1rel":
            idx = row["r_idx"]
        elif kind == "causal":
            idx = row["c_idx"]
        else:
            idx = qmap.get(key)
            if idx is None:
                continue
        obs, sample = dataset.sampler.sample_sequence(idx)
        pred = np.asarray(predict_sparse_action(policy, obs, device), dtype=np.float64)
        pred = pred.reshape(-1, pred.shape[-1])[: fair.HORIZON]
        gt = np.asarray(sample["sparse"], dtype=np.float64)[: fair.HORIZON]
        if len(pred) < fair.HORIZON or len(gt) < fair.HORIZON:
            continue
        normal = row["normals"]
        if kind == "r1rel":
            xyz, _rot, dn = r1rel_parts(pred, row["anchor_p"], row["anchor_R"], normal)
            _lx, _lr, ldn = r1rel_parts(gt, row["anchor_p"], row["anchor_R"], normal)
        else:
            xyz = tcp_from_velocity(field, pred, row)
            dn = pred[:, 2]
            ldn = gt[:, 2]
        err = xyz - row["demo"]
        along = np.sum(err * normal, axis=1)
        tang = np.linalg.norm(err - along[:, None] * normal, axis=1) * 1e3
        items.append({
            "episode": row["episode"],
            "pos": np.linalg.norm(err, axis=1) * 1e3,
            "tang": tang,
            "dn": np.abs(dn - ldn) * 1e3,
            "force": np.abs(-K_RUN * (dn - ldn)),
        })
    print(f"[tcp] {name} {len(items)}", flush=True)
    return items


def main() -> None:
    _check()
    OUT.mkdir(parents=True, exist_ok=True)
    field = fair.FixBMoldField(fair.FIELD_PATH)
    apron = cKDTree(field.vertices[np.asarray(np.load(fair.FIELD_PATH)["is_apron"]).astype(bool)])
    label = distribution(field, apron)
    (OUT / "score.json").write_text(json.dumps({"label_k3000": label}, indent=2))
    print(json.dumps(label, indent=2), flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    specs = [
        ("causal", fair.CAUSAL_ZARR, fair.CAUSAL_NORM, "14308b72", 10, [
            ("causal e25", both.CAUSAL_DIR / "epoch=0025-val_loss=0.058.ckpt"),
            ("causal e199", both.CAUSAL_DIR / "latest.ckpt"),
        ]),
        ("r1rel", fair.R1REL_ZARR, fair.R1REL_NORM, "2c20fba56851", 21, [
            ("r1rel e25", fair.R1REL_CKPT),
            ("r1rel last", both.TRAIN / "oct_v35_h4s_odom_r1rel/checkpoints/latest.ckpt"),
        ]),
        ("heading", str(both.DATA / "r4vt_heading_tilt/acp_replay_buffer.zarr"),
         SCRIPTS / "assets/oct_v35_h4s_r4vt_action_normalizer_minmax_heading_tilt_swing_sc.pkl",
         "e93b8756", 8, [
            ("heading e25", both.HEAD_DIR / "epoch=0025-val_loss=0.059.ckpt"),
            ("heading latest", both.HEAD_DIR / "latest.ckpt"),
        ]),
    ]
    datasets = {}
    for kind, zarr_path, norm, sha, dim, ckpts in specs:
        policy, cfg = load_policy_and_cfg(str(ckpts[0][1]), device)
        datasets[kind] = both.build_full(cfg, zarr_path, norm, sha, dim)
        del policy
        torch.cuda.empty_cache()
    rows = both.prepare(datasets["causal"], datasets["r1rel"])
    table = {}
    for kind, _z, _n, _s, _d, ckpts in specs:
        qmap = fair.query_map(datasets[kind])
        for label_name, path in ckpts:
            policy, _cfg = load_policy_and_cfg(str(path), device)
            table[label_name] = pack(score_one(
                label_name, policy, datasets[kind], kind, field, rows, device, qmap
            ))
            del policy
            torch.cuda.empty_cache()
    report = {
        "position": "foot integrated with v, plus the query tangent offset. δn is not in the point.",
        "dn": "absolute error versus the checkpoint label, mm. force_n is that error times 3000 N/m.",
        "n_queries": len(rows),
        "table": table,
        "label_k3000": label,
    }
    (OUT / "score.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()

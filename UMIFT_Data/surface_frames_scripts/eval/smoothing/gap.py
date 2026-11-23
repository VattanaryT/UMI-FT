#!/usr/bin/env python3
"""Angle between the smooth field normal and the raw mesh normal. No retraining."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr
from scipy.spatial import cKDTree

ROOT = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames")
SCRIPTS = ROOT / "UMIFT_Data/surface_frames_scripts"
sys.path[:0] = [
    str(SCRIPTS / "eval"),
    str(SCRIPTS),
    str(ROOT / "PyriteEnvSuites/scripts"),
    str(ROOT / "PyriteML"),
    str(ROOT / "PyriteConfig"),
    str(ROOT / "PyriteUtility"),
    str(ROOT),
]

import compare_r1rel as fair
from fixb_smooth_field import SmoothFrameField
from run_policy_open_loop_val_export import load_policy_and_cfg

OUT = Path(__file__).resolve().parent
DATA = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr")
FIELD = DATA / "shared/fixb_field.npz"
SMOOTH = DATA / "shared/fixb_smooth_frames.npz"
ZARR = DATA / "r4vt_causal_foot/acp_replay_buffer.zarr"
VAL = {9, 12, 28, 37, 41}
DROP = {35, 44}
FORCE_N = 8.0
FLANGE_M = 0.015
UP = np.array([0.0, -1.0, 0.0])
REGIONS = ("crown", "fillet", "wall", "flange")
STEPS = (1, 6, 16, 64)


def _check() -> None:
    tilted = np.array([0.0, np.sin(np.radians(10.0)), np.cos(np.radians(10.0))])
    got = float(np.degrees(np.arccos(np.clip(tilted[2], -1.0, 1.0))))
    assert abs(got - 10.0) < 1e-6


def angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=1), -1.0, 1.0)))


def region_of(normal: np.ndarray, foot: np.ndarray, apron: cKDTree) -> np.ndarray:
    n = normal / np.clip(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12, None)
    tilt = np.degrees(np.arccos(np.clip(n @ UP, -1.0, 1.0)))
    name = np.full(len(n), "wall", dtype=object)
    name[tilt < 25.0] = "crown"
    name[(tilt >= 25.0) & (tilt < 55.0)] = "fillet"
    near = apron.query(foot, workers=-1)[0] < FLANGE_M
    name[near] = "flange"
    return name


def normals_at(field: SmoothFrameField, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    got = field.query(pts)
    _x, vid, w = field.closest(pts)
    raw = np.einsum("qk,qkd->qd", w, field.normals[vid])
    return got["n_out"].reshape(-1, 3), raw, got["x_s"].reshape(-1, 3)


def contact_rows(field: SmoothFrameField, apron: cKDTree) -> dict[str, np.ndarray]:
    root = zarr.open(ZARR, "r")["data"]
    angles = []
    regions = []
    for key in sorted(root.keys(), key=lambda s: int(s.split("_")[-1])):
        ep = int(key.split("_")[-1])
        if ep in DROP or ep in VAL:
            continue
        group = root[key]
        robot_t = np.asarray(group["robot_time_stamps_0"]).reshape(-1)
        tcp = np.asarray(group["ts_pose_fb_0"], dtype=np.float64)[:, :3]
        wrench_t = np.asarray(group["wrench_time_stamps_left_0"]).reshape(-1)
        force = np.linalg.norm(np.asarray(group["wrench_left_0"], dtype=np.float64)[:, :3], axis=1)
        j = np.clip(np.searchsorted(wrench_t, robot_t), 1, len(wrench_t) - 1)
        left = np.abs(robot_t - wrench_t[j - 1]) <= np.abs(robot_t - wrench_t[j])
        j = np.where(left, j - 1, j)
        take = force[j] >= FORCE_N
        if not np.any(take):
            continue
        pts = tcp[take]
        field_n, raw_n, foot = normals_at(field, pts)
        angles.append(angle_deg(field_n, raw_n))
        regions.append(region_of(raw_n, foot, apron))
        print(f"[smooth] episode {ep} contact {int(take.sum())}", flush=True)
    return {"angle": np.concatenate(angles), "region": np.concatenate(regions)}


def summarize(angle: np.ndarray, region: np.ndarray) -> dict:
    rows = {}
    for name in REGIONS:
        sel = region == name
        x = angle[sel]
        rows[name] = {
            "n": int(sel.sum()),
            "median": None if not sel.any() else float(np.median(x)),
            "p90": None if not sel.any() else float(np.percentile(x, 90)),
            "gt5": None if not sel.any() else float(np.mean(x > 5.0)),
            "gt10": None if not sel.any() else float(np.mean(x > 10.0)),
        }
    rows["all"] = {
        "n": int(len(angle)),
        "median": float(np.median(angle)),
        "p90": float(np.percentile(angle, 90)),
        "gt5": float(np.mean(angle > 5.0)),
        "gt10": float(np.mean(angle > 10.0)),
    }
    return rows


def fair_by_region(field: SmoothFrameField, apron: cKDTree) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    causal_path, causal_policy, causal_cfg, _failed = fair.load_causal(device)
    causal_ds, stride, horizon = fair.build_dataset(
        causal_cfg, fair.CAUSAL_ZARR, fair.CAUSAL_NORM, "14308b72", 10
    )
    r1_policy, r1_cfg = load_policy_and_cfg(str(fair.R1REL_CKPT), device)
    r1_ds, r_stride, r_h = fair.build_dataset(
        r1_cfg, fair.R1REL_ZARR, fair.R1REL_NORM, "2c20fba56851", 21
    )
    if stride != r_stride or horizon != r_h:
        raise SystemExit("causal and r1rel horizons differ")
    mold = fair.FixBMoldField(fair.FIELD_PATH)
    queries = fair.common_queries(causal_ds, r1_ds, VAL)
    print(f"[smooth] fair queries {len(queries)} on {sorted(VAL)}", flush=True)
    per_ep = fair.score(
        mold, fair.long_axis(mold.vertices), causal_policy, causal_ds,
        r1_policy, r1_ds, queries, device, stride,
    )
    del causal_policy, r1_policy
    torch.cuda.empty_cache()
    bag = {name: {step: {"causal_pos": [], "causal_rot": [], "r1_pos": [], "r1_rot": []} for step in STEPS} for name in REGIONS}
    for rows in per_ep.values():
        for row in rows:
            demo = np.asarray(row["xyz"]["demo"], dtype=np.float64)
            field_n, raw_n, foot = normals_at(field, demo)
            names = region_of(raw_n, foot, apron)
            for step in STEPS:
                i = step - 1
                slot = bag[names[i]][step]
                slot["causal_pos"].append(float(row["causal"]["pos_mm"][i]))
                slot["causal_rot"].append(float(row["causal"]["rot_deg"][i]))
                slot["r1_pos"].append(float(row["r1rel"]["pos_mm"][i]))
                slot["r1_rot"].append(float(row["r1rel"]["rot_deg"][i]))
    out = {}
    for name in REGIONS:
        out[name] = {}
        for step in STEPS:
            slot = bag[name][step]
            def mean(key, slot=slot):
                return None if not slot[key] else float(np.mean(slot[key]))
            out[name][str(step)] = {
                "n": len(slot["causal_pos"]),
                "causal_pos": mean("causal_pos"),
                "causal_rot": mean("causal_rot"),
                "r1rel_pos": mean("r1_pos"),
                "r1rel_rot": mean("r1_rot"),
            }
    out["ckpt"] = {"causal": str(causal_path), "r1rel": str(fair.R1REL_CKPT)}
    return out


def histogram(angle: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=120)
    ax.hist(angle, bins=40, color="#4f7399", edgecolor="white")
    ax.axvline(5, color="#e67e22", lw=1.2, label="5°")
    ax.axvline(10, color="#c0392b", lw=1.2, label="10°")
    ax.set_xlabel("field normal vs raw mesh normal (deg)")
    ax.set_ylabel("training contact samples")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    _check()
    blob = np.load(FIELD)
    field = SmoothFrameField(FIELD, SMOOTH)
    apron = cKDTree(field.vertices[np.asarray(blob["is_apron"]).astype(bool)])
    rows = contact_rows(field, apron)
    gap = summarize(rows["angle"], rows["region"])
    histogram(rows["angle"], OUT / "gap_hist.png")
    errors = fair_by_region(field, apron)
    report = {
        "field_normal": "40 mm smooth sidecar, outward",
        "raw_normal": "unsmoothed mesh vertex normal at the same closest point",
        "contact": "|F|>=8 N, training episodes, val and 35/44 left out",
        "gap": gap,
        "fair": errors,
    }
    (OUT / "smoothing.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

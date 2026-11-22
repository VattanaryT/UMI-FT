#!/usr/bin/env python3
"""Variance of the training position and rotation labels in three frames.

Field position is one control step: (v1 dt, v2 dt, δn), in meters.
World position is the fixture-frame step of ts_pose_fb_0.
r1rel position is that same step in the current TCP frame, which is the
relative action translation. Rotation is (ψ, a, b) against the rotvec of the
TCP step, in the TCP frame and in the fixture frame.

Episodes 35 and 44 are excluded. Every 4th robot sample.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr
from scipy.spatial.transform import Rotation

ROOT = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames")
SCRIPTS = ROOT / "UMIFT_Data/surface_frames_scripts"
sys.path.insert(0, str(SCRIPTS))

from fixb_labels import pose7_to_Rp  # noqa: E402

ZARR = (
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "r4vt_heading_tilt/acp_replay_buffer.zarr"
)
OUT = SCRIPTS / "eval/frame_variance.json"
MD = SCRIPTS / "eval/frame_variance.md"
PNG = SCRIPTS / "eval/frame_variance.png"
SKIP = {35, 44}
STRIDE = 4
UP = np.array([0.0, -1.0, 0.0])
REGIONS = ("crown", "radius", "flange", "wall_near", "wall_far")


def across_hat(feet):
    """Horizontal direction across the cloud, toward +Z."""
    centered = feet - feet.mean(0)
    axis = np.linalg.eigh(np.cov(centered.T))[1][:, -1]
    if axis[0] < 0.0:
        axis = -axis
    horizontal = np.array([0.0, 0.0, 1.0])
    dropped = horizontal - np.dot(horizontal, axis) * axis
    dropped[1] = 0.0
    hat = dropped / max(float(np.linalg.norm(dropped)), 1e-12)
    return hat if hat[2] >= 0.0 else -hat


def regions_of(normal, foot, bottom, hat):
    n = normal / np.clip(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12, None)
    ang = np.degrees(np.arccos(np.clip(n @ UP, -1.0, 1.0)))
    name = np.full(len(n), "crown", dtype=object)
    name[(ang >= 25.0) & (ang < 55.0)] = "radius"
    name[ang >= 55.0] = np.where(n[ang >= 55.0] @ hat >= 0.0, "wall_near", "wall_far")
    name[(ang < 25.0) & (foot[:, 1] >= bottom)] = "flange"
    return name


def _pack(rows):
    return {key: np.concatenate([row[key] for row in rows]) for key in rows[0]}


def load():
    root = zarr.open(ZARR, "r")
    raw = root.attrs.get("conversion_info", {})
    info = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
    data = root["data"]
    feet = []
    for ep in range(50):
        if ep in SKIP or f"episode_{ep}" not in data:
            continue
        feet.append(np.asarray(data[f"episode_{ep}"]["mold_x_s_0"][::STRIDE], np.float64))
    feet = np.concatenate(feet)
    bottom = float(np.percentile(feet[:, 1], 90))
    hat = across_hat(feet)
    rows = []
    for ep in range(50):
        if ep in SKIP or f"episode_{ep}" not in data:
            continue
        g = data[f"episode_{ep}"]
        sl = slice(None, None, STRIDE)
        pose = np.asarray(g["ts_pose_fb_0"][sl], np.float64)
        v = np.asarray(g["fixb_v_par_0"][sl], np.float64)
        dn = np.asarray(g["fixb_delta_n_0"][sl], np.float64).reshape(-1)
        tilt = np.asarray(g["fixb_heading_tilt_0"][sl], np.float64)
        normal = np.asarray(g["mold_normal_0"][sl], np.float64)
        foot = np.asarray(g["mold_x_s_0"][sl], np.float64)
        t = np.asarray(g["robot_time_stamps_0"][sl], np.float64).reshape(-1)
        n = min(len(pose), len(v), len(dn), len(tilt), len(normal), len(foot), len(t))
        if n < 3:
            continue
        pose, v, dn, tilt, normal, foot, t = (
            pose[:n], v[:n], dn[:n], tilt[:n], normal[:n], foot[:n], t[:n]
        )
        R, p = pose7_to_Rp(pose)
        dt = np.diff(t)
        ok = (dt > 1e-4) & (dt < 0.2)
        dp = np.diff(p, axis=0)
        body = np.einsum("nij,nj->ni", R[:-1].transpose(0, 2, 1), dp)
        rel = np.einsum("nji,njk->nik", R[:-1], R[1:])
        world_rel = np.einsum("nij,njk->nik", R[1:], R[:-1].transpose(0, 2, 1))
        psi = np.degrees(np.arctan2(tilt[:, 0], tilt[:, 1]))
        ab = np.degrees(tilt[:, 2:4])
        dpsi = (np.diff(psi) + 180.0) % 360.0 - 180.0
        rows.append(
            {
                "field": np.column_stack([v[:-1, 0] * dt, v[:-1, 1] * dt, np.diff(dn)])[ok],
                "field_label": np.column_stack([v[:-1, 0], v[:-1, 1], dn[:-1]])[ok],
                "world": dp[ok],
                "r1rel": body[ok],
                "tilt": np.column_stack([dpsi, np.diff(ab[:, 0]), np.diff(ab[:, 1])])[ok],
                "tilt_label": np.column_stack([psi[:-1], ab[:-1]])[ok],
                "r1rel_rot": np.degrees(Rotation.from_matrix(rel[ok]).as_rotvec()),
                "world_rot": np.degrees(Rotation.from_matrix(world_rel[ok]).as_rotvec()),
                "region": regions_of(normal[:-1], foot[:-1], bottom, hat)[ok],
            }
        )
        print(f"episode {ep} {int(ok.sum())}", flush=True)
    return _pack(rows), info


def _stats(block):
    block = np.asarray(block, np.float64)
    block = block[np.isfinite(block).all(1)]
    var = block.var(0)
    total = float(var.sum())
    share = (var / total).tolist() if total > 0 else [0.0, 0.0, 0.0]
    cov = np.cov(block.T)
    eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eig_sum = float(eig.sum())
    pca = (eig / eig_sum).tolist() if eig_sum > 0 else [0.0, 0.0, 0.0]
    return {
        "n": int(len(block)),
        "std": [round(float(v), 5) for v in np.sqrt(var)],
        "share": [round(float(v), 4) for v in share],
        "pca": [round(float(v), 4) for v in pca],
    }


def summarize(packed):
    keys = {
        "field_step_m": "field",
        "field_label": "field_label",
        "world_m": "world",
        "r1rel_m": "r1rel",
        "tilt_step_deg": "tilt",
        "tilt_label_deg": "tilt_label",
        "r1rel_rot_deg": "r1rel_rot",
        "world_rot_deg": "world_rot",
    }
    out = {"overall": {}, "by_region": {}}
    for name, key in keys.items():
        out["overall"][name] = _stats(packed[key])
    for region in REGIONS:
        mask = packed["region"] == region
        out["by_region"][region] = {
            name: _stats(packed[key][mask]) for name, key in keys.items()
        }
    return out


def chart(report):
    specs = (
        ("v1 dt", "field_step_m", 0),
        ("v2 dt", "field_step_m", 1),
        ("Δδn", "field_step_m", 2),
        ("world x", "world_m", 0),
        ("world y", "world_m", 1),
        ("world z", "world_m", 2),
        ("r1rel x", "r1rel_m", 0),
        ("r1rel y", "r1rel_m", 1),
        ("r1rel z", "r1rel_m", 2),
        ("Δψ", "tilt_step_deg", 0),
        ("Δa", "tilt_step_deg", 1),
        ("Δb", "tilt_step_deg", 2),
        ("r1rel rx", "r1rel_rot_deg", 0),
        ("r1rel ry", "r1rel_rot_deg", 1),
        ("r1rel rz", "r1rel_rot_deg", 2),
    )
    labels = [item[0] for item in specs]
    shares = [report["overall"][key]["share"][i] for _label, key, i in specs]
    colors = ["#4aa3ff"] * 3 + ["#888"] * 3 + ["#3ddc97"] * 3 + ["#ff4d6a"] * 3 + ["#ff9f1a"] * 3
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.bar(labels, shares, color=colors)
    ax.set_ylabel("share of variance")
    ax.set_ylim(0, 1)
    ax.set_title("Training-label variance by axis")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(PNG, dpi=120)
    plt.close(fig)


def _line(stats):
    share = "/".join(f"{v:.2f}" for v in stats["share"])
    pca = "/".join(f"{v:.2f}" for v in stats["pca"])
    std = "/".join(f"{v:.4g}" for v in stats["std"])
    return f"n={stats['n']} std {std} share {share} pca {pca}"


def write_md(report, info):
    lines = [
        "# Frame variance of the training labels",
        "",
        "The chart is one robot step. Field position is `(v1 dt, v2 dt, Δδn)` in meters,",
        "next to the fixture-frame step and the same step in the current TCP frame.",
        "Rotation is the step of `(ψ, a, b)` in degrees, next to the TCP rotvec.",
        "`field_label` is the stored `(v1, v2, δn)` in m/s, m/s, m.",
        "`tilt_label_deg` is the stored `(ψ, a, b)`, not the step.",
        "Episodes 35 and 44 are out. Every 4th sample.",
        "",
        "`robot0_mold_geom` is signed distance, outward normal, and `R_field` rot6d.",
        "It does not contain surface coordinates `(u, v)`. The foot point is `mold_x_s_0`.",
        "",
        f"Wrench: conversion_info says `{info.get('pose_and_wrench_body_frame', 'tcp')}`.",
        "The stored `wrench_left_0` is the TCP frame. The relative-obs adjoint is",
        "the identity at the newest pose, so the policy still sees the TCP wrench.",
        "",
        "## Overall",
        "",
    ]
    for name, stats in report["overall"].items():
        lines.append(f"- {name}: {_line(stats)}")
    lines += ["", "## By region", ""]
    for region, blob in report["by_region"].items():
        lines.append(f"### {region}")
        for name, stats in blob.items():
            lines.append(f"- {name}: {_line(stats)}")
    lines.append("")
    MD.write_text("\n".join(lines))


def main():
    packed, info = load()
    report = summarize(packed)
    report["conversion_info_wrench"] = info.get("pose_and_wrench_body_frame")
    report["mold_geom"] = "signed_m, normal_xyz, R_field_rot6d. No (u, v)."
    OUT.write_text(json.dumps(report, indent=2))
    write_md(report, info)
    chart(report)
    print(f"wrote {OUT}", flush=True)
    for name, stats in report["overall"].items():
        print(name, stats["share"], "pca", stats["pca"], flush=True)


if __name__ == "__main__":
    main()

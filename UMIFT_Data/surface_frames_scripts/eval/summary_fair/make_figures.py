#!/usr/bin/env python3
"""Figures for eval/summary_fair. Reads existing zarrs and the sweep json. Writes png only."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

ROOT = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames")


def time_based_moving_average(wrench, timestamps, window_duration_s=1.0):
    wrench = np.asarray(wrench, dtype=np.float64)
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    total = np.vstack([np.zeros((1, wrench.shape[1])), np.cumsum(wrench, axis=0)])
    left = np.searchsorted(ts, ts - window_duration_s, side="left")
    count = np.arange(1, len(ts) + 1) - left
    return (total[1:] - total[left]) / count[:, None]

HERE = Path(__file__).resolve().parent
DATA = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr")
MESH = DATA / "shared/fixb_field.npz"
OLD = DATA / "r4vt_smooth_tcp_track_clean/acp_replay_buffer.zarr"
NEW = DATA / "r4vt_causal_foot_carry/acp_replay_buffer.zarr"
K3000 = DATA / "r4vt_causal_foot_carry_k3000/acp_replay_buffer.zarr"
SWEEP = ROOT / "UMIFT_Data/surface_frames_scripts/eval/smooth_sweep/sweep.json"
SKIP = {9, 12, 28, 35, 37, 41, 44}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def _check_frame(frame: np.ndarray, outward: np.ndarray) -> None:
    normal = frame[:, 2]
    if float(np.dot(normal, outward)) < 0.9:
        raise RuntimeError("field column 2 is not the outward normal")
    if abs(float(np.dot(frame[:, 0], frame[:, 2]))) > 1e-3:
        raise RuntimeError("field axes are not orthogonal")


def field_frame() -> None:
    blob = np.load(MESH)
    vertices = np.asarray(blob["vertices"], dtype=np.float64)
    frames = np.asarray(blob["frames"], dtype=np.float64)
    outward = np.asarray(blob["normals"], dtype=np.float64)
    apron = np.asarray(blob["is_apron"]).astype(bool) if "is_apron" in blob.files else np.zeros(len(vertices), dtype=bool)
    _check_frame(frames[len(vertices) // 2], outward[len(vertices) // 2])
    body = np.flatnonzero(~apron)
    center = float(np.median(vertices[body, 0]))
    band = body[np.abs(vertices[body, 0] - center) < 0.003]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.scatter(vertices[band, 2], -vertices[band, 1], s=4, c="#9aa3ad", linewidths=0)
    up_axis = np.array([0.0, -1.0, 0.0])
    tilt = np.degrees(np.arccos(np.clip(outward[band] @ up_axis, -1.0, 1.0)))
    i = int(band[int(np.argmin(np.abs(tilt - 50.0)))])
    origin = np.array([vertices[i, 2], -vertices[i, 1]])
    into = -frames[i, :, 2]
    side = frames[i, :, 1]
    into_2 = np.array([into[2], -into[1]])
    side_2 = np.array([side[2], -side[1]])
    into_2 /= max(np.linalg.norm(into_2), 1e-9)
    side_2 /= max(np.linalg.norm(side_2), 1e-9)
    length = 0.03

    def arrow(vec, color, name):
        ax.annotate(
            "", xy=origin + vec * length, xytext=origin,
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6),
        )
        ax.text(*(origin + vec * (length + 0.004)), name, color=color)

    arrow(into_2, "#b85c38", "tool axis")
    arrow(side_2, "#2a6f4e", "across")
    lean = into_2 * np.cos(np.radians(25)) + side_2 * np.sin(np.radians(25))
    arrow(lean, "#6b4c9a", "tilt a")
    ax.scatter([origin[0]], [origin[1]], s=30, c="#1f4e79", zorder=5)
    ax.text(origin[0], origin[1] + 0.012, "heading, out of the page", color="#1f4e79", ha="center")
    ax.set_aspect("equal")
    ax.set_xlabel("across the mold")
    ax.set_ylabel("up")
    fig.tight_layout()
    fig.savefig(HERE / "field_frame.png", dpi=140)
    plt.close(fig)


def smoothing() -> None:
    rows = json.loads(SWEEP.read_text())
    regions = ["crown", "fillet", "wall", "flange"]
    radii = [row["radius_mm"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    x = np.arange(len(regions))
    width = 0.24
    for i, row in enumerate(rows):
        med = [row["gap_deg"][name]["median"] for name in regions]
        axes[0].bar(x + (i - 1) * width, med, width, label=f"{row['radius_mm']:.0f} mm")
    axes[0].set_xticks(x, regions)
    axes[0].set_ylabel("normal vs raw mesh, median deg")
    axes[0].legend(frameon=False)
    axes[1].bar([str(int(r)) for r in radii], [row["jumps_gt30"] for row in rows], color="#4c78a8")
    axes[1].set_xlabel("smooth radius, mm")
    axes[1].set_ylabel("frame jumps over 30°")
    fig.tight_layout()
    fig.savefig(HERE / "smoothing.png", dpi=140)
    plt.close(fig)


def carry() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    axes[0].bar(["before", "after"], [1268, 0], color=["#d36b4f", "#4c78a8"])
    axes[0].text(1, 40, "0", ha="center", color="#4c78a8")
    axes[0].set_ylabel("frame jumps over 30°")
    axes[1].bar(["before", "after"], [53.7, 3.3], color=["#d36b4f", "#4c78a8"])
    axes[1].set_ylabel("ΔR-masked steps, % of kept chunks")
    fig.tight_layout()
    fig.savefig(HERE / "carry.png", dpi=140)
    plt.close(fig)


def _contact_dn(path: Path) -> np.ndarray:
    root = zarr.open(path, "r")["data"]
    parts = []
    for ep_id in range(50):
        if ep_id in SKIP:
            continue
        ep = root[f"episode_{ep_id}"]
        robot_t = np.asarray(ep["robot_time_stamps_0"]).reshape(-1)
        wrench_t = np.asarray(ep["wrench_time_stamps_left_0"]).reshape(-1)
        wrench = np.asarray(ep["wrench_left_0"], dtype=np.float64)
        smooth = time_based_moving_average(wrench, wrench_t, 1.0)
        j = np.clip(np.searchsorted(wrench_t, robot_t), 1, len(wrench_t) - 1)
        left = np.abs(robot_t - wrench_t[j - 1]) <= np.abs(robot_t - wrench_t[j])
        force = np.linalg.norm(smooth[np.where(left, j - 1, j), :3], axis=1)
        dn = np.asarray(ep["fixb_delta_n_0"], dtype=np.float64).reshape(-1)
        parts.append(dn[force >= 8.0] * 1e3)
    return np.concatenate(parts)


def delta_n() -> None:
    old = _contact_dn(OLD)
    new = _contact_dn(NEW)
    if not (np.median(new) < 0.0 and np.median(new) > -20.0):
        raise RuntimeError(f"new δn median {np.median(new):.2f} mm is outside the carry report")
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    lo = float(np.percentile(np.concatenate([old, new]), 1))
    hi = float(np.percentile(np.concatenate([old, new]), 99))
    bins = np.linspace(lo, hi, 80)
    ax.hist(old, bins=bins, density=True, histtype="step", linewidth=1.6, label="previous label", color="#d36b4f")
    ax.hist(new, bins=bins, density=True, histtype="step", linewidth=1.6, label="−(F · n) / 3000", color="#4c78a8")
    ax.set_xlabel("δn, mm, contact")
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(HERE / "dn.png", dpi=140)
    plt.close(fig)


def _stiffness(path: Path) -> np.ndarray:
    root = zarr.open(path, "r")["data"]
    return np.concatenate([
        np.asarray(root[f"episode_{i}"]["stiffness_0"], dtype=np.float64).reshape(-1)
        for i in range(50)
    ])


def stiffness() -> None:
    before = _stiffness(NEW)
    after = _stiffness(K3000)
    if np.any(after != 3000.0):
        raise RuntimeError("k3000 zarr is not constant 3000")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    axes[0].hist(before, bins=40, color="#d36b4f")
    axes[0].set_xlabel("stiffness")
    axes[0].set_ylabel("samples")
    axes[0].set_title("before")
    axes[1].hist(after, bins=[2990, 3010], color="#4c78a8")
    axes[1].set_xlim(0, 3200)
    axes[1].set_xlabel("stiffness")
    axes[1].set_title("after, every sample is 3000")
    fig.tight_layout()
    fig.savefig(HERE / "stiffness.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    field_frame()
    smoothing()
    carry()
    delta_n()
    stiffness()
    print("wrote", HERE)

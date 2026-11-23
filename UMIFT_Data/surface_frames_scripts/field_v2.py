#!/usr/bin/env python3
"""Field v2 labels. New files only. Does not submit training.

t1 follows the mold PCA long axis, toward the end nearer the right-arm base.
TCP position obs is in that same frame. Foot stays closest-point.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "UMIFT_Data/surface_frames_scripts"
sys.path[:0] = [str(SCRIPTS), str(ROOT / "PyriteML"), str(ROOT / "PyriteConfig"), str(ROOT / "PyriteUtility"), str(ROOT)]

import zarr  # noqa: E402

from causal_foot import CausalFoot  # noqa: E402
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer  # noqa: E402
from field_v2_dataset import mold_local  # noqa: E402
from fixb_build_datasets import MOLD_KEYS, link_tree, replace_array  # noqa: E402
from fixb_field import FixBMoldField, orthonormal_frames  # noqa: E402
from fixb_labels import delta_R_from_field_tool, fixb_pose7_command, pose7_to_Rp, so3_to_rot6d_rows  # noqa: E402
from fixb_smooth_frame_labels import EXCLUDE, LONG_DT  # noqa: E402
from fixb_tcp_track import project_tcp_velocity  # noqa: E402
from fixb_tcp_track_clean import _kept  # noqa: E402
from heading_tilt import geodesic_deg  # noqa: E402
from mold_aware_runtime import rot6d_rows_to_R  # noqa: E402
from PyriteUtility.data_pipeline.wrench_baseline_utils import time_based_moving_average  # noqa: E402

DATA = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr")
TRAIN_FIELD = DATA / "shared/fixb_field.npz"
SMOOTH = SCRIPTS / "eval/smooth_sweep/frames_r020.npz"
# The 20 mm sidecar was written in the compare worktree and is not in this checkout.
if not SMOOTH.is_file():
    SMOOTH = Path(
        "/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames/"
        "UMIFT_Data/surface_frames_scripts/eval/smooth_sweep/frames_r020.npz"
    )
THIN_FIELD = Path(
    "/mnt/data/umidata/ft/held_out_mold/thin_stringer/transform_check/wall_fix/fixb_field_wall_vertical.npz"
)
SRC = DATA / "r4vt_smooth_tcp_track_clean/acp_replay_buffer.zarr"
OUT = DATA / "r4vt_field_v2_k3000"
NORM = SCRIPTS / "assets/oct_v35_h4s_r4vt_action_normalizer_minmax_field_v2_k3000.pkl"
HELD = SCRIPTS / "eval/thin_heldout"
SNAP = SCRIPTS / "eval/thin_frames"
K = 3000.0
CARRY_DEG = 30.0
UP = np.array([0.0, -1.0, 0.0])
# Right-arm base origin in the fixture frame. Inverse of T_right_base_from_fixture.
_R = np.array(
    [[0.999981, 0.006024, -0.001401], [-0.006045, 0.999865, -0.015283], [0.001309, 0.015291, 0.999882]]
)
_T = np.array([0.093641, 0.888888, 0.413707])
BASE = -_R.T @ _T
INWARD = np.diag([1.0, -1.0, -1.0])
OBS_BEFORE = [
    "rgb_0",
    "robot0_eef_pos  fixture xyz, meters",
    "robot0_eef_rot_axis_angle  fixture rot6d",
    "robot0_eef_wrench_left  TCP wrench",
    "robot0_eef_wrench_right  zeros",
    "robot0_mold_geom  sd, outward normal, field rot6d",
]
OBS_AFTER = [
    "rgb_0",
    "robot0_eef_pos  mold PCA xyz, meters (origin body centroid, +x toward the right-arm base)",
    "robot0_eef_rot_axis_angle  fixture rot6d",
    "robot0_eef_wrench_left  TCP wrench",
    "robot0_eef_wrench_right  zeros",
    "robot0_mold_geom  sd, outward normal, field rot6d with the new t1",
]


def mold_frame(vertices: np.ndarray, apron: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PCA of the body. +x points at the end nearer the right-arm base."""
    body = vertices[~apron] if np.any(apron) else vertices
    origin = body.mean(0)
    _, _, vt = np.linalg.svd(body - origin, full_matrices=False)
    long = vt[0].copy()
    width = vt[1].copy()
    proj = (body - origin) @ long
    hi = body[proj >= np.quantile(proj, 0.94)].mean(0)
    lo = body[proj <= np.quantile(proj, 0.06)].mean(0)
    if np.linalg.norm(hi - BASE) > np.linalg.norm(lo - BASE):
        long *= -1.0
    if float(np.dot(width, UP)) < 0.0:
        width *= -1.0
    axes = np.column_stack([long, width, np.cross(long, width)])
    return origin, axes


def reorient(frames: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """Keep column 2. Point t1 along axes[:, 0], falling back to the width axis."""
    frames = np.asarray(frames, dtype=np.float64)
    normal = frames[:, :, 2]
    normal = normal / np.clip(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12, None)
    long = np.broadcast_to(axes[:, 0], normal.shape).copy()
    tangent = long - normal * np.sum(long * normal, axis=1, keepdims=True)
    weak = np.linalg.norm(tangent, axis=1) < 0.25
    if np.any(weak):
        width = np.broadcast_to(axes[:, 1], normal.shape).copy()
        tangent[weak] = width[weak] - normal[weak] * np.sum(width[weak] * normal[weak], axis=1, keepdims=True)
    return orthonormal_frames(tangent, normal)


def _load_field(path: Path):
    data = np.load(path)
    vertices = np.asarray(data["vertices"], dtype=np.float64)
    frames = np.asarray(data["frames"], dtype=np.float64).reshape(-1, 3, 3)
    apron = np.asarray(data["is_apron"]).astype(bool) if "is_apron" in data.files else np.zeros(len(vertices), bool)
    return vertices, frames, apron


def _tangent_stats(frames: np.ndarray, axes: np.ndarray, apron: np.ndarray) -> dict:
    body = ~apron if np.any(apron) else np.ones(len(frames), dtype=bool)
    t1 = frames[body, :, 0]
    return {
        "long": np.round(axes[:, 0], 4).tolist(),
        "mean_t1": np.round(t1.mean(0), 4).tolist(),
        "mean_t1_dot_long": round(float(np.mean(t1 @ axes[:, 0])), 4),
        "mean_t1x": round(float(t1[:, 0].mean()), 4),
    }


def _span(points: np.ndarray) -> dict:
    return {
        "min": np.round(points.min(0), 4).tolist(),
        "max": np.round(points.max(0), 4).tolist(),
        "median": np.round(np.median(points, axis=0), 4).tolist(),
    }


def _tcp(zarr_path: Path) -> np.ndarray:
    root = zarr.open(str(zarr_path), "r")
    chunks = []
    for name in root["data"].keys():
        ep = int(name.split("_")[-1])
        if ep in EXCLUDE:
            continue
        pose = np.asarray(root["data"][name]["ts_pose_fb_0"], dtype=np.float64)
        chunks.append(pose[:, :3])
    return np.concatenate(chunks)


def _thin_tcp() -> dict[str, np.ndarray]:
    kept = json.loads((HELD / "contact_wall_v2/kept.json").read_text())
    out = {}
    for letter, folder, key in (("C", "C_first", "0"), ("A", "A_full", "1")):
        ep = zarr.open(str(HELD / folder / "replay.zarr"), "r")["data"]["episode_0"]
        robot_t = np.asarray(ep["robot_time_stamps_0"]).reshape(-1)
        rgb0 = float(np.asarray(ep["rgb_time_stamps_0"]).reshape(-1)[0])
        pose = np.asarray(ep["ts_pose_fb_0"], dtype=np.float64)
        times = [t for t, keep in zip(kept[key]["t"], kept[key]["keep"]) if keep]
        rel = robot_t - rgb0
        index = [int(np.argmin(np.abs(rel - t))) for t in times]
        out[letter] = pose[index, :3]
    return out


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise SystemExit(message)


def gates() -> dict:
    train_v, train_f, train_a = _load_field(TRAIN_FIELD)
    thin_v, thin_f, thin_a = _load_field(THIN_FIELD)
    train_axes = mold_frame(train_v, train_a)
    thin_axes = mold_frame(thin_v, thin_a)
    train_new = reorient(train_f, train_axes[1])
    thin_new = reorient(thin_f, thin_axes[1])
    train_stats = _tangent_stats(train_new, train_axes[1], train_a)
    thin_stats = _tangent_stats(thin_new, thin_axes[1], thin_a)
    smooth = np.load(SMOOTH)
    smooth_new = reorient(np.asarray(smooth["frames"], dtype=np.float64), train_axes[1])
    if smooth_new.shape[0] != len(train_v):
        raise SystemExit(f"20 mm frames {smooth_new.shape} do not match {len(train_v)} vertices")
    smooth_stats = _tangent_stats(smooth_new, train_axes[1], train_a)
    print("[field_v2] tangent training", json.dumps(train_stats), flush=True)
    print("[field_v2] tangent training 20mm", json.dumps(smooth_stats), flush=True)
    print("[field_v2] tangent thin", json.dumps(thin_stats), flush=True)
    _require(train_stats["long"][0] > 0.95 and train_stats["mean_t1x"] > 0.90, "training t1 left +X")
    _require(
        thin_stats["mean_t1_dot_long"] > 0.90 and thin_stats["mean_t1x"] > 0.3,
        "thin t1 is not along the part",
    )
    _require(float(smooth["radius_m"]) == 0.02, f"smooth radius is {float(smooth['radius_m'])}, not 20 mm")

    train_tcp = _tcp(SRC)
    thin_tcp = _thin_tcp()
    thin_all = np.concatenate([thin_tcp["C"], thin_tcp["A"]])
    before = {
        "training": _span(train_tcp),
        "thin": _span(thin_all),
    }
    after = {
        "training": _span(mold_local(train_tcp, *train_axes)),
        "thin": _span(mold_local(thin_all, *thin_axes)),
        "C": _span(mold_local(thin_tcp["C"], *thin_axes)),
        "A": _span(mold_local(thin_tcp["A"], *thin_axes)),
    }
    print("[field_v2] obs before", json.dumps(OBS_BEFORE), flush=True)
    print("[field_v2] obs after", json.dumps(OBS_AFTER), flush=True)
    print("[field_v2] absolute xyz", json.dumps(before), flush=True)
    print("[field_v2] mold xyz", json.dumps(after), flush=True)
    return {
        "train_axes": train_axes,
        "thin_axes": thin_axes,
        "train_frames": train_new,
        "thin_frames": thin_new,
        "smooth_frames": smooth_new,
        "train_vertices": train_v,
        "thin_vertices": thin_v,
        "train_apron": train_a,
        "thin_apron": thin_a,
        "mold_xyz": after,
    }


def _draw(ax, vertices, frames, index, title: str, length: float, view: np.ndarray) -> None:
    import matplotlib.pyplot as plt  # noqa: F401

    cloud = view[::12]
    ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=1.6, c="#4d5c68", alpha=0.55, linewidths=0)
    origin = vertices[index]
    for column, color in ((0, "#d32020"), (1, "#1f8a3b"), (2, "#1d4ed8")):
        direction = frames[index, :, column]
        ax.quiver(
            origin[:, 0], origin[:, 1], origin[:, 2],
            direction[:, 0], direction[:, 1], direction[:, 2],
            length=length, normalize=True, color=color, arrow_length_ratio=0.28, linewidth=1.1,
        )
    center = view.mean(0)
    radius = float((view.max(0) - view.min(0)).max()) / 2 * 1.05
    for setter, value in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        setter(value - radius, value + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=-58)
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()


def _fps(points: np.ndarray, n: int) -> np.ndarray:
    chosen = [int(np.argmax(points[:, 0]))]
    nearest = np.full(len(points), np.inf)
    for _ in range(n - 1):
        nearest = np.minimum(nearest, np.linalg.norm(points - points[chosen[-1]], axis=1))
        chosen.append(int(np.argmax(nearest)))
    return np.asarray(chosen)


def _stroke(points: np.ndarray, n: int) -> np.ndarray:
    kept = [0]
    for i in range(1, len(points)):
        if np.linalg.norm(points[i] - points[kept[-1]]) >= 0.025:
            kept.append(i)
    take = np.linspace(0, len(kept) - 1, min(n, len(kept))).astype(int)
    return np.asarray(kept)[take]


def snaps(spec: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree

    SNAP.mkdir(parents=True, exist_ok=True)
    thin_body = np.flatnonzero(~spec["thin_apron"])
    train_body = np.flatnonzero(~spec["train_apron"])
    c_pose = np.asarray(
        zarr.open(str(HELD / "C_first/replay.zarr"), "r")["data"]["episode_0"]["ts_pose_fb_0"]
    )
    t_pose = np.asarray(zarr.open(str(SRC), "r")["data"]["episode_0"]["ts_pose_fb_0"])

    def on_stroke(pose, vertices):
        idx = _stroke(np.asarray(pose[:, :3], dtype=np.float64), 36)
        _dist, vert = cKDTree(vertices).query(pose[idx, :3])
        return vert

    panels = (
        ("thin_stringer.png", spec["thin_vertices"], spec["thin_frames"], thin_body, on_stroke(c_pose, spec["thin_vertices"]), "thin stringer"),
        ("training_stringer.png", spec["train_vertices"], spec["train_frames"], train_body, on_stroke(t_pose, spec["train_vertices"]), "training stringer"),
    )
    for name, vertices, frames, body, stroke, title in panels:
        fig = plt.figure(figsize=(11.5, 5.2))
        left, right = fig.add_subplot(121, projection="3d"), fig.add_subplot(122, projection="3d")
        picks = body[_fps(vertices[body], 30)]
        _draw(left, vertices, frames, picks, f"{title}: 30 surface points", 0.028, vertices[body])
        _draw(right, vertices, frames, stroke, f"{title}: demo stroke", 0.028, vertices[body])
        fig.tight_layout()
        fig.savefig(SNAP / name, dpi=130)
        plt.close(fig)
        print(f"[field_v2] {SNAP / name}", flush=True)
    cap = np.arange(len(spec["thin_vertices"]) - 513, len(spec["thin_vertices"]))
    fig = plt.figure(figsize=(6.2, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    _draw(ax, spec["thin_vertices"], spec["thin_frames"], cap[::16], "thin end cap", 0.012, spec["thin_vertices"][cap])
    center = spec["thin_vertices"][cap].mean(0)
    radius = float((spec["thin_vertices"][cap].max(0) - spec["thin_vertices"][cap].min(0)).max()) / 1.4
    for setter, value in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        setter(value - radius, value + radius)
    fig.tight_layout()
    fig.savefig(SNAP / "thin_end_cap.png", dpi=130)
    plt.close(fig)
    print(f"[field_v2] {SNAP / 'thin_end_cap.png'}", flush=True)


def thin_gt(spec: dict) -> None:
    """Ground truth through the thin scorer, with the new t1 on the kept queries."""
    sys.path[:0] = [str(HELD), str(HELD.parent), str(HELD.parent.parent)]
    import heldout  # noqa: E402
    import input_check  # noqa: E402

    field = FixBMoldField(str(THIN_FIELD))
    field.frames = spec["thin_frames"]
    kept = json.loads((HELD / "contact_wall_v2/kept.json").read_text())
    report = {}
    for letter, folder, key, episode in (("C", "C_first", "0", "episode_0.npz"), ("A", "A_full", "1", "episode_1.npz")):
        demo = input_check._load(HELD / folder / "replay.zarr")
        hit = field.query(demo["foot"])
        inward = np.einsum("nij,jk->nik", hit["R"], INWARD)
        demo["frame"] = inward
        demo["dR"] = so3_to_rot6d_rows(delta_R_from_field_tool(inward, demo["tool_R"]))
        blob = np.load(HELD / "export/causal_e25" / episode)
        windows = np.asarray(blob["action_robot_idx"], dtype=np.int64)
        rel = [round(float(demo["robot_t"][int(i)] - demo["rgb_t"][0]), 3) for i in windows[:, 0]]
        want = {round(float(t), 3) for t, keep in zip(kept[key]["t"], kept[key]["keep"]) if keep}
        chosen = [windows[i] for i, t in enumerate(rel) if t in want]
        _require(len(chosen) == len(want), f"{letter} kept {len(chosen)} does not match {len(want)}")
        scored = input_check._score_gt(field, demo, spec["thin_axes"][1][:, 0], chosen)
        report[letter] = {"n": scored["n"], "pos": scored["pos"]["mean"], "rot": scored["rot"]["mean"]}
        print(f"[field_v2] thin GT {letter}", json.dumps(report[letter]), flush=True)
        _require(scored["pos"]["mean"]["1"] <= 20.0, f"{letter} step-1 position {scored['pos']['mean']['1']} mm")
        _require(scored["rot"]["mean"]["1"] <= 0.5, f"{letter} step-1 rotation {scored['rot']['mean']['1']} deg")
    return report


def carry_frames(frames: np.ndarray, tool: np.ndarray) -> np.ndarray:
    out = np.array(frames, dtype=np.float64, copy=True)
    for i in range(1, len(out)):
        ang = float(geodesic_deg(out[i - 1][None], out[i][None])[0])
        moved = float(geodesic_deg(tool[i - 1][None], tool[i][None])[0])
        if ang > CARRY_DEG and moved < CARRY_DEG:
            out[i] = out[i - 1]
    return out


def smooth_force(group, robot_t):
    wrench_t = np.asarray(group["wrench_time_stamps_left_0"]).reshape(-1)
    wrench = np.asarray(group["wrench_left_0"], dtype=np.float64)
    smooth = time_based_moving_average(wrench, wrench_t, 1.0)
    j = np.clip(np.searchsorted(wrench_t, robot_t), 1, len(wrench_t) - 1)
    left = np.abs(robot_t - wrench_t[j - 1]) <= np.abs(robot_t - wrench_t[j])
    return smooth[np.where(left, j - 1, j), :3]


def _roundtrip(frame: np.ndarray, tool: np.ndarray) -> np.ndarray:
    packed = so3_to_rot6d_rows(delta_R_from_field_tool(frame, tool)).astype(np.float32)
    got = np.einsum("nij,njk->nik", frame, rot6d_rows_to_R(packed.astype(np.float64)))
    return geodesic_deg(got, tool)


def _fit(rows: np.ndarray) -> str:
    if NORM.exists():
        raise SystemExit(f"refusing to overwrite {NORM}")
    lo = rows.min(0).astype(np.float32)
    hi = rows.max(0).astype(np.float32)
    span = np.maximum(hi - lo, np.float32(1e-7))
    scale = (np.float32(2.0) / span).astype(np.float32)
    offset = (np.float32(-1.0) - scale * lo).astype(np.float32)
    # A constant 3000 overflows the float32 offset. Same pin as the k3000 normalizer.
    scale[3] = np.float32(2.0e7)
    offset[3] = np.float32(-1.0) - scale[3] * np.float32(K)
    stats = {
        "min": lo, "max": hi,
        "mean": rows.mean(0).astype(np.float32),
        "std": rows.std(0).astype(np.float32),
    }
    lin = LinearNormalizer()
    lin["action"] = SingleFieldLinearNormalizer.create_manual(scale, offset, stats)
    value = float(scale[3] * K + offset[3])
    if abs(value) > 1e-4 or not np.all(rows[:, 3] == K):
        raise SystemExit(f"stiffness did not pin to 3000 ({value})")
    NORM.parent.mkdir(parents=True, exist_ok=True)
    with NORM.open("wb") as handle:
        pickle.dump(lin, handle)
    digest = hashlib.sha256()
    with NORM.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sbatch(sha: str) -> None:
    path = SCRIPTS / "fixb_train_oct_v35_r4vt_field_v2_k3000_e200_klone.sbatch"
    text = (SCRIPTS / "config/task/umift_oct_fixb_r4vt_field_v2.yaml").read_text().replace("SHA_HERE", sha)
    (SCRIPTS / "config/task/umift_oct_fixb_r4vt_field_v2.yaml").write_text(text)
    host = "/gscratch/scrubbed/vtevy"
    data = f"{host}/umidata/mold_aware_datasets/stringer_v3.5_fixb_odom_corr"
    code = f"{host}/UMI-FT-causal_foot"
    path.write_text(f"""#!/bin/bash
#SBATCH --job-name=fixb-r4vt-field-v2-k3000
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=320G
#SBATCH --time=24:00:00
#SBATCH --chdir=/gscratch/scrubbed/vtevy
#SBATCH --output=/gscratch/scrubbed/vtevy/logs/fixb/%j.out
#SBATCH --error=/gscratch/scrubbed/vtevy/logs/fixb/%j.err

# Field v2 of the k3000 carry run. Do not submit from this file.
# 20 mm smooth field, 30° frame carry, stiffness 3000, new PCA t1.
# robot0_eef_pos is the mold PCA position.

set -euo pipefail

SIF="${{SIF:-/gscratch/weirdlab/vtevy/images/umift-train_v3.sif}}"
UMIFT_HOST="${{UMIFT_HOST:-{code}}}"
DATA_ROOT="${{DATA_ROOT:-{data}}}"
CKPT_HOST="${{CKPT_HOST:-{host}/training_outputs/oct_v35_h4s_r4vt_field_v2_k3000}}"
WANDB_ENV="${{WANDB_ENV:-/gscratch/weirdlab/vtevy/wandb.env}}"
SCRIPTS="/opt/umift/UMIFT_Data/surface_frames_scripts"
NORM_FILE="oct_v35_h4s_r4vt_action_normalizer_minmax_field_v2_k3000.pkl"
NORM_SHA="{sha}"
RUN_TITLE="oct-v35-h4s-r4vt-field-v2-k3000_e200_bs64_valtopk_${{SLURM_JOB_ID}}"

test -f "${{SIF}}"
test -f "${{UMIFT_HOST}}/UMIFT_Data/surface_frames_scripts/fixb_r4vt_dataset.py"
test -f "${{UMIFT_HOST}}/UMIFT_Data/surface_frames_scripts/field_v2_dataset.py"
test -f "${{UMIFT_HOST}}/UMIFT_Data/surface_frames_scripts/config/task/umift_oct_fixb_r4vt_field_v2.yaml"
test -f "${{UMIFT_HOST}}/UMIFT_Data/surface_frames_scripts/assets/${{NORM_FILE}}"
test -d "${{DATA_ROOT}}/r4vt_field_v2_k3000/acp_replay_buffer.zarr"
test -f "${{DATA_ROOT}}/shared/long_dt_gt_100ms.npz"

mkdir -p "${{CKPT_HOST}}" /gscratch/scrubbed/vtevy/logs/fixb

module load apptainer

apptainer exec --nv "${{SIF}}" python -c \\
  'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'

APPTAINER_ARGS=(
  --nv
  --pwd /opt/umift/PyriteML
  --bind "${{UMIFT_HOST}}:/opt/umift"
  --bind "${{DATA_ROOT}}:/data"
  --bind "${{CKPT_HOST}}:/checkpoints"
  --env UMIFT_ROOT=/opt/umift
  --env PYRITE_DATASET_FOLDERS=/data
  --env PYRITE_CHECKPOINT_FOLDERS=/checkpoints
  --env PYTHONPATH="${{SCRIPTS}}:/opt/umift:/opt/umift/PyriteConfig:/opt/umift/PyriteUtility:/opt/umift/PyriteML"
  --env HYDRA_FULL_ERROR=1
)

if [[ -f "${{WANDB_ENV}}" ]]; then
  APPTAINER_ARGS+=(--env-file "${{WANDB_ENV}}")
else
  echo "WARNING: ${{WANDB_ENV}} not found; wandb may be offline."
fi

apptainer exec "${{APPTAINER_ARGS[@]}}" "${{SIF}}" \\
  accelerate launch --mixed_precision=bf16 --num-processes=1 \\
    /opt/umift/PyriteML/train.py --config-name=train_umift_workspace \\
    "hydra.searchpath=[file://${{SCRIPTS}}/config]" \\
    task=umift_oct_fixb_r4vt_field_v2 \\
    task.dataset_path=/data/r4vt_field_v2_k3000/acp_replay_buffer.zarr \\
    "task.dataset.action_normalizer_path=${{SCRIPTS}}/assets/${{NORM_FILE}}" \\
    task.dataset.action_normalizer_sha256="${{NORM_SHA}}" \\
    task.dataset.reduced_r4=true \\
    task.dataset.long_dt_mask_path=/data/shared/long_dt_gt_100ms.npz \\
    "task.dataset.exclusion_stats_path=/checkpoints/${{RUN_TITLE}}/exclusion_stats.json" \\
    task.dataset.pose_jump_flag_path=null \\
    "task.dataset.exclude_episodes=[35,44]" \\
    dataloader.batch_size=64 \\
    dataloader.prefetch_factor=8 \\
    dataloader.num_workers=32 \\
    val_dataloader.batch_size=64 \\
    val_dataloader.num_workers=4 \\
    val_dataloader.shuffle=True \\
    training.num_epochs=200 \\
    'training.val_at=[0,5,10,15,20,25,30,35,40,60,80,100,120,140,160,180,199]' \\
    'training.checkpoint_at=[0,5,10,15,20,25,30,35,40,60,80,100,120,140,160,180,199]' \\
    training.log_gradient_norm=true \\
    training.log_per_dim_action_loss=true \\
    checkpoint.topk.monitor_key=val_loss \\
    checkpoint.topk.mode=min \\
    checkpoint.topk.k=2 \\
    'checkpoint.topk.format_str="epoch={{epoch:04d}}-val_loss={{val_loss:.3f}}.ckpt"' \\
    checkpoint.save_last_ckpt=true \\
    task.drop_depth_inputs=true \\
    task.sparse_obs_rgb_down_sample_steps=4 \\
    task.sparse_obs_rgb_horizon=16 \\
    task.sparse_obs_depth_down_sample_steps=4 \\
    task.sparse_obs_depth_horizon=16 \\
    task.sparse_obs_low_dim_down_sample_steps=16 \\
    task.sparse_obs_low_dim_horizon=51 \\
    task.sparse_obs_mold_down_sample_steps=16 \\
    task.sparse_obs_mold_horizon=51 \\
    task.sparse_obs_wrench_down_sample_steps=1 \\
    task.sparse_obs_wrench_horizon=118 \\
    task.sparse_action_down_sample_steps=1 \\
    task.sparse_action_horizon=64 \\
    task.sparse_obs_gripper_down_sample_steps=1 \\
    task.sparse_obs_gripper_horizon=1 \\
    task.sparse_obs_ultrawide_down_sample_steps=5 \\
    task.sparse_obs_ultrawide_horizon=4 \\
    logging.name="${{RUN_TITLE}}" \\
    logging.group=oct_v35_h4s_field_v2 \\
    "logging.tags=[oct_v35_h4s,r4vt,field_v2,k3000,r4_10,october_only,exbad,no_depth,l40s,e200,bs64,valtopk]" \\
    "hydra.run.dir=/checkpoints/${{RUN_TITLE}}"
""")
    print(f"[field_v2] {path}", flush=True)


def _commands() -> None:
    tar = "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_field_v2_k3000.tar"
    remote = "/gscratch/scrubbed/vtevy/umidata/mold_aware_datasets/stringer_v3.5_fixb_odom_corr"
    code = "/gscratch/scrubbed/vtevy/UMI-FT-causal_foot/UMIFT_Data/surface_frames_scripts"
    ssh = 'ssh -o ControlMaster=auto -o ControlPersist=8h'
    print("\n".join([
        f"tar -C /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr -cf {tar} r4vt_field_v2_k3000",
        f"{ssh} vtevy@klone.hyak.uw.edu 'mkdir -p {remote} {code}/assets {code}/config/task'",
        f"rsync -a -e \"{ssh}\" {tar} vtevy@klone.hyak.uw.edu:{remote}/",
        f"{ssh} vtevy@klone.hyak.uw.edu 'tar -xf {remote}/r4vt_field_v2_k3000.tar -C {remote}'",
        f"rm {tar}",
        f"rsync -a -e \"{ssh}\" {SCRIPTS}/fixb_train_oct_v35_r4vt_field_v2_k3000_e200_klone.sbatch vtevy@klone.hyak.uw.edu:{code}/",
        f"rsync -a -e \"{ssh}\" {NORM} vtevy@klone.hyak.uw.edu:{code}/assets/",
        f"rsync -a -e \"{ssh}\" {SCRIPTS}/field_v2_dataset.py vtevy@klone.hyak.uw.edu:{code}/",
        f"rsync -a -e \"{ssh}\" {SCRIPTS}/config/task/umift_oct_fixb_r4vt_field_v2.yaml vtevy@klone.hyak.uw.edu:{code}/config/task/",
        f"sbatch {code}/fixb_train_oct_v35_r4vt_field_v2_k3000_e200_klone.sbatch",
    ]), flush=True)


def build(spec: dict) -> None:
    out_zarr = OUT / "acp_replay_buffer.zarr"
    if out_zarr.exists() or NORM.exists():
        raise SystemExit(f"refusing to overwrite {out_zarr} or {NORM}")
    frames_path = OUT / "frames_pca_r020.npz"
    OUT.mkdir(parents=True)
    np.savez(frames_path, frames=spec["smooth_frames"].astype(np.float32), radius_m=np.float32(0.02))
    origin, axes = spec["train_axes"]
    long_blob = np.load(LONG_DT)
    long_idx: dict[int, set[int]] = {}
    for ep, idx in zip(long_blob["flagged_episode"], long_blob["flagged_index"]):
        long_idx.setdefault(int(ep), set()).add(int(idx))
    tracker = CausalFoot(TRAIN_FIELD, frames_path)
    link_tree(SRC, out_zarr)
    root = zarr.open(str(out_zarr), "a")
    root.attrs["label_version"] = "field_v2_k3000_v1"
    root.attrs["smooth_frames"] = str(frames_path)
    root.attrs["delta_n"] = "-(F_smooth · n_out) / 3000, wrench moving average 1 s"
    root.attrs["frame_carry_deg"] = CARRY_DEG
    root.attrs["stiffness"] = "3000 on every row"
    root.attrs["field_v2_axes"] = {
        "origin": origin.tolist(),
        "axes": axes.tolist(),
        "end": "right_base",
        "base_fixture": BASE.tolist(),
    }
    rows = []
    trip = []
    for ep_id in range(50):
        ep = root["data"][f"episode_{ep_id}"]
        robot_t = np.asarray(ep["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
        rgb_t = np.asarray(ep["rgb_time_stamps_0"], dtype=np.float64).reshape(-1)
        rotation, position = pose7_to_Rp(np.asarray(ep["ts_pose_fb_0"], dtype=np.float64))
        foot, frame = tracker.track(position)
        frame = carry_frames(reorient(frame, axes), rotation)
        err = _roundtrip(frame, rotation)
        trip.append(err)
        if float(np.median(err)) > 0.05 or float(err.max()) > 1.0:
            raise SystemExit(f"episode {ep_id} GT round-trip median {np.median(err):.3f} max {err.max():.3f}")
        outward = -frame[:, :, 2]
        force = smooth_force(ep, robot_t)
        fn = np.einsum("ni,ni->n", np.einsum("nij,nj->ni", rotation, force), outward)
        depth = -fn / K
        velocity, dt = project_tcp_velocity(frame, position, robot_t, rgb_t)
        delta_r = so3_to_rot6d_rows(delta_R_from_field_tool(frame, rotation))
        curved = tracker.field.query(foot, with_curvature=True)
        pack = {
            "mold_x_s_0": foot.astype(np.float32),
            "mold_R_field_0": frame.reshape(-1, 9).astype(np.float32),
            "mold_R_field_rot6d_0": so3_to_rot6d_rows(frame).astype(np.float32),
            "mold_normal_0": outward.astype(np.float32),
            "mold_signed_distance_0": np.sum((position - foot) * outward, axis=1).astype(np.float32).reshape(-1, 1),
            "mold_curvature_0": np.asarray(curved["curv"], dtype=np.float32),
            "fixb_v_par_0": velocity.astype(np.float32),
            "fixb_delta_n_0": depth.reshape(-1, 1).astype(np.float32),
            "fixb_dt_next_0": dt.reshape(-1, 1).astype(np.float32),
        }
        for name in MOLD_KEYS:
            replace_array(ep, name, pack[name], ep[name])
        replace_array(ep, "ts_pose_command_0", fixb_pose7_command(velocity, depth).astype(np.float32), ep["ts_pose_command_0"])
        replace_array(ep, "fixb_delta_R_rot6d_0", delta_r.astype(np.float32), ep["fixb_delta_R_rot6d_0"])
        stored = ep["stiffness_0"]
        replace_array(ep, "stiffness_0", np.full(stored.shape, np.float32(K)), stored)
        ep.attrs["label_version"] = "field_v2_k3000_v1"
        if ep_id not in EXCLUDE:
            use = _kept(ep, ep_id, long_idx, np.zeros(len(position), dtype=bool))
            if np.any(use):
                rows.append(np.column_stack([
                    velocity[use, 0], velocity[use, 1], depth[use], np.full(int(use.sum()), K), delta_r[use],
                ]))
        print(f"[field_v2] ep {ep_id:02d} roundtrip max {float(err.max()):.4f}", flush=True)
    got = np.concatenate(rows).astype(np.float32)
    err = np.concatenate(trip)
    print(f"[field_v2] GT round-trip median {float(np.median(err)):.5f} max {float(err.max()):.5f}", flush=True)
    sha = _fit(got)
    print(f"[field_v2] normalizer {NORM} {sha} rows {len(got)}", flush=True)
    _sbatch(sha)
    _commands()


def _self_check() -> None:
    axes = np.eye(3)
    out = reorient(np.eye(3)[None], axes)
    if not np.allclose(out[0, :, 0], [1.0, 0.0, 0.0], atol=1e-8):
        raise RuntimeError("t1 did not follow +X")
    flipped = axes.copy()
    flipped[:, 0] *= -1
    out = reorient(np.eye(3)[None], flipped)
    if not np.allclose(out[0, :, 0], [-1.0, 0.0, 0.0], atol=1e-8):
        raise RuntimeError("t1 did not follow a flipped axis")
    cap = np.array([0.0, 0.0, 1.0])
    frame = np.stack([np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), cap], axis=1)[None]
    # long axis parallel to the normal: t1 falls back to +Y (the width axis).
    tall = np.column_stack([cap, np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])])
    out = reorient(frame, tall)
    if abs(float(np.dot(out[0, :, 0], [0.0, 1.0, 0.0]))) < 0.99:
        raise RuntimeError("cap fallback missed the width axis")


def main() -> None:
    _self_check()
    spec = gates()
    snaps(spec)
    after = spec["mold_xyz"]
    lo, hi = np.asarray(after["training"]["min"]), np.asarray(after["training"]["max"])
    thin_lo, thin_hi = np.asarray(after["thin"]["min"]), np.asarray(after["thin"]["max"])
    _require(
        bool(np.all(thin_lo >= lo - 0.01) and np.all(thin_hi <= hi + 0.01)),
        "thin mold position is outside training",
    )
    thin_gt(spec)
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        return
    build(spec)


if __name__ == "__main__":
    main()

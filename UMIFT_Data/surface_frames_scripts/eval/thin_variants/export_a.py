#!/usr/bin/env python3
"""Recording C, variant A TCP, split score. New files only.

Images and wrench come from the existing remap acp zarr. Poses are replaced
with the variant A solve. Recordings A/B/D have no acp zarr; this does not build one.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames")
SCRIPTS = ROOT / "UMIFT_Data/surface_frames_scripts"
sys.path[:0] = [
    str(SCRIPTS / "eval" / "tcp_dn"),
    str(SCRIPTS / "eval" / "both_ckpt"),
    str(SCRIPTS / "eval"),
    str(SCRIPTS),
    str(ROOT / "UMIFT_Data/scripts"),
    str(ROOT / "PyriteEnvSuites/scripts"),
    str(ROOT / "PyriteML"),
    str(ROOT / "PyriteConfig"),
    str(ROOT / "PyriteUtility"),
    str(ROOT),
]

import torch  # noqa: E402
import zarr  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import compare_r1rel as fair  # noqa: E402
import compare_thin as thin  # noqa: E402
import score as split  # noqa: E402
import score_both as both  # noqa: E402
from convert_unified_t265_wild_to_umift_zarr import _resample_pose7  # noqa: E402
from diffusion_policy.common.drop_depth_inputs import apply_drop_depth_inputs  # noqa: E402
from diffusion_policy.common.drop_scan_inputs import apply_drop_scan_inputs  # noqa: E402
from diffusion_policy.common.sampler_umift import sparse_action_indices_from_rgb_timestamps  # noqa: E402
from fixb_field import FixBMoldField  # noqa: E402
from fixb_labels import delta_R_from_field_tool, pose7_to_Rp, so3_to_rot6d_rows  # noqa: E402
import heading_tilt  # noqa: E402
from heading_tilt import long_axis, pack_tilt  # noqa: E402
from run_policy_open_loop_val_export import load_policy_and_cfg, predict_sparse_action  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "variant_A"
DST = OUT / "replay.zarr"
MASK = OUT / "long_dt_empty.npz"
REPORT = OUT / "score.json"
POSE = Path(
    "/mnt/data/umidata/ft/held_out_mold/thin_stringer/transform_check/"
    "variant_resolve/A/2026-11-19-15.49.52_ee_master.npz"
)
WALL = Path(
    "/mnt/data/umidata/ft/held_out_mold/thin_stringer/transform_check/"
    "wall_fix/fixb_field_wall_vertical.npz"
)
UNIFIED = Path(
    "/mnt/data/umidata/ft/datasets/flat_stringer/zarr/2026-11-19-15.49.52_unified.zarr"
)
K = 3000.0
STEPS = (1, 6, 16, 64)
H = 64


def _e_long(normal, axis):
    """Same carry as heading_tilt.e_long_sequence. That one calls float() on a length-1 array."""
    normal = np.asarray(normal, dtype=np.float64).reshape(-1, 3)
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    dropped, length = heading_tilt._tangent(np.broadcast_to(axis, normal.shape), normal)
    angle = np.degrees(np.arcsin(np.clip(length[:, 0], 0.0, 1.0)))
    good = angle >= heading_tilt.DEGEN_DEG
    direction = np.zeros_like(normal)
    usable = length[:, 0] > 1e-8
    direction[usable] = dropped[usable] / length[usable]
    horizontal = axis - np.dot(axis, heading_tilt.UP) * heading_tilt.UP
    h_norm = float(np.linalg.norm(horizontal))
    horizontal = horizontal / h_norm if h_norm > 1e-8 else np.array([0.0, 0.0, 1.0])
    last = None
    for i in range(len(direction)):
        if good[i]:
            last = direction[i].copy()
            continue
        seed = last if last is not None else horizontal
        tangent, size = heading_tilt._tangent(seed, normal[i])
        size = float(np.reshape(size, ()))
        if size < 1e-8:
            tangent, size = heading_tilt._tangent(np.array([0.0, 0.0, 1.0]), normal[i])
            size = float(np.reshape(size, ()))
        direction[i] = tangent.reshape(3) / max(size, 1e-12)
        last = direction[i].copy()
    return direction


def _check_cluster(field) -> dict:
    """The 66 mm median is the opening hover. Three later tag frames sit on the mesh."""
    blob = np.load(POSE)
    pos = np.asarray(blob["T_master_tcp"])[:, :3, 3]
    times = np.asarray(blob["timestamp_sec_unified"]).reshape(-1)
    tag = np.asarray(blob["n_tags_visible"]).astype(int) >= 1
    g = zarr.open(UNIFIED, "r")["sensors"]["ft_wrench"]
    ft = np.asarray(g["timestamp_sec_unified"])
    force = np.column_stack([np.asarray(g[k]) for k in ("force_x", "force_y", "force_z")])
    j = np.clip(np.searchsorted(ft, times), 1, len(ft) - 1)
    left = np.abs(times - ft[j - 1]) <= np.abs(times - ft[j])
    mag = np.linalg.norm(force[np.where(left, j - 1, j)], axis=1)
    dist = np.linalg.norm(pos - field.closest(pos)[0], axis=1) * 1e3
    early = tag & (times < 1.0)
    late = tag & (times >= 1.0)
    if early.sum() != 11 or late.sum() != 3:
        raise RuntimeError(f"tag split {early.sum()} {late.sum()}")
    if np.median(dist[early]) < 60 or np.max(dist[late]) > 5:
        raise RuntimeError("tag-frame distances are not an opening cluster")
    if np.median(mag[early]) > 8 or np.min(mag[late]) < 8:
        raise RuntimeError("opening tag frames are not the unloaded ones")
    return {
        "early_n": 11,
        "early_dist_mm_median": round(float(np.median(dist[early])), 1),
        "early_force_n_median": round(float(np.median(mag[early])), 1),
        "late_n": 3,
        "late_dist_mm": [round(float(v), 1) for v in dist[late]],
        "all_tag_dist_mm_median": round(float(np.median(dist[tag])), 1),
        "contact_dist_mm_median": round(float(np.median(dist[mag >= 8])), 1),
    }


def _dataset(cfg, zarr_path, norm, sha_prefix, action_dim):
    ds_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    apply_drop_scan_inputs(ds_cfg)
    apply_drop_depth_inputs(ds_cfg)
    OmegaConf.update(ds_cfg, "task.dataset_path", zarr_path, merge=False)
    OmegaConf.update(ds_cfg, "task.dataset.dataset_path", zarr_path, merge=False)
    OmegaConf.update(ds_cfg, "task.dataset.action_normalizer_path", str(norm), merge=False)
    OmegaConf.update(ds_cfg, "task.dataset.long_dt_mask_path", str(MASK), merge=False)
    for key in ("exclusion_stats_path", "filter_stats_path", "pose_jump_flag_path"):
        if key in ds_cfg.task.dataset:
            OmegaConf.update(ds_cfg, f"task.dataset.{key}", None, merge=False)
    sha = str(ds_cfg.task.dataset.action_normalizer_sha256)
    dim = int(ds_cfg.task.shape_meta.action.shape[0])
    if not sha.startswith(sha_prefix) or dim != action_dim:
        raise SystemExit(f"sha {sha} dim {dim}, expected {sha_prefix} dim {action_dim}")
    dataset = fair.hydra_instantiate(ds_cfg.task.dataset)
    if len(dataset.sampler.indices) == 0:
        dataset = dataset.get_validation_dataset()
    return dataset


def _write(field) -> dict:
    if DST.exists() or REPORT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copytree(thin.SRC, DST)
    root = zarr.open(DST, "r+")
    ep = root["data"]["episode_0"]
    rgb_t = np.asarray(ep["rgb_time_stamps_0"]).reshape(-1)
    robot_t = np.asarray(ep["robot_time_stamps_0"]).reshape(-1)
    blob = np.load(POSE)
    src_t = np.asarray(blob["timestamp_sec_unified"]).reshape(-1)
    if len(src_t) != len(rgb_t) or np.max(np.abs(src_t - rgb_t)) > 1e-6:
        raise SystemExit("variant A times do not match the remap rgb clock")
    pose = _resample_pose7(src_t, np.asarray(blob["pose7"], dtype=np.float64), robot_t)
    ep["ts_pose_fb_0"][:] = pose.astype(ep["ts_pose_fb_0"].dtype)
    tool_R, tool_p = pose7_to_Rp(pose)
    geom = field.query(tool_p, with_curvature=True)
    foot = np.asarray(geom["x_s"], dtype=np.float64).reshape(-1, 3)
    outward = np.asarray(geom["R"], dtype=np.float64).reshape(-1, 3, 3)
    frame = thin.inward_frame(outward)
    normal = outward[:, :, 2].copy()
    n = len(robot_t)
    ep["wrench_time_stamps_0"] = np.asarray(ep["wrench_time_stamps_left_0"])
    ep["ts_pose_command_0"] = pose.astype(np.float32)
    ep["ts_pose_virtual_target_0"] = pose.astype(np.float32)
    ep["stiffness_0"] = np.zeros(n, dtype=np.float32)
    ep["mold_signed_distance_0"] = np.asarray(geom["sd"], dtype=np.float32).reshape(-1, 1)
    ep["mold_normal_0"] = normal.astype(np.float32)
    ep["mold_R_field_rot6d_0"] = so3_to_rot6d_rows(frame).astype(np.float32)
    ep["mold_x_s_0"] = foot.astype(np.float32)
    ep["mold_R_field_0"] = frame.astype(np.float32)
    axis = long_axis(field.vertices)
    tilt = pack_tilt(tool_R, normal, _e_long(normal, axis))
    ep["fixb_heading_tilt_0"] = tilt.astype(np.float32)
    ep["fixb_delta_R_rot6d_0"] = so3_to_rot6d_rows(
        delta_R_from_field_tool(frame, tool_R)
    ).astype(np.float32)
    if np.asarray(ep["gripper_0"]).shape[0] != n:
        ep["gripper_0"] = np.zeros((n, 1), dtype=np.float32)
    np.savez(MASK, flagged_episode=np.zeros(0, np.int64), flagged_index=np.zeros(0, np.int64))
    wrench_t = np.asarray(ep["wrench_time_stamps_left_0"]).reshape(-1)
    wrench = np.asarray(ep["wrench_left_0"], dtype=np.float64)
    from PyriteUtility.data_pipeline.wrench_baseline_utils import time_based_moving_average

    smooth = time_based_moving_average(wrench, wrench_t, 1.0)[:, :3]
    return {
        "robot_t": robot_t,
        "rgb_t": rgb_t,
        "tool_p": tool_p,
        "tool_R": tool_R,
        "foot": foot,
        "frame": frame,
        "normal": normal,
        "smooth": smooth,
        "wrench_t": wrench_t,
        "cutoff": float(rgb_t[0]) + 0.5 * float(rgb_t[-1] - rgb_t[0]),
        "axis": axis,
    }


def _queries(dataset, demo):
    rows = []
    for idx, (episode, rgb) in enumerate(dataset.sampler.indices):
        rgb = int(rgb)
        if demo["rgb_t"][rgb] >= demo["cutoff"]:
            continue
        index, _times = sparse_action_indices_from_rgb_timestamps(
            demo["robot_t"], demo["rgb_t"], rgb, H, 1
        )
        index = np.asarray(index, dtype=np.int64)
        if len(index) < H or demo["robot_t"][index[-1]] >= demo["cutoff"]:
            continue
        rows.append((idx, rgb, index))
    return rows


def _dn(demo, index):
    t = demo["robot_t"][index]
    j = np.clip(np.searchsorted(demo["wrench_t"], t), 1, len(demo["wrench_t"]) - 1)
    left = np.abs(t - demo["wrench_t"][j - 1]) <= np.abs(t - demo["wrench_t"][j])
    j = np.where(left, j - 1, j)
    force = np.einsum("nij,nj->ni", demo["tool_R"][index], demo["smooth"][j])
    return -np.einsum("ni,ni->n", force, demo["normal"][index]) / K


def _score(name, policy, dataset, field, demo, rows, device):
    pos = []
    tang = []
    dn = []
    for idx, _rgb, index in rows:
        obs, _sample = dataset.sampler.sample_sequence(idx)
        pred = np.asarray(predict_sparse_action(policy, obs, device), dtype=np.float64)
        pred = pred.reshape(-1, pred.shape[-1])[:H]
        if pred.shape[0] < H or pred.shape[1] < 3:
            raise RuntimeError(f"{name} action {pred.shape}")
        times = demo["robot_t"][index]
        dts = np.diff(times, append=times[-1])
        dts[-1] = dts[-2]
        i0 = int(index[0])
        row = {
            "foot0": demo["foot"][i0],
            "frame0": demo["frame"][i0],
            "anchor_p": demo["tool_p"][i0],
            "dts": dts,
        }
        xyz = split.tcp_from_velocity(field, pred, row)
        err = xyz - demo["tool_p"][index]
        along = np.einsum("ni,ni->n", err, demo["normal"][index])
        pos.append(np.linalg.norm(err, axis=1) * 1e3)
        tang.append(np.linalg.norm(err - along[:, None] * demo["normal"][index], axis=1) * 1e3)
        dn.append(np.abs(pred[:, 2] - _dn(demo, index)) * 1e3)
    pos, tang, dn = np.stack(pos), np.stack(tang), np.stack(dn)
    def at(block):
        return {str(s): round(float(block[:, s - 1].mean()), 2) for s in STEPS}
    return {"n": int(len(rows)), "pos_mm": at(pos), "tangential_mm": at(tang), "dn_mm": at(dn),
            "force_n": {k: round(v * K / 1e3, 2) for k, v in at(dn).items()}}


def main() -> None:
    split._check()
    thin._self_check()
    good = _e_long(np.array([[0.0, -1.0, 0.0]]), np.array([1.0, 0.0, 0.0]))
    ref = heading_tilt.e_long_sequence(np.array([[0.0, -1.0, 0.0]]), np.array([1.0, 0.0, 0.0]))[0]
    if not np.allclose(good, ref):
        raise RuntimeError("e_long disagrees on a flat normal")
    _e_long(np.array([[1.0, 0.0, 0.0]]), np.array([1.0, 0.0, 0.0]))
    field = FixBMoldField(str(WALL))
    cluster = _check_cluster(field)
    demo = _write(field)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    specs = [
        ("causal e25", both.CAUSAL_DIR / "epoch=0025-val_loss=0.058.ckpt",
         fair.CAUSAL_NORM, "14308b72", 10, "causal"),
        ("causal e199", both.CAUSAL_DIR / "latest.ckpt",
         fair.CAUSAL_NORM, "14308b72", 10, "causal"),
        ("heading e25", both.HEAD_DIR / "epoch=0025-val_loss=0.059.ckpt",
         thin.HEAD_NORM, "e93b8756", 8, "heading"),
        ("heading e100", both.HEAD_DIR / "latest.ckpt",
         thin.HEAD_NORM, "e93b8756", 8, "heading"),
    ]
    datasets = {}
    queries = {}
    for label, path, norm, sha, dim, kind in specs:
        if not path.is_file() or path.stat().st_size < 2e9:
            raise SystemExit(f"bad checkpoint {path}")
        if kind in datasets:
            continue
        _policy, cfg = load_policy_and_cfg(str(path), device)
        datasets[kind] = _dataset(cfg, str(DST), norm, sha, dim)
        queries[kind] = _queries(datasets[kind], demo)
        print(f"[export-a] {kind} queries {len(queries[kind])}", flush=True)
        del _policy
        torch.cuda.empty_cache()
    rgb = set(queries["causal"][i][1] for i in range(len(queries["causal"])))
    rgb &= {row[1] for row in queries["heading"]}
    if len(rgb) < 2:
        raise SystemExit(f"only {len(rgb)} shared queries")
    table = {}
    for label, path, _norm, _sha, _dim, kind in specs:
        policy, _cfg = load_policy_and_cfg(str(path), device)
        use = [row for row in queries[kind] if row[1] in rgb]
        table[label] = _score(label, policy, datasets[kind], field, demo, use, device)
        print(f"[export-a] {label} {table[label]}", flush=True)
        del policy
        torch.cuda.empty_cache()
    report = {
        "recording": "2026-11-19-15.49.52",
        "variant": "A",
        "mesh": str(WALL),
        "zarr": str(DST),
        "position": "foot walked by v, plus the query TCP tangent offset. δn is not in the point.",
        "dn": "error versus -(F·n_out)/3000, mm. force_n is that error times 3000.",
        "queries": "first half of the recording, 64-step chunks that finish inside it",
        "tag_vs_contact": cluster,
        "skipped": "A/B/D recordings have unified zarrs only. No acp replay buffer to restamp.",
        "table": table,
    }
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps(table, indent=2), flush=True)


if __name__ == "__main__":
    main()

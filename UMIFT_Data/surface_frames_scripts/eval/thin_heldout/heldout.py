#!/usr/bin/env python3
"""Thin held-out score. New paths only. Reads a checkpoint list.

  python heldout.py [checkpoints.json]

Recording C is the first half of contact. Recording A is the full take.
Both use variant A poses and the wall-vertical mesh. A missing optional
checkpoint (job id not on disk yet) is skipped.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames")
SCRIPTS = ROOT / "UMIFT_Data/surface_frames_scripts"
SURF = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames/UMIFT_Data/surface_frames_scripts")
sys.path[:0] = [
    str(SCRIPTS / "eval" / "tcp_dn"),
    str(SCRIPTS / "eval"),
    str(SCRIPTS),
    str(ROOT / "UMIFT_Data"),
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
from scipy.spatial.transform import Rotation  # noqa: E402

import compare_r1rel as fair  # noqa: E402
import compare_thin as thin  # noqa: E402
import heading_tilt  # noqa: E402
import score as split  # noqa: E402
from convert_unified_t265_wild_to_umift_zarr import (  # noqa: E402
    _resample_pose7,
    build_wrench_keys,
)
from diffusion_policy.common.drop_depth_inputs import apply_drop_depth_inputs  # noqa: E402
from diffusion_policy.common.drop_scan_inputs import apply_drop_scan_inputs  # noqa: E402
from diffusion_policy.common.sampler_umift import sparse_action_indices_from_rgb_timestamps  # noqa: E402
from fixb_field import FixBMoldField, orthonormal_frames  # noqa: E402
from fixb_labels import delta_R_from_field_tool, pose7_to_Rp, so3_to_rot6d_rows  # noqa: E402
from heading_tilt import geodesic_deg, long_axis, pack_tilt, rotation_from_tilt  # noqa: E402
from mold_aware_runtime import rot6d_rows_to_R  # noqa: E402
from PyriteUtility.data_pipeline.wrench_baseline_utils import time_based_moving_average  # noqa: E402
from run_policy_open_loop_val_export import load_policy_and_cfg, predict_sparse_action  # noqa: E402
from umift.processing.umift_image_prep import resize_hwc_uint8  # noqa: E402

HERE = Path(__file__).resolve().parent
WALL = Path(
    "/mnt/data/umidata/ft/held_out_mold/thin_stringer/transform_check/"
    "wall_fix/fixb_field_wall_vertical.npz"
)
VARIANT = Path(
    "/mnt/data/umidata/ft/held_out_mold/thin_stringer/transform_check/variant_resolve/A"
)
UNIFIED = Path("/mnt/data/umidata/ft/datasets/flat_stringer/zarr")
C_SRC = SCRIPTS / "eval/thin_variants/variant_A/replay.zarr"
MASK = HERE / "long_dt_empty.npz"
REPORT = HERE / "score.json"
EXPORT = HERE / "export"
TRAIN = Path("/mnt/data/umidata/ft/training_outputs")
C_STAMP = "2026-11-19-15.49.52"
A_STAMP = "2026-11-19-15.43.29"
K = 3000.0
H = 64
STEPS = (1, 6, 16, 64)
MIN_CKPT = 2_000_000_000
I6 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
KNOWN_NORM = {
    "14308b72": fair.CAUSAL_NORM,
    "e93b8756": thin.HEAD_NORM,
    "2c20fba56851": fair.R1REL_NORM,
}
EXPECT = {"causal": ("14308b72", 10), "heading": ("e93b8756", 8), "r1rel": ("2c20fba56851", 21)}
_WALK_CHECKED = False


def _check() -> None:
    turned = Rotation.from_rotvec([0.0, 0.0, np.pi / 2]).as_matrix()
    got = float(geodesic_deg(np.eye(3)[None], turned[None])[0])
    if abs(got - 90.0) > 1e-6:
        raise RuntimeError(f"geodesic {got}")
    heading_tilt._self_check()
    flat = _e_long(np.array([[0.0, -1.0, 0.0]]), np.array([1.0, 0.0, 0.0]))
    ref = heading_tilt.e_long_sequence(np.array([[0.0, -1.0, 0.0]]), np.array([1.0, 0.0, 0.0]))[0]
    if not np.allclose(flat, ref):
        raise RuntimeError("e_long disagrees on a flat normal")
    _e_long(np.array([[1.0, 0.0, 0.0]]), np.array([1.0, 0.0, 0.0]))
    split._check()


def _e_long(normal, axis):
    """heading_tilt.e_long_sequence crashes when the tangent length is a 1-vector."""
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


def _walk(field, action, foot0, frame0, start_p, dts):
    """rollout_tool with δn zeroed, also returning the inward frame at each step."""
    global _WALK_CHECKED
    velocity = np.asarray(action[:, 0:2], dtype=np.float64)
    foot = np.asarray(foot0, dtype=np.float64).reshape(3).copy()
    frame = np.asarray(frame0, dtype=np.float64).reshape(3, 3).copy()
    _landed, n0, _t1 = fair._hit(field, foot)
    sign = 1.0 if float(np.dot(n0, frame[:, 2])) >= 0.0 else -1.0
    tangent = frame[:, :2].T @ (np.asarray(start_p, dtype=np.float64).reshape(3) - foot)
    dts = np.asarray(dts, dtype=np.float64).reshape(-1)
    pos = np.zeros((len(velocity), 3))
    frames = np.zeros((len(velocity), 3, 3))
    for k in range(len(velocity)):
        frames[k] = frame
        pos[k] = foot + frame[:, :2] @ tangent
        if k + 1 == len(velocity):
            break
        guess = foot + frame[:, :2] @ velocity[k] * float(dts[k])
        landed, normal, t1 = fair._hit(field, guess)
        if float(np.dot(sign * normal, frame[:, 2])) < 0.5:
            foot = foot + frame[:, :2] @ (frame[:, :2].T @ (guess - foot))
            continue
        foot = landed
        frame = fair._align(orthonormal_frames(t1, sign * normal), frame)
    if not _WALK_CHECKED:
        other = split.tcp_from_velocity(
            field, action, {"foot0": foot0, "frame0": frame0, "anchor_p": start_p, "dts": dts}
        )
        if not np.allclose(pos, other, atol=1e-8):
            raise RuntimeError("walked tool path does not match the split scorer")
        _WALK_CHECKED = True
    return pos, frames


def _nearest(times, query):
    j = np.clip(np.searchsorted(times, query), 1, len(times) - 1)
    left = np.abs(query - times[j - 1]) <= np.abs(query - times[j])
    return np.where(left, j - 1, j)


def _col(group, key):
    return np.asarray(group[key]).reshape(-1)


def _put(group, key, value):
    group[key] = np.asarray(value)


def _mold(pose, field, axis):
    tool_R, tool_p = pose7_to_Rp(pose)
    geom = field.query(tool_p, with_curvature=True)
    foot = np.asarray(geom["x_s"], dtype=np.float64).reshape(-1, 3)
    outward = np.asarray(geom["R"], dtype=np.float64).reshape(-1, 3, 3)
    frame = thin.inward_frame(outward)
    normal = outward[:, :, 2].copy()
    tilt = pack_tilt(tool_R, normal, _e_long(normal, axis))
    delta = so3_to_rot6d_rows(delta_R_from_field_tool(frame, tool_R))
    return {
        "mold_signed_distance_0": np.asarray(geom["sd"], dtype=np.float32).reshape(-1, 1),
        "mold_normal_0": normal.astype(np.float32),
        "mold_R_field_rot6d_0": so3_to_rot6d_rows(frame).astype(np.float32),
        "mold_x_s_0": foot.astype(np.float32),
        "mold_R_field_0": frame.astype(np.float32),
        "fixb_heading_tilt_0": tilt.astype(np.float32),
        "fixb_delta_R_rot6d_0": delta.astype(np.float32),
    }


def _slice_c(field_unused, t_mid: float) -> Path:
    del field_unused
    dst = HERE / "C_first" / "replay.zarr"
    if (dst / "data" / "episode_0" / "rgb_0").exists():
        print(f"[heldout] reuse {dst}", flush=True)
        return dst
    if dst.exists():
        raise SystemExit(f"partial zarr, not touching {dst}")
    src = zarr.open(C_SRC, "r")["data"]["episode_0"]
    rgb_t = _col(src, "rgb_time_stamps_0")
    robot_t = _col(src, "robot_time_stamps_0")
    wrench_t = _col(src, "wrench_time_stamps_0")
    limit = float(rgb_t[0]) + float(t_mid)
    keep = {
        "robot": robot_t <= limit + 1e-6,
        "rgb": rgb_t <= limit + 1e-6,
        "wrench": wrench_t <= limit + 1e-6,
    }
    clocks = {
        "rgb_0": "rgb", "rgb_time_stamps_0": "rgb", "depth_0": "rgb", "depth_time_stamps_0": "rgb",
        "ultrawide_0": "rgb", "ultrawide_time_stamps_0": "rgb",
        "ts_pose_fb_0": "robot", "robot_time_stamps_0": "robot", "ts_pose_command_0": "robot",
        "ts_pose_virtual_target_0": "robot", "stiffness_0": "robot", "gripper_0": "robot",
        "gripper_time_stamps_0": "robot", "mold_R_field_0": "robot", "mold_R_field_rot6d_0": "robot",
        "mold_normal_0": "robot", "mold_signed_distance_0": "robot", "mold_x_s_0": "robot",
        "fixb_heading_tilt_0": "robot", "fixb_delta_R_rot6d_0": "robot",
        "wrench_left_0": "wrench", "wrench_right_0": "wrench", "wrench_concat_0": "wrench",
        "wrench_left_coinft_0": "wrench", "wrench_right_coinft_0": "wrench",
        "wrench_concat_coinft_0": "wrench", "wrench_time_stamps_0": "wrench",
        "wrench_time_stamps_left_0": "wrench", "wrench_time_stamps_right_0": "wrench",
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open(dst, "w")
    ep = root.create_group("data").create_group("episode_0")
    for key, clock in clocks.items():
        _put(ep, key, np.asarray(src[key])[keep[clock]])
    n_rgb = int(keep["rgb"].sum())
    _put(ep, "map_to_d_idx_0", np.arange(n_rgb, dtype=np.int64).reshape(-1, 1))
    _put(ep, "map_to_uw_idx_0", np.arange(n_rgb, dtype=np.int64).reshape(-1, 1))
    meta = root.create_group("meta")
    counts = {"rgb": n_rgb, "depth": n_rgb, "ultrawide": n_rgb,
              "robot": int(keep["robot"].sum()), "gripper": int(keep["robot"].sum()),
              "wrench": int(keep["wrench"].sum())}
    for name, n in counts.items():
        _put(meta, f"episode_{name}0_len", np.array([n], dtype=np.int64))
    print(f"[heldout] C sliced to {t_mid}s  rgb {n_rgb} robot {counts['robot']}", flush=True)
    return dst


def _build_a(field, axis) -> Path:
    dst = HERE / "A_full" / "replay.zarr"
    if (dst / "data" / "episode_0" / "rgb_0").exists():
        print(f"[heldout] reuse {dst}", flush=True)
        return dst
    if dst.exists():
        raise SystemExit(f"partial zarr, not touching {dst}")
    uni = zarr.open(UNIFIED / f"{A_STAMP}_unified.zarr", "r")
    color = uni["camera"]["color"]
    rgb_t = np.asarray(color["timestamp_sec_unified"], dtype=np.float64).reshape(-1)
    rgb = resize_hwc_uint8(np.asarray(color["frames"]))
    if len(rgb) != len(rgb_t):
        raise RuntimeError("recording A rgb and timestamps differ")
    odom_t = np.asarray(uni["sensors"]["t265_odom"]["timestamp_sec_unified"], dtype=np.float64).reshape(-1)
    side = np.load(VARIANT / f"{A_STAMP}_ee_master.npz")
    pose = _resample_pose7(
        np.asarray(side["timestamp_sec_unified"]).reshape(-1),
        np.asarray(side["pose7"], dtype=np.float64),
        odom_t,
    )
    ft = uni["sensors"]["ft_wrench"]
    wrench_t = np.asarray(ft["timestamp_sec_unified"], dtype=np.float64).reshape(-1)
    wrench = build_wrench_keys({k: np.asarray(ft[k]) for k in (
        "force_x", "force_y", "force_z", "torque_x", "torque_y", "torque_z")},
        frame_convention="umift_std_v2")
    n = len(odom_t)
    n_rgb = len(rgb_t)
    dst.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open(dst, "w")
    ep = root.create_group("data").create_group("episode_0")
    _put(ep, "rgb_0", rgb)
    _put(ep, "rgb_time_stamps_0", rgb_t.reshape(-1, 1))
    _put(ep, "depth_0", np.zeros((n_rgb, 224, 224, 3), dtype=np.float16))
    _put(ep, "depth_time_stamps_0", rgb_t.reshape(-1, 1))
    _put(ep, "ultrawide_0", np.zeros((n_rgb, 224, 224, 3), dtype=np.uint8))
    _put(ep, "ultrawide_time_stamps_0", rgb_t.reshape(-1, 1))
    _put(ep, "map_to_d_idx_0", np.arange(n_rgb, dtype=np.int64).reshape(-1, 1))
    _put(ep, "map_to_uw_idx_0", np.arange(n_rgb, dtype=np.int64).reshape(-1, 1))
    _put(ep, "robot_time_stamps_0", odom_t.reshape(-1, 1))
    _put(ep, "ts_pose_fb_0", pose)
    _put(ep, "ts_pose_command_0", pose.astype(np.float32))
    _put(ep, "ts_pose_virtual_target_0", pose.astype(np.float32))
    _put(ep, "stiffness_0", np.zeros(n, dtype=np.float32))
    _put(ep, "gripper_0", np.zeros((n, 1), dtype=np.float64))
    _put(ep, "gripper_time_stamps_0", odom_t.reshape(-1, 1))
    for key, value in _mold(pose, field, axis).items():
        _put(ep, key, value)
    for key, value in wrench.items():
        _put(ep, key, value)
    _put(ep, "wrench_time_stamps_0", wrench_t.reshape(-1, 1))
    _put(ep, "wrench_time_stamps_left_0", wrench_t.reshape(-1, 1))
    _put(ep, "wrench_time_stamps_right_0", wrench_t.reshape(-1, 1))
    meta = root.create_group("meta")
    for name, count in (("rgb", n_rgb), ("depth", n_rgb), ("ultrawide", n_rgb),
                        ("robot", n), ("gripper", n), ("wrench", len(wrench_t))):
        _put(meta, f"episode_{name}0_len", np.array([count], dtype=np.int64))
    print(f"[heldout] A built  rgb {n_rgb} robot {n}", flush=True)
    return dst


def _demo(path: Path, cutoff):
    ep = zarr.open(path, "r")["data"]["episode_0"]
    robot_t = _col(ep, "robot_time_stamps_0")
    rgb_t = _col(ep, "rgb_time_stamps_0")
    tool_R, tool_p = pose7_to_Rp(np.asarray(ep["ts_pose_fb_0"], dtype=np.float64))
    wrench_t = _col(ep, "wrench_time_stamps_0")
    wrench = np.asarray(ep["wrench_left_0"], dtype=np.float64)
    smooth = time_based_moving_average(wrench, wrench_t, 1.0)[:, :3]
    return {
        "robot_t": robot_t,
        "rgb_t": rgb_t,
        "tool_p": tool_p,
        "tool_R": tool_R,
        "foot": np.asarray(ep["mold_x_s_0"], dtype=np.float64).reshape(-1, 3),
        "frame": np.asarray(ep["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3),
        "normal": np.asarray(ep["mold_normal_0"], dtype=np.float64).reshape(-1, 3),
        "wrench": wrench,
        "wrench_t": wrench_t,
        "smooth": smooth,
        "force": np.linalg.norm(wrench[:, :3], axis=1),
        "cutoff": cutoff,
    }


def _dataset(cfg, zarr_path, norm):
    ds_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    apply_drop_scan_inputs(ds_cfg)
    apply_drop_depth_inputs(ds_cfg)
    OmegaConf.update(ds_cfg, "task.dataset_path", str(zarr_path), merge=False)
    OmegaConf.update(ds_cfg, "task.dataset.dataset_path", str(zarr_path), merge=False)
    OmegaConf.update(ds_cfg, "task.dataset.action_normalizer_path", str(norm), merge=False)
    OmegaConf.update(ds_cfg, "task.dataset.long_dt_mask_path", str(MASK), merge=False)
    for key in ("exclusion_stats_path", "filter_stats_path", "pose_jump_flag_path"):
        if key in ds_cfg.task.dataset:
            OmegaConf.update(ds_cfg, f"task.dataset.{key}", None, merge=False)
    dataset = fair.hydra_instantiate(ds_cfg.task.dataset)
    if len(dataset.sampler.indices) == 0:
        dataset = dataset.get_validation_dataset()
    return dataset


def _queries(dataset, demo):
    rows = []
    cutoff = demo["cutoff"]
    for idx, pair in enumerate(dataset.sampler.indices):
        rgb = int(pair[1])
        if cutoff is not None and demo["rgb_t"][rgb] > cutoff:
            continue
        index, _times = sparse_action_indices_from_rgb_timestamps(
            demo["robot_t"], demo["rgb_t"], rgb, H, 1
        )
        index = np.asarray(index, dtype=np.int64)
        if len(index) < H or index[-1] >= len(demo["robot_t"]) or index[0] < 0:
            continue
        if cutoff is not None and demo["robot_t"][index[-1]] > cutoff:
            continue
        rows.append((idx, rgb, index))
    return rows


def _dn(demo, index):
    j = _nearest(demo["wrench_t"], demo["robot_t"][index])
    force = np.einsum("nij,nj->ni", demo["tool_R"][index], demo["smooth"][j])
    return -np.einsum("ni,ni->n", force, demo["normal"][index]) / K


def _dts(demo, index):
    times = demo["robot_t"][index]
    dts = np.diff(times, append=times[-1])
    dts[-1] = dts[-2]
    return dts


def _mean(block):
    return {str(step): round(float(block[:, step - 1].mean()), 2) for step in STEPS}


def _wall_anchor(field, tcp):
    """Closest point on whatever mesh `field` holds, as an inward mold frame."""
    geom = field.query(np.asarray(tcp, dtype=np.float64).reshape(1, 3))
    outward = np.asarray(geom["R"], dtype=np.float64).reshape(1, 3, 3)
    return np.asarray(geom["x_s"], dtype=np.float64).reshape(3), thin.inward_frame(outward)[0]


def _score_one(decode, policy, dataset, field, demo, rows, device, axis, foot_field=None, on_query=None, foot_mode=None):
    """foot_field replaces the mesh the foot is snapped to. None keeps the stored foot.

    foot_mode "aligned" uses field.direction (set per query). Default keeps closest-point.
    """
    pos, rot, dn, tilt_a, tilt_b = [], [], [], [], []
    paths, actions, labels, force, indices, rgbs = [], [], [], [], [], []
    for n_done, (idx, rgb, index) in enumerate(rows):
        obs, _sample = dataset.sampler.sample_sequence(idx)
        pred = np.asarray(predict_sparse_action(policy, obs, device), dtype=np.float64)
        pred = pred.reshape(-1, pred.shape[-1])[:H]
        if pred.shape != (H, pred.shape[-1]):
            raise RuntimeError(f"action {pred.shape}")
        if on_query is not None:
            on_query(pred, obs, rgb, index)
        i0 = int(index[0])
        dts = _dts(demo, index)
        if decode == "r1rel":
            xyz, got_R = fair.r1rel_pose(pred, demo["tool_p"][i0], demo["tool_R"][i0])
            dn_row = None
            ab = None
        else:
            if foot_mode == "aligned":
                import aligned_foot
                j = int(_nearest(demo["wrench_t"], np.array([demo["robot_t"][i0]]))[0])
                field.direction = aligned_foot.direction_from(
                    demo["tool_R"][i0],
                    demo["tool_R"][i0] @ demo["smooth"][j],
                    float(np.linalg.norm(demo["smooth"][j])),
                )
                foot0, frame0 = _wall_anchor(field, demo["tool_p"][i0])
                use = field
            elif foot_field is None:
                foot0, frame0, use = demo["foot"][i0], demo["frame"][i0], field
            else:
                foot0, frame0 = _wall_anchor(foot_field, demo["tool_p"][i0])
                use = foot_field
            xyz, frames = _walk(use, pred, foot0, frame0, demo["tool_p"][i0], dts)
            n_out = -frames[:, :, 2]
            e_long = _e_long(n_out, axis)
            if decode == "causal":
                got_R = np.einsum("nij,njk->nik", frames, rot6d_rows_to_R(pred[:, 4:10]))
            else:
                got_R = rotation_from_tilt(n_out, e_long, pred[:, 4:8])
            dn_row = np.abs(pred[:, 2] - _dn(demo, index)) * 1e3
            ab = np.degrees(np.abs(
                pack_tilt(got_R, n_out, e_long)[:, 2:4]
                - pack_tilt(demo["tool_R"][index], n_out, e_long)[:, 2:4]
            ))
        err = np.linalg.norm(xyz - demo["tool_p"][index], axis=1) * 1e3
        turn = geodesic_deg(got_R, demo["tool_R"][index])
        pos.append(err)
        rot.append(turn)
        paths.append(xyz)
        actions.append(pred)
        force.append(demo["force"][_nearest(demo["wrench_t"], demo["robot_t"][index])])
        indices.append(index)
        rgbs.append(rgb)
        if dn_row is not None:
            dn.append(dn_row)
            tilt_a.append(ab[:, 0])
            tilt_b.append(ab[:, 1])
            label = np.zeros((H, 27))
            label[:, 21:27] = I6
            label[:, 2] = _dn(demo, index)
            wide = np.zeros((H, 27))
            wide[:, 21:27] = I6
            wide[:, 0:3] = pred[:, 0:3]
            wide[:, 18] = pred[:, 3]
            labels.append(wide)
            actions[-1] = wide
        else:
            label = np.zeros_like(pred)
            label[:, 3:9] = I6
            labels.append(label)
        if n_done and n_done % 40 == 0:
            print(f"[heldout] {n_done}/{len(rows)}", flush=True)
    out = {
        "n": len(rows),
        "pos_mm": _mean(np.stack(pos)),
        "rot_deg": _mean(np.stack(rot)),
        "_pos": np.stack(pos),
        "_rot": np.stack(rot),
        "pred_pos": np.stack(paths),
        "pred_action": np.stack(actions),
        "label_action": np.stack(labels),
        "force_n": np.stack(force),
        "action_robot_idx": np.stack(indices),
        "query_rgb_idx": np.asarray(rgbs, dtype=np.int64),
    }
    if dn:
        out["dn_mm"] = _mean(np.stack(dn))
        out["tilt_a_deg"] = _mean(np.stack(tilt_a))
        out["tilt_b_deg"] = _mean(np.stack(tilt_b))
        out["_dn"] = np.stack(dn)
        out["_ta"] = np.stack(tilt_a)
        out["_tb"] = np.stack(tilt_b)
    return out


def _resolve(entry, root: Path):
    path = entry.get("path")
    if path:
        found = Path(path)
        if found.is_file() and found.stat().st_size >= MIN_CKPT:
            return found
        if entry.get("optional"):
            return None
        raise SystemExit(f"missing or truncated {path}")
    job = str(entry.get("job") or "")
    hits = [p for p in root.glob(f"**/*{job}*/checkpoints/*.ckpt") if p.stat().st_size >= MIN_CKPT]
    if not hits:
        return None
    want = int(entry.get("epoch") or 40)

    def epoch_of(ckpt: Path):
        match = re.search(r"epoch=(\d+)", ckpt.name)
        if match:
            return int(match.group(1))
        return fair.ckpt_epoch(ckpt)

    ranked = sorted(hits, key=lambda p: abs((epoch_of(p) if epoch_of(p) is not None else 10**6) - want))
    return ranked[0]


def _norm_of(cfg, kind: str):
    sha = str(cfg.task.dataset.action_normalizer_sha256)
    dim = int(cfg.task.shape_meta.action.shape[0])
    expect = EXPECT.get(kind)
    if expect and (not sha.startswith(expect[0]) or dim != expect[1]):
        raise SystemExit(f"{kind} sha {sha[:12]} dim {dim}")
    decode = {10: "causal", 8: "heading", 21: "r1rel"}.get(dim)
    if decode is None:
        return None
    path = next((known for prefix, known in KNOWN_NORM.items() if sha.startswith(prefix)), None)
    if path is None:
        path = Path(str(cfg.task.dataset.action_normalizer_path))
        if not path.is_file():
            path = SCRIPTS / "assets" / path.name
        if not path.is_file():
            return None
    return path, dim, decode, sha


def _slug(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")


def _save_export(name, entry, ckpt, rows_by_letter, scored):
    dest = EXPORT / _slug(name)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    val = None
    match = re.search(r"val_loss=([0-9]+(?:\.[0-9]+)?)", ckpt.name)
    if match:
        val = float(match.group(1))
    (dest / "settings.json").write_text(json.dumps({
        "role": scored["C"]["decode"] if "C" in scored else scored["A"]["decode"],
        "checkpoint_epoch": entry.get("epoch"),
        "val_loss": val,
        "execution_horizon": 16,
        "query_stride": 1,
        "ckpt": str(ckpt),
        "dataset": "",
    }))
    for letter, episode in (("C", 0), ("A", 1)):
        block = scored[letter]
        demo = rows_by_letter[letter]["demo"]
        n = block["n"]
        np.savez_compressed(
            dest / f"episode_{episode}.npz",
            robot_time_s=demo["robot_t"],
            rgb_time_s=demo["rgb_t"],
            tcp_xyz=demo["tool_p"],
            tcp_R=demo["tool_R"],
            force_series_t=demo["wrench_t"],
            force_series=demo["force"],
            bad_robot=np.zeros(len(demo["robot_t"]), dtype=bool),
            query_time_s=demo["rgb_t"][block["query_rgb_idx"]],
            query_rgb_idx=block["query_rgb_idx"],
            pred_action=block["pred_action"],
            label_action=block["label_action"],
            pred_pos=block["pred_pos"],
            n_valid=np.full(n, H, dtype=np.int64),
            force_n=block["force_n"],
            bad_span=np.zeros((n, H), dtype=bool),
            long_dt=np.zeros((n, H), dtype=bool),
            action_robot_idx=block["action_robot_idx"],
            split=np.array("first half" if letter == "C" else "full"),
        )


def _public(block):
    keys = ("n", "pos_mm", "rot_deg", "dn_mm", "tilt_a_deg", "tilt_b_deg")
    return {key: block[key] for key in keys if key in block}


def _pool(blocks):
    def cat(key):
        parts = [block[key] for block in blocks if key in block]
        return np.concatenate(parts, axis=0) if parts else None

    out = {"n": int(sum(block["n"] for block in blocks)), "pos_mm": _mean(cat("_pos")), "rot_deg": _mean(cat("_rot"))}
    dn, ta, tb = cat("_dn"), cat("_ta"), cat("_tb")
    if dn is not None:
        out["dn_mm"] = _mean(dn)
        out["tilt_a_deg"] = _mean(ta)
        out["tilt_b_deg"] = _mean(tb)
    return out


def _line(name, block):
    def seq(key):
        if key not in block:
            return " " * 23
        return "/".join(f"{block[key][str(s)]:6.1f}" for s in STEPS)

    return f"{name:16} {block['n']:4d}  {seq('pos_mm')}  {seq('dn_mm')}  {seq('rot_deg')}  {seq('tilt_a_deg')}  {seq('tilt_b_deg')}"


def _print_table(title, rows):
    print(f"\n{title}", flush=True)
    print(f"{'model':16} {'n':4}  {'pos mm 1/6/16/64':23}  {'dn mm 1/6/16/64':23}  "
          f"{'rot deg 1/6/16/64':23}  {'tilt a deg':23}  {'tilt b deg':23}", flush=True)
    for name, block in rows:
        print(_line(name, block), flush=True)


def _page_check():
    sys.path.insert(0, str(SURF))
    import view_openloop as viewer

    exports = [viewer.load_export(path) for path in sorted(EXPORT.iterdir()) if (path / "settings.json").is_file()]
    for episode in (0, 1):
        payload = viewer.episode_payload(exports, episode)
        if len(payload["runs"]) != len(exports):
            raise RuntimeError(f"episode {episode} runs {len(payload['runs'])}")
    print(f"[heldout] page payload ok  {len(exports)} runs", flush=True)


def main() -> None:
    _check()
    list_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "checkpoints.json"
    spec = json.loads(list_path.read_text())
    root = Path(spec.get("train_root") or TRAIN)
    found, skipped = [], []
    for entry in spec["models"]:
        path = _resolve(entry, root)
        if path is None:
            skipped.append(entry["name"])
            print(f"[heldout] skip {entry['name']}", flush=True)
            continue
        found.append((entry, path))
    if not found:
        raise SystemExit("no checkpoints")
    if not MASK.is_file():
        np.savez(MASK, flagged_episode=np.zeros(0, np.int64), flagged_index=np.zeros(0, np.int64))
    t_mid = json.loads((SCRIPTS / "eval/thin_variants/contact_half.json").read_text())["episodes"]["C"]["t_mid_s"]
    field = FixBMoldField(str(WALL))
    axis = long_axis(field.vertices)
    c_path = _slice_c(field, t_mid)
    a_path = _build_a(field, axis)
    recordings = {
        "C": {"path": c_path, "demo": _demo(c_path, _demo(c_path, None)["rgb_t"][0] + float(t_mid))},
        "A": {"path": a_path, "demo": _demo(a_path, None)},
    }
    recordings["C"]["demo"]["axis"] = axis
    recordings["A"]["demo"]["axis"] = axis
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = {}
    report = {"recordings": {}, "skipped": skipped, "list": str(list_path)}
    tables = {"C": [], "A": [], "combined": []}
    for entry, ckpt in found:
        policy, cfg = load_policy_and_cfg(str(ckpt), device)
        resolved = _norm_of(cfg, entry.get("kind") or "auto")
        if resolved is None:
            skipped.append(entry["name"])
            print(f"[heldout] skip {entry['name']}  unsupported action or normalizer", flush=True)
            del policy
            torch.cuda.empty_cache()
            continue
        norm, _dim, decode, sha = resolved
        scored = {}
        for letter, rec in recordings.items():
            key = (str(rec["path"]), str(norm))
            if key not in cache:
                cache[key] = _dataset(cfg, rec["path"], norm)
                cache[key + ("q",)] = _queries(cache[key], rec["demo"])
                print(f"[heldout] {letter} {decode} queries {len(cache[key + ('q',)])}", flush=True)
            rows = cache[key + ("q",)]
            if not rows:
                raise RuntimeError(f"no queries for {letter} {entry['name']}")
            print(f"[heldout] score {entry['name']} {letter}", flush=True)
            block = _score_one(decode, policy, cache[key], field, rec["demo"], rows, device, axis)
            block["decode"] = decode
            scored[letter] = block
        pooled = _pool([scored["C"], scored["A"]])
        _save_export(entry["name"], entry, ckpt, recordings, scored)
        for letter in ("C", "A"):
            public = _public(scored[letter])
            tables[letter].append((entry["name"], public))
            report["recordings"].setdefault(letter, {})[entry["name"]] = {
                **public, "ckpt": str(ckpt), "sha": sha[:12], "decode": decode,
            }
        tables["combined"].append((entry["name"], pooled))
        report["recordings"].setdefault("combined", {})[entry["name"]] = pooled
        del policy
        torch.cuda.empty_cache()
    report["window"] = {
        "C": f"first half of contact, t <= {t_mid} s, history from 0 kept",
        "A": "full recording",
        "variant": "A",
        "mesh": str(WALL),
        "position": "foot walked by v plus the query TCP tangent offset. δn is not in the point.",
        "dn": "|(pred δn) - (-(F·n_out)/3000)| mm. r1rel has no δn.",
        "rotation": "geodesic degrees, predicted tool rotation vs demo TCP",
        "tilt": "pack_tilt a/b of predicted and demo rotations, same walked frame, degrees. r1rel has none.",
        "steps": list(STEPS),
    }
    report["skipped"] = skipped
    REPORT.write_text(json.dumps(report, indent=2))
    _print_table(f"C  first half of contact (t <= {t_mid} s)  variant A", tables["C"])
    _print_table("A  full recording  variant A", tables["A"])
    _print_table("combined  unweighted over queries", tables["combined"])
    if skipped:
        print("skipped:", ", ".join(skipped), flush=True)
    _page_check()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Held-out thin stringer: causal-foot, r1rel, heading-tilt, and holding still.

The policy replay is the training-mount remap of demo 2026-11-19-15.49.52.
The second half of that recording has no AprilTags, so only the first half
of its duration is used, and only chunks that finish inside that half.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames")
SCRIPTS = ROOT / "UMIFT_Data/surface_frames_scripts"
sys.path[:0] = [str(SCRIPTS / "eval"), str(SCRIPTS), str(ROOT / "PyriteEnvSuites/scripts"),
                str(ROOT / "PyriteML"), str(ROOT / "PyriteConfig"), str(ROOT / "PyriteUtility"), str(ROOT)]

import torch
import zarr
from omegaconf import OmegaConf
from scipy.spatial import cKDTree

import compare_r1rel as base
from diffusion_policy.common.drop_depth_inputs import apply_drop_depth_inputs
from diffusion_policy.common.drop_scan_inputs import apply_drop_scan_inputs
from diffusion_policy.common.sampler_umift import sparse_action_indices_from_rgb_timestamps
from fixb_field import FixBMoldField
from fixb_labels import delta_R_from_field_tool, pose7_to_Rp, so3_to_rot6d_rows
from fixb_tcp_track import project_tcp_velocity
from heading_tilt import e_long_sequence, long_axis, pack_tilt, rotation_from_tilt
from mold_aware_runtime import rot6d_rows_to_R
from run_policy_open_loop_val_export import load_policy_and_cfg, predict_sparse_action

OUT = SCRIPTS / "eval" / "thin"
SRC = Path(
    "/mnt/data/umidata/ft/held_out_mold/thin_stringer/transform_check/"
    "remap_to_training/acp_replay_buffer.zarr"
)
FIELD_PATH = Path("/mnt/data/umidata/ft/held_out_mold/thin_stringer/field/fixb_field.npz")
MESH = Path("/mnt/data/umidata/ft/held_out_mold/thin_stringer/thin_web_fill_fixture_pts_oriented.ply")
TRAIN = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "r4vt_causal_foot/acp_replay_buffer.zarr"
)
HEAD_CKPT = Path(
    "/mnt/data/umidata/ft/training_outputs/"
    "oct-v35-h4s-r4vt-heading-tilt_sc_e200_bs64_valtopk_43448962/checkpoints/"
    "epoch=0025-val_loss=0.059.ckpt"
)
HEAD_NORM = SCRIPTS / "assets/oct_v35_h4s_r4vt_action_normalizer_minmax_heading_tilt_swing_sc.pkl"
ROBUST_CKPT = Path(
    "/mnt/data/umidata/ft/training_outputs/"
    "oct_v35_h4s_r4vt_heading_tilt_sc_robustab_mem128/epoch=0025-val_loss=0.073.ckpt"
)
ROBUST_NORM = SCRIPTS / "assets/oct_v35_h4s_r4vt_action_normalizer_minmax_heading_tilt_swing_sc_robust_ab.pkl"
SKIP = {9, 12, 28, 37, 41, 35, 44}
REGIONS = ("crown", "radius", "flange", "operator wall", "far wall")
MODELS = ("causal", "heading", "robust", "r1rel", "zero", "label")
POLICY_NAMES = ("causal", "heading", "robust", "r1rel")
COLORS = {
    "demo": "#4aa3ff", "causal": "#ff9f1a", "heading": "#7b61ff", "robust": "#c77dff",
    "r1rel": "#ff4d6a", "zero": "#8a8a8a", "label": "#2a9d8f",
}
KEYS = (
    ("pos_mm", "position mm"),
    ("rot_deg", "rotation deg"),
    ("a_deg", "wall-lean a deg"),
    ("surface_mm", "surface-distance error mm"),
    ("penetrate_mm", "penetration mm"),
)


def _self_check() -> None:
    base._self_check()
    dur = 22.0
    if abs(0.5 * dur - 11.0) > 1e-9:
        raise RuntimeError("half-duration cutoff")
    outward = np.eye(3)
    inward = outward @ np.diag([1.0, -1.0, -1.0])
    if np.dot(inward[:, 2], outward[:, 2]) > 0 or abs(np.linalg.det(inward) - 1) > 1e-9:
        raise RuntimeError("inward frame flip")


def inward_frame(rotation: np.ndarray) -> np.ndarray:
    """Query frames have z outward. The training mold frame has z inward."""
    flip = np.diag([1.0, -1.0, -1.0])
    return np.einsum("...ij,jk->...ik", rotation, flip)


def prepare(field: FixBMoldField) -> dict:
    src = zarr.open(SRC, "r")
    ep = src["data"]["episode_0"]
    robot_t = np.asarray(ep["robot_time_stamps_0"]).reshape(-1)
    rgb_t = np.asarray(ep["rgb_time_stamps_0"]).reshape(-1)
    pose = np.asarray(ep["ts_pose_fb_0"], dtype=np.float64)
    tool_R, tool_p = pose7_to_Rp(pose)
    geom = field.query(tool_p, with_curvature=True)
    foot = np.asarray(geom["x_s"], dtype=np.float64).reshape(-1, 3)
    outward_R = np.asarray(geom["R"], dtype=np.float64).reshape(-1, 3, 3)
    frame = inward_frame(outward_R)
    clearance = np.asarray(geom["sd"], dtype=np.float64).reshape(-1)
    normal = outward_R[:, :, 2].copy()
    curv = np.asarray(geom["curv"], dtype=np.float64).reshape(-1, 4)
    lift = np.linalg.norm(foot + clearance[:, None] * normal - tool_p, axis=1)
    dst_path = OUT / "replay.zarr"
    if dst_path.exists():
        import shutil
        shutil.rmtree(dst_path)
    dst = zarr.open(dst_path, "w")
    dst.create_group("meta")
    for key, value in src["meta"].items():
        dst["meta"][key] = np.asarray(value)
    group = dst.create_group("data/episode_0")
    for key, value in ep.items():
        group[key] = np.asarray(value)
    n = len(robot_t)
    if np.asarray(group["gripper_0"]).shape[0] != n:
        group["gripper_0"] = np.zeros((n, 1), dtype=np.float32)
    group["wrench_time_stamps_0"] = np.asarray(ep["wrench_time_stamps_left_0"])
    group["ts_pose_command_0"] = pose.astype(np.float32)
    group["ts_pose_virtual_target_0"] = pose.astype(np.float32)
    group["stiffness_0"] = np.zeros(n, dtype=np.float32)
    group["mold_signed_distance_0"] = clearance.astype(np.float32).reshape(-1, 1)
    group["mold_normal_0"] = normal.astype(np.float32)
    group["mold_R_field_rot6d_0"] = so3_to_rot6d_rows(frame).astype(np.float32)
    group["mold_x_s_0"] = foot.astype(np.float32)
    group["mold_R_field_0"] = frame.astype(np.float32)
    tilt = pack_tilt(tool_R, normal, e_long_sequence(normal, long_axis(field.vertices))[0])
    group["fixb_heading_tilt_0"] = tilt.astype(np.float32)
    delta = delta_R_from_field_tool(frame, tool_R)
    group["fixb_delta_R_rot6d_0"] = so3_to_rot6d_rows(delta).astype(np.float32)
    mask = OUT / "long_dt_empty.npz"
    np.savez(mask, flagged_episode=np.zeros(0, np.int64), flagged_index=np.zeros(0, np.int64))
    t0 = float(rgb_t[0])
    dur = float(rgb_t[-1] - t0)
    return {
        "zarr": str(dst_path),
        "mask": str(mask),
        "robot_t": robot_t,
        "rgb_t": rgb_t,
        "tool_p": tool_p,
        "tool_R": tool_R,
        "foot": foot,
        "frame": frame,
        "clearance": clearance,
        "normal": normal,
        "curv": curv,
        "t0": t0,
        "dur": dur,
        "cutoff": t0 + 0.5 * dur,
        "lift_mm": float(np.median(lift) * 1e3),
    }


def build_dataset(cfg, zarr_path, norm, sha_prefix, action_dim, mask):
    ds_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    apply_drop_scan_inputs(ds_cfg)
    apply_drop_depth_inputs(ds_cfg)
    OmegaConf.update(ds_cfg, "task.dataset_path", zarr_path, merge=False)
    OmegaConf.update(ds_cfg, "task.dataset.dataset_path", zarr_path, merge=False)
    OmegaConf.update(ds_cfg, "task.dataset.action_normalizer_path", str(norm), merge=False)
    OmegaConf.update(ds_cfg, "task.dataset.long_dt_mask_path", mask, merge=False)
    OmegaConf.update(ds_cfg, "task.dataset.exclusion_stats_path", None, merge=False)
    sha = str(ds_cfg.task.dataset.action_normalizer_sha256)
    dim = int(ds_cfg.task.shape_meta.action.shape[0])
    if not sha.startswith(sha_prefix) or dim != action_dim:
        raise SystemExit(f"sha {sha} dim {dim}, expected {sha_prefix} dim {action_dim}")
    dataset = base.hydra_instantiate(ds_cfg.task.dataset)
    if len(dataset.sampler.indices) == 0:
        dataset = dataset.get_validation_dataset()
    return dataset


def queries(dataset, demo):
    chosen = []
    last = -10**9
    horizon = 64
    for idx, (episode, rgb) in enumerate(dataset.sampler.indices):
        rgb = int(rgb)
        if demo["rgb_t"][rgb] >= demo["cutoff"]:
            continue
        # Stride 64 leaves one window: the 4 s lookback plus a 64-step chunk
        # has to finish before 11 s. Stride 16 keeps the windows that fit.
        if rgb - last < 16:
            continue
        index, _times = sparse_action_indices_from_rgb_timestamps(
            demo["robot_t"], demo["rgb_t"], rgb, horizon, 1
        )
        index = np.asarray(index, dtype=np.int64)
        if len(index) < horizon or demo["robot_t"][index[-1]] >= demo["cutoff"]:
            continue
        chosen.append((idx, rgb, index))
        last = rgb
    if not chosen:
        raise SystemExit("no first-half chunks finish before the cutoff")
    return chosen


def label_action(demo):
    velocity, _dt = project_tcp_velocity(demo["frame"], demo["tool_p"], demo["robot_t"], demo["rgb_t"])
    delta = delta_R_from_field_tool(demo["frame"], demo["tool_R"])
    action = np.zeros((len(velocity), 10), dtype=np.float64)
    action[:, 0:2] = velocity
    action[:, 2] = np.einsum("ni,ni->n", demo["tool_p"] - demo["foot"], demo["frame"][:, :, 2])
    action[:, 4:10] = so3_to_rot6d_rows(delta)
    return action


def heading_rotation(action, normal, axis):
    e_long, _angle = e_long_sequence(normal, axis)
    return rotation_from_tilt(normal, e_long, action[:, 4:8])


def pose_at(field, action, index, demo, kind):
    if kind == "r1rel":
        anchor = int(index[0])
        return base.r1rel_pose(action, demo["tool_p"][anchor], demo["tool_R"][anchor])
    times = demo["robot_t"][index]
    dts = np.diff(times, append=times[-1])
    if len(dts) > 1:
        dts[-1] = dts[-2]
    i0 = int(index[0])
    pos = base.rollout_tool(
        field, action, demo["foot"][i0], demo["frame"][i0], demo["tool_p"][i0], dts
    )
    if kind == "heading":
        rot = heading_rotation(action, demo["normal"][index], long_axis(field.vertices))
        return pos, rot
    delta = rot6d_rows_to_R(np.asarray(action[:, 4:10], dtype=np.float64))
    rot = np.einsum("nij,njk->nik", demo["frame"][index], delta)
    return pos, rot


def region_of(normal, foot, tree):
    up = np.array([0.0, -1.0, 0.0])
    ang = np.degrees(np.arccos(np.clip(normal @ up, -1.0, 1.0)))
    dist, _idx = tree.query(foot, k=1)
    name = np.empty(len(foot), dtype=object)
    name[:] = "radius"
    name[ang < 25.0] = "crown"
    name[ang >= 55.0] = np.where(normal[ang >= 55.0, 2] >= 0.0, "operator wall", "far wall")
    name[dist < 0.015] = "flange"
    return name


def score(field, axis, policies, demo, rows):
    per = []
    labels = label_action(demo)
    tree = cKDTree(field.vertices[np.asarray(field.meta and True) and _apron(field)])
    for n, (idx, rgb, index) in enumerate(rows, start=1):
        demo_x = demo["tool_p"][index]
        demo_R = demo["tool_R"][index]
        region = region_of(demo["normal"][index], demo["foot"][index], tree)
        poses = {}
        for name, (policy, dataset, kind) in policies.items():
            obs, _action = dataset.sampler.sample_sequence(idx)
            pred = np.asarray(predict_sparse_action(policy, obs, policies["device"]), dtype=np.float64)
            pred = pred.reshape(-1, pred.shape[-1])[:64]
            poses[name] = pose_at(field, pred, index, demo, kind)
        poses["zero"] = (
            np.broadcast_to(demo_x[0], demo_x.shape).copy(),
            np.broadcast_to(demo_R[0], demo_R.shape).copy(),
        )
        lx, lR = pose_at(field, labels[index], index, demo, "causal")
        poses["label"] = (lx, lR)
        row = {"t": float(demo["rgb_t"][rgb] - demo["t0"]), "region": region, "xyz": {}, "R": {}}
        for name, (xyz, rot) in poses.items():
            row[name] = base.metrics(xyz, rot, demo_x, demo_R, field, axis)
            row["xyz"][name] = xyz
            row["R"][name] = rot
        row["xyz"]["demo"] = demo_x
        row["R"]["demo"] = demo_R
        per.append(row)
        print(f"[thin] chunk {n}/{len(rows)} t={row['t']:.2f}s", flush=True)
    return per


def _apron(field):
    blob = np.load(FIELD_PATH)
    return blob["is_apron"].astype(bool)


def mean_at(rows, model, key, step, region=None):
    vals = []
    for row in rows:
        value = row[model][key][step - 1]
        if region is None or row["region"][step - 1] == region:
            vals.append(float(value))
    if not vals:
        return None, 0
    return float(np.mean(vals)), len(vals)


def tilt_at(rows, model, step, region=None):
    vals = []
    for row in rows:
        if "tilt_a" not in row or model not in row["tilt_a"]:
            continue
        if region is None or row["region"][step - 1] == region:
            vals.append(float(row["tilt_a"][model][step - 1]))
    if not vals:
        return None, 0
    return float(np.mean(vals)), len(vals)


def consistency(field, demo):
    """Neighbor spread of the action at similar normal, curvature, and flange distance."""
    apron = cKDTree(field.vertices[_apron(field)])
    thin = _steps(demo["tool_p"], demo["tool_R"], demo["frame"], demo["foot"], demo["normal"],
                  demo["curv"], demo["robot_t"], demo["rgb_t"], apron, demo["cutoff"], long_axis(field.vertices))
    train_field = FixBMoldField(
        "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz"
    )
    train_apron = cKDTree(train_field.vertices[np.load(
        "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz"
    )["is_apron"].astype(bool)])
    root = zarr.open(TRAIN, "r")["data"]
    parts = []
    for ep in range(50):
        if ep in SKIP or f"episode_{ep}" not in root:
            continue
        g = root[f"episode_{ep}"]
        tool_R, tool_p = pose7_to_Rp(np.asarray(g["ts_pose_fb_0"], dtype=np.float64))
        frame = np.asarray(g["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3)
        foot = np.asarray(g["mold_x_s_0"], dtype=np.float64).reshape(-1, 3)
        normal = np.asarray(g["mold_normal_0"], dtype=np.float64).reshape(-1, 3)
        robot_t = np.asarray(g["robot_time_stamps_0"]).reshape(-1)
        rgb_t = np.asarray(g["rgb_time_stamps_0"]).reshape(-1)
        # Curvature is not stored. One scalar from the normal's change is enough
        # only if we query; use the stored frame and a zero curvature placeholder
        # replaced below by the shape-operator mean when the array exists.
        if "mold_curvature_0" in g:
            curv = np.asarray(g["mold_curvature_0"], dtype=np.float64).reshape(-1, 4)
        else:
            curv = np.zeros((len(tool_p), 4))
        parts.append(_steps(
            tool_p, tool_R, frame, foot, normal, curv, robot_t, rgb_t, train_apron, None,
            long_axis(train_field.vertices),
        ))
    train = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    take = np.arange(0, len(train["geom"]), 4)
    train = {key: value[take] for key, value in train.items()}
    return {"train": _spread(train), "thin": _spread(thin), "n_train": int(len(train["geom"])), "n_thin": int(len(thin["geom"]))}


def _steps(tool_p, tool_R, frame, foot, normal, curv, robot_t, rgb_t, apron, cutoff, axis):
    velocity, dt = project_tcp_velocity(frame, tool_p, robot_t, rgb_t)
    delta = delta_R_from_field_tool(frame, tool_R)
    tilt = pack_tilt(tool_R, normal, e_long_sequence(normal, axis)[0])
    depth = np.sum((tool_p - foot) * frame[:, :, 2], axis=1)
    rgb_i = np.searchsorted(robot_t, rgb_t)
    rgb_i = rgb_i[(rgb_i > 0) & (rgb_i < len(robot_t) - 1)]
    if cutoff is not None:
        rgb_i = rgb_i[robot_t[rgb_i] < cutoff]
    step = np.einsum("nij,nj->ni", np.transpose(tool_R[rgb_i], (0, 2, 1)), tool_p[rgb_i + 1] - tool_p[rgb_i])
    rel = np.einsum("nij,njk->nik", np.transpose(tool_R[rgb_i], (0, 2, 1)), tool_R[rgb_i + 1])
    from scipy.spatial.transform import Rotation
    rotvec = Rotation.from_matrix(rel).as_rotvec()
    ok = (dt[rgb_i] > 0.02) & (dt[rgb_i] < 0.2)
    idx = rgb_i[ok]
    dist, _ = apron.query(foot[idx], k=1)
    mean_k = 0.5 * (curv[idx, 0] + curv[idx, 1])
    geom = np.column_stack([normal[idx], mean_k, dist])
    field_pos = np.column_stack([velocity[idx, 0] * dt[idx], velocity[idx, 1] * dt[idx], depth[idx]])
    field_rot = np.column_stack([np.arctan2(tilt[idx, 0], tilt[idx, 1]), tilt[idx, 2], tilt[idx, 3]])
    return {"geom": geom, "field_pos": field_pos, "field_rot": field_rot, "rel_pos": step[ok], "rel_rot": rotvec[ok]}


def _spread(pack, k=16):
    geom = np.asarray(pack["geom"], dtype=np.float64)
    mu, sig = geom.mean(0), geom.std(0)
    sig[sig < 1e-9] = 1.0
    tree = cKDTree((geom - mu) / sig)
    kk = min(k + 1, len(geom))
    _d, nn = tree.query((geom - mu) / sig, k=kk)
    out = {}
    for key in ("field_pos", "field_rot", "rel_pos", "rel_rot"):
        value = np.asarray(pack[key], dtype=np.float64)
        neigh = value[nn[:, 1:]]
        out[key] = neigh.std(axis=1).mean(axis=0)
    return out


def write_report(meta, rows, curves, spread):
    lines = [
        "# Thin stringer, first half, held-out mold",
        "",
        f"Zarr: `{SRC}` (training-mount remap of demo 2026-11-19-15.49.52, one episode).",
        f"Mesh: `{MESH}`.",
        f"Field: `{FIELD_PATH}`.",
        "",
        f"Cutoff: t < {meta['cutoff_s']:.3f} s, half of the {meta['dur_s']:.3f} s recording.",
        "The second half has no AprilTags. A chunk is kept only when its 64th step",
        "is still before that cutoff. RGB stride is 16, not 64: after the 4 s",
        f"lookback, stride 64 leaves one window. {meta['n']} chunks.",
        "Causal-foot and both heading-tilt models roll the chunk out from the query",
        "TCP. v1/v2 walk the foot on the field. The action's own δn lifts that foot.",
        "Later demonstrated feet are not used. r1rel is the predicted TCP from the",
        "same query pose. Zero holds that pose. The label is the recorded v and the",
        "TCP's own normal offset, rolled out the same way.",
        "",
        f"Causal-foot epoch {meta['causal_epoch']} `{meta['causal_ckpt']}`.",
        f"r1rel epoch {meta['r1rel_epoch']} `{meta['r1rel_ckpt']}`.",
        f"Heading-tilt 43448962 epoch {meta['heading_epoch']} `{meta['heading_ckpt']}`.",
        f"Heading-tilt robust-ab epoch {meta['robust_epoch']} `{meta['robust_ckpt']}`.",
        "",
        "## All regions",
        "",
        "| metric | step | causal | heading | robust | r1rel | zero | label |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, title in KEYS:
        for step in base.REPORT_STEPS:
            cells = [f"{curves[m][key][step]:.2f}" for m in MODELS]
            lines.append(f"| {title} | {step} | {' | '.join(cells)} |")
    lines += ["", "## Per region", ""]
    for region in REGIONS:
        lines += [
            f"### {region}",
            "",
            "| metric | step | causal | heading | robust | r1rel | zero | n |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for key, title in KEYS:
            for step in base.REPORT_STEPS:
                cells = []
                n = 0
                for model in ("causal", "heading", "robust", "r1rel", "zero"):
                    value, n = mean_at(rows, model, key, step, region)
                    cells.append("—" if value is None else f"{value:.2f}")
                lines.append(f"| {title} | {step} | {' | '.join(cells)} | {n} |")
        lines.append("")
    lines += [
        "## Tilt a, heading models",
        "",
        "Absolute error of the action's tilt `a`, degrees, against the tilt of the",
        "demonstrated tool. Steps 1, 6 and 64.",
        "",
        "| region | step | heading e25 | robust-ab e25 | n |",
        "|---|---:|---:|---:|---:|",
    ]
    for region in ("all",) + REGIONS:
        for step in (1, 6, 64):
            cells = []
            n = 0
            for model in ("heading", "robust"):
                value, n = tilt_at(rows, model, step, None if region == "all" else region)
                cells.append("—" if value is None else f"{value:.2f}")
            lines.append(f"| {region} | {step} | {' | '.join(cells)} | {n} |")
    lines.append("")
    lines += [
        "## Label consistency",
        "",
        "Spread is the mean, over steps, of the standard deviation of the 16 nearest",
        "steps in (outward normal, mean curvature, distance to the flange).",
        "Field position is (v1·dt, v2·dt, δn) in mm. Field rotation is (ψ, a, b) in degrees.",
        "r1rel position is the one-step tool displacement in the tool frame, in mm.",
        "r1rel rotation is that step's rotvec, in degrees.",
        f"Training steps {spread['n_train']} (fixb, val and episodes 35 and 44 left out).",
        f"Thin steps {spread['n_thin']} (first half only).",
        "",
        "| action | train | thin |",
        "|---|---|---|",
    ]
    names = {
        "field_pos": ("v1·dt mm", "v2·dt mm", "δn mm"),
        "field_rot": ("ψ deg", "a deg", "b deg"),
        "rel_pos": ("r1rel x mm", "r1rel y mm", "r1rel z mm"),
        "rel_rot": ("r1rel rx deg", "r1rel ry deg", "r1rel rz deg"),
    }
    scale = {"field_pos": 1e3, "field_rot": 180 / np.pi, "rel_pos": 1e3, "rel_rot": 180 / np.pi}
    for key, labels in names.items():
        for i, label in enumerate(labels):
            a = float(spread["train"][key][i]) * scale[key]
            b = float(spread["thin"][key][i]) * scale[key]
            lines.append(f"| {label} | {a:.2f} | {b:.2f} |")
    lines += [
        "",
        "Pictures show the first 16 steps. The straight line is tool z.",
        "",
    ]
    (OUT / "compare_thin.md").write_text("\n".join(lines))


def render(field_path, rows, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from view_oct import decimate_mesh

    blob = np.load(field_path)
    vertices, faces = decimate_mesh(blob["vertices"], blob["faces"], blob["is_apron"])
    def plot3(xyz):
        xyz = np.asarray(xyz, dtype=np.float64)
        return xyz[..., 0], xyz[..., 2], -xyz[..., 1]
    vx, vy, vz = plot3(vertices)
    tris = np.stack([np.column_stack([vx[f], vy[f], vz[f]]) for f in faces])

    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.4), dpi=120)
    step = np.arange(1, 65)
    for ax, (key, title) in zip(axes.ravel(), KEYS):
        for model in MODELS:
            ax.plot(step, [curves[model][key][s] for s in step], color=COLORS[model], lw=1.3,
                    ls="--" if model == "label" else "-", label=model)
        for mark in base.REPORT_STEPS:
            ax.axvline(mark, color="#ddd", lw=0.6)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=8)
    axes.ravel()[-1].axis("off")
    axes.ravel()[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "errors.png")
    plt.close(fig)

    # Consistency bars are written by the caller via curves["spread"] if present.
    row = rows[0]
    fig = plt.figure(figsize=(8.2, 6.2), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(Poly3DCollection(tris, alpha=0.35, facecolor="#9aabb6", edgecolor="none"))
    for model in ("demo", "causal", "heading", "robust", "r1rel"):
        xs, ys, zs = plot3(row["xyz"][model][:16])
        ax.plot(xs, ys, zs, color=COLORS[model], lw=2.2, label=model)
        o = row["xyz"][model][15]
        tip = o + 0.08 * row["R"][model][15][:, 2]
        ax.plot(*plot3(np.stack([o, tip])), color=COLORS[model], lw=3.0)
    pts = np.concatenate([row["xyz"][m][:16] for m in ("demo", "causal", "heading", "robust", "r1rel")])
    x, y, z = plot3(pts)
    c = np.array([np.median(x), np.median(y), np.median(z)])
    span = float(np.percentile(np.abs(np.stack([x - c[0], y - c[1], z - c[2]])), 98)) * 1.4
    ax.set_xlim(c[0] - span, c[0] + span)
    ax.set_ylim(c[1] - span, c[1] + span)
    ax.set_zlim(c[2] - span, c[2] + span)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=-70)
    ax.set_title(f"thin stringer at {row['t']:.1f} s, first 16 steps, line is tool z", fontsize=11)
    ax.legend(fontsize=8)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(OUT / "path.png")
    plt.close(fig)

    frame_dir = OUT / "frames"
    frame_dir.mkdir(exist_ok=True)
    for old in frame_dir.glob("frame_*.png"):
        old.unlink()
    for i, row in enumerate(rows):
        fig = plt.figure(figsize=(8.2, 6.2), dpi=110)
        ax = fig.add_subplot(111, projection="3d")
        ax.add_collection3d(Poly3DCollection(tris, alpha=0.35, facecolor="#9aabb6", edgecolor="none"))
        for model in ("demo", "causal", "heading", "robust", "r1rel"):
            xs, ys, zs = plot3(row["xyz"][model][:16])
            ax.plot(xs, ys, zs, color=COLORS[model], lw=2.2, label=model)
            o = row["xyz"][model][15]
            tip = o + 0.08 * row["R"][model][15][:, 2]
            ax.plot(*plot3(np.stack([o, tip])), color=COLORS[model], lw=3.0)
        pts = np.concatenate([row["xyz"][m][:16] for m in ("demo", "causal", "heading", "robust", "r1rel")])
        x, y, z = plot3(pts)
        c = np.array([np.median(x), np.median(y), np.median(z)])
        span = max(float(np.percentile(np.abs(np.stack([x - c[0], y - c[1], z - c[2]])), 98)) * 1.4, 0.02)
        ax.set_xlim(c[0] - span, c[0] + span)
        ax.set_ylim(c[1] - span, c[1] + span)
        ax.set_zlim(c[2] - span, c[2] + span)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=18, azim=-70)
        ax.set_title(f"t={row['t']:.1f} s — line is tool z", fontsize=11)
        ax.legend(fontsize=8)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(frame_dir / f"frame_{i:03d}.png")
        plt.close(fig)


def spread_figure(spread):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels, train, thin = [], [], []
    names = (
        ("field_pos", ("v1 dt", "v2 dt", "dn"), 1e3),
        ("field_rot", ("psi", "a", "b"), 180 / np.pi),
        ("rel_pos", ("body x", "body y", "body z"), 1e3),
        ("rel_rot", ("rx", "ry", "rz"), 180 / np.pi),
    )
    for key, labs, scale in names:
        for i, lab in enumerate(labs):
            labels.append(lab)
            train.append(float(spread["train"][key][i]) * scale)
            thin.append(float(spread["thin"][key][i]) * scale)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 3.6), dpi=120)
    ax.bar(x - 0.18, train, width=0.36, label="fixb train", color="#4aa3ff")
    ax.bar(x + 0.18, thin, width=0.36, label="thin first half", color="#ff9f1a")
    ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("neighbor std")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Action spread given local geometry")
    fig.tight_layout()
    fig.savefig(OUT / "consistency.png")
    plt.close(fig)


def main() -> None:
    _self_check()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[thin] device", device, flush=True)
    field = FixBMoldField(FIELD_PATH)
    axis = long_axis(field.vertices)
    demo = prepare(field)
    print(
        f"[thin] cutoff {demo['cutoff'] - demo['t0']:.3f} s of {demo['dur']:.3f} s, "
        f"lift {demo['lift_mm']:.2f} mm",
        flush=True,
    )
    causal_path, causal_policy, causal_cfg, _failed = base.load_causal(device)
    head_policy, head_cfg = load_policy_and_cfg(str(HEAD_CKPT), device)
    robust_policy, robust_cfg = load_policy_and_cfg(str(ROBUST_CKPT), device)
    r1_policy, r1_cfg = load_policy_and_cfg(str(base.R1REL_CKPT), device)
    specs = {
        "causal": (causal_policy, causal_cfg, base.CAUSAL_NORM, "14308b72", 10, "causal"),
        "heading": (head_policy, head_cfg, HEAD_NORM, "e93b8756", 8, "heading"),
        "robust": (robust_policy, robust_cfg, ROBUST_NORM, "3f9b16b1", 8, "heading"),
        "r1rel": (r1_policy, r1_cfg, base.R1REL_NORM, "2c20fba56851", 21, "r1rel"),
    }
    policies = {"device": device}
    rows_q = None
    for name, (policy, cfg, norm, sha, dim, kind) in specs.items():
        dataset = build_dataset(cfg, demo["zarr"], norm, sha, dim, demo["mask"])
        if rows_q is None:
            rows_q = queries(dataset, demo)
            print(f"[thin] chunks {len(rows_q)}", flush=True)
        policies[name] = (policy, dataset, kind)
    # sample_sequence indices differ per dataset if filters differ. Recompute per dataset.
    # The RGB ids are what must match. Rebuild queries inside score from the first dataset
    # and look up the same rgb in the others.
    base_rows = rows_q
    rgb_index = {}
    for name in POLICY_NAMES:
        _policy, dataset, _kind = policies[name]
        rgb_index[name] = {int(rgb): i for i, (_ep, rgb) in enumerate(dataset.sampler.indices)}
    aligned = []
    for idx, rgb, index in base_rows:
        if all(rgb in rgb_index[name] for name in POLICY_NAMES):
            aligned.append((rgb, index, {name: rgb_index[name][rgb] for name in POLICY_NAMES}))
    if not aligned:
        raise SystemExit("the three datasets share no first-half queries")
    print(f"[thin] shared chunks {len(aligned)}", flush=True)
    labels = label_action(demo)
    tree = cKDTree(field.vertices[_apron(field)])
    per = []
    for n, (rgb, index, loc) in enumerate(aligned, start=1):
        demo_x = demo["tool_p"][index]
        demo_R = demo["tool_R"][index]
        region = region_of(demo["normal"][index], demo["foot"][index], tree)
        row = {
            "t": float(demo["rgb_t"][rgb] - demo["t0"]),
            "region": region,
            "xyz": {"demo": demo_x},
            "R": {"demo": demo_R},
            "tilt_a": {},
        }
        e_long, _angle = e_long_sequence(demo["normal"][index], axis)
        label_a = np.degrees(pack_tilt(demo_R, demo["normal"][index], e_long)[:, 2])
        for name in POLICY_NAMES:
            policy, dataset, kind = policies[name]
            obs, _action = dataset.sampler.sample_sequence(loc[name])
            pred = np.asarray(predict_sparse_action(policy, obs, device), dtype=np.float64)
            pred = pred.reshape(-1, pred.shape[-1])[:64]
            xyz, rot = pose_at(field, pred, index, demo, kind)
            row[name] = base.metrics(xyz, rot, demo_x, demo_R, field, axis)
            row["xyz"][name] = xyz
            row["R"][name] = rot
            if name in ("heading", "robust"):
                row["tilt_a"][name] = np.abs(np.degrees(pred[:, 6]) - label_a)
        xyz, rot = pose_at(field, labels[index], index, demo, "causal")
        row["label"] = base.metrics(xyz, rot, demo_x, demo_R, field, axis)
        row["xyz"]["zero"] = np.broadcast_to(demo_x[0], demo_x.shape).copy()
        row["R"]["zero"] = np.broadcast_to(demo_R[0], demo_R.shape).copy()
        row["zero"] = base.metrics(row["xyz"]["zero"], row["R"]["zero"], demo_x, demo_R, field, axis)
        row["xyz"]["label"] = xyz
        row["R"]["label"] = rot
        per.append(row)
        print(f"[thin] chunk {n}/{len(aligned)} t={row['t']:.2f}s", flush=True)
    del causal_policy, head_policy, robust_policy, r1_policy
    torch.cuda.empty_cache()
    curves = {m: {key: {} for key, _title in KEYS} for m in MODELS}
    for model in MODELS:
        for key, _title in KEYS:
            for step in range(1, 65):
                value, _n = mean_at(per, model, key, step)
                curves[model][key][step] = value
    spread = consistency(field, demo)
    meta = {
        "cutoff_s": demo["cutoff"] - demo["t0"],
        "dur_s": demo["dur"],
        "n": len(per),
        "lift_mm": demo["lift_mm"],
        "causal_epoch": base.ckpt_epoch(causal_path),
        "causal_ckpt": str(causal_path),
        "r1rel_epoch": base.ckpt_epoch(base.R1REL_CKPT),
        "r1rel_ckpt": str(base.R1REL_CKPT),
        "heading_epoch": base.ckpt_epoch(HEAD_CKPT),
        "heading_ckpt": str(HEAD_CKPT),
        "robust_epoch": base.ckpt_epoch(ROBUST_CKPT),
        "robust_ckpt": str(ROBUST_CKPT),
    }
    write_report(meta, per, curves, spread)
    render(FIELD_PATH, per, curves)
    spread_figure(spread)
    import shutil
    import subprocess
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg and any((OUT / "frames").glob("frame_*.png")):
        subprocess.check_call([
            ffmpeg, "-y", "-framerate", "2", "-i", str(OUT / "frames" / "frame_%03d.png"),
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(OUT / "wall.mp4"),
        ])
        subprocess.check_call([
            ffmpeg, "-y", "-i", str(OUT / "wall.mp4"),
            "-vf", "fps=2,scale=720:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
            str(OUT / "wall.gif"),
        ])
    payload = {
        "meta": meta,
        "mean": {m: {k: {str(s): curves[m][k][s] for s in base.REPORT_STEPS} for k, _t in KEYS} for m in MODELS},
        "spread": {
            side: {key: np.asarray(spread[side][key]).tolist() for key in spread[side]}
            for side in ("train", "thin")
        },
    }
    (OUT / "compare_thin.json").write_text(json.dumps(payload, indent=2))
    print(f"[thin] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

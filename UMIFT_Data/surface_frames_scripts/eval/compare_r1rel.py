#!/usr/bin/env python3
"""World-frame tool-pose comparison of causal-foot and r1rel.

Same validation episodes, same RGB query times. Both action chunks are turned
into a fixture-frame tool pose and scored against the demonstrated TCP.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames")
SCRIPTS = ROOT / "UMIFT_Data/surface_frames_scripts"
sys.path[:0] = [
    str(SCRIPTS),
    str(ROOT / "PyriteEnvSuites/scripts"),
    str(ROOT / "PyriteML"),
    str(ROOT / "PyriteConfig"),
    str(ROOT / "PyriteUtility"),
    str(ROOT),
]

import torch
import zarr
from omegaconf import OmegaConf

from diffusion_policy.common.drop_depth_inputs import apply_drop_depth_inputs
from diffusion_policy.common.drop_scan_inputs import apply_drop_scan_inputs
from diffusion_policy.common.sampler_umift import sparse_action_indices_from_rgb_timestamps
from fixb_field import FixBMoldField, orthonormal_frames
from fixb_labels import pose7_to_Rp
from heading_tilt import e_long_sequence, geodesic_deg, long_axis, pack_tilt
from mold_aware_runtime import rot6d_rows_to_R
from openloop_causal_foot import held_out_episodes
from run_policy_open_loop_val_export import load_policy_and_cfg, predict_sparse_action
import PyriteUtility.spatial_math.spatial_utilities as su
from PyriteConfig.tasks.umift.umift_type_conversions import action21_postprocess

OUT = SCRIPTS / "eval"
DATA = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr")
FIELD_PATH = DATA / "shared/fixb_field.npz"
LONG_DT = str(DATA / "shared/long_dt_gt_100ms.npz")
CAUSAL_ZARR = str(DATA / "r4vt_causal_foot/acp_replay_buffer.zarr")
CAUSAL_NORM = SCRIPTS / "assets/oct_v35_h4s_r4vt_action_normalizer_minmax_causal_foot.pkl"
CAUSAL_DIR = Path(
    "/mnt/data/umidata/ft/training_outputs/oct_v35_h4s_r4vt_causal_foot/"
    "oct-v35-h4s-r4vt-causal-foot_e200_bs64_valtopk_43390127/checkpoints"
)
# Last saved epoch first. Epoch 199 is the file named latest.ckpt.
CAUSAL_CANDIDATES = (
    CAUSAL_DIR / "latest.ckpt",
    CAUSAL_DIR / "epoch=0120-val_loss=0.148.ckpt",
    CAUSAL_DIR / "epoch=0040-val_loss=0.063.ckpt",
    CAUSAL_DIR / "epoch=0025-val_loss=0.058.ckpt",
)
R1REL_ZARR = str(DATA / "r1rel_world/acp_replay_buffer.zarr")
R1REL_NORM = SCRIPTS / "assets/oct_v35_h4s_r1rel_odom_action_pose_normalizer_2c20fba56851.pkl"
# Lowest local val loss on the odom r1rel run. The e200 train-loss top-k kept
# epoch 195 (val 0.152) and deleted this epoch; the val-loss file is epoch 25.
R1REL_CKPT = Path(
    "/mnt/data/umidata/ft/training_outputs/oct_v35_h4s_odom_r1rel_valtopk/"
    "checkpoints/epoch=0025-val_loss=0.032.ckpt"
)
DT = 1.0 / 15.0
STRIDE = 64
HORIZON = 64
REPORT_STEPS = (1, 6, 16, 64)  # 1-indexed positions in the chunk
UP = np.array([0.0, -1.0, 0.0])
WALL_DEG = 55.0
MODELS = ("causal", "label", "r1rel", "zero")


def _self_check() -> None:
    from heading_tilt import _self_check as tilt_check

    tilt_check()
    eye = np.eye(3)[None]
    if float(geodesic_deg(eye, eye)[0]) != 0.0:
        raise RuntimeError("geodesic of identity is not 0")

    class _Flat:
        normals = np.array([[0.0, 0.0, 1.0]])
        frames = np.eye(3)[None]

        def closest(self, pts):
            p = np.asarray(pts, dtype=np.float64).reshape(1, 3)
            return p, np.zeros((1, 3), dtype=np.int64), np.ones((1, 3))

    action = np.zeros((3, 10))
    action[:, 2] = 0.05
    action[0, 0] = 1.0
    pos = rollout_tool(
        _Flat(), action, np.zeros(3), np.eye(3), np.array([0.0, 0.02, 0.05]), np.full(3, 0.1)
    )
    if np.linalg.norm(pos[0] - np.array([0.0, 0.02, 0.05])) > 1e-6:
        raise RuntimeError("rollout left the query pose")
    if np.linalg.norm(pos[1] - np.array([0.1, 0.02, 0.05])) > 1e-6:
        raise RuntimeError("rollout did not integrate v and the predicted clearance")


def hydra_instantiate(node):
    import hydra

    return hydra.utils.instantiate(node)


def build_dataset(cfg, zarr_path, norm, sha_prefix, action_dim):
    ds_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    apply_drop_scan_inputs(ds_cfg)
    apply_drop_depth_inputs(ds_cfg)
    OmegaConf.update(ds_cfg, "task.dataset_path", zarr_path, merge=False)
    OmegaConf.update(ds_cfg, "task.dataset.dataset_path", zarr_path, merge=False)
    OmegaConf.update(ds_cfg, "task.dataset.action_normalizer_path", str(norm), merge=False)
    OmegaConf.update(ds_cfg, "task.dataset.long_dt_mask_path", LONG_DT, merge=False)
    OmegaConf.update(ds_cfg, "task.dataset.exclusion_stats_path", None, merge=False)
    sha = str(ds_cfg.task.dataset.action_normalizer_sha256)
    dim = int(ds_cfg.task.shape_meta.action.shape[0])
    if not sha.startswith(sha_prefix) or dim != action_dim:
        raise SystemExit(f"sha {sha} dim {dim}, expected {sha_prefix} dim {action_dim}")
    dataset = hydra_instantiate(ds_cfg.task.dataset)
    stride = int(ds_cfg.task.shape_meta.sample.action.sparse.down_sample_steps)
    horizon = int(ds_cfg.task.shape_meta.sample.action.sparse.horizon)
    return dataset.get_validation_dataset(), stride, horizon


def load_causal(device):
    errors = []
    for path in CAUSAL_CANDIDATES:
        try:
            policy, cfg = load_policy_and_cfg(str(path), device)
        except Exception as exc:
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
            print(f"[compare] unreadable {path.name}: {exc}", flush=True)
            continue
        return path, policy, cfg, errors
    raise SystemExit("no causal-foot checkpoint loaded\n" + "\n".join(errors))


def ckpt_epoch(path: Path) -> int | None:
    """Epoch stored in the checkpoint pickle. None if the blob is not a small int."""
    import io
    import pickle
    import zipfile

    class U(pickle.Unpickler):
        def persistent_load(self, pid):
            return None

    raw = zipfile.ZipFile(path).read("archive/data.pkl")
    i = raw.rfind(b"epoch")
    window = raw[i : i + 80]
    # codecs.encode(latin1) of PROTO4 BININT1 <epoch> STOP, written as utf-8.
    marker = window.find(b"\xc2\x80\x04K")
    if marker < 0:
        return None
    lead = window[marker + 4]
    if lead < 128:
        return int(lead)
    return ((lead & 0x1F) << 6) | (window[marker + 5] & 0x3F)


def episode_arrays(zarr_path: str, episode: int):
    group = zarr.open(zarr_path, "r")["data"][f"episode_{episode}"]
    rotation, position = pose7_to_Rp(np.asarray(group["ts_pose_fb_0"], dtype=np.float64))
    robot_t = np.asarray(group["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
    rgb_t = np.asarray(group["rgb_time_stamps_0"], dtype=np.float64).reshape(-1)
    return position, rotation, robot_t, rgb_t


def query_map(dataset) -> dict[tuple[int, int], int]:
    found = {}
    for idx, pair in enumerate(dataset.sampler.indices):
        episode, rgb = int(pair[0]), int(pair[1])
        found[(episode, rgb)] = idx
    return found


def common_queries(causal_ds, r1rel_ds, episodes) -> list[tuple[int, int, int, int]]:
    """(episode, rgb, causal_idx, r1rel_idx) every STRIDE rgb frames, both samplers."""
    causal = query_map(causal_ds)
    r1rel = query_map(r1rel_ds)
    chosen = []
    last = {}
    keys = sorted(set(causal) & set(r1rel))
    for episode, rgb in keys:
        if episode not in episodes:
            continue
        prev = last.get(episode)
        if prev is not None and rgb - prev < STRIDE:
            continue
        chosen.append((episode, rgb, causal[(episode, rgb)], r1rel[(episode, rgb)]))
        last[episode] = rgb
    if not chosen:
        raise SystemExit("no shared validation queries")
    return chosen


def _hit(field, point):
    hit, vid, w = field.closest(np.asarray(point, dtype=np.float64).reshape(1, 3))
    n = np.asarray(w[0] @ field.normals[vid[0]], dtype=np.float64)
    n = n / max(float(np.linalg.norm(n)), 1e-12)
    t1 = np.asarray(w[0] @ field.frames[vid[0], :, 0], dtype=np.float64)
    return np.asarray(hit, dtype=np.float64).reshape(3), n, t1


def _align(new, prev):
    new = np.array(new, dtype=np.float64, copy=True)
    if float(np.dot(new[:, 2], prev[:, 2])) < 0.0:
        new = new @ np.diag([1.0, -1.0, -1.0])
    if float(np.dot(new[:, 0], prev[:, 0])) < 0.0:
        new = new @ np.diag([-1.0, -1.0, 1.0])
    return new


def _walked_frames(field, action, foot0, frame0, dts):
    """Field frames along v. Used when the caller has no label frames."""
    velocity = np.asarray(action[:, 0:2], dtype=np.float64)
    foot = np.asarray(foot0, dtype=np.float64).reshape(3).copy()
    frame = np.asarray(frame0, dtype=np.float64).reshape(3, 3).copy()
    _landed, n0, _t1 = _hit(field, foot)
    sign = 1.0 if float(np.dot(n0, frame[:, 2])) >= 0.0 else -1.0
    dts = np.asarray(dts, dtype=np.float64).reshape(-1)
    frames = np.empty((len(velocity), 3, 3))
    for k in range(len(velocity)):
        frames[k] = frame
        if k + 1 == len(velocity):
            break
        guess = foot + frame[:, :2] @ velocity[k] * float(dts[k])
        landed, n, t1 = _hit(field, guess)
        if float(np.dot(sign * n, frame[:, 2])) < 0.5:
            foot = foot + frame[:, :2] @ (frame[:, :2].T @ (guess - foot))
            continue
        foot = landed
        frame = _align(orthonormal_frames(t1, sign * n), frame)
    return frames


def rollout_tool(field, action, foot0, frame0, start_p, dts, frames=None):
    """Tool point of one chunk. v integrates from the query TCP. δn is not added.

    Pass the label frames when they are known. Otherwise the foot is walked so
    each step has a tangent basis. dt must be the label builder's control step.
    """
    from r4vt_horizon import integrate_tangent

    if frames is None:
        frames = _walked_frames(field, action, foot0, frame0, dts)
    return integrate_tangent(np.asarray(action[:, 0:2], dtype=np.float64), frames, dts, start_p)


def causal_pose(field, action, anchor_p, feet, frames, dts):
    """Open-loop tool position, rotation in the demonstrated mold frame."""
    frames = np.asarray(frames, dtype=np.float64)
    pos = rollout_tool(field, action, feet[0], frames[0], anchor_p, dts, frames=frames)
    from r4vt_horizon import compose_field_delta

    return pos, compose_field_delta(frames, action[:, 4:10])


def r1rel_pose(action, anchor_p, anchor_R):
    base = np.eye(4, dtype=np.float64)
    base[:3, :3] = anchor_R
    base[:3, 3] = anchor_p
    command, _vt, _stiff, _eoat = action21_postprocess(
        np.asarray(action, dtype=np.float64), [base], [0]
    )
    pose = su.SE3_to_pose7(command[0])
    se3 = su.pose7_to_SE3(pose)
    return se3[:, :3, 3], se3[:, :3, :3]


def target_index(robot_t, rgb_t, rgb, stride, horizon):
    index, _times = sparse_action_indices_from_rgb_timestamps(
        robot_t, rgb_t, rgb, horizon, stride
    )
    return np.asarray(index, dtype=np.int64)


def surface_of(field, xyz):
    """Signed clearance along the outward normal, meters. Positive is outside."""
    geom = field.query(np.asarray(xyz, dtype=np.float64))
    normal = np.asarray(geom["R"][..., 2], dtype=np.float64)
    if getattr(field, "orientation_z_inward", False):
        normal = -normal
    point = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    foot = np.asarray(geom["x_s"], dtype=np.float64).reshape(-1, 3)
    clearance = np.sum((point - foot) * normal.reshape(-1, 3), axis=1)
    return clearance, normal.reshape(-1, 3), foot


def wall_mask(normal) -> np.ndarray:
    up = UP / np.linalg.norm(UP)
    cos = np.clip(np.sum(normal * up, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(cos)) >= WALL_DEG


def lean_a(rotation, normal, axis):
    e_long, _angle = e_long_sequence(normal, axis)
    return pack_tilt(rotation, normal, e_long)[:, 2]


def metrics(pred_x, pred_R, demo_x, demo_R, field, axis):
    pos = np.linalg.norm(pred_x - demo_x, axis=1) * 1e3
    rot = geodesic_deg(pred_R, demo_R)
    demo_clear, demo_n, _foot = surface_of(field, demo_x)
    pred_clear, _n, _f = surface_of(field, pred_x)
    # One chart for both poses: the wall under the demonstration.
    a_err = np.degrees(np.abs(lean_a(pred_R, demo_n, axis) - lean_a(demo_R, demo_n, axis)))
    return {
        "pos_mm": pos,
        "rot_deg": rot,
        "a_deg": a_err,
        "surface_mm": np.abs(pred_clear - demo_clear) * 1e3,
        "penetrate_mm": np.maximum(0.0, -pred_clear) * 1e3,
        "clear_mm": pred_clear * 1e3,
        "demo_clear_mm": demo_clear * 1e3,
        "wall": wall_mask(demo_n),
    }


def mean_of(rows, key):
    return np.mean(np.stack([row[key] for row in rows]), axis=0)


def mold_at(episode):
    group = zarr.open(CAUSAL_ZARR, "r")["data"][f"episode_{episode}"]
    foot = np.asarray(group["mold_x_s_0"], dtype=np.float64).reshape(-1, 3)
    frame = np.asarray(group["mold_R_field_0"], dtype=np.float64).reshape(-1, 3, 3)
    clearance = np.asarray(group["mold_signed_distance_0"], dtype=np.float64).reshape(-1)
    return foot, frame, clearance


def score(field, axis, causal_policy, causal_ds, r1rel_policy, r1rel_ds, queries, device, stride):
    causal_cache = {}
    r1rel_cache = {}
    mold_cache = {}
    per_ep = {}
    checked = False
    for n, (episode, rgb, c_idx, r_idx) in enumerate(queries, start=1):
        if episode not in causal_cache:
            causal_cache[episode] = episode_arrays(CAUSAL_ZARR, episode)
            r1rel_cache[episode] = episode_arrays(R1REL_ZARR, episode)
            mold_cache[episode] = mold_at(episode)
        c_pos, c_rot, c_rt, c_rgb = causal_cache[episode]
        r_pos, r_rot, r_rt, r_rgb = r1rel_cache[episode]
        if c_rt.shape != r_rt.shape or np.max(np.abs(c_rt - r_rt)) > 1e-6:
            raise SystemExit(f"episode {episode} timestamps differ between the zarrs")
        c_obs, c_action = causal_ds.sampler.sample_sequence(c_idx)
        r_obs, r_action = r1rel_ds.sampler.sample_sequence(r_idx)
        c_pred = np.asarray(predict_sparse_action(causal_policy, c_obs, device), dtype=np.float64)
        r_pred = np.asarray(predict_sparse_action(r1rel_policy, r_obs, device), dtype=np.float64)
        c_gt = np.asarray(c_action["sparse"], dtype=np.float64)
        r_gt = np.asarray(r_action["sparse"], dtype=np.float64)
        horizon = min(HORIZON, len(c_pred), len(r_pred), len(c_gt), len(r_gt))
        if horizon < HORIZON:
            print(f"[compare] skip ep {episode} rgb {rgb}: chunk {horizon}", flush=True)
            continue
        c_pred, r_pred, c_gt, r_gt = c_pred[:horizon], r_pred[:horizon], c_gt[:horizon], r_gt[:horizon]
        index = target_index(r_rt, r_rgb, rgb, stride, horizon)
        if len(index) < horizon:
            print(f"[compare] skip ep {episode} rgb {rgb}: targets {len(index)}", flush=True)
            continue
        anchor_i = int(np.clip(np.searchsorted(r_rt, r_rgb[rgb]), 0, len(r_pos) - 1))
        anchor_p = r_pos[anchor_i]
        anchor_R = r_rot[anchor_i]
        p0 = np.asarray(c_obs["sparse"]["robot0_eef_pos"][-1], dtype=np.float64).reshape(3)
        r0 = rot6d_rows_to_R(
            np.asarray(c_obs["sparse"]["robot0_eef_rot_axis_angle"][-1], dtype=np.float64)
        )
        if not checked:
            label_x, label_R = r1rel_pose(r_gt, anchor_p, anchor_R)
            demo_x = r_pos[index]
            gap = np.linalg.norm(label_x - demo_x, axis=1)
            anchor_gap = float(np.linalg.norm(p0 - anchor_p)) * 1e3
            print(
                f"[compare] r1rel label vs demo TCP median {np.median(gap) * 1e3:.3f} mm "
                f"max {np.max(gap) * 1e3:.3f} mm; causal obs vs anchor {anchor_gap:.3f} mm",
                flush=True,
            )
            if float(np.median(gap)) > 0.002:
                raise SystemExit("r1rel label does not reconstruct the demo TCP at these indices")
            if anchor_gap > 20.0:
                raise SystemExit("causal observation pose is not the query TCP")
        demo_x = r_pos[index]
        demo_R = r_rot[index]
        times = r_rt[index]
        dts = np.empty(len(index))
        dts[:-1] = np.diff(times)
        dts[-1] = dts[-2]
        feet, frames, _clearances = mold_cache[episode]
        cx, cR = causal_pose(field, c_pred, anchor_p, feet[index], frames[index], dts)
        lx, lR = causal_pose(field, c_gt, anchor_p, feet[index], frames[index], dts)
        rx, rR = r1rel_pose(r_pred, anchor_p, anchor_R)
        zx = np.broadcast_to(anchor_p, demo_x.shape).copy()
        zR = np.broadcast_to(anchor_R, demo_R.shape).copy()
        if not checked:
            label_rot = float(np.median(geodesic_deg(lR, demo_R)))
            label_pos = float(np.linalg.norm(lx[0] - demo_x[0]) * 1e3)
            label_later = float(np.median(np.linalg.norm(lx[1:] - demo_x[1:], axis=1)) * 1e3)
            print(
                f"[compare] causal label replay step 1: {label_pos:.3f} mm; "
                f"later position median {label_later:.3f} mm; rotation median {label_rot:.3f} deg",
                flush=True,
            )
            if label_rot > 2.0:
                raise SystemExit("causal label rotation does not match the demonstrated tool")
            print(
                f"[compare] open-loop label position floor step 1 {label_pos:.1f} mm, "
                f"later median {label_later:.1f} mm",
                flush=True,
            )
            checked = True
        row = {
            "episode": episode,
            "rgb": rgb,
            "t": float(r_rgb[rgb] - r_rgb[0]),
            "causal": metrics(cx, cR, demo_x, demo_R, field, axis),
            "label": metrics(lx, lR, demo_x, demo_R, field, axis),
            "r1rel": metrics(rx, rR, demo_x, demo_R, field, axis),
            "zero": metrics(zx, zR, demo_x, demo_R, field, axis),
            "xyz": {"demo": demo_x, "causal": cx, "r1rel": rx, "zero": zx},
            "R": {"demo": demo_R, "causal": cR, "r1rel": rR, "zero": zR},
        }
        per_ep.setdefault(episode, []).append(row)
        if n % 10 == 0 or n == len(queries):
            print(f"[compare] {n}/{len(queries)}", flush=True)
    return per_ep


def grand_mean(per_ep, model, key, wall=False):
    """Mean of per-episode means, shape (horizon,)."""
    curves = []
    counts = []
    for rows in per_ep.values():
        if wall:
            kept = []
            for row in rows:
                mask = row["causal"]["wall"]
                if not np.any(mask):
                    continue
                values = row[model][key].copy()
                values[~mask] = np.nan
                kept.append(values)
            if not kept:
                continue
            with np.errstate(all="ignore"):
                curves.append(np.nanmean(np.stack(kept), axis=0))
            counts.append(int(np.sum([np.any(r["causal"]["wall"]) for r in rows])))
        else:
            curves.append(mean_of([row[model] for row in rows], key))
            counts.append(len(rows))
    if not curves:
        return None, 0
    return np.mean(np.stack(curves), axis=0), int(sum(counts))


def episode_at(rows, model, key, step):
    return float(np.mean([row[model][key][step - 1] for row in rows]))


def write_report(meta, per_ep, curves):
    steps = REPORT_STEPS
    keys = (
        ("pos_mm", "position mm"),
        ("rot_deg", "rotation deg"),
        ("a_deg", "wall-lean a deg"),
        ("surface_mm", "surface-distance error mm"),
        ("penetrate_mm", "penetration mm"),
    )
    lines = [
        "# Causal-foot vs r1rel, world-frame tool pose",
        "",
        f"Causal-foot: `{meta['causal_ckpt']}` (epoch {meta['causal_epoch']}).",
        f"r1rel: `{meta['r1rel_ckpt']}` (epoch {meta['r1rel_epoch']}, val loss 0.032).",
        "",
        meta["causal_note"],
        meta["r1rel_note"],
        "",
        "Both chunks start from the query TCP. r1rel's 21-d relative pose is",
        "`action21_postprocess` on that pose; its recorded action matches the",
        "demonstrated TCP exactly. Causal-foot integrates v1/v2 on the field from",
        "the query foot and lifts by its own δn. The query's tangent offset is",
        "kept. Later demonstrated feet are not used. Rotation is still the",
        "demonstrated mold frame at that time times ΔR. The `label` column is the",
        "recorded action decoded the same way. On the training set δn is not the",
        "TCP height (median miss about 28 mm), so the label position is not zero.",
        "",
        f"Queries are all five validation episodes, RGB stride {STRIDE}, one",
        "non-overlapping 64-step chunk. The table is the mean of the per-episode",
        "means. Lower is better. Zero holds the tool at the query pose.",
        "",
        "Wall-lean `a` is the lean across the stringer, in degrees, using the mold",
        "normal under the demonstration for every column. Surface distance is how",
        "far the predicted clearance misses the demonstrated clearance. Penetration",
        "is how far the TCP is inside the mold.",
        f"The demonstrated TCP sits {meta['demo_clear_mm']:.1f} mm outside the mold on these queries (median).",
        "",
        "## Summary",
        "",
        "| metric | step | causal-foot | label floor | r1rel | zero |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    wins = []
    losses = []
    for key, title in keys:
        for step in steps:
            cells = []
            values = {}
            for model in MODELS:
                value = float(curves[model][key][step - 1])
                values[model] = value
                cells.append(f"{value:.2f}")
            # Step 1 position is the measured TCP for causal-foot, not a prediction.
            if not (step == 1 and key in ("pos_mm", "surface_mm")):
                if values["r1rel"] + 1e-9 < values["causal"]:
                    wins.append(
                        f"{title} at step {step} ({values['r1rel']:.2f} vs causal {values['causal']:.2f})"
                    )
                elif values["causal"] + 1e-9 < values["r1rel"]:
                    losses.append(
                        f"{title} at step {step} (causal {values['causal']:.2f} vs r1rel {values['r1rel']:.2f})"
                    )
            lines.append(f"| {title} | {step} | {' | '.join(cells)} |")
    lines += ["", "### On wall queries only", ""]
    lines.append("| metric | step | causal-foot | label floor | r1rel | zero | n queries |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for key, title in keys[:3]:
        for step in steps:
            cells = []
            for model in MODELS:
                curve = curves["wall"][model][key]
                cells.append("—" if curve is None else f"{float(curve[step - 1]):.2f}")
            lines.append(f"| {title} | {step} | {' | '.join(cells)} | {curves['wall_n']} |")
    lines += ["", "## Where r1rel is closer", ""]
    if wins:
        lines += [f"- {item}" for item in wins]
    else:
        lines.append("r1rel is not closer than causal-foot on any summary cell above.")
    lines += ["", "## Where causal-foot is closer", ""]
    if losses:
        lines += [f"- {item}" for item in losses]
    else:
        lines.append("Causal-foot is not closer than r1rel on any summary cell above.")
    lines += ["", "## Per episode at step 16", ""]
    lines.append("| episode | n | causal pos | r1rel pos | causal rot | r1rel rot | causal a | r1rel a | wall steps |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for episode, rows in sorted(per_ep.items()):
        wall_n = int(sum(np.sum(row["causal"]["wall"]) for row in rows))
        lines.append(
            f"| {episode} | {len(rows)} | {episode_at(rows, 'causal', 'pos_mm', 16):.1f} | "
            f"{episode_at(rows, 'r1rel', 'pos_mm', 16):.1f} | "
            f"{episode_at(rows, 'causal', 'rot_deg', 16):.1f} | "
            f"{episode_at(rows, 'r1rel', 'rot_deg', 16):.1f} | "
            f"{episode_at(rows, 'causal', 'a_deg', 16):.1f} | "
            f"{episode_at(rows, 'r1rel', 'a_deg', 16):.1f} | {wall_n} |"
        )
    lines += [
        "",
        "Step 1 is the pose at the query time, step 6 and step 16 are inside the",
        "chunk, and step 64 is the last action. The mold drawing uses the same",
        "decimated mesh as `view_openloop.py`. The pictures show the first 16",
        "steps of one chunk. The straight line at the end is the tool z axis.",
        "",
    ]
    (OUT / "compare_r1rel.md").write_text("\n".join(lines))


def _to_plot(xyz):
    xyz = np.asarray(xyz, dtype=np.float64)
    return xyz[..., 0], xyz[..., 2], -xyz[..., 1]


def render(field_npz, per_ep, curves):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from view_oct import decimate_mesh

    blob = np.load(field_npz)
    vertices, faces = decimate_mesh(blob["vertices"], blob["faces"], blob["is_apron"])
    vx, vy, vz = _to_plot(vertices)
    tris = np.stack([np.column_stack([vx[f], vy[f], vz[f]]) for f in faces])
    colors = {
        "demo": "#4aa3ff", "causal": "#ff9f1a", "label": "#2a9d8f",
        "r1rel": "#ff4d6a", "zero": "#8a8a8a",
    }

    def add_mesh(ax):
        coll = Poly3DCollection(tris, alpha=0.35, facecolor="#9aabb6", edgecolor="none")
        ax.add_collection3d(coll)

    def glyph(ax, origin, rotation, color, scale=0.08):
        # Tool z is the press axis, drawn longer than view_openloop's 40 mm so tilt reads.
        o = np.asarray(origin, dtype=np.float64).reshape(3)
        R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        tip = o + scale * R[:, 2]
        xs, ys, zs = _to_plot(np.stack([o, tip]))
        ax.plot(xs, ys, zs, color=color, lw=3.0)

    def limits(ax, points):
        pts = np.concatenate(points, axis=0)
        x, y, z = _to_plot(pts)
        cx, cy, cz = np.median(x), np.median(y), np.median(z)
        span = float(np.percentile(np.abs(np.stack([x - cx, y - cy, z - cz])), 98)) * 1.3
        ax.set_xlim(cx - span, cx + span)
        ax.set_ylim(cy - span, cy + span)
        ax.set_zlim(cz - span, cz + span)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=18, azim=-70)

    # Error grid.
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.4), dpi=120)
    series = (
        ("pos_mm", "position error (mm)"),
        ("rot_deg", "rotation error (deg)"),
        ("a_deg", "wall-lean a error (deg)"),
        ("surface_mm", "surface miss (mm)"),
        ("penetrate_mm", "penetration (mm)"),
    )
    step = np.arange(1, HORIZON + 1)
    for ax, (key, title) in zip(axes.ravel(), series):
        for model, label, style in (
            ("label", "label floor", "--"),
            ("causal", "causal-foot", "-"),
            ("r1rel", "r1rel", "-"),
            ("zero", "zero", "-"),
        ):
            ax.plot(step, curves[model][key], color=colors[model], label=label, lw=1.4, ls=style)
        for mark in REPORT_STEPS:
            ax.axvline(mark, color="#ddd", lw=0.6)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("step")
        ax.tick_params(labelsize=8)
    axes.ravel()[-1].axis("off")
    axes.ravel()[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Error against the demonstrated tool pose", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "errors.png")
    plt.close(fig)

    # Pick a wall episode and two others.
    wall_count = []
    for episode, rows in per_ep.items():
        wall_count.append((sum(int(np.sum(r["causal"]["wall"])) for r in rows), episode))
    wall_count.sort(reverse=True)
    wall_ep = wall_count[0][1]
    others = [ep for _n, ep in wall_count if ep != wall_ep][:2]
    show = [wall_ep] + others

    def one_chunk(rows, wall):
        if wall:
            ranked = sorted(rows, key=lambda row: int(np.sum(row["causal"]["wall"][:16])), reverse=True)
            return ranked[0]
        return rows[len(rows) // 2]

    def draw_chunk(row, title, path, zoom_pts):
        fig = plt.figure(figsize=(8.2, 6.2), dpi=120)
        ax = fig.add_subplot(111, projection="3d")
        add_mesh(ax)
        for model, label in (("demo", "demo"), ("causal", "causal-foot"), ("r1rel", "r1rel")):
            xs, ys, zs = _to_plot(row["xyz"][model][:16])
            ax.plot(xs, ys, zs, color=colors[model], lw=2.4, label=label)
            glyph(ax, row["xyz"][model][15], row["R"][model][15], colors[model])
        limits(ax, [p[:16] for p in zoom_pts])
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    for episode in show:
        rows = per_ep[episode]
        row = one_chunk(rows, episode == wall_ep)
        tag = "wall" if episode == wall_ep else "path"
        pts = [row["xyz"][m] for m in ("demo", "causal", "r1rel")]
        draw_chunk(
            row,
            f"episode {episode} ({tag}) at {row['t']:.0f} s, first 16 steps, line is tool z",
            OUT / f"ep{episode}.png",
            pts,
        )

    # Video: a handful of chunks on the wall episode, glyphs at the tip.
    wall_rows = [row for row in per_ep[wall_ep] if np.any(row["causal"]["wall"][:16])]
    if len(wall_rows) < 2:
        wall_rows = per_ep[wall_ep]
    take = wall_rows[:: max(1, len(wall_rows) // 8)][:8]
    frame_dir = OUT / "frames"
    frame_dir.mkdir(exist_ok=True)
    for old in frame_dir.glob("frame_*.png"):
        old.unlink()
    for i, row in enumerate(take):
        fig = plt.figure(figsize=(8.2, 6.2), dpi=110)
        ax = fig.add_subplot(111, projection="3d")
        add_mesh(ax)
        for model, label in (("demo", "demo"), ("causal", "causal-foot"), ("r1rel", "r1rel")):
            xs, ys, zs = _to_plot(row["xyz"][model][:16])
            ax.plot(xs, ys, zs, color=colors[model], lw=2.4, label=label)
            glyph(ax, row["xyz"][model][15], row["R"][model][15], colors[model])
        limits(ax, [row["xyz"][m][:16] for m in ("demo", "causal", "r1rel")])
        ax.set_title(f"episode {wall_ep} at {row['t']:.1f} s — line is tool z", fontsize=11)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(frame_dir / f"frame_{i:03d}.png")
        plt.close(fig)
    return wall_ep, show


def encode_video():
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("[compare] no ffmpeg, png frames only", flush=True)
        return
    frames = OUT / "frames" / "frame_%03d.png"
    mp4 = OUT / "wall.mp4"
    gif = OUT / "wall.gif"
    subprocess.check_call(
        [ffmpeg, "-y", "-framerate", "4", "-i", str(frames),
         "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4)],
    )
    subprocess.check_call(
        [ffmpeg, "-y", "-i", str(mp4),
         "-vf", "fps=4,scale=720:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
         str(gif)],
    )


def main() -> None:
    _self_check()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[compare] device", device, flush=True)
    field = FixBMoldField(FIELD_PATH)
    axis = long_axis(field.vertices)
    causal_path, causal_policy, causal_cfg, failed = load_causal(device)
    causal_epoch = ckpt_epoch(causal_path)
    print(f"[compare] causal {causal_path.name} epoch {causal_epoch}", flush=True)
    causal_ds, c_stride, c_h = build_dataset(causal_cfg, CAUSAL_ZARR, CAUSAL_NORM, "14308b72", 10)
    r1rel_policy, r1rel_cfg = load_policy_and_cfg(str(R1REL_CKPT), device)
    r1rel_epoch = ckpt_epoch(R1REL_CKPT)
    print(f"[compare] r1rel {R1REL_CKPT.name} epoch {r1rel_epoch}", flush=True)
    r1rel_ds, r_stride, r_h = build_dataset(r1rel_cfg, R1REL_ZARR, R1REL_NORM, "2c20fba56851", 21)
    if c_stride != r_stride or c_h != r_h or c_h < HORIZON:
        raise SystemExit(f"horizon mismatch causal {c_h}x{c_stride} r1rel {r_h}x{r_stride}")
    episodes = held_out_episodes(Path(CAUSAL_ZARR))
    queries = common_queries(causal_ds, r1rel_ds, set(episodes))
    print(f"[compare] episodes {episodes} queries {len(queries)} stride {c_stride}", flush=True)
    per_ep = score(
        field, axis, causal_policy, causal_ds, r1rel_policy, r1rel_ds, queries, device, c_stride
    )
    del causal_policy, r1rel_policy
    torch.cuda.empty_cache()
    keys = ("pos_mm", "rot_deg", "a_deg", "surface_mm", "penetrate_mm")
    curves = {model: {} for model in MODELS}
    curves["wall"] = {model: {} for model in MODELS}
    for model in MODELS:
        for key in keys:
            curve, _n = grand_mean(per_ep, model, key, wall=False)
            curves[model][key] = curve
            wcurve, wn = grand_mean(per_ep, model, key, wall=True)
            curves["wall"][model][key] = wcurve
    curves["wall_n"] = wn
    clears = np.concatenate([row["causal"]["demo_clear_mm"] for rows in per_ep.values() for row in rows])
    meta = {
        "causal_ckpt": str(causal_path),
        "causal_epoch": causal_epoch,
        "causal_failed": failed,
        "causal_note": (
            f"Epoch {causal_epoch} loaded from `{causal_path.name}`. "
            "The zip opens and torch.load reads the weights. "
            "This local file is a complete epoch 199, so it is the checkpoint that was scored."
            + ((" Earlier files that did not load: " + "; ".join(failed)) if failed else "")
        ),
        "r1rel_ckpt": str(R1REL_CKPT),
        "r1rel_epoch": r1rel_epoch,
        "r1rel_note": (
            "Best local odom r1rel checkpoint by validation loss: epoch 25, val loss 0.032 "
            f"(`{R1REL_CKPT.name}`). On the 200-epoch odom run the minimum was also epoch 25 "
            "(val 0.0333); that file was deleted by train-loss top-k. The surviving "
            "`oct_v35_h4s_odom_r1rel/checkpoints/latest.ckpt` is epoch 195, val loss 0.152."
        ),
        "episodes": episodes,
        "n_queries": sum(len(v) for v in per_ep.values()),
        "demo_clear_mm": float(np.median(clears)),
    }
    # JSON without the pose arrays.
    summary = {
        "meta": meta,
        "steps": list(REPORT_STEPS),
        "mean": {
            model: {key: np.round(curves[model][key], 4).tolist() for key in keys}
            for model in MODELS
        },
    }
    (OUT / "compare_r1rel.json").write_text(json.dumps(summary, indent=2))
    write_report(meta, per_ep, curves)
    wall_ep, show = render(FIELD_PATH, per_ep, curves)
    encode_video()
    print(f"[compare] wrote {OUT} wall episode {wall_ep} stills {show}", flush=True)


if __name__ == "__main__":
    main()

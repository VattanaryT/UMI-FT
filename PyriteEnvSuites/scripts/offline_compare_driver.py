#!/usr/bin/env python3
"""Offline comparison for one checkpoint.

Without ``--model`` this only writes ``settings.json``. With ``--model`` it
loads that checkpoint, samples a few validation queries, predicts, decodes,
and writes scores. It calls the existing loaders and decoders. It does not
import the env runner or the playback script.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

# Copy checks are per channel. A position pass does not pass rotation.
TOL_TIME_S = 1e-4
TOL_POS_M = 1e-5
TOL_ROT_DEG = 1e-3
TOL_STIFFNESS = 1e-3

# Decoded TCP position is the fixture-frame origin of ts_pose_fb for every
# model. Rotation is not one quantity:
#   hist, r1rel — command rotation from action21_postprocess
#   r4          — R_field @ predicted ΔR
#   r2          — field heading, not a predicted tool rotation
METRIC_GROUPS = {
    "tcp_position_mm": (
        "Fixture-frame TCP position versus later measured ts_pose_fb. "
        "hist/r1rel decode the command position. r2/r4 decode x_d."
    ),
    "command_rotation_deg": "hist and r1rel. Geodesic degrees.",
    "predicted_tool_rotation_deg": "r4. Geodesic degrees.",
    "field_heading_deg": "r2 only. Not pooled with command or ΔR rotation.",
    "stiffness_scalar": (
        "Unnormalized action dim 18. Same labeller scalar if stiffness_0 matches."
    ),
    "virtual_target": "hist and r1rel only.",
}

REPEAT_ATOL = 1e-5
CKPT_ROOT = Path(
    os.environ.get(
        "PYRITE_CHECKPOINT_FOLDERS", "/mnt/data/umidata/ft/training_outputs"
    )
)
DATA_ROOT = Path("/mnt/data/umidata/ft")
FIXB_SHARED = DATA_ROOT / "mold_aware_datasets" / "stringer_v3.5_fixb" / "shared"

# Decoder kind follows the stored action, not a shared postprocess.
MODELS = {
    "hist": {
        "ckpt": "stringer_v3.5/hist/no_dagger/epoch=0095-train_loss=0.016.ckpt",
        "dataset": DATA_ROOT / "datasets" / "stringer_v3.5" / "acp_replay_buffer.zarr",
        "decoder": "command",
        "rotation_group": "command_rotation_deg",
    },
    "r1rel": {
        "ckpt": "oct_v35_h4s/r1rel_world/epoch=0025-train_loss=0.029.ckpt",
        "dataset": DATA_ROOT
        / "mold_aware_datasets"
        / "stringer_v3.5_fixb"
        / "r1_world_world"
        / "acp_replay_buffer.zarr",
        "decoder": "command",
        "rotation_group": "command_rotation_deg",
    },
    "r2": {
        "ckpt": "oct_v35_h4s/r2_world_fixb/epoch=0030-train_loss=0.021.ckpt",
        "dataset": DATA_ROOT
        / "mold_aware_datasets"
        / "stringer_v3.5_fixb"
        / "r2_world_fixb"
        / "acp_replay_buffer.zarr",
        "decoder": "fixb",
        "fixb_run": "r2_world_fixb",
        "rotation_group": "field_heading_deg",
    },
    "r4": {
        "ckpt": "oct_v35_h4s/r4_world_fixb_tilt/epoch=0030-train_loss=0.022.ckpt",
        "dataset": DATA_ROOT
        / "mold_aware_datasets"
        / "stringer_v3.5_fixb"
        / "r4_world_fixb_tilt"
        / "acp_replay_buffer.zarr",
        "decoder": "fixb",
        "fixb_run": "r4_world_fixb_tilt",
        "rotation_group": "predicted_tool_rotation_deg",
    },
}

INTENTIONAL_DIFFERENCES = (
    "Fair hist feeds recorded depth. The stock exporter zeros depth_0.",
    "r1rel has no depth_0, so the stock zeroing leaves its sample unchanged.",
    "Each query is reseeded from (model, query id) before its draw.",
    "Physical dt is the diff of RGB action-target timestamps. "
    "The stock zarr clock uses raw_time_step_s.",
    "r2 and r4 positions come from integrate_fixb. They do not go through "
    "action21_postprocess.",
    "These episodes are the training validation split, not a held-out test.",
)


def query_seed(model_key: str, query_id: int) -> int:
    """Seed for one query. Other queries do not enter the hash."""
    digest = hashlib.sha256(f"{model_key}:{int(query_id)}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFFFFFF


def seeded_randn(model_key: str, query_id: int, shape: Tuple[int, ...]):
    """Reseed, then draw. A skipped query does not change this draw."""
    import torch

    torch.manual_seed(query_seed(model_key, query_id))
    return torch.randn(shape)


def seeded_call(model_key: str, query_id: int, fn: Callable[[], Any]) -> Any:
    """Reseed the global RNG, then call fn. Matches a stock predict after the same seed."""
    import torch

    torch.manual_seed(query_seed(model_key, query_id))
    return fn()


def channel_check(name: str, diff: np.ndarray, tol: float) -> Dict[str, Any]:
    max_abs = float(np.max(np.abs(np.asarray(diff, dtype=np.float64))))
    return {"name": name, "max_abs": max_abs, "tol": float(tol), "ok": max_abs <= tol}


def agreement_report(
    *,
    time_s: np.ndarray,
    pos_m: np.ndarray,
    rot_deg: np.ndarray,
    stiffness: np.ndarray,
) -> Dict[str, Any]:
    channels = {
        "time_s": channel_check("time_s", time_s, TOL_TIME_S),
        "position_m": channel_check("position_m", pos_m, TOL_POS_M),
        "rotation_deg": channel_check("rotation_deg", rot_deg, TOL_ROT_DEG),
        "stiffness": channel_check("stiffness", stiffness, TOL_STIFFNESS),
    }
    return {"channels": channels, "ok": all(c["ok"] for c in channels.values())}


def geodesic_deg(ra: np.ndarray, rb: np.ndarray) -> np.ndarray:
    """Angle of ra^T rb, in degrees. Same relative order as fixb rot_angle_deg."""
    rel = np.einsum("...ji,...jk->...ik", ra, rb)
    tr = np.trace(rel, axis1=-2, axis2=-1)
    cos = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def integration_dt(times: np.ndarray) -> np.ndarray:
    """dt used by fixb_open_loop_metrics.decode. The last column is unused."""
    times = np.asarray(times, dtype=np.float64)
    dt = np.diff(times, axis=1)
    return np.concatenate([dt, dt[:, -1:]], axis=1)


def _scripts_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "UMIFT_Data" / "surface_frames_scripts"


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decode_command_like_stock(raw_action, base_se3, id_list: Sequence[int] = (0,)):
    """Existing stock decoder. Dim 21 command and virtual-target poses."""
    module = _load_script(
        "run_policy_open_loop_val_export",
        Path(__file__).resolve().parent / "run_policy_open_loop_val_export.py",
    )
    return module.decode_horizon_actions(
        raw_action=np.asarray(raw_action),
        base_se3=base_se3,
        id_list=id_list,
        action_dim=int(np.asarray(raw_action).shape[-1]),
    )


def decode_fixb_like_metrics(run, action, times, p0, r0, field, r_off):
    """Existing anchored Fix B decoder, including its integration dt."""
    scripts = str(_scripts_dir())
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from fixb_open_loop_metrics import decode

    return decode(run, action, times, p0, r0, field, r_off)


def tcp_pose7_to_Rp(pose7: np.ndarray):
    """Same wxyz layout as fixb_labels.pose7_to_Rp. Returns R, p."""
    from scipy.spatial.transform import Rotation

    p = np.asarray(pose7, dtype=np.float64).reshape(-1, 7)
    rot = Rotation.from_quat(p[:, [4, 5, 6, 3]]).as_matrix()
    return rot, p[:, :3].copy()


def prepare_obs(obs: Dict[str, Any], zero_depth: bool) -> Dict[str, Any]:
    """Copy the sample. Depth is zeroed only when zero_depth is true."""
    out = copy.deepcopy(obs)
    if zero_depth:
        sparse = out.get("sparse", out)
        if isinstance(sparse, dict) and "depth_0" in sparse:
            sparse["depth_0"] = np.zeros_like(np.asarray(sparse["depth_0"]))
    return out


def replacement_asset(stored: str) -> Optional[str]:
    """Local asset with the same filename when the stored path is missing."""
    path = Path(str(stored))
    if path.is_file():
        return None
    local = _scripts_dir() / "assets" / path.name
    if not local.is_file():
        raise FileNotFoundError(f"missing {stored} and no asset named {path.name}")
    return str(local)


def action_targets(rgb_times, action_times, rgb_id: int, horizon: int, stride: int):
    """RGB-grid action indices. Padded steps are not valid."""
    from PyriteML.diffusion_policy.common.sampler_umift import (
        sparse_action_indices_from_rgb_timestamps,
    )

    idx, times = sparse_action_indices_from_rgb_timestamps(
        action_times=action_times,
        rgb_times=rgb_times,
        rgb_id=int(rgb_id),
        action_horizon=int(horizon),
        rgb_frame_stride=int(stride),
    )
    n_valid = int(len(idx))
    if n_valid == 0:
        raise ValueError(f"rgb_id {rgb_id} has no action targets")
    if n_valid < horizon:
        idx = np.concatenate([idx, np.repeat(idx[-1:], horizon - n_valid)])
        times = np.concatenate([times, np.repeat(times[-1:], horizon - n_valid)])
    return (
        np.asarray(idx[:horizon], dtype=np.int64),
        np.asarray(times[:horizon], dtype=np.float64),
        n_valid,
    )


def check_smoke_arrays(raw, times, pos, rot, valid, action_dim: int) -> list:
    """Shape, finite-value, and timestamp problems. Empty means the arrays pass."""
    raw = np.asarray(raw)
    times = np.asarray(times, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64)
    rot = np.asarray(rot, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    problems = []
    if raw.ndim != 3 or raw.shape[-1] != action_dim:
        problems.append(f"raw shape {raw.shape}, expected (Q, H, {action_dim})")
    q = raw.shape[0] if raw.ndim == 3 else -1
    exec_h = times.shape[-1] if times.ndim == 2 else -1
    if times.shape != (q, exec_h) or not np.isfinite(times).all():
        problems.append(f"times shape {times.shape} finite={np.isfinite(times).all()}")
    if pos.shape != (q, exec_h, 3) or not np.isfinite(pos).all():
        problems.append(f"pos shape {pos.shape}")
    if rot.shape != (q, exec_h, 3, 3) or not np.isfinite(rot).all():
        problems.append(f"rot shape {rot.shape}")
    if valid.shape != (q, exec_h):
        problems.append(f"valid shape {valid.shape}")
    if raw.size and not np.isfinite(raw).all():
        problems.append("raw action has a non-finite value")
    if valid.size and not valid.any():
        problems.append("no valid action steps")
    for row, mask in zip(times, valid):
        kept = row[mask]
        if kept.size > 1 and not np.all(np.diff(kept) > 0):
            problems.append("action timestamps are not strictly increasing")
            break
    return problems


def rotation_matrices_ok(rot: np.ndarray, tol: float = 1e-4) -> bool:
    rot = np.asarray(rot, dtype=np.float64)
    det = np.linalg.det(rot)
    gram = np.einsum("...ji,...jk->...ik", rot, rot)
    eye = np.eye(3)
    return bool(
        np.max(np.abs(det - 1.0)) <= tol and np.max(np.abs(gram - eye)) <= tol
    )


def physical_scores(pos, rot, pos_ref, rot_ref, valid) -> Dict[str, float]:
    """Mean TCP error on valid steps. Position and rotation stay separate."""
    pos = np.asarray(pos, dtype=np.float64)
    rot = np.asarray(rot, dtype=np.float64)
    pos_ref = np.asarray(pos_ref, dtype=np.float64)
    rot_ref = np.asarray(rot_ref, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if pos.shape != pos_ref.shape or rot.shape != rot_ref.shape:
        raise ValueError(f"shape {pos.shape} {rot.shape} vs {pos_ref.shape} {rot_ref.shape}")
    if valid.shape != pos.shape[:2] or not valid.any():
        raise ValueError("valid mask does not select any steps")
    dist_mm = np.linalg.norm(pos - pos_ref, axis=-1) * 1000.0
    ang = geodesic_deg(rot, rot_ref)
    return {
        "position_mm_mean": float(dist_mm[valid].mean()),
        "rotation_deg_mean": float(np.asarray(ang)[valid].mean()),
        "n_steps": int(valid.sum()),
    }


def masked_mse(pred, gt, valid) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if pred.shape != gt.shape or valid.shape != pred.shape[:2]:
        raise ValueError(f"mse shapes {pred.shape} {gt.shape} {valid.shape}")
    err = pred[valid] - gt[valid]
    return float(np.mean(err**2))


def require_fresh_dir(path: Path) -> Path:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"output directory already exists: {path}")
    path.mkdir(parents=True)
    return path


def settings_record(run: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    record = {
        "inference": "not run" if run is None else "smoke",
        "tolerances": {
            "time_s": TOL_TIME_S,
            "position_m": TOL_POS_M,
            "rotation_deg": TOL_ROT_DEG,
            "stiffness": TOL_STIFFNESS,
            "repeat_abs": REPEAT_ATOL,
        },
        "seed": "sha256(model:episode:rgb_id) before each query",
        "metric_groups": METRIC_GROUPS,
        "intentional_differences": list(INTENTIONAL_DIFFERENCES),
        "quartet": {key: spec["ckpt"] for key, spec in MODELS.items()},
        "stored_dataset_path": "/data/acp_replay_buffer.zarr",
        "validation_not_test": (
            "Episodes 9, 12, 28, 37, 41 are the seed-42 validation split."
        ),
    }
    if run is not None:
        record["run"] = run
    return record


def write_record(out: Path, run: Optional[Dict[str, Any]] = None) -> Path:
    path = Path(out) / "settings.json"
    path.write_text(json.dumps(settings_record(run), indent=2) + "\n")
    return path


def action_spec(cfg) -> Tuple[int, int, int]:
    meta = cfg.shape_meta if "shape_meta" in cfg else cfg.task.shape_meta
    sparse = meta.sample.action.sparse
    return int(meta.action.shape[0]), int(sparse.horizon), int(sparse.down_sample_steps)


def repair_dataset_cfg(cfg, dataset_path: Path, out: Path) -> list:
    """Point the in-memory config at local files. Does not edit the checkpoint."""
    from omegaconf import OmegaConf

    repairs = []
    stored = str(cfg.task.dataset.dataset_path)
    OmegaConf.update(cfg, "task.dataset.dataset_path", str(dataset_path), merge=False)
    repairs.append({"dataset_path": {"stored": stored, "local": str(dataset_path)}})
    for key in (
        "pose_jump_flag_path",
        "action_normalizer_path",
        "r2_action_normalizer_path",
    ):
        if key in cfg.task.dataset and cfg.task.dataset[key]:
            local = replacement_asset(str(cfg.task.dataset[key]))
            if local is not None:
                OmegaConf.update(cfg, f"task.dataset.{key}", local, merge=False)
                repairs.append({key: local})
    if "filter_stats_path" in cfg.task.dataset and cfg.task.dataset.filter_stats_path:
        dest = str(out / "filter_stats.json")
        OmegaConf.update(cfg, "task.dataset.filter_stats_path", dest, merge=False)
        repairs.append({"filter_stats_path": dest})
    return repairs


def select_queries(sampler, episode_id: int, n: int):
    chosen = []
    episodes = set()
    for gidx, (epi, rgb) in enumerate(sampler.indices):
        episodes.add(int(epi))
        if int(epi) == int(episode_id):
            chosen.append((int(gidx), int(rgb)))
            if len(chosen) == n:
                break
    if int(episode_id) not in episodes:
        raise RuntimeError(
            f"episode {episode_id} is not in the val split: {sorted(episodes)}"
        )
    if len(chosen) < n:
        raise RuntimeError(f"episode {episode_id} has {len(chosen)} queries, need {n}")
    return chosen


def load_tcp(zarr_path: Path, episode_id: int) -> Dict[str, Any]:
    import zarr

    group = zarr.open(str(zarr_path), mode="r")["data"][f"episode_{episode_id}"]
    rot, pos = tcp_pose7_to_Rp(np.asarray(group["ts_pose_fb_0"][:]))
    out = {
        "R": rot,
        "p": pos,
        "stiffness": np.asarray(group["stiffness_0"][:], dtype=np.float64).reshape(-1),
        "rgb_t": np.asarray(group["rgb_time_stamps_0"][:], dtype=np.float64).reshape(-1),
        "robot_t": np.asarray(group["robot_time_stamps_0"][:], dtype=np.float64).reshape(-1),
        "vt_p": None,
    }
    if "ts_pose_virtual_target_0" in group:
        _vr, vp = tcp_pose7_to_Rp(np.asarray(group["ts_pose_virtual_target_0"][:]))
        out["vt_p"] = vp
    return out


def predict_recorded(policy, obs, device, stock) -> np.ndarray:
    """Same batching as the stock exporter, without its depth zeroing."""
    obs = prepare_obs(obs, zero_depth=False)

    def to_batch(value):
        tensor = stock.torch.as_tensor(np.asarray(value)).to(device)
        return stock.rearrange(tensor, "... -> 1 ...")

    batched = stock.dict_apply(obs, to_batch)
    with stock.torch.no_grad():
        result = policy.predict_action(batched)
    key = "sparse" if result.get("sparse") is not None else "action"
    return result[key][0].detach().to("cpu").numpy()


def _normalized(normalizer, rows: np.ndarray) -> np.ndarray:
    import torch

    values = normalizer.normalize(torch.as_tensor(rows, dtype=torch.float32))
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float64)


def run_model(out: Path, model_key: str, episode_id: int, n_queries: int, exec_h: int, device_name: str) -> int:
    """One checkpoint, one episode, a few queries. Writes npz, settings, and report."""
    import torch

    spec = MODELS[model_key]
    ckpt = CKPT_ROOT / spec["ckpt"]
    dataset_path = Path(spec["dataset"])
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    if not dataset_path.is_dir():
        raise FileNotFoundError(dataset_path)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda was requested but is not available")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    scripts = str(_scripts_dir())
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    stock = _load_script(
        "run_policy_open_loop_val_export",
        Path(__file__).resolve().parent / "run_policy_open_loop_val_export.py",
    )
    print(f"loading {ckpt}", flush=True)
    policy, cfg = stock.load_policy_and_cfg(str(ckpt), device)
    action_dim, horizon, stride = action_spec(cfg)
    exec_h = min(int(exec_h), horizon)
    repairs = repair_dataset_cfg(cfg, dataset_path, out)
    print(f"building dataset {dataset_path}", flush=True)
    _train, val_dataset = stock.build_val_dataset(cfg, str(dataset_path))
    chosen = select_queries(val_dataset.sampler, episode_id, n_queries)
    tcp = load_tcp(dataset_path, episode_id)
    sampler = val_dataset.sampler

    field = r_off = None
    if spec["decoder"] == "fixb":
        from fixb_field import FixBMoldField

        field = FixBMoldField(FIXB_SHARED / "fixb_field.npz")
        meta = json.loads((FIXB_SHARED / "fixb_meta.json").read_text())
        r_off = np.asarray(meta["R_off"], dtype=np.float64)

    def predict_pass():
        raws, gts, bases, idx_rows, time_rows, n_valid, q_robot, depth = (
            [], [], [], [], [], [], [], False,
        )
        for gidx, rgb_id in chosen:
            obs, action = sampler.sample_sequence(gidx)
            sparse = obs.get("sparse", obs)
            depth = depth or (isinstance(sparse, dict) and "depth_0" in sparse)
            info = stock.get_query_base_info(sampler, gidx, [0])
            seed_key = f"{model_key}:{episode_id}"
            raw = seeded_call(
                seed_key,
                rgb_id,
                lambda obs=obs: predict_recorded(policy, obs, device, stock),
            )
            targets = action_targets(
                tcp["rgb_t"], tcp["robot_t"], rgb_id, exec_h, stride
            )
            raws.append(np.asarray(raw))
            gts.append(np.asarray(action["sparse"], dtype=np.float64))
            bases.append(info["base_se3"])
            idx_rows.append(targets[0])
            time_rows.append(targets[1])
            n_valid.append(targets[2])
            q_robot.append(
                int(np.clip(np.searchsorted(tcp["robot_t"], tcp["rgb_t"][rgb_id]), 0, len(tcp["robot_t"]) - 1))
            )
        return raws, gts, bases, idx_rows, time_rows, n_valid, q_robot, depth

    print(f"predicting {n_queries} queries twice", flush=True)
    raws, gts, bases, idx_rows, time_rows, n_valid, q_robot, depth = predict_pass()
    raws_again, *_rest = predict_pass()
    raw = np.stack(raws)
    raw_again = np.stack(raws_again)
    repeat = float(np.max(np.abs(raw - raw_again)))
    gt = np.stack(gts)[:, :exec_h]
    raw_exec = raw[:, :exec_h]
    times = np.stack(time_rows)
    idx = np.stack(idx_rows)
    valid = np.arange(exec_h)[None, :] < np.asarray(n_valid)[:, None]

    if spec["decoder"] == "command":
        positions, rotations, vt_pos, decoded_stiff = [], [], [], []
        for row, base in zip(raw_exec, bases):
            cmd, vt, stiff, _grip = stock.decode_horizon_actions(row, base, [0], action_dim)
            rot, pos = tcp_pose7_to_Rp(cmd)
            _vr, vp = tcp_pose7_to_Rp(vt)
            positions.append(pos)
            rotations.append(rot)
            vt_pos.append(vp)
            decoded_stiff.append(np.asarray(stiff, dtype=np.float64).reshape(-1))
        pos = np.stack(positions)
        rot = np.stack(rotations)
        vt_pos = np.stack(vt_pos)
        decoded_stiff = np.stack(decoded_stiff)
    else:
        q = np.asarray(q_robot)
        pos, rot = decode_fixb_like_metrics(
            spec["fixb_run"],
            raw_exec,
            times,
            tcp["p"][q],
            tcp["R"][q],
            field,
            r_off,
        )
        vt_pos = None
        decoded_stiff = raw_exec[..., 18]

    problems = check_smoke_arrays(raw, times, pos, rot, valid, action_dim)
    if not rotation_matrices_ok(rot):
        problems.append("decoded rotation is not a rotation matrix")
    if repeat > REPEAT_ATOL:
        problems.append(f"repeat max abs {repeat} exceeds {REPEAT_ATOL}")
    if spec["decoder"] == "command":
        stiff_gap = float(np.max(np.abs(decoded_stiff - raw_exec[..., 18])))
        if stiff_gap > 1e-4:
            problems.append(f"decoded stiffness differs from action dim 18 by {stiff_gap}")
        for base, q in zip(bases, q_robot):
            gap_p = float(np.linalg.norm(np.asarray(base[0])[:3, 3] - tcp["p"][q]))
            gap_r = float(np.asarray(geodesic_deg(np.asarray(base[0])[:3, :3], tcp["R"][q])).reshape(-1)[0])
            if gap_p > TOL_POS_M or gap_r > TOL_ROT_DEG:
                problems.append(
                    f"query TCP and zarr ts_pose_fb differ by {gap_p:.3g} m and {gap_r:.3g} deg"
                )
                break

    scores = physical_scores(pos, rot, tcp["p"][idx], tcp["R"][idx], valid)
    scores["rotation_group"] = spec["rotation_group"]
    scores["stiffness_mae"] = float(
        np.abs(raw_exec[..., 18] - tcp["stiffness"][idx])[valid].mean()
    )
    scores["label_action_mse"] = masked_mse(raw_exec, gt, valid)
    kept_pred = raw_exec[valid]
    kept_gt = gt[valid]
    normalizer = stock.get_action_normalizer(policy)
    norm_err = _normalized(normalizer, kept_pred) - _normalized(normalizer, kept_gt)
    scores["label_naction_mse"] = float(np.mean(norm_err**2))
    if vt_pos is not None and tcp["vt_p"] is not None:
        vt_mm = np.linalg.norm(vt_pos - tcp["vt_p"][idx], axis=-1) * 1000.0
        scores["virtual_target_mm_mean"] = float(vt_mm[valid].mean())

    report = {
        "ok": not problems,
        "problems": problems,
        "model": model_key,
        "episode": episode_id,
        "n_queries": n_queries,
        "exec_h": exec_h,
        "action_dim": action_dim,
        "action_horizon": horizon,
        "rgb_frame_stride": stride,
        "rgb_ids": [rgb for _g, rgb in chosen],
        "repeat_max_abs": repeat,
        "depth": "recorded" if depth else "absent",
        "decoder": spec["decoder"],
        "scores": scores,
        "repairs": repairs,
    }
    np.savez_compressed(
        out / f"episode_{episode_id}.npz",
        raw_action=raw,
        raw_action_repeat=raw_again,
        gt_action=np.stack(gts),
        action_target_time_s=times,
        action_robot_idx=idx,
        valid=valid,
        decoded_pos_m=pos,
        decoded_rot=rot,
        rgb_id=np.asarray([rgb for _g, rgb in chosen], dtype=np.int64),
    )
    write_record(
        out,
        {
            "model": model_key,
            "checkpoint": str(ckpt),
            "dataset": str(dataset_path),
            "episode": episode_id,
            "n_queries": n_queries,
            "exec_h": exec_h,
            "device": device_name,
            "depth": report["depth"],
            "decoder": spec["decoder"],
            "rotation_group": spec["rotation_group"],
            "repairs": repairs,
        },
    )
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"{model_key} episode {episode_id}: raw {raw.shape} "
        f"repeat {repeat:.3g} pos {scores['position_mm_mean']:.2f} mm "
        f"{scores['rotation_group']} {scores['rotation_deg_mean']:.2f} deg "
        f"ok={report['ok']}"
    )
    if problems:
        print("\n".join(problems))
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", choices=sorted(MODELS))
    parser.add_argument("--episode", type=int, default=9)
    parser.add_argument("--n-queries", type=int, default=3)
    parser.add_argument("--exec-h", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.n_queries < 1 or args.exec_h < 1:
        raise SystemExit("--n-queries and --exec-h must be positive")
    require_fresh_dir(args.out)
    if args.model is None:
        write_record(args.out)
        print(f"wrote {args.out / 'settings.json'}; inference not run")
        return 0
    return run_model(
        args.out, args.model, args.episode, args.n_queries, args.exec_h, args.device
    )


if __name__ == "__main__":
    raise SystemExit(main())

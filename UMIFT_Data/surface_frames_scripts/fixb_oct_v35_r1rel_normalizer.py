#!/usr/bin/env python3
"""Fit the October r1 relative-action normalizer and run the pre-submit checks.

Uses the same train chunks as umift_oct_fixb_r1 (pose_frames relative). Saves
action + robot0_eef_pos + robot0_eef_rot_axis_angle. Does not submit a job.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from diffusion_policy.common.normalize_util import (
    array_to_stats,
    concatenate_normalizer,
    get_identity_normalizer_from_stat,
    get_range_normalizer_from_stat,
)
from diffusion_policy.dataset.umift_dataset import UmiFTDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from PyriteConfig.tasks.umift.umift_type_conversions import action21_postprocess
from PyriteUtility.spatial_math import spatial_utilities as su

from fixb_dataset import R1REL_NORMALIZER_KEYS
from fixb_pose_jump import FLAG_NPZ, action_indices_for_query, keep_weight, load_flags

# Range-normalized action dims. Rot6d stays an identity normalizer.
RANGE_SLICES = ((0, 3), (9, 12), (18, 19), (19, 20), (20, 21))
P_LO = 0.5
P_HI = 99.5

ROOT = Path(__file__).resolve().parents[2]
TRAIN_CFG = ROOT / "PyriteML" / "diffusion_policy" / "config"
SCRIPTS_CFG = Path(__file__).resolve().parent / "config"
ASSETS = Path(__file__).resolve().parent / "assets"
ZARR = (
    "/mnt/data/umidata/ft/mold_aware_datasets/"
    "stringer_v3.5_fixb/r1_world_world/acp_replay_buffer.zarr"
)


def _compose(task: str):
    with initialize_config_dir(version_base=None, config_dir=str(TRAIN_CFG)):
        cfg = compose(
            config_name="train_umift_workspace",
            overrides=[
                f"hydra.searchpath=[file://{SCRIPTS_CFG}]",
                f"task={task}",
            ],
        )
    return OmegaConf.to_container(cfg.task, resolve=True)


def _dataset(task_cfg: dict) -> UmiFTDataset:
    return UmiFTDataset(
        shape_meta=task_cfg["shape_meta"],
        dataset_path=task_cfg["dataset_path"],
        sparse_query_frequency_down_sample_steps=2,
        action_padding=True,
        temporally_independent_normalization=False,
        seed=42,
        val_ratio=0.10,
        normalize_wrench=True,
        action_gap_threshold_mult=3.0,
        scan_pre_first_fill="zeros",
        max_abs_pos_m=2.0,
    )


def _minmax(normalizer, key: str):
    stats = normalizer[key].get_input_stats()
    mn = stats["min"].detach().float().cpu().numpy().reshape(-1)
    mx = stats["max"].detach().float().cpu().numpy().reshape(-1)
    return mn, mx


def _clip_percentile(values: np.ndarray) -> np.ndarray:
    lo = np.percentile(values, P_LO, axis=0)
    hi = np.percentile(values, P_HI, axis=0)
    return np.clip(values, lo, hi).astype(np.float32)


def _fit_action(rows: np.ndarray):
    pieces = []
    cursor = 0
    for start, stop in RANGE_SLICES:
        if start != cursor:
            pieces.append(
                get_identity_normalizer_from_stat(array_to_stats(rows[:, cursor:start]))
            )
        clipped = _clip_percentile(rows[:, start:stop])
        pieces.append(get_range_normalizer_from_stat(array_to_stats(clipped)))
        cursor = stop
    if cursor != rows.shape[1]:
        pieces.append(
            get_identity_normalizer_from_stat(array_to_stats(rows[:, cursor:]))
        )
    return concatenate_normalizer(pieces)


def _collect_unflagged(ds: UmiFTDataset, flags: dict):
    ds.sampler.ignore_rgb(True)
    actions = []
    positions = []
    rotations = []
    n_kept = 0
    n_dropped = 0
    for i, (epi_id, rgb_id) in enumerate(ds.sampler.indices):
        _obs, action_sample = ds.sampler.sample_sequence(i)
        action = np.asarray(action_sample["sparse"], dtype=np.float32)
        sparse = _obs["sparse"]
        pos = np.asarray(sparse["robot0_eef_pos"], dtype=np.float32)
        rot = np.asarray(sparse["robot0_eef_rot_axis_angle"], dtype=np.float32)
        action_idx = action_indices_for_query(ds, int(epi_id), int(rgb_id))
        keep = keep_weight(flags[f"episode_{int(epi_id)}"], action_idx, action.shape[0])
        actions.append(action[keep > 0])
        positions.append(pos)
        rotations.append(rot.reshape(-1, rot.shape[-1]))
        n_kept += int(np.sum(keep > 0))
        n_dropped += int(np.sum(keep == 0))
        if i % 4000 == 0:
            print(f"[collect] {i}/{len(ds.sampler.indices)}", flush=True)
    print(f"[collect] unflagged action steps {n_kept}  dropped {n_dropped}")
    pos_h = np.stack(positions, axis=0)
    return (
        np.concatenate(actions, axis=0),
        pos_h.reshape(-1, pos_h.shape[-1]),
        np.concatenate(rotations, axis=0),
        pos_h,
    )


def _print_action_table(rel_norm, abs_norm):
    rel_mn, rel_mx = _minmax(rel_norm, "action")
    abs_mn, abs_mx = _minmax(abs_norm, "action")
    print("action dim  new_min  new_max  previous_min  previous_max")
    for i in range(21):
        print(
            f"{i:5d}  {rel_mn[i]:10.4g}  {rel_mx[i]:10.4g}  "
            f"{abs_mn[i]:10.4g}  {abs_mx[i]:10.4g}"
        )
    for key in ("robot0_eef_pos", "robot0_eef_rot_axis_angle"):
        rmn, rmx = _minmax(rel_norm, key)
        amn, amx = _minmax(abs_norm, key)
        print(f"{key} rel min {np.array2string(rmn, precision=4)} max {np.array2string(rmx, precision=4)}")
        print(f"{key} abs min {np.array2string(amn, precision=4)} max {np.array2string(amx, precision=4)}")


def _sanity(ds: UmiFTDataset) -> None:
    batch = torch.utils.data.DataLoader(ds, batch_size=2, num_workers=0).__iter__().__next__()
    action = batch["action"]["sparse"][0].numpy()
    step0 = action[0]
    pos_mm = float(np.linalg.norm(step0[:3]) * 1000.0)
    rot = step0[3:9]
    ident = np.array([1, 0, 0, 0, 1, 0], dtype=np.float64)
    rot_err = float(np.linalg.norm(rot - ident))
    print(
        f"[sanity batch] first-step |xyz|={pos_mm:.3f} mm  "
        f"rot6d_l2_vs_identity={rot_err:.4g}  action0[:9]={np.array2string(step0[:9], precision=4)}"
    )
    geom = batch["obs"]["sparse"]["robot0_mold_geom"]
    print(
        f"[sanity batch] robot0_mold_geom in shape_meta="
        f"{'robot0_mold_geom' in ds.shape_meta['obs']} "
        f"batch shape={tuple(geom.shape)}"
    )
    if pos_mm > 20.0:
        raise RuntimeError(f"first action step is {pos_mm:.1f} mm, not a few mm")

    val = ds.get_validation_dataset()
    sampler = val.sampler
    orig = sampler.action_to_action_sample
    stash = {}

    def _wrap(action_sparse, action_dense, id_list, base_pose):
        stash["abs"] = np.asarray(action_sparse, dtype=np.float64).copy()
        stash["base"] = [np.asarray(b, dtype=np.float64).copy() for b in base_pose]
        return orig(action_sparse, action_dense, id_list, base_pose)

    sampler.action_to_action_sample = _wrap
    _obs, action_sample = sampler.sample_sequence(0)
    rel = np.asarray(action_sample["sparse"], dtype=np.float64)
    mats, _vt, _stiff, _eoat = action21_postprocess(rel, stash["base"], [0])
    recon = su.SE3_to_pose9(mats[0])
    demo = stash["abs"][:, :9]
    pos_err_mm = np.linalg.norm(recon[:, :3] - demo[:, :3], axis=1) * 1000.0
    print(
        f"[sanity roundtrip] n={len(pos_err_mm)} max={pos_err_mm.max():.4f} mm "
        f"median={np.median(pos_err_mm):.4f} mm"
    )
    if float(pos_err_mm.max()) > 1.0:
        raise RuntimeError(f"round-trip error {pos_err_mm.max():.3f} mm exceeds 1 mm")


def main() -> int:
    rel_cfg = _compose("umift_oct_fixb_r1rel")
    frames = rel_cfg["shape_meta"]["pose_frames"]
    if frames["obs"] != "relative" or frames["action"] != "relative":
        raise RuntimeError(f"r1rel pose_frames {frames}")
    if rel_cfg["dataset_path"] != ZARR:
        raise RuntimeError("dataset_path is not the October r1 zarr")
    sample = rel_cfg["shape_meta"]["sample"]
    if int(sample["action"]["sparse"]["horizon"]) != 24:
        raise RuntimeError("action horizon is not 24")
    if int(sample["obs"]["sparse"]["robot0_mold_geom"]["horizon"]) != 5:
        raise RuntimeError("mold horizon is not 5")

    print("[load] relative")
    rel_ds = _dataset(rel_cfg)
    keys = list(rel_ds.replay_buffer["data"].keys())
    val_ids = [int(k.split("_")[-1]) for k, m in zip(keys, rel_ds.val_mask) if m]
    n_train = len(rel_ds)
    n_val = len(rel_ds.get_validation_dataset())
    print(f"[chunks] train={n_train} val={n_val} val_episodes={val_ids}")
    if n_train != 20207 or n_val != 2055:
        raise RuntimeError(f"chunk counts {n_train}/{n_val}, expected 20207/2055")
    if sorted(val_ids) != [9, 12, 28, 37, 41]:
        raise RuntimeError(f"val episodes {val_ids}")

    print("[sanity]")
    _sanity(rel_ds)

    flags = load_flags(FLAG_NPZ)
    print("[fit] action: unflagged steps, 0.5–99.5 percentile scale only")
    print("[fit] eef_pos: unclipped min/max (percentile scale saturates >1%)")
    action_rows, pos_rows, rot_rows, pos_h = _collect_unflagged(rel_ds, flags)
    pct = _clip_percentile(pos_rows)
    # Bounds the percentile fit would have used. Training values are not clipped.
    pct_min = pct.min(axis=0)
    pct_max = pct.max(axis=0)
    outside = (pos_h < pct_min) | (pos_h > pct_max)
    print(
        f"[eef_pos] scalar saturate {outside.mean():.4%}  "
        f"any-dim {outside.any(axis=2).mean():.4%}"
    )
    for t in range(pos_h.shape[1]):
        step = outside[:, t]
        mag = np.linalg.norm(pos_h[:, t], axis=1)
        print(
            f"[eef_pos] horizon[{t}] saturate {step.mean():.4%}  "
            f"|p| p50/p99/max {np.percentile(mag,[50,99]).round(4)} {mag.max():.4f}"
        )
    saved = LinearNormalizer()
    saved["action"] = _fit_action(action_rows)
    saved["robot0_eef_pos"] = get_range_normalizer_from_stat(array_to_stats(pos_rows))
    saved["robot0_eef_rot_axis_angle"] = get_identity_normalizer_from_stat(
        array_to_stats(rot_rows)
    )
    previous_path = ASSETS / (
        "oct_v35_r1rel_action_pose_normalizer_b51ef4dc779f.pkl"
    )
    with previous_path.open("rb") as f:
        previous = pickle.load(f)
    _print_action_table(saved, previous)
    ASSETS.mkdir(parents=True, exist_ok=True)
    tmp = ASSETS / "oct_v35_r1rel_action_pose_normalizer_tmp.pkl"
    with tmp.open("wb") as f:
        pickle.dump(saved, f)
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    final = ASSETS / f"oct_v35_r1rel_action_pose_normalizer_{digest[:12]}.pkl"
    tmp.replace(final)
    print(f"[wrote] {final}")
    print(f"[sha256] {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

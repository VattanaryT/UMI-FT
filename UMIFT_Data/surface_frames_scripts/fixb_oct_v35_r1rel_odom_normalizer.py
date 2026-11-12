#!/usr/bin/env python3
"""Fit the r1rel h4s normalizer on the odom-corrected October zarr.

Same fit as fixb_oct_v35_r1rel_normalizer.py --h4s: action range dims use the
0.5–99.5 percentile, robot0_eef_pos uses the unclipped range, rot6d stays
identity. Samples are the train queries that remain after bad spans and
long waits. Episodes 35 and 44 stay. The old pose-jump flags are not used.
Writes a new pickle. Does not open an existing pickle for writing.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from diffusion_policy.common.normalize_util import (
    array_to_stats,
    get_identity_normalizer_from_stat,
    get_range_normalizer_from_stat,
)
from diffusion_policy.dataset.umift_dataset import UmiFTDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer

from fixb_oct_v35_r1rel_normalizer import _fit_action, _sanity
from fixb_r4vt_dataset import drop_excluded_queries

ROOT = Path(__file__).resolve().parents[2]
TRAIN_CFG = ROOT / "PyriteML" / "diffusion_policy" / "config"
SCRIPTS_CFG = Path(__file__).resolve().parent / "config"
ASSETS = Path(__file__).resolve().parent / "assets"
ZARR = (
    "/mnt/data/umidata/ft/mold_aware_datasets/"
    "stringer_v3.5_fixb_odom_corr/r1rel_world/acp_replay_buffer.zarr"
)
LONG_DT = (
    "/mnt/data/umidata/ft/mold_aware_datasets/"
    "stringer_v3.5_fixb_odom_corr/shared/long_dt_gt_100ms.npz"
)
STEM = "oct_v35_h4s_r1rel_odom_action_pose_normalizer"


def main() -> int:
    with initialize_config_dir(version_base=None, config_dir=str(TRAIN_CFG)):
        cfg = compose(
            config_name="train_umift_workspace",
            overrides=[
                f"hydra.searchpath=[file://{SCRIPTS_CFG}]",
                "task=umift_oct_fixb_r1rel_h4s",
            ],
        )
    task = OmegaConf.to_container(cfg.task, resolve=True)
    frames = task["shape_meta"]["pose_frames"]
    if frames["obs"] != "relative" or frames["action"] != "relative":
        raise RuntimeError(f"r1rel pose_frames {frames}")
    sample = task["shape_meta"]["sample"]
    action_h = int(sample["action"]["sparse"]["horizon"])
    mold_h = int(sample["obs"]["sparse"]["robot0_mold_geom"]["horizon"])
    if action_h != 64 or mold_h != 51:
        raise RuntimeError(f"horizons action={action_h} mold={mold_h}")

    ds = UmiFTDataset(
        shape_meta=task["shape_meta"],
        dataset_path=ZARR,
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
    _sanity(ds)
    stats = drop_excluded_queries(ds.sampler, ZARR, [], LONG_DT, tag="r1rel-fit")
    if stats["dropped_episode"] != 0:
        raise RuntimeError("r1rel fit dropped an episode")
    ds.sampler.ignore_rgb(True)
    actions = []
    positions = []
    rotations = []
    for i, _ids in enumerate(ds.sampler.indices):
        obs, action_sample = ds.sampler.sample_sequence(i)
        action = np.asarray(action_sample["sparse"], dtype=np.float32)
        sparse = obs["sparse"]
        pos = np.asarray(sparse["robot0_eef_pos"], dtype=np.float32)
        rot = np.asarray(sparse["robot0_eef_rot_axis_angle"], dtype=np.float32)
        actions.append(action)
        positions.append(pos)
        rotations.append(rot.reshape(-1, rot.shape[-1]))
        if i % 4000 == 0:
            print(f"[collect] {i}/{len(ds.sampler.indices)}", flush=True)
    action_rows = np.concatenate(actions, axis=0)
    pos_rows = np.concatenate(positions, axis=0)
    rot_rows = np.concatenate(rotations, axis=0)
    print(
        f"[fit] kept_queries={stats['kept']} action_steps={action_rows.shape[0]} "
        f"eef_rows={pos_rows.shape[0]}"
    )
    saved = LinearNormalizer()
    saved["action"] = _fit_action(action_rows)
    saved["robot0_eef_pos"] = get_range_normalizer_from_stat(array_to_stats(pos_rows))
    saved["robot0_eef_rot_axis_angle"] = get_identity_normalizer_from_stat(
        array_to_stats(rot_rows)
    )
    blob = pickle.dumps(saved)
    digest = hashlib.sha256(blob).hexdigest()
    final = ASSETS / f"{STEM}_{digest[:12]}.pkl"
    if final.exists():
        raise RuntimeError(f"refusing to overwrite {final}")
    final.write_bytes(blob)
    scale = saved["action"].params_dict["scale"].detach().float().cpu().numpy().reshape(-1)
    offset = saved["action"].params_dict["offset"].detach().float().cpu().numpy().reshape(-1)
    print(f"[wrote] {final}")
    print(f"[sha256] {digest}")
    print(f"[action xyz] scale={scale[:3]} offset={offset[:3]}")
    print(f"[kept] {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

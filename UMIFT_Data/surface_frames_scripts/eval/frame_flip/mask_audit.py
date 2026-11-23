#!/usr/bin/env python3
"""Causal-foot keep mask vs the ΔR loss mask. Reads the training config. No writes to data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames")
SCRIPTS = ROOT / "UMIFT_Data/surface_frames_scripts"
sys.path[:0] = [
    str(SCRIPTS),
    str(ROOT / "PyriteML"),
    str(ROOT / "PyriteConfig"),
    str(ROOT / "PyriteUtility"),
    str(ROOT),
]

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from fixb_r4_chunk_filter import episode_masks_from_zarr

ZARR = (
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "r4vt_causal_foot/acp_replay_buffer.zarr"
)
OUT = Path(__file__).resolve().parent / "mask_audit.json"
CFG = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames/PyriteML/diffusion_policy/config")
TASKS = SCRIPTS / "config"


def raw_jumps() -> dict:
    import zarr

    root = zarr.open(ZARR, "r")["data"]
    n = n_jump = 0
    n_ex = n_jump_ex = 0
    per = {}
    for key in sorted(root.keys(), key=lambda s: int(s.split("_")[-1])):
        if not key.startswith("episode_"):
            continue
        ep = int(key.split("_")[-1])
        jump = episode_masks_from_zarr(root[key])["field_jump"]
        per[str(ep)] = {"n": int(len(jump)), "jump": int(jump.sum())}
        n += len(jump)
        n_jump += int(jump.sum())
        if ep not in (35, 44):
            n_ex += len(jump)
            n_jump_ex += int(jump.sum())
    return {
        "samples": n,
        "field_jump_samples": n_jump,
        "field_jump_fraction": float(n_jump / max(n, 1)),
        "excluding_35_44_samples": n_ex,
        "excluding_35_44_jumps": n_jump_ex,
        "excluding_35_44_fraction": float(n_jump_ex / max(n_ex, 1)),
        "per_episode_jumps": {k: v["jump"] for k, v in per.items() if v["jump"]},
    }


def training_config():
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=str(CFG), version_base=None)
    cfg = compose(
        config_name="train_umift_workspace",
        overrides=[
            f"hydra.searchpath=[file://{TASKS}]",
            "task=umift_oct_fixb_r4vt_causal_foot",
            "task.dataset_path=" + ZARR,
            "task.dataset.dataset_path=" + ZARR,
            "task.dataset.exclusion_stats_path=null",
            "task.dataset.pose_jump_flag_path=null",
            "task.dataset.exclude_episodes=[35,44]",
            "task.dataset.reduced_r4=true",
        ],
    )
    return cfg


def loss_fraction(dataset) -> dict:
    n_steps = n_masked = 0
    H = int(dataset.shape_meta["sample"]["action"]["sparse"]["horizon"])
    D = int(dataset.shape_meta["action"]["shape"][0])
    cols = slice(4, 10) if D == 10 else slice(21, 27)
    for epi_id, rgb_id in dataset.sampler.indices:
        w = dataset._delta_R_weight_for_query(int(epi_id), int(rgb_id), H, D)
        n_steps += H
        n_masked += int(np.sum(w[:, cols] == 0.0) // 6)
    return {
        "chunks": len(dataset.sampler.indices),
        "steps": n_steps,
        "masked_steps": n_masked,
        "fraction": float(n_masked / max(n_steps, 1)),
    }


def main() -> None:
    cfg = training_config()
    ds = cfg.task.dataset
    mask_on = bool(ds.get("mask_delta_R_after_field_jumps"))
    report = {
        "config_target": str(ds._target_),
        "mask_delta_R_after_field_jumps": mask_on,
        "field_jump_deg": float(ds.get("field_jump_deg") or 30),
        "exclude_episodes": [int(v) for v in ds.exclude_episodes],
        "long_dt_mask": str(ds.long_dt_mask_path),
        "pose_jump_flag_path": None if ds.get("pose_jump_flag_path") in (None, "null") else str(ds.pose_jump_flag_path),
        "raw": raw_jumps(),
        "keep_mask": (
            "drop_excluded_queries drops episodes 35 and 44, bad_span_0, and long-dt. "
            "It does not drop field jumps. filter_sampler_indices is not called."
        ),
    }
    from hydra.utils import instantiate

    dataset = instantiate(OmegaConf.to_container(ds, resolve=True))
    # UmiFTDatasetR4 logs the loss mask before R4VT drops queries.
    # sampler.indices here is the keep mask training actually samples.
    report["loss_mask_on_kept_chunks"] = loss_fraction(dataset)
    report["printed_545"] = {
        "masked": 667257,
        "steps": 1224000,
        "chunks": 19125,
        "when": "UmiFTDatasetR4.__init__, before drop_excluded_queries",
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "raw"}, indent=2))
    print("raw jumps", report["raw"]["field_jump_samples"], "/", report["raw"]["samples"])


if __name__ == "__main__":
    main()

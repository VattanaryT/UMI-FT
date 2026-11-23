#!/usr/bin/env python3
"""ΔR-masked share on kept train chunks, same function the trainer logs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames")
SCRIPTS = ROOT / "UMIFT_Data/surface_frames_scripts"
sys.path[:0] = [str(SCRIPTS), str(ROOT / "PyriteML"), str(ROOT / "PyriteConfig"), str(ROOT / "PyriteUtility"), str(ROOT)]

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

TRAIN_CFG = ROOT / "PyriteML/diffusion_policy/config"
SCRIPTS_CFG = SCRIPTS / "config"
OUT = Path(__file__).resolve().parent


def share(zarr: str, norm: str, sha: str, tag: str) -> dict:
    stats_path = OUT / f"exclusion_{tag}.json"
    with initialize_config_dir(version_base=None, config_dir=str(TRAIN_CFG)):
        cfg = compose(
            config_name="train_umift_workspace",
            overrides=[
                f"hydra.searchpath=[file://{SCRIPTS_CFG}]",
                "task=umift_oct_fixb_r4vt_causal_foot",
                f"task.dataset_path={zarr}",
                f"task.dataset.dataset_path={zarr}",
                f"task.dataset.action_normalizer_path={norm}",
                f"task.dataset.action_normalizer_sha256={sha}",
                f"task.dataset.exclusion_stats_path={stats_path}",
                "task.dataset.pose_jump_flag_path=null",
            ],
        )
    dataset = instantiate(cfg.task.dataset)
    before = dict(dataset._r4_mask_stats)
    after = dataset._log_delta_R_mask_fraction(train=True)
    return {"tag": tag, "before_drop": before, "kept": after, "exclusion": dataset._exclusion_stats["train"]}


def main() -> None:
    zarr, norm, sha, tag = sys.argv[1:5]
    row = share(zarr, norm, sha, tag)
    path = OUT / f"mask_{tag}.json"
    path.write_text(json.dumps(row, indent=2))
    print(json.dumps({
        "tag": tag,
        "before_drop_frac": row["before_drop"]["frac_delta_R_steps_masked"],
        "kept_frac": row["kept"]["frac_delta_R_steps_masked"],
        "kept_chunks": row["kept"]["n_chunks"],
        "exclusion": row["exclusion"],
    }, indent=2))


if __name__ == "__main__":
    main()

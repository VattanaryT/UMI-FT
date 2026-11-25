#!/usr/bin/env python3
"""How many lift windows the sv4b and sv4b-lift samplers see in one epoch.

Does not train. Builds the training sampler only.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "UMIFT_Data/surface_frames_scripts"
sys.path[:0] = [
    str(SCRIPTS),
    str(ROOT / "PyriteML"),
    str(ROOT / "PyriteConfig"),
    str(ROOT / "PyriteUtility"),
    str(ROOT),
]

from field_v2n_v3r_liftos_dataset import repeat_plan  # noqa: E402


def _dataset(task: str, overrides: list[str]):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
    import hydra

    GlobalHydra.instance().clear()
    config_dir = str(ROOT / "PyriteML/diffusion_policy/config")
    search = f"hydra.searchpath=[file://{SCRIPTS}/config]"
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="train_umift_workspace",
            overrides=[search, f"task={task}", *overrides],
        )
    OmegaConf.set_struct(cfg.task.dataset, False)
    os.environ.setdefault("PYRITE_CHECKPOINT_FOLDERS", "/mnt/data/umidata/ft/training_outputs")
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    return dataset


def _report(name: str, plan: dict) -> None:
    print(
        f"{name}: unique_lift {plan['lift']}  other {plan['other']}  "
        f"lift_rows_per_epoch {plan['slots']}  epoch_len {plan['epoch']}  "
        f"fraction {plan['fraction']:.4f}"
    )


def main() -> None:
    base = _dataset(
        "umift_oct_fixb_r4vt_field_v2n_v3r_liftos_sv4b",
        ["task.dataset.exclusion_stats_path=null"],
    )
    plan15 = base._lift_plan
    _report("sv4b 15%", plan15)
    plan30 = repeat_plan(plan15["lift"], plan15["other"], 0.30)
    plan30["lift"] = plan15["lift"]
    plan30["other"] = plan15["other"]
    _report("sv4b-lift 30% same ends", plan30)
    lifted = _dataset(
        "umift_oct_fixb_r4vt_field_v2n_v3r_liftos_sv4b_lift",
        ["task.dataset.exclusion_stats_path=null"],
    )
    _report("sv4b-lift dataset", lifted._lift_plan)
    assets = SCRIPTS / "assets"
    october = _dataset(
        "umift_oct_fixb_r4vt_field_v2n_v3r_liftos",
        [
            "task.dataset.exclusion_stats_path=null",
            f"task.dataset.lift_index_path={assets / 'lift_isolated_ends.npz'}",
            "task.dataset.action_normalizer_path="
            f"{assets / 'oct_v35_h4s_r4vt_action_normalizer_minmax_field_v2n_v3r_liftos.pkl'}",
        ],
    )
    _report("liftos 15%", october._lift_plan)


if __name__ == "__main__":
    main()

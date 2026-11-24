"""field_v3r_dataset imports and samples with open3d blocked."""

import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "UMIFT_Data" / "surface_frames_scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "PyriteML"))

from diffusion_policy.model.common.normalizer import LinearNormalizer  # noqa: E402


def _batch(tree):
    if isinstance(tree, dict):
        return {key: _batch(value) for key, value in tree.items()}
    return tree.unsqueeze(0).float()


def test_rot6d_matches_the_label_functions():
    from fixb_labels import so3_to_rot6d_rows as old_pack
    from mold_aware_runtime import rot6d_rows_to_R as old_unpack
    from rot6d import rot6d_rows_to_R, so3_to_rot6d_rows

    rng = np.random.default_rng(0)
    mats = []
    for _ in range(8):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        mats.append(q)
    rotation = np.stack(mats)
    packed = so3_to_rot6d_rows(rotation)
    assert np.array_equal(packed, old_pack(rotation))
    assert np.allclose(rot6d_rows_to_R(packed), old_unpack(packed))


def test_import_and_one_real_batch():
    saved = sys.modules.get("open3d", None)
    sys.modules["open3d"] = None
    sys.modules.pop("field_v3r_dataset", None)
    sys.modules.pop("rot6d", None)
    try:
        from field_v3r_dataset import UmiFTDatasetFieldV3R

        assert sys.modules["open3d"] is None
        GlobalHydra.instance().clear()
        train_cfg = ROOT / "PyriteML" / "diffusion_policy" / "config"
        with initialize_config_dir(version_base=None, config_dir=str(train_cfg)):
            cfg = compose(
                config_name="train_umift_workspace",
                overrides=[
                    f"hydra.searchpath=[file://{SCRIPTS / 'config'}]",
                    "task=umift_oct_fixb_r4vt_field_v3r",
                    "task.dataset.exclusion_stats_path=null",
                    "hydra.job.chdir=false",
                ],
            )
        OmegaConf.resolve(cfg)
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, UmiFTDatasetFieldV3R)
        item = dataset[0]
        batch = _batch({"obs": item["obs"], "action": item["action"]})
        if "action_loss_weight" in item:
            batch["action_loss_weight"] = _batch(item["action_loss_weight"])
        action = batch["action"]["sparse"]
        assert action.shape[0] == 1 and action.shape[-1] == 10
        assert torch.isfinite(action).all()
        policy = hydra.utils.instantiate(cfg.policy)
        normalizer = LinearNormalizer()
        fit = dict(batch["obs"]["sparse"])
        fit["action"] = action
        normalizer.fit(fit)
        policy.set_normalizer(normalizer)
        loss = policy.compute_loss(batch, None)
        assert torch.isfinite(loss)
        assert sys.modules["open3d"] is None
    finally:
        if saved is None:
            sys.modules.pop("open3d", None)
        else:
            sys.modules["open3d"] = saved

"""r1rel no-RGB: obs keys are the parent minus rgb, and one real batch steps."""

import sys
from pathlib import Path

import hydra
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "UMIFT_Data" / "surface_frames_scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "PyriteML"))

from diffusion_policy.common.drop_depth_inputs import apply_drop_depth_inputs
from diffusion_policy.common.drop_rgb_inputs import apply_drop_rgb_inputs
from diffusion_policy.common.drop_scan_inputs import apply_drop_scan_inputs
from diffusion_policy.model.common.normalizer import LinearNormalizer

PARENT = "umift_oct_fixb_r1rel_odom_h4s"
CHILD = "umift_oct_fixb_r1rel_odom_h4s_norgb"
RUN = "oct_v35_h4s_r1rel_norgb"


def _compose(task):
    GlobalHydra.instance().clear()
    train_cfg = ROOT / "PyriteML" / "diffusion_policy" / "config"
    with initialize_config_dir(version_base=None, config_dir=str(train_cfg)):
        cfg = compose(
            config_name="train_umift_workspace",
            overrides=[
                f"hydra.searchpath=[file://{SCRIPTS / 'config'}]",
                f"task={task}",
                "task.dataset.exclusion_stats_path=null",
                "hydra.job.chdir=false",
            ],
        )
    apply_drop_scan_inputs(cfg)
    apply_drop_depth_inputs(cfg)
    apply_drop_rgb_inputs(cfg)
    return cfg


def _batch(tree):
    if isinstance(tree, dict):
        return {key: _batch(value) for key, value in tree.items()}
    return tree.unsqueeze(0).float()


def test_obs_keys_are_parent_minus_rgb():
    parent = set(_compose(PARENT).task.shape_meta.obs)
    child_cfg = _compose(CHILD)
    child = set(child_cfg.task.shape_meta.obs)
    assert parent - child == {"rgb_0", "rgb_time_stamps_0"}
    assert child - parent == set()
    assert "rgb_0" in child_cfg.task.shape_meta.sample.obs.sparse
    assert "rgb_0" in child_cfg.task.shape_meta.raw
    text = (SCRIPTS / "fixb_train_oct_v35_r1rel_norgb_klone.sbatch").read_text()
    assert "--account=cse" in text
    assert "--time=24:00:00" in text
    assert "--partition=gpu-l40s" in text
    assert "checkpoint.topk.k=2" in text
    assert "checkpoint.save_last_ckpt=true" in text
    assert f"logging.name={RUN}" in text


def test_forward_on_a_real_batch():
    cfg = _compose(CHILD)
    OmegaConf.resolve(cfg)
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    policy = hydra.utils.instantiate(cfg.policy)
    assert policy.obs_encoder.rgb_keys == []
    assert not any("vit" in name or "clip" in name for name, _ in policy.named_parameters())
    item = dataset[0]
    batch = _batch({"obs": item["obs"], "action": item["action"]})
    if "action_loss_weight" in item:
        batch["action_loss_weight"] = _batch(item["action_loss_weight"])
    normalizer = LinearNormalizer()
    fit = dict(batch["obs"]["sparse"])
    fit["action"] = batch["action"]["sparse"]
    normalizer.fit(fit)
    policy.set_normalizer(normalizer)
    loss = policy.compute_loss(batch, None)
    assert torch.isfinite(loss)

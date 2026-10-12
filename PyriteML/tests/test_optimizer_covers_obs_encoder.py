"""Regression: every trainable obs_encoder parameter must be in the optimizer.

Previously only the UNet and the rgb/depth backbones were optimised; the scan_heatmap
CNN, wrench encoders, modality-attention transformer, linear_projection and
position_embedding silently stayed at their random initialisation.
"""

import os

import hydra
import pytest
import torch
from hydra import compose, initialize_config_dir

CONFIG_DIR = os.path.join(
    os.path.dirname(__file__), "..", "diffusion_policy", "config"
)


def _workspace(task, *extra):
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath(CONFIG_DIR)):
        cfg = compose(
            config_name="train_umift_workspace",
            overrides=[f"task={task}", *extra],
        )
    cls = hydra.utils.get_class(cfg._target_)
    return cls(cfg, output_dir="/tmp/_unused_ws")


def _opt_ids(ws):
    return {id(p) for g in ws.optimizer.param_groups for p in g["params"]}


@pytest.mark.parametrize("task", ["umift_ss_v2", "umift_ss_v2_crop"])
def test_all_trainable_obs_encoder_params_in_optimizer(task):
    ws = _workspace(task)
    ids = _opt_ids(ws)
    missing = [
        n
        for n, p in ws.model.obs_encoder.named_parameters()
        if p.requires_grad and id(p) not in ids
    ]
    assert not missing, f"not optimised: {missing[:5]} (+{len(missing) - 5} more)"
    scan = [
        p
        for n, p in ws.model.obs_encoder.named_parameters()
        if "scan_heatmap_model_map" in n
    ]
    assert scan and all(id(p) in ids for p in scan)
    # no parameter is listed in two groups
    all_params = [p for g in ws.optimizer.param_groups for p in g["params"]]
    assert len(all_params) == len({id(p) for p in all_params})


def test_scan_cnn_actually_updates():
    ws = _workspace("umift_ss_v2_crop")
    enc = ws.model.obs_encoder.scan_heatmap_model_map["scan_heatmap_0"]
    before = enc.backbone[0].weight.detach().clone()
    x = torch.rand(4, 3, 64, 64)
    loss = enc(x).pow(2).mean()
    loss.backward()
    ws.optimizer.step()
    assert (enc.backbone[0].weight.detach() - before).abs().max() > 0


def test_legacy_flag_restores_old_behaviour():
    ws = _workspace("umift_ss_v2_crop", "training.train_obs_encoder_rest=false")
    ids = _opt_ids(ws)
    scan = [
        p
        for n, p in ws.model.obs_encoder.named_parameters()
        if "scan_heatmap_model_map" in n
    ]
    assert scan and not any(id(p) in ids for p in scan)

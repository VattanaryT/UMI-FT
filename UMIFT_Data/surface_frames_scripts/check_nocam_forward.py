#!/usr/bin/env python3
"""Build the nocam policy on CPU and run one training forward. No dataset, no GPU."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-causal_foot")
SCRIPTS = ROOT / "UMIFT_Data/surface_frames_scripts"
sys.path[:0] = [
    str(SCRIPTS),
    str(ROOT / "PyriteML"),
    str(ROOT / "PyriteConfig"),
    str(ROOT / "PyriteUtility"),
    str(ROOT),
]

from diffusion_policy.common.drop_depth_inputs import apply_drop_depth_inputs
from diffusion_policy.common.drop_scan_inputs import apply_drop_scan_inputs
from diffusion_policy.model.common.normalizer import LinearNormalizer


def main() -> None:
    torch.set_num_threads(4)
    overrides = [
        f"hydra.searchpath=[file://{SCRIPTS}/config]",
        "task=umift_oct_fixb_r4vt_causal_foot_fixedk3000",
        "policy._target_=nocam_encoder.NoWristRgbPolicy",
        "policy.obs_encoder._target_=nocam_encoder.NoWristRgbEncoder",
        "task.drop_depth_inputs=true",
        "task.sparse_obs_rgb_horizon=16",
        "task.sparse_obs_low_dim_horizon=51",
        "task.sparse_obs_mold_horizon=51",
        "task.sparse_obs_wrench_horizon=118",
        "task.sparse_action_horizon=64",
        "training.device=cpu",
    ]
    with initialize_config_dir(
        config_dir=str(ROOT / "PyriteML/diffusion_policy/config"),
        version_base=None,
    ):
        cfg = compose(config_name="train_umift_workspace", overrides=overrides)
    apply_drop_scan_inputs(cfg)
    apply_drop_depth_inputs(cfg)
    OmegaConf.resolve(cfg)
    sha = cfg.task.dataset.action_normalizer_sha256
    if not str(sha).startswith("9a697f0c"):
        raise SystemExit(f"normalizer sha is {sha}")
    policy = hydra_instantiate(cfg.policy)
    policy.eval()
    encoder = policy.obs_encoder
    print("rgb_keys", list(encoder.rgb_keys), flush=True)
    print("wrench_keys", list(encoder.wrench_keys), flush=True)
    print("low_dim_keys", list(encoder.low_dim_keys), flush=True)
    if encoder.rgb_keys:
        raise SystemExit("encoder still has a wrist RGB branch")
    needed = {
        "robot0_eef_wrench_left",
        "robot0_eef_wrench_right",
        "robot0_eef_pos",
        "robot0_eef_rot_axis_angle",
        "robot0_mold_geom",
    }
    if not needed <= set(encoder.wrench_keys) | set(encoder.low_dim_keys):
        raise SystemExit(f"missing low-dim inputs: {needed - set(encoder.low_dim_keys) - set(encoder.wrench_keys)}")

    sparse = {
        "rgb_0": torch.rand(2, 16, 3, 224, 224),
        "robot0_eef_pos": torch.randn(2, 51, 3),
        "robot0_eef_rot_axis_angle": torch.randn(2, 51, 6),
        "robot0_eef_wrench_left": torch.randn(2, 118, 6),
        "robot0_eef_wrench_right": torch.randn(2, 118, 6),
        "robot0_mold_geom": torch.randn(2, 51, 10),
    }
    action = torch.randn(2, 64, 10)
    seen = {}

    real_forward = encoder.forward

    def spy(obs_dict):
        seen["keys"] = sorted(obs_dict)
        return real_forward(obs_dict)

    encoder.forward = spy
    norm = LinearNormalizer()
    norm.fit({"action": action, **{key: value for key, value in sparse.items() if key != "rgb_0"}})
    policy.set_normalizer(norm, None)
    loss = policy.compute_loss(
        {"obs": {"sparse": sparse}, "action": {"sparse": action}},
        {"start_training_dense": False},
    )
    if "rgb_0" in seen["keys"]:
        raise SystemExit(f"encoder saw rgb_0: {seen['keys']}")
    if not torch.isfinite(loss):
        raise SystemExit(f"loss is not finite: {loss}")
    print(f"forward_ok loss={float(loss):.6f} encoder_keys={seen['keys']}", flush=True)


def hydra_instantiate(node):
    import hydra

    return hydra.utils.instantiate(node)


if __name__ == "__main__":
    main()

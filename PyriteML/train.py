"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import pathlib
import sys

import hydra
from omegaconf import OmegaConf

from diffusion_policy.common.drop_scan_inputs import apply_drop_scan_inputs
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str(
        pathlib.Path(__file__).parent.joinpath("diffusion_policy", "config")
    ),
)
def main(cfg: OmegaConf):
    # Before resolve, shape_meta interpolations still point at task.shape_meta,
    # so one deletion removes the scan branch from the policy and the dataset.
    apply_drop_scan_inputs(cfg)
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()


if __name__ == "__main__":
    main()

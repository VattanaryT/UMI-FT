#!/usr/bin/env python3
"""e199 page. Reuses heading_tilt_panel without editing it."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames")
SCRIPTS = ROOT / "UMIFT_Data/surface_frames_scripts"
sys.path[:0] = [str(SCRIPTS / "eval"), str(SCRIPTS)]

import heading_tilt_panel as panel

panel.CKPT = (
    "/mnt/data/umidata/ft/training_outputs/"
    "oct-v35-h4s-r4vt-heading-tilt_sc_e200_bs64_valtopk_43448962/checkpoints/"
    "epoch=0199-val_loss=0.237.ckpt"
)
panel.OUT = SCRIPTS / "eval/heading_tilt_43448962_e199_ep9_ep28.html"
_page = panel.page


def page(pack):
    html = _page(pack)
    return html.replace("epoch 100", "epoch 199")


panel.page = page


if __name__ == "__main__":
    panel.main()

#!/usr/bin/env python3
"""Viser for the lift-override episode-9 traces. Port 8812. Leaves 8810 and 8811."""

import sys
from pathlib import Path

SIM = Path("/home/boeing/umi_ft_ws/worktrees/UMI-FT-field_v3r/closedloop_sim")
sys.path.insert(0, str(SIM))

import scene_view

OUT = Path(__file__).resolve().parent / "out_lift_ep9" / "view"
scene_view.OUT = OUT
scene_view.PORT = 8812
_glob = Path.glob


def _glob_ep(self, pattern):
    if pattern == "policy_*_fixb_ep40.npz":
        pattern = "policy_*_fixb_ep9.npz"
    return _glob(self, pattern)


Path.glob = _glob_ep
scene_view.main()

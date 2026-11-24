#!/usr/bin/env python3
"""Viser for the new checkpoint rollouts. Port 8812. Does not touch 8811."""

from pathlib import Path

import scene_view

OUT = Path(__file__).resolve().parent / "out_ckpt_abc" / "closed"
scene_view.OUT = OUT
scene_view.PORT = 8812
_glob = Path.glob


def _glob_ep(self, pattern):
    if pattern == "policy_*_fixb_ep40.npz":
        pattern = "policy_*_full_fixb_ep*.npz"
    return _glob(self, pattern)


Path.glob = _glob_ep
scene_view.main()

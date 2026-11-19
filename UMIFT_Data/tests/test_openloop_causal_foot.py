"""A causal-foot label action, decoded by the runner, matches the label."""

import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "surface_frames_scripts"
sys.path.insert(0, str(SCRIPTS))

import openloop_causal_foot as cf  # noqa: E402


def test_normalizer_sha_and_held_out_episodes():
    assert cf.file_sha256(cf.CAUSAL_NORM) == cf.CAUSAL_SHA
    assert cf.held_out_episodes() == [9, 12, 28, 37, 41]


def test_gt_action_decodes_to_about_zero():
    result = cf.score_labels([9])
    summary = result["summary"]["all"]
    for key in ("position_mm", "rotation_deg", "delta_n_mm", "tilt_deg"):
        pair = summary[key]
        assert pair["n"] > 0
        assert pair["median"] < 1e-3, (key, pair)
        assert pair["p90"] < 1e-3, (key, pair)

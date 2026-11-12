"""GT replay pushes the same chunks the offline runner-mode ensembler gets."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "surface_frames_scripts"
ROOT = SCRIPTS.parents[1]
for entry in (SCRIPTS, ROOT, ROOT / "PyriteML", ROOT / "PyriteUtility"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import replay_gt  # noqa: E402
from gt_replay import GtReplay, apply_normalizer  # noqa: E402


def _reference(arm: str, episode: int, exec_h: int = 6):
    """The action_blend / replay_horizon chunk loop, written out."""
    if arm == "r4vt":
        group = replay_gt._open_episode(replay_gt.R4VT_ZARR, episode)
        scale, offset = replay_gt.load_action_scale(replay_gt.R4VT_NORMALIZER, replay_gt.R4VT_SHA)
        with_delta_r = True
    else:
        group = replay_gt._open_episode(replay_gt.R1REL_ZARR, episode)
        scale, offset = replay_gt.load_action_scale(replay_gt.R1REL_NORMALIZER, replay_gt.R1REL_SHA)
        with_delta_r = False
    robot_t = np.asarray(group["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
    rgb_t = np.asarray(group["rgb_time_stamps_0"], dtype=np.float64).reshape(-1)
    demo = np.asarray(group["ts_pose_fb_0"], dtype=np.float64)
    actions = []
    for indices in replay_gt._windows(robot_t, rgb_t, exec_h):
        action = replay_gt.roundtrip_action(
            replay_gt._action_rows(group, indices, with_delta_r=with_delta_r),
            scale,
            offset,
        )
        if arm == "r1rel":
            import PyriteUtility.spatial_math.spatial_utilities as su
            from PyriteConfig.tasks.umift.umift_type_conversions import (
                action21_to_action_sample,
            )

            base = su.pose7_to_SE3(demo[int(indices[0])].reshape(1, 7))[0]
            relative = action21_to_action_sample(action, np.zeros((0,)), [0], [base])["sparse"]
            action = replay_gt.roundtrip_action(relative, scale, offset)
        actions.append(np.asarray(action, dtype=np.float64))
    return actions


def test_chunks_match_offline_runner_mode():
    for arm in ("r4vt", "r1rel"):
        for episode in (0, 9):
            replay = GtReplay(arm, episode, exec_h=6, normalizer=None)
            reference = _reference(arm, episode, 6)
            offline, _windows = replay_gt.pushed_action_chunks(arm, episode, 6)
            assert len(replay.actions) == len(reference) == len(offline) > 0
            for got, ref, off in zip(replay.actions, reference, offline):
                assert got.shape[0] == replay_gt.ACTION_HORIZON
                assert got.shape[1] == (27 if arm == "r4vt" else 21)
                np.testing.assert_allclose(got, ref, atol=0.0, rtol=0.0)
                np.testing.assert_allclose(got, off, atol=0.0, rtol=0.0)


def test_normalizer_roundtrip_is_identity():
    import torch
    from diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer

    scale = torch.tensor([2.0, 0.5, 4.0], dtype=torch.float32)
    offset = torch.tensor([-0.25, 0.5, 1.0], dtype=torch.float32)
    stats = {
        "min": torch.zeros(3),
        "max": torch.ones(3),
        "mean": torch.zeros(3),
        "std": torch.ones(3),
    }
    normalizer = SingleFieldLinearNormalizer.create_manual(scale, offset, stats)
    action = np.array([[0.2, -0.4, 0.8], [1.0, 0.0, -0.5]], dtype=np.float64)
    got = apply_normalizer(normalizer, action)
    np.testing.assert_allclose(got, action, atol=1e-5)

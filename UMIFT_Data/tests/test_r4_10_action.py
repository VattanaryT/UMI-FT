"""10-d causal-foot action: column slice, passthrough, jump mask, normalizer pin."""

import numpy as np
import torch
import zarr

from PyriteConfig.tasks.umift.umift_type_conversions import (
    action_r4_10_from_action27,
    action_r4_10_to_action_sample,
    raw_to_action21,
)
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from fixb_dataset import UmiFTDatasetR4, apply_shared_action_normalizer


NORM = (
    "/home/boeing/umi_ft_ws/worktrees/UMI-FT-causal_foot/"
    "UMIFT_Data/surface_frames_scripts/assets/"
    "oct_v35_h4s_r4vt_action_normalizer_minmax_causal_foot.pkl"
)
SHA = "14308b72d55600b4359dfa67d10aedab378e42ad26f2bbcac988eed4d97a1ac2"


def test_slice_keeps_velocity_depth_stiffness_and_delta_r():
    row = np.arange(27, dtype=np.float32)
    got = action_r4_10_from_action27(row)
    assert list(got) == [0, 1, 2, 18, 21, 22, 23, 24, 25, 26]


def test_sample_leaves_the_10_vector_unchanged():
    action = np.array([[0.2, -0.1, -0.05, 500.0, 0.9, 0.1, 0, 0, 1, 0]], np.float32)
    out = action_r4_10_to_action_sample(action, np.zeros((0, 1, 10)), [0], [np.eye(4)])
    assert np.allclose(out["sparse"], action)


class _Jump:
    _r4_mask = True
    _r4_field_jump = {"episode_0": np.array([False, True])}

    def _action_indices_for_query(self, epi_id, rgb_id):
        return np.array([0, 1])


def test_field_jump_zeros_the_10d_delta_r_dims():
    weights = UmiFTDatasetR4._delta_R_weight_for_query(_Jump(), 0, 0, 2, 10)
    assert weights.shape == (2, 10)
    assert np.all(weights[0] == 1)
    assert np.all(weights[1, :4] == 1)
    assert np.all(weights[1, 4:] == 0)


def test_field_jump_still_zeros_the_27d_delta_r_dims():
    weights = UmiFTDatasetR4._delta_R_weight_for_query(_Jump(), 0, 0, 2, 27)
    assert np.all(weights[1, :21] == 1)
    assert np.all(weights[1, 21:] == 0)


def test_pin_replaces_every_dim_including_delta_r():
    online = LinearNormalizer()
    ones = np.ones(10, dtype=np.float32)
    online["action"] = SingleFieldLinearNormalizer.create_manual(
        ones,
        np.zeros(10, dtype=np.float32),
        {key: ones.copy() for key in ("min", "max", "mean", "std")},
    )
    apply_shared_action_normalizer(online, NORM, SHA, "r4_10")
    scale = online.params_dict["action"]["scale"].detach().float().cpu().reshape(-1)
    assert scale.numel() == 10
    assert not torch.allclose(scale[4:], torch.ones(6))


def test_episode_slice_matches_the_stored_labels():
    path = (
        "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
        "r4vt_causal_foot/acp_replay_buffer.zarr"
    )
    ep = zarr.open(path, "r")["data"]["episode_40"]
    out = {}
    raw_to_action21(ep, out, [0])
    got = action_r4_10_from_action27(out["action"])
    velocity = np.asarray(ep["fixb_v_par_0"][:8], dtype=np.float64)
    depth = np.asarray(ep["fixb_delta_n_0"][:8], dtype=np.float64).reshape(-1)
    stiff = np.asarray(ep["stiffness_0"][:8], dtype=np.float64).reshape(-1)
    delta_r = np.asarray(ep["fixb_delta_R_rot6d_0"][:8], dtype=np.float64)
    assert np.allclose(got[:8, 0:2], velocity, atol=1e-5)
    assert np.allclose(got[:8, 2], depth, atol=1e-5)
    assert np.allclose(got[:8, 3], stiff, atol=1e-4)
    assert np.allclose(got[:8, 4:10], delta_r, atol=1e-5)

"""Ground-truth label replay: normalizer round trip, then decode, before chunking."""

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "surface_frames_scripts"
sys.path.insert(0, str(SCRIPTS))

import replay_analysis as ra  # noqa: E402
import replay_gt as rg  # noqa: E402

TOL_POS_M = 1e-5
TOL_ROT_DEG = 1e-3


def test_r1rel_chunk_before_ensemble():
    if not rg.R1REL_ZARR.exists():
        pytest.skip("r1rel zarr is not on this machine")
    err = rg.r1rel_chunk_errors(episode=0, rgb_id=100)
    assert err["pos_m"] <= TOL_POS_M
    assert err["rot_deg"] <= TOL_ROT_DEG


def test_r4vt_one_step_before_ensemble():
    if not rg.R4VT_ZARR.exists():
        pytest.skip("r4vt zarr is not on this machine")
    err = rg.r4vt_label_step_errors(episode=0, index=6314)
    assert err["depth_m"] <= TOL_POS_M
    assert err["foot_m"] <= TOL_POS_M
    assert err["foot_next_m"] <= TOL_POS_M
    assert err["rot_deg"] <= TOL_ROT_DEG


def test_ensemble_identical_chunks_stay_put():
    from scipy.spatial.transform import Rotation

    quat_xyzw = Rotation.from_euler("z", 10.0, degrees=True).as_quat()
    pose = np.zeros((8, 7), dtype=np.float64)
    pose[:, 0] = 0.1
    pose[:, 3:] = quat_xyzw[[3, 0, 1, 2]]
    stiff = np.full(8, 40.0)
    chunks = [
        {"pose7": pose.copy(), "nominal_pose7": pose.copy(), "stiffness": stiff.copy()}
        for _ in range(3)
    ]
    blended = rg.blend_executed(chunks, execution_horizon=6)
    assert blended["pose7"].shape[0] == 18
    pos = np.linalg.norm(blended["pose7"][:, :3] - pose[0, :3], axis=1)
    assert float(pos.max()) <= TOL_POS_M
    got = Rotation.from_quat(blended["pose7"][:, [4, 5, 6, 3]]).as_matrix()
    want = Rotation.from_euler("z", 10.0, degrees=True).as_matrix()
    ang = rg.geodesic_deg(got, np.broadcast_to(want, got.shape))
    assert float(np.max(ang)) <= TOL_ROT_DEG
    assert np.allclose(blended["stiffness"], 40.0)


def test_bad_span_excluded_from_clean():
    err = np.array([1.0, 100.0, 3.0])
    contact = np.array([True, True, True])
    bad = np.array([False, True, False])
    long_dt = np.array([False, False, True])
    buckets = rg.stiffness_buckets(err, contact, bad, long_dt)
    assert buckets["clean_contact"]["n"] == 1
    assert buckets["clean_contact"]["median"] == 1.0
    assert buckets["bad_contact"]["n"] == 1
    assert buckets["bad_contact"]["median"] == 100.0
    assert buckets["clean_free"]["n"] == 0
    assert buckets["n_long_dt"] == 1


def test_mean_rotation_takes_the_short_arc():
    from scipy.spatial.transform import Rotation

    plus = Rotation.from_euler("z", 170.0, degrees=True).as_matrix()
    minus = Rotation.from_euler("z", -170.0, degrees=True).as_matrix()
    got = rg.mean_rotation(np.stack([plus, minus]), np.ones(2))
    target = Rotation.from_euler("z", 180.0, degrees=True).as_matrix()
    assert float(rg.geodesic_deg(got, target)) < 1e-6


def test_action_blend_of_one_chunk_keeps_the_row():
    from fixb_labels import so3_to_rot6d_rows
    from scipy.spatial.transform import Rotation

    action = np.zeros((4, 27), dtype=np.float64)
    action[:, 0:3] = np.array([0.2, -0.1, -0.04])
    action[:, 21:27] = so3_to_rot6d_rows(
        Rotation.from_euler("z", 25.0, degrees=True).as_matrix()
    )
    blended = rg.blend_action_rows([action], [0], [1])
    assert np.allclose(blended[0, 0:3], action[1, 0:3])
    assert np.allclose(blended[0, 21:27], action[1, 21:27])


def test_t1_flip_is_a_half_turn_about_the_normal():
    eye = np.eye(3)
    flip = eye @ np.diag([-1.0, -1.0, 1.0])
    assert bool(ra.t1_flipped(flip, eye))
    assert not bool(ra.t1_flipped(eye, eye))
    yaw = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert not bool(ra.t1_flipped(yaw, eye))
    assert bool(ra.t1_swapped(yaw, eye))


def test_val_split_follows_string_sorted_keys():
    if not rg.R1REL_ZARR.exists() or not rg.R4VT_ZARR.exists():
        pytest.skip("odom-corr zarr is not on this machine")
    assert rg.val_episode_ids(rg.R1REL_ZARR) == [9, 12, 28, 37, 41]
    assert rg.val_episode_ids(rg.R4VT_ZARR) == [9, 12, 28, 37, 41]
    assert rg.split_name(0, rg.R1REL_ZARR) == "train"
    assert rg.split_name(9, rg.R4VT_ZARR) == "val"


def test_normalizer_sha_mismatch_raises():
    with pytest.raises(RuntimeError, match="sha256"):
        rg.load_action_scale(rg.R4VT_NORMALIZER, "0" * 64)

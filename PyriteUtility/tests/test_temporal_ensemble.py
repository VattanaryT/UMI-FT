"""Unit tests for PoseChunkEnsembler / temporal action-chunk blending."""

from __future__ import annotations

import numpy as np
import pytest

from PyriteUtility.planning_control.temporal_ensemble import (
    PoseChunkEnsembler,
    ensemble_chunks_offline,
    make_horizon_weights,
)


def _pose7(xyz, quat=(1.0, 0.0, 0.0, 0.0)):
    return np.array([*xyz, *quat], dtype=np.float64)


def _make_chunk(start_xyz, horizon=12, step_delta=(0.01, 0.0, 0.0)):
    """Linear pose path of length ``horizon`` starting at ``start_xyz``."""
    poses = np.zeros((horizon, 7), dtype=np.float64)
    stiff = np.full(horizon, 100.0, dtype=np.float64)
    for i in range(horizon):
        xyz = (
            start_xyz[0] + step_delta[0] * i,
            start_xyz[1] + step_delta[1] * i,
            start_xyz[2] + step_delta[2] * i,
        )
        poses[i] = _pose7(xyz)
    return {"pose7": poses, "stiffness": stiff}


def test_make_horizon_weights_uniform_and_exp():
    w0 = make_horizon_weights(4, mode="exp", decay_m=0.0)
    np.testing.assert_allclose(w0, np.ones(4))

    wu = make_horizon_weights(4, mode="uniform")
    np.testing.assert_allclose(wu, np.ones(4))

    we = make_horizon_weights(4, mode="exp", decay_m=1.0)
    assert we[0] > we[1] > we[2] > we[3]
    np.testing.assert_allclose(we[0], 1.0)


def test_hard_cut_boundary_jump_reduced_by_ensemble():
    """Two overlapping chunks that disagree at the execution boundary.

    Chunk A predicts steps 0..11 along +X.
    Chunk B (queried after executing 4 steps) predicts steps 4..15 but with a
    discontinuous +Z offset injected on its first step — the classic hard-cut
    MPC jump. Ensembling should shrink the boundary delta.
    """
    horizon = 12
    exec_h = 4

    chunk_a = _make_chunk((0.0, 0.0, 0.0), horizon=horizon)
    # Chunk B continues from where A would have continued, but with a z jump
    # of 5 cm at its first executed step (absolute step 4).
    chunk_b = _make_chunk((0.04, 0.0, 0.05), horizon=horizon)

    # Hard cut: take first exec_h of each, concatenate.
    hard = np.concatenate(
        [chunk_a["pose7"][:exec_h], chunk_b["pose7"][:exec_h]], axis=0
    )
    hard_boundary_dz = abs(hard[exec_h, 2] - hard[exec_h - 1, 2])
    assert hard_boundary_dz > 0.04  # ~5cm injected jump

    ensembled = ensemble_chunks_offline(
        chunks=[chunk_a, chunk_b],
        execution_horizon=exec_h,
        pose_keys=("pose7",),
        linear_keys=("stiffness",),
        weight_mode="uniform",
        decay_m=0.0,
    )
    soft = ensembled["pose7"]
    soft_boundary_dz = abs(soft[exec_h, 2] - soft[exec_h - 1, 2])

    # Absolute step 4 is averaged between chunk_a[4] (z=0) and chunk_b[0] (z=0.05).
    assert soft_boundary_dz < hard_boundary_dz
    assert soft_boundary_dz < 0.03  # below typical max_incre_m


def test_quaternion_sign_flip_is_handled():
    """Opposite-hemisphere quaternions representing the same rotation should blend."""
    horizon = 4
    exec_h = 2
    # Identity quaternion vs its antipodal counterpart.
    pose_a = np.tile(_pose7((0, 0, 0), (1, 0, 0, 0)), (horizon, 1))
    pose_b = np.tile(_pose7((0, 0, 0), (-1, 0, 0, 0)), (horizon, 1))

    ens = PoseChunkEnsembler(pose_keys=("pose7",), linear_keys=(), weight_mode="uniform")
    ens.push(0, {"pose7": pose_a})
    ens.push(0, {"pose7": pose_b})
    out = ens.pop(exec_h)
    quat = out["pose7"][0, 3:7]
    # Average of q and -q (after sign alignment) stays unit and near identity.
    np.testing.assert_allclose(np.linalg.norm(quat), 1.0, atol=1e-6)
    assert abs(quat[0]) > 0.99


def test_single_chunk_pop_matches_raw():
    chunk = _make_chunk((1.0, 2.0, 3.0), horizon=8)
    ens = PoseChunkEnsembler(
        pose_keys=("pose7",),
        linear_keys=("stiffness",),
        weight_mode="exp",
        decay_m=0.5,
    )
    ens.push(0, chunk)
    out = ens.pop(4)
    np.testing.assert_allclose(out["pose7"], chunk["pose7"][:4], atol=1e-9)
    np.testing.assert_allclose(out["stiffness"], chunk["stiffness"][:4], atol=1e-9)


def test_pop_without_push_raises():
    ens = PoseChunkEnsembler(pose_keys=("pose7",))
    with pytest.raises(KeyError):
        ens.pop(1)


def test_reset_clears_state():
    ens = PoseChunkEnsembler(pose_keys=("pose7",), linear_keys=("stiffness",))
    ens.push(0, _make_chunk((0, 0, 0), horizon=6))
    assert ens.buffered_steps == 6
    ens.reset()
    assert ens.buffered_steps == 0
    assert ens.next_pop_step == 0

"""Hist stiffness rebuild: scalar ensemble, then a pure soft axis."""

import numpy as np

from PyriteUtility.planning_control.temporal_ensemble import (
    PoseChunkEnsembler,
    blend_pose_chunk,
)
from mold_aware_runtime import hist_stiffness_enabled, rebuild_hist_stiffness


def _pose(xyz, quat=(1.0, 0.0, 0.0, 0.0)):
    return np.array([[xyz[0], xyz[1], xyz[2], *quat]], dtype=np.float64)


def _trans_eigs(packed, step):
    block = packed[:, 6 * step : 6 * step + 6]
    return np.sort(np.linalg.eigvalsh(block[:3, :3]))


def test_flag_is_off_by_default():
    assert hist_stiffness_enabled({}) is False
    assert hist_stiffness_enabled({"UMIFT_HIST_STIFFNESS": "0"}) is False
    assert hist_stiffness_enabled({"UMIFT_HIST_STIFFNESS": "1"}) is True


def test_rebuild_keeps_a_pure_axis_and_the_new_gains():
    nominal = _pose((0.0, 0.0, 0.0))
    virtual = _pose((0.0, 0.0, 0.01))
    packed = rebuild_hist_stiffness(nominal, virtual, [200.0])
    assert np.allclose(_trans_eigs(packed, 0), [200.0, 3000.0, 3000.0])
    block = packed[:, :6]
    rot = np.sort(np.linalg.eigvalsh(block[3:, 3:]))
    assert np.allclose(rot, [20.0, 20.0, 100.0])


def test_scalar_blend_does_not_tilt_the_axis():
    """Two chunks press along different axes. Averaging the 6x6 tilts it."""
    nominal = np.zeros((2, 7), dtype=np.float64)
    nominal[:, 3] = 1.0
    virtual_a = nominal.copy()
    virtual_a[:, 2] = 0.01
    virtual_b = nominal.copy()
    virtual_b[:, 0] = 0.01
    k = np.array([200.0, 200.0])
    matrix_a = rebuild_hist_stiffness(nominal, virtual_a, k)
    matrix_b = rebuild_hist_stiffness(nominal, virtual_b, k)
    # Same constructor the runner uses, but with the 6x6 as the linear field.
    averaged = PoseChunkEnsembler(
        pose_keys=("pose7", "nominal_pose7"),
        linear_keys=("stiffness",),
        weight_mode="exp",
        decay_m=0.005,
    )
    per_step_a = matrix_a.reshape(6, 2, 6).transpose(1, 0, 2)
    blend_pose_chunk(
        averaged,
        0,
        {"pose7": virtual_a, "nominal_pose7": nominal, "stiffness": per_step_a},
        1,
    )
    per_step_b = matrix_b.reshape(6, 2, 6).transpose(1, 0, 2)
    mixed = blend_pose_chunk(
        averaged,
        1,
        {"pose7": virtual_b, "nominal_pose7": nominal, "stiffness": per_step_b},
        1,
    )
    mixed_k = np.asarray(mixed["stiffness"][0])
    gap = np.sort(np.linalg.eigvalsh(mixed_k[:3, :3]))
    assert gap[2] - gap[1] > 100.0

    scalar = PoseChunkEnsembler(
        pose_keys=("pose7", "nominal_pose7"),
        linear_keys=("stiffness",),
        weight_mode="exp",
        decay_m=0.005,
    )
    blend_pose_chunk(
        scalar,
        0,
        {"pose7": virtual_a, "nominal_pose7": nominal, "stiffness": k},
        1,
    )
    blended = blend_pose_chunk(
        scalar,
        1,
        {"pose7": virtual_b, "nominal_pose7": nominal, "stiffness": k},
        1,
    )
    rebuilt = rebuild_hist_stiffness(
        blended["nominal_pose7"], blended["pose7"], blended["stiffness"]
    )
    pure = _trans_eigs(rebuilt, 0)
    assert pure[2] - pure[1] < 1e-6
    assert np.allclose(pure[1:], [3000.0, 3000.0])

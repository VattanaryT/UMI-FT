"""Smooth orientation field. No robot, no mold file."""

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "surface_frames_scripts"
sys.path.insert(0, str(SCRIPTS))

from fixb_smooth_field import (  # noqa: E402
    SMOOTH_RADIUS_M,
    neighbor_t1_deg,
    propagate_t1,
    smooth_normals,
)
from mold_aware_runtime import MoldAwareRuntime, r4_safety_limits  # noqa: E402


def test_barycentric_column_is_the_last_axis():
    frames = np.zeros((4, 3, 3))
    frames[:, :, 2] = [0.0, 0.0, 1.0]
    frames[:, 2, :] = [9.0, 8.0, 1.0]
    vid = np.array([[0, 1, 2]])
    assert np.allclose(frames[vid][..., 2], [0.0, 0.0, 1.0])


def test_planted_tangent_flip_has_no_seam():
    t1 = np.tile(np.array([1.0, 0.0, 0.0]), (6, 1))
    t1[3:] *= -1.0
    edges = np.array([[i, i + 1] for i in range(5)])
    fixed = propagate_t1(t1, edges)
    assert neighbor_t1_deg(fixed, edges) < 1.0


def test_wide_smooth_does_not_flip_across_five_mm():
    # Two point clouds meeting at a 90° normal jump. Radius is the mold default.
    a = np.stack([np.linspace(-0.04, 0.0, 20), np.zeros(20), np.zeros(20)], axis=1)
    b = np.stack([np.zeros(20), np.zeros(20), np.linspace(0.0, 0.04, 20)], axis=1)
    pts = np.concatenate([a, b])
    nrm = np.concatenate([np.tile([0.0, 0.0, 1.0], (20, 1)), np.tile([1.0, 0.0, 0.0], (20, 1))])
    sm = smooth_normals(pts, nrm, np.ones(len(pts)), SMOOTH_RADIUS_M)
    # Samples 5 mm either side of the corner, interpolated from the smoothed cloud.
    from scipy.spatial import cKDTree

    tree = cKDTree(pts)

    def at(p):
        d, i = tree.query(p, k=6, distance_upper_bound=SMOOTH_RADIUS_M)
        w = np.where(np.isfinite(d), 1.0 / np.maximum(d, 1e-4) ** 2, 0.0)
        n = (sm[np.clip(i, 0, len(sm) - 1)] * w[:, None]).sum(0)
        return n / np.linalg.norm(n)

    left = at(np.array([-0.005, 0.0, 0.002]))
    right = at(np.array([0.002, 0.0, 0.005]))
    turn = float(np.degrees(np.arccos(np.clip(np.dot(left, right), -1.0, 1.0))))
    assert turn < 30.0


class _Inward:
    """z = -Z, outward normal +Z, surface on z = 0."""

    orientation_z_inward = True

    def query(self, p, **_ignored):
        p = np.asarray(p, dtype=np.float64).reshape(-1, 3)
        R = np.broadcast_to(np.diag([1.0, -1.0, -1.0]), (len(p), 3, 3)).copy()
        foot = p.copy()
        foot[:, 2] = 0.0
        return {"x_s": foot, "R": R, "sd": p[:, 2].copy()}

    def track_step(self, p, p_prev, x_prev, R_prev):
        hit = self.query(p)
        return hit["x_s"], hit["R"]


def test_positive_delta_n_stays_outside_when_z_points_in():
    runtime = MoldAwareRuntime(
        field=_Inward(),
        R_off=np.eye(3),
        run_type="r4_world_fixb_tilt",
        limits=r4_safety_limits(),
        dry_run=False,
    )
    action = np.zeros((1, 27))
    action[0, 2] = 0.015
    action[0, 21:27] = [1, 0, 0, 0, 1, 0]
    decoded = runtime.decode_chunk(
        action, np.array([0.0, 0.0, 0.02]), np.eye(3), np.array([1.0 / 15.0]), apply_safety=True
    )
    assert decoded["x_d"][0, 2] == pytest.approx(0.015)
    assert decoded["n_out"][0, 2] == pytest.approx(1.0)
    assert decoded["R_field"][0, 2, 2] == pytest.approx(-1.0)
    assert not bool(decoded["face_hold"][0])

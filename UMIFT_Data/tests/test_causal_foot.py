"""Geometry of the shared causal foot. No mold mesh."""

import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "surface_frames_scripts"
sys.path.insert(0, str(SCRIPTS))

from causal_foot import _closest_on_triangles, double_reflect  # noqa: E402
from fixb_field import orthonormal_frames  # noqa: E402


def test_double_reflection_keeps_a_tangent_on_the_new_normal():
    n0 = np.array([0.0, 0.0, 1.0])
    t1 = np.array([1.0, 0.0, 0.0])
    n1 = np.array([0.0, 1.0, 0.0])
    carried = double_reflect(t1, n0, n1)
    assert abs(float(np.dot(carried, n1))) < 1e-8
    assert abs(float(np.linalg.norm(carried)) - 1.0) < 1e-8
    same = double_reflect(t1, n0, n0)
    assert np.allclose(same, t1)


def test_closest_point_drops_the_height_above_a_triangle():
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]])
    point, weights, _tri = _closest_on_triangles(
        np.array([0.25, 0.25, 1.0]), verts, faces, np.array([0])
    )
    assert np.allclose(point, [0.25, 0.25, 0.0])
    assert abs(float(weights.sum()) - 1.0) < 1e-8


def test_frame_uses_t2_as_n_cross_t1():
    n = np.array([0.0, -1.0, 0.0])
    t1 = np.array([1.0, 0.0, 0.0])
    frame = orthonormal_frames(t1, n)
    assert np.allclose(frame[:, 2], n)
    assert np.allclose(frame[:, 1], np.cross(n, frame[:, 0]))

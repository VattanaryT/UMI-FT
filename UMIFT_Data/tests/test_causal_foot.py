"""Geometry of the shared causal foot. No mold mesh."""

import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "surface_frames_scripts"
sys.path.insert(0, str(SCRIPTS))

from causal_foot import _closest_on_triangles, double_reflect  # noqa: E402
from fixb_field import orthonormal_frames  # noqa: E402
from mold_aware_runtime import (  # noqa: E402
    TCP_RESET_GAP_M,
    _integrate_causal_tcp,
    reset_carried_tcp,
)


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


def test_tcp_reset_stays_off_until_the_flag_is_set(monkeypatch):
    carried = np.array([0.02, 0.0, 0.0])
    measured = np.zeros(3)
    assert float(np.linalg.norm(carried - measured)) > TCP_RESET_GAP_M
    monkeypatch.delenv("UMIFT_RESET_TCP_TO_MEASURED", raising=False)
    assert np.allclose(reset_carried_tcp(carried, measured), carried)
    monkeypatch.setenv("UMIFT_RESET_TCP_TO_MEASURED", "1")
    assert np.allclose(reset_carried_tcp(carried, measured), measured)
    monkeypatch.setenv("UMIFT_RESET_TCP_TO_MEASURED", "1")
    close = np.array([0.010, 0.0, 0.0])
    assert np.allclose(reset_carried_tcp(close, measured), close)


def test_replay_tcp_step_moves_the_internal_tcp_and_policy_follows_the_command():
    """A replay step is the demo displacement. No step means the command."""

    class _Foot:
        def step(self, tcp, prev):
            point = np.asarray(tcp, dtype=np.float64).reshape(3).copy()
            point[2] = 0.0
            return point, np.eye(3)

    class _Field:
        _causal_foot = _Foot()

    p0 = np.zeros(3)
    v = np.zeros((2, 2))
    delta_n = np.array([-0.04, -0.04])
    dt = np.full(2, 1.0 / 15.0)
    disp = np.array([[0.01, 0.0, 0.0], [0.02, 0.0, 0.0]])
    replay = _integrate_causal_tcp(_Field(), p0, v, delta_n, dt, None, None, None, disp)
    policy = _integrate_causal_tcp(_Field(), p0, v, delta_n, dt, None, None, None, None)
    assert np.allclose(replay["tcp_carry"], [0.03, 0.0, 0.0])
    # Outward is -z, so a -40 mm press carries the policy TCP to z = +40 mm.
    assert abs(float(policy["tcp_carry"][2]) - 0.04) < 1e-8
    assert not np.allclose(policy["tcp_carry"], replay["tcp_carry"])


def test_frame_uses_t2_as_n_cross_t1():
    n = np.array([0.0, -1.0, 0.0])
    t1 = np.array([1.0, 0.0, 0.0])
    frame = orthonormal_frames(t1, n)
    assert np.allclose(frame[:, 2], n)
    assert np.allclose(frame[:, 1], np.cross(n, frame[:, 0]))

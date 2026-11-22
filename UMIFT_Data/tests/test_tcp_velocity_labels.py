"""TCP-velocity labels: local foot, and a harmonic off-surface frame."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import view_oct
import view_smooth_frame
from fixb_diffused_field import relax_quaternions
from fixb_field import orthonormal_frames
from fixb_tcp_track import accept_local, project_tcp_velocity, track_local_feet


def _frame(normal):
    z = -np.asarray(normal, dtype=np.float64)
    return orthonormal_frames(np.array([1.0, 0.0, 0.0]), z)


def test_tcp_velocity_is_the_tool_motion_on_t1():
    R = np.repeat(_frame([0.0, 0.0, 1.0])[None], 3, axis=0)
    p = np.array([[0.0, 0.0, 0.01], [0.01, 0.0, 0.01], [0.02, 0.0, 0.01]])
    robot_t = np.array([0.0, 0.1, 0.2])
    rgb_t = np.array([0.0, 0.1, 0.2])
    v, dt = project_tcp_velocity(R, p, robot_t, rgb_t)
    assert dt[0] == pytest.approx(0.1)
    assert v[0, 0] == pytest.approx(0.1)
    assert v[0, 1] == pytest.approx(0.0)


def test_local_foot_refuses_a_crown_wall_hop():
    def query(pts):
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        x = np.empty_like(pts)
        R = np.empty((len(pts), 3, 3))
        crown = _frame([0.0, 1.0, 0.0])
        wall = _frame([0.0, 0.0, 1.0])
        for i, row in enumerate(pts):
            hopped = 0.02 < row[0] < 0.03 and abs(row[1] - 0.01) < 1e-9 and abs(row[2] - 0.02) < 1e-9
            if hopped:
                x[i] = [row[0], 0.01, 0.0]
                R[i] = wall
            else:
                x[i] = [row[0], 0.0, row[2]]
                R[i] = crown
        return x, R

    t = np.linspace(0.0, 0.05, 11)
    p = np.column_stack([t, np.full_like(t, 0.01), np.full_like(t, 0.02)])
    x, R, held = track_local_feet(query, p)
    assert held > 0
    assert np.max(np.abs(x[:, 1])) < 1e-9
    assert np.max(np.abs(x[:, 2] - 0.02)) < 1e-9
    jump = np.linalg.norm(np.diff(x, axis=0), axis=1)
    assert np.max(jump) < 0.020
    assert not accept_local(np.array([0.025, 0.01, 0.0]), _frame([0.0, 0.0, 1.0]), x[4], R[4])


def test_one_sample_cannot_catch_the_other_sheet():
    crown = _frame([0.0, 1.0, 0.0])

    def query(pts):
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        x = np.array(pts, copy=True)
        x[:, 1] = 0.0
        return x, np.repeat(crown[None], len(pts), axis=0)

    p = np.array([[0.0, 0.01, 0.0], [0.05, 0.01, 0.0]])
    x, _, held = track_local_feet(query, p)
    assert held == 1
    assert np.linalg.norm(x[1] - x[0]) <= 0.006 + 1e-6


def test_relax_keeps_a_harmonic_line():
    left = Rotation.from_matrix(np.eye(3)).as_quat()
    right = Rotation.from_rotvec([0.0, 0.4, 0.0]).as_quat()
    q = np.zeros((5, 3, 3, 4), dtype=np.float64)
    for i in range(5):
        q[i] = (1.0 - i / 4.0) * left + (i / 4.0) * right
    q[2, 1, 1] = left
    free = np.zeros((5, 3, 3), dtype=bool)
    free[2, 1, 1] = True
    relax_quaternions(q, free, n_iter=50, tol=1e-8)
    mid = q[2, 1, 1]
    mid /= np.linalg.norm(mid)
    want = 0.5 * (left + right)
    want /= np.linalg.norm(want)
    assert abs(float(np.dot(mid, want))) > 0.999


def test_viewer_adds_buttons_without_editing_view_oct():
    assert 'id="track"' in view_smooth_frame.PAGE
    assert 'id="dof"' in view_smooth_frame.PAGE
    assert 'id="trackclean"' in view_smooth_frame.PAGE
    assert "glitchRect" in view_smooth_frame.PAGE
    assert 'id="track"' not in view_oct.PAGE
    assert 'id="trackclean"' not in view_oct.PAGE
    assert "glitchRect" not in view_oct.PAGE
    assert 'let mode = "smooth"' in view_smooth_frame.PAGE
    assert "z from up" in view_smooth_frame.PAGE
    assert "triadTraces" in view_smooth_frame.PAGE
    assert 'id="foot"' in view_smooth_frame.PAGE
    assert "tcp to foot" in view_smooth_frame.PAGE
    assert "frame n" in view_smooth_frame.PAGE
    assert 'id="foot"' not in view_oct.PAGE
    assert "yaxis4" in view_smooth_frame.PAGE
    assert "z from up" not in view_oct.PAGE
    assert "triadTraces" not in view_oct.PAGE


def test_world_tool_orientation_of_identity_and_up():
    from scipy.spatial.transform import Rotation

    pose = np.zeros((2, 7))
    pose[:, 3] = 1.0
    series = view_smooth_frame.world_tool_series(pose)
    assert np.allclose(series["z_up"], 90.0)
    assert np.allclose(series["yaw"], 0.0)
    # Columns are tool x, y, z. Tool z is world up, tool x stays +X.
    R = np.column_stack(([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]))
    q = Rotation.from_matrix(R).as_quat()
    up = np.zeros((1, 7))
    up[0, :3] = [0.1, 0.2, 0.3]
    up[0, 3] = q[3]
    up[0, 4:7] = q[:3]
    aimed = view_smooth_frame.world_tool_series(up)
    assert abs(float(aimed["z_up"][0])) < 1e-6
    assert abs(float(aimed["yaw"][0])) < 1e-6

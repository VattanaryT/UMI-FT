"""Geometry checked against scipy and hand-built frames, not against saved outputs.

hypothesis is not installed. Random cases use numpy Generator(0).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

SCRIPTS = Path(__file__).resolve().parents[1] / "surface_frames_scripts"
sys.path.insert(0, str(SCRIPTS))

from fixb_field import FixBMoldField, orthonormal_frames  # noqa: E402
from fixb_labels import pose7_to_Rp, so3_to_rot6d_rows  # noqa: E402
from heading_tilt import frame_zero, pack_tilt, rotation_from_tilt  # noqa: E402
from mold_aware_runtime import rot6d_rows_to_R  # noqa: E402

FIELD = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz"
)
RNG = np.random.default_rng(0)
UP = np.array([0.0, -1.0, 0.0])


def _scipy_deg(a, b):
    rel = np.einsum("...ji,...jk->...ik", a, b)
    return np.degrees(Rotation.from_matrix(rel.reshape(-1, 3, 3)).magnitude())


def _so3(n=8):
    return Rotation.random(n, random_state=0).as_matrix()


def test_pose7_matches_a_hand_90_degree_yaw():
    # 90° about +z. scipy xyzw is [0, 0, sin(45°), cos(45°)]. pose7 stores wxyz.
    half = np.sqrt(0.5)
    pose = np.array([0.1, -0.2, 0.3, half, 0.0, 0.0, half])
    got, p = pose7_to_Rp(pose)
    expect = Rotation.from_quat([0.0, 0.0, half, half]).as_matrix()
    assert np.allclose(p, [0.1, -0.2, 0.3])
    assert np.allclose(got[0], expect, atol=1e-12)


def test_pose7_round_trip_matches_scipy():
    mats = _so3()
    quat_xyzw = Rotation.from_matrix(mats).as_quat()
    pose = np.concatenate([RNG.normal(size=(len(mats), 3)), quat_xyzw[:, 3:4], quat_xyzw[:, :3]], axis=1)
    got, _p = pose7_to_Rp(pose)
    assert np.allclose(got, mats, atol=1e-10)


def test_rot6d_round_trip_matches_scipy():
    mats = _so3()
    packed = so3_to_rot6d_rows(mats)
    # The 6 numbers are the first two rows, which are what Gram-Schmidt rebuilds.
    assert np.allclose(packed.reshape(-1, 2, 3), mats[:, :2, :], atol=1e-12)
    got = rot6d_rows_to_R(packed)
    assert _scipy_deg(got, mats).max() < 1e-6


def test_plane_closest_point_is_straight_down():
    # Unit square in z=0. Outward normal is +z. The query is 0.4 above an interior point.
    v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    frames = np.broadcast_to(np.eye(3), (4, 3, 3)).copy()
    path = Path("/tmp/plane_fixb_field.npz")
    np.savez(
        path,
        vertices=v,
        faces=faces,
        normals=np.tile([0.0, 0.0, 1.0], (4, 1)),
        frames=frames,
        curv_tensor=np.zeros((4, 3, 3)),
    )
    field = FixBMoldField(path)
    got = field.query(np.array([0.2, 0.3, 0.4]))
    assert np.allclose(got["x_s"], [0.2, 0.3, 0.0], atol=1e-6)
    assert np.allclose(got["R"][:, 2], [0.0, 0.0, 1.0], atol=1e-6)
    assert abs(float(got["sd"]) - 0.4) < 1e-6


def test_cylinder_and_sphere_frames_match_the_cross_product():
    # Cylinder axis +z, radius 1, at (1, 0, z): outward normal +x, generator +z.
    # t2 = n × t1 = (0, -1, 0).
    cyl = orthonormal_frames(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
    expect = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]).T
    assert np.allclose(cyl, expect, atol=1e-12)
    # Sphere at the north pole. Tangent +x, t2 = n × t1 = +y.
    north = orthonormal_frames(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    assert np.allclose(north, np.eye(3), atol=1e-12)
    # A random sphere point. The normal is the radius, and t2 is n × t1.
    p = RNG.normal(size=3)
    n = p / np.linalg.norm(p)
    t1 = np.cross(n, np.array([0.0, 0.0, 1.0]))
    t1 /= np.linalg.norm(t1)
    got = orthonormal_frames(t1, n)
    assert np.allclose(got[:, 2], n, atol=1e-12)
    assert np.allclose(got[:, 1], np.cross(n, got[:, 0]), atol=1e-12)


def test_random_frames_are_rotations():
    n = RNG.normal(size=(16, 3))
    t1 = RNG.normal(size=(16, 3))
    frames = orthonormal_frames(t1, n)
    gram = np.einsum("nij,nkj->nik", frames, frames)
    assert np.allclose(gram, np.eye(3), atol=1e-8)
    assert np.allclose(np.linalg.det(frames), 1.0, atol=1e-8)


def test_zero_tilt_frame_is_the_hand_basis():
    # n_out points up. e_long points +x. side = n × e_long = +z, so tool y = -z and tool z = down.
    n = np.array([[0.0, -1.0, 0.0]])
    axis = np.array([[1.0, 0.0, 0.0]])
    got = rotation_from_tilt(n, axis, np.array([[0.0, 1.0, 0.0, 0.0]]))
    expect = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]).T
    assert np.allclose(got[0], expect, atol=1e-12)
    assert np.allclose(frame_zero(axis, n)[0], expect, atol=1e-12)


def test_heading_tilt_round_trip_matches_scipy():
    n = np.array([[0.0, -1.0, 0.0]])
    axis = np.array([[1.0, 0.0, 0.0]])
    base = frame_zero(axis, n)[0]
    extras = Rotation.random(6, random_state=1).as_matrix()
    tools = np.einsum("ij,njk->nik", base, extras)
    packed = pack_tilt(tools, np.broadcast_to(n, (len(tools), 3)), np.broadcast_to(axis, (len(tools), 3)))
    back = rotation_from_tilt(np.broadcast_to(n, (len(tools), 3)), np.broadcast_to(axis, (len(tools), 3)), packed)
    assert _scipy_deg(tools, back).max() < 1e-6


def test_real_mold_frames_stay_rotations_and_neighbors_stay_close():
    if not FIELD.is_file():
        pytest.skip("fixb field is not on this machine")
    field = FixBMoldField(FIELD)
    idx = np.linspace(0, len(field.frames) - 1, 300, dtype=int)
    frames = field.frames[idx]
    gram = np.einsum("nij,nkj->nik", frames, frames)
    assert np.allclose(gram, np.eye(3), atol=1e-6)
    assert np.all(np.linalg.det(frames) > 0.0)
    crown = int(np.argmax(field.normals @ UP))
    nbrs = np.unique(field.faces[np.any(field.faces == crown, axis=1)])
    angs = [_scipy_deg(field.frames[crown], field.frames[int(j)])[0] for j in nbrs if int(j) != crown]
    # Adjacent triangles on the crown. A flipped tangent would be ~180°.
    assert max(angs) < 15.0

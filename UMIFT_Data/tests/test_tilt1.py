"""1-tilt encode/decode on real causal-foot episodes.

Fails if the far side (n_out against table +Z) keeps the zero heading:
tool x would stay on +e_long, and +a would lean away from the flange.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import zarr
from scipy.spatial.transform import Rotation

SCRIPTS = Path(__file__).resolve().parents[1] / "surface_frames_scripts"
sys.path.insert(0, str(SCRIPTS))

from fixb_labels import pose7_to_Rp  # noqa: E402
from heading_tilt import (  # noqa: E402
    _Rz,
    e_long_sequence,
    frame_zero,
    geodesic_deg,
    long_axis,
    pack_tilt,
)
from tilt1 import SIDE_Z, heading_half, pack_tilt1, rotation_from_tilt1  # noqa: E402

ROOT = Path("/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr")
ZARR = ROOT / "r4vt_causal_foot" / "acp_replay_buffer.zarr"
FIELD = ROOT / "shared" / "fixb_field.npz"
EPISODES = (0, 9, 31)
DOWN = np.array([0.0, 1.0, 0.0])


def _frames(group, axis):
    rotation, _position = pose7_to_Rp(np.asarray(group["ts_pose_fb_0"]))
    normal = np.asarray(group["mold_normal_0"], dtype=np.float64)
    direction, _angle = e_long_sequence(normal, axis)
    return rotation, normal, direction


def _snapped(normal, direction, angle):
    """R_zero @ Rz(0 or π from the side) @ Rx(angle). Independent of tilt1's decoder."""
    half = heading_half(normal)
    zero = frame_zero(direction, normal)
    spin = Rotation.from_rotvec(
        np.column_stack([np.asarray(angle, dtype=np.float64), np.zeros(len(half)), np.zeros(len(half))])
    ).as_matrix()
    return np.einsum("nij,njk,nkl->nil", zero, _Rz(half), spin)


@pytest.fixture(scope="module")
def axis():
    if not FIELD.is_file() or not ZARR.exists():
        pytest.skip("causal-foot zarr or field is not on this machine")
    return long_axis(np.load(FIELD)["vertices"])


def test_side_holds_through_the_crown_until_z_is_clear():
    """A deadband sample keeps the last side. An episode opens on its first clear side."""
    near = np.zeros((4, 3))
    near[:, 2] = [SIDE_Z + 0.2, 0.0, -(SIDE_Z - 0.1), -(SIDE_Z + 0.05)]
    assert np.allclose(heading_half(near), [0.0, 0.0, 0.0, np.pi])
    opening = np.zeros((3, 3))
    opening[:, 2] = [0.1, -0.1, -(SIDE_Z + 0.2)]
    assert np.allclose(heading_half(opening), [np.pi, np.pi, np.pi])
    carried = np.zeros((2, 3))
    carried[:, 2] = [-0.1, -(SIDE_Z + 0.2)]
    assert np.allclose(heading_half(carried, previous=0.0), [0.0, np.pi])


def test_far_side_heading_is_half_turn_and_positive_a_leans_down():
    """Near side (n_out toward +Z) stays at 0. Far side is π. +a tips tool z down."""
    along = np.array([[1.0, 0.0, 0.0]])
    for normal in (np.array([[0.0, 0.0, 1.0]]), np.array([[0.0, 0.0, -1.0]])):
        assert heading_half(normal)[0] == (0.0 if normal[0, 2] > 0.0 else np.pi)
        upright = rotation_from_tilt1(normal, along, [0.0])
        leaned = rotation_from_tilt1(normal, along, [0.3])
        assert float((leaned[0, :, 2] - upright[0, :, 2]) @ DOWN) > 0.0
        assert geodesic_deg(leaned, _snapped(normal, along, [0.3])).max() < 1e-6


def test_literal_lean_on_a_real_frame_round_trips(axis):
    group = zarr.open(str(ZARR), "r")["data"]["episode_0"]
    _rotation, normal, direction = _frames(group, axis)
    angle = np.full(len(normal), 0.4)
    expect = _snapped(normal, direction, angle)
    assert geodesic_deg(rotation_from_tilt1(normal, direction, angle), expect).max() < 1e-6
    assert np.max(np.abs(pack_tilt1(expect, normal, direction) - 0.4)) < 1e-6


def test_action_is_velocity_press_stiffness_and_lean(axis):
    """Fails if a is written over v1/v2/δn, or if ψ/b are appended."""
    from tilt1 import action_tilt1_from_group

    group = zarr.open(str(ZARR), "r")["data"]["episode_9"]
    rotation, normal, direction = _frames(group, axis)
    half = heading_half(normal)
    zero = frame_zero(direction, normal)
    relative = np.einsum("nji,njk->nik", zero, rotation)
    swung = np.einsum("nji,njk->nik", _Rz(half), relative)
    lean = np.arctan2(swung[:, 2, 1] - swung[:, 1, 2], swung[:, 1, 1] + swung[:, 2, 2])
    command = np.asarray(group["ts_pose_command_0"])
    stiff = np.asarray(group["stiffness_0"]).reshape(-1)
    action = action_tilt1_from_group(group, axis)
    assert action.shape == (len(command), 5)
    assert np.allclose(action[:, 0], command[:, 0])
    assert np.allclose(action[:, 1], command[:, 1])
    assert np.allclose(action[:, 2], command[:, 2])
    assert np.allclose(action[:, 3], stiff)
    assert np.max(np.abs(action[:, 4] - lean.astype(np.float32))) < 1e-5


def test_side_snap_matches_demo_heading_hemisphere(axis):
    """The far side is the demo's 180° heading, not a coin flip along the stringer."""
    root = zarr.open(str(ZARR), "r")
    hit = total = 0
    for ep in EPISODES:
        rotation, normal, direction = _frames(root["data"][f"episode_{ep}"], axis)
        psi = np.arctan2(pack_tilt(rotation, normal, direction)[:, 0], pack_tilt(rotation, normal, direction)[:, 1])
        demo_half = np.where(np.cos(psi) < 0.0, np.pi, 0.0)
        hit += int(np.count_nonzero(np.isclose(heading_half(normal), demo_half)))
        total += len(psi)
    assert hit / total > 0.9


def test_real_episode_lean_round_trips_with_field_heading(axis):
    root = zarr.open(str(ZARR), "r")
    for ep in EPISODES:
        rotation, normal, direction = _frames(root["data"][f"episode_{ep}"], axis)
        lean = pack_tilt1(rotation, normal, direction)
        got = rotation_from_tilt1(normal, direction, lean)
        assert geodesic_deg(got, _snapped(normal, direction, lean)).max() < 1e-6
        again = pack_tilt1(got, normal, direction)
        assert np.max(np.abs(again - lean)) < 1e-6
        assert geodesic_deg(got, rotation_from_tilt1(normal, direction, again)).max() < 1e-6

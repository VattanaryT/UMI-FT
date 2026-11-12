"""Geometry of the Fix B foot, frame, and one-step decode. No mesh."""

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "surface_frames_scripts"
sys.path.insert(0, str(SCRIPTS))

from fixb_field import integrate_fixb, orthonormal_frames  # noqa: E402
from fixb_labels import next_control_times  # noqa: E402
from fixb_labels_tracked import episode_arrays_tracked  # noqa: E402

R4VT = (
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "r4vt_world_fixb_tilt/acp_replay_buffer.zarr"
)
R1REL = (
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/"
    "r1rel_world/acp_replay_buffer.zarr"
)


class PlaneZ:
    """z = 0 plane. Outward normal is +z."""

    def project(self, p):
        out = np.asarray(p, dtype=np.float64).copy()
        out[..., 2] = 0.0
        return out

    def query(self, p, with_curvature=False):
        p = np.asarray(p, dtype=np.float64)
        eye = np.eye(3, dtype=np.float64)
        out = {
            "x_s": self.project(p),
            "R": np.broadcast_to(eye, p.shape[:-1] + (3, 3)).copy(),
            "sd": p[..., 2].copy(),
        }
        if with_curvature:
            out["curv"] = np.zeros(p.shape[:-1] + (4,), dtype=np.float64)
        return out


def _poses(xyz: np.ndarray) -> np.ndarray:
    pose = np.zeros((len(xyz), 7), dtype=np.float64)
    pose[:, :3] = xyz
    pose[:, 3] = 1.0
    return pose


def test_flat_plane_signed_distance_is_height():
    n = 400
    robot_t = np.arange(n) * 0.005
    rgb_t = robot_t[::10]
    height = 0.04
    xyz = np.column_stack([np.zeros(n), np.zeros(n), np.full(n, height)])
    ep = {
        "robot_time_stamps_0": robot_t,
        "rgb_time_stamps_0": rgb_t,
        "ts_pose_fb_0": _poses(xyz),
    }
    arr = episode_arrays_tracked(PlaneZ(), ep)
    sd = arr["mold_signed_distance_0"].reshape(-1)
    normal = arr["mold_normal_0"]
    foot = arr["mold_x_s_0"]
    assert np.allclose(sd, height, atol=1e-5)
    assert np.allclose(normal, [0.0, 0.0, 1.0], atol=1e-6)
    assert np.allclose(foot[:, 2], 0.0, atol=1e-6)


def test_sphere_and_cylinder_frames_point_outward():
    radial = np.array([1.0, 1.0, 1.0])
    t1 = np.array([-1.0, 1.0, 0.0])
    sphere = orthonormal_frames(t1[None], radial[None])[0]
    n_hat = radial / np.linalg.norm(radial)
    assert np.allclose(sphere[:, 2], n_hat, atol=1e-8)
    assert abs(np.linalg.det(sphere) - 1.0) < 1e-8
    assert np.allclose(sphere.T @ sphere, np.eye(3), atol=1e-8)
    x_s = sphere[:, 2]
    sd = float(np.dot(1.25 * x_s - x_s, sphere[:, 2]))
    assert abs(sd - 0.25) < 1e-8

    cylinder = orthonormal_frames(np.array([[0.0, 0.0, 1.0]]), np.array([[3.0, 0.0, 0.0]]))[0]
    assert np.allclose(cylinder[:, 2], [1.0, 0.0, 0.0], atol=1e-8)
    assert abs(np.linalg.det(cylinder) - 1.0) < 1e-8
    assert np.allclose(cylinder.T @ cylinder, np.eye(3), atol=1e-8)


def test_straight_line_tangent_velocity():
    n = 400
    robot_t = np.arange(n) * 0.005
    rgb_t = robot_t[::10]
    speed = 0.05
    xyz = np.column_stack([speed * robot_t, np.zeros(n), np.full(n, 0.02)])
    ep = {
        "robot_time_stamps_0": robot_t,
        "rgb_time_stamps_0": rgb_t,
        "ts_pose_fb_0": _poses(xyz),
    }
    v = episode_arrays_tracked(PlaneZ(), ep)["fixb_v_par_0"]
    mid = v[40:-40]
    assert np.allclose(mid[:, 0], speed, atol=1e-3)
    assert np.allclose(mid[:, 1], 0.0, atol=1e-3)


def test_one_step_decode_within_1mm():
    p0 = np.array([[0.10, -0.20, 0.03]])
    v = np.array([[[0.04, 0.01], [0.0, 0.0]]])
    dn = np.array([[0.03, 0.03]])
    dt = np.array([[0.05, 0.05]])
    dec = integrate_fixb(PlaneZ(), p0, v, dn, dt)
    expect_s = np.array([[0.10, -0.20, 0.0], [0.102, -0.1995, 0.0]])
    expect_d = expect_s + np.array([0.0, 0.0, 0.03])
    assert np.max(np.abs(dec["x_s"][0] - expect_s)) < 1e-3
    assert np.max(np.abs(dec["x_d"][0] - expect_d)) < 1e-3


def test_episode0_index_6314_matches_stored_delta_n_and_v():
    zarr = pytest.importorskip("zarr")
    if not Path(R4VT).exists() or not Path(R1REL).exists():
        pytest.skip("odom-corr r4vt zarr is not on this machine")
    a = zarr.open(R4VT, "r")["data"]["episode_0"]
    b = zarr.open(R1REL, "r")["data"]["episode_0"]
    i = 6314
    robot_t = np.asarray(a["robot_time_stamps_0"], dtype=np.float64).reshape(-1)
    rgb_t = np.asarray(a["rgb_time_stamps_0"], dtype=np.float64).reshape(-1)
    t_next = next_control_times(robot_t, rgb_t)
    j = int(np.clip(np.searchsorted(robot_t, t_next[i]), 0, len(robot_t) - 1))
    x_s = np.asarray(a["mold_x_s_0"][i], dtype=np.float64)
    x_next = np.asarray(a["mold_x_s_0"][j], dtype=np.float64)
    R = np.asarray(a["mold_R_field_0"][i], dtype=np.float64).reshape(3, 3)
    p_vt = np.asarray(b["ts_pose_virtual_target_0"][i, :3], dtype=np.float64)
    dt = float(t_next[i] - robot_t[i])
    dn = float(np.dot(p_vt - x_s, R[:, 2]))
    v = (R[:, :2].T @ (x_next - x_s)) / dt
    stored_dn = float(np.asarray(a["fixb_delta_n_0"][i]).reshape(-1)[0])
    stored_v = np.asarray(a["fixb_v_par_0"][i], dtype=np.float64).reshape(2)
    assert abs(dn - stored_dn) < 1e-5
    assert np.allclose(v, stored_v, atol=1e-4)
    assert abs(stored_dn - (-0.153244)) < 1e-4
    assert np.allclose(stored_v, [-0.013059, -0.106350], atol=1e-4)

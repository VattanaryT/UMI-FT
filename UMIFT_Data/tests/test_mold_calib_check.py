"""Mold calibration check: pose conversion, thresholds, offset summary."""

import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[1] / "surface_frames_scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from fixb_labels import pose7_to_Rp
from mold_aware_runtime import DEFAULT_FIELD
from mold_calib_check import (
    expected_xyz,
    fit_rigid_offset,
    load_points,
    measure_contact,
    normal_error_deg,
    point_passed,
    pose7_base_to_fixture,
    resolve_field,
    rotvec_to_pose7,
    spot_distance_mm,
)
from PyriteEnvSuites.env_runners.umift_env_runner import pose7_batch_to_fixture


def test_train_field_is_the_runtime_default():
    assert resolve_field("train", None) == Path(DEFAULT_FIELD)


def test_field_flag_overrides_mold():
    path = Path("/tmp/other_fixb_field.npz")
    assert resolve_field("heldout", path) == path


def test_rotvec_pose7_matches_runtime_quat_layout():
    pose7 = rotvec_to_pose7(np.array([0.1, -0.2, 0.3, 0.0, 0.0, np.pi / 2]))
    R, p = pose7_to_Rp(pose7.reshape(1, 7))
    assert np.allclose(p[0], [0.1, -0.2, 0.3])
    assert np.allclose(R[0, :, 0], [0.0, 1.0, 0.0], atol=1e-8)
    assert np.allclose(R[0, :, 2], [0.0, 0.0, 1.0], atol=1e-8)


def test_normal_error_is_zero_when_tool_z_is_into_the_mold():
    angle, err = normal_error_deg(np.array([0.0, -1.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    assert angle == 180.0
    assert err == 0.0
    assert point_passed(sd_mm=1.0, normal_err_deg=err, sd_fail_mm=3.0, normal_fail_deg=5.0)
    assert not point_passed(sd_mm=4.0, normal_err_deg=0.0, sd_fail_mm=3.0, normal_fail_deg=5.0)
    assert not point_passed(sd_mm=0.0, normal_err_deg=6.0, sd_fail_mm=3.0, normal_fail_deg=5.0)


def test_yaml_defines_six_corners_in_fixture_axes():
    catalog = load_points()
    labels = [p["label"] for p in catalog["points"]]
    assert labels == [
        "flange_l_near",
        "flange_l_far",
        "flange_r_near",
        "flange_r_far",
        "wall_l_near",
        "wall_r_far",
    ]
    assert np.allclose(catalog["stringer_axis"], [1.0, 0.0, 0.0])
    assert np.allclose(catalog["across_axis"], [0.0, 0.0, 1.0])
    assert np.allclose(catalog["crown_normal"], [0.0, -1.0, 0.0])
    assert catalog["spot_warn_mm"] == 20.0
    by_label = {p["label"]: p for p in catalog["points"]}
    left = expected_xyz(by_label["flange_l_near"], "train")
    right = expected_xyz(by_label["flange_r_near"], "train")
    far = expected_xyz(by_label["flange_l_far"], "train")
    assert left[2] > right[2]
    assert left[0] > far[0]


def test_rigid_offset_recovers_translation_and_yaw():
    feet = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.0, 0.0, 0.3],
            [0.4, 0.0, -0.2],
            [0.05, -0.03, 0.1],
            [0.35, -0.03, -0.05],
        ]
    )
    center = feet.mean(axis=0)
    t = np.array([0.004, 0.001, -0.002])
    omega = np.array([0.0, -0.02, 0.0])
    tcp = feet + t + np.cross(omega, feet - center)
    fit = fit_rigid_offset(feet, tcp, [1, 0, 0], [0, 0, 1], [0, -1, 0])
    assert np.allclose(fit["t_m"], t, atol=1e-9)
    assert abs(fit["along_m"] - 0.004) < 1e-9
    assert abs(fit["across_m"] - (-0.002)) < 1e-9
    assert abs(fit["crown_m"] - (-0.001)) < 1e-9
    assert abs(fit["yaw_deg"] - np.degrees(0.02)) < 1e-6
    assert fit["resid_rms_m"] < 1e-9


def test_spot_distance_is_millimeters():
    assert abs(spot_distance_mm([0.0, 0.0, 0.0], [0.02, 0.0, 0.0]) - 20.0) < 1e-9


def test_fixture_map_matches_the_runner():
    rng = np.random.default_rng(0)
    pose7 = np.zeros((2, 7))
    pose7[:, 0:3] = rng.normal(size=(2, 3))
    pose7[:, 3] = 1.0
    T = np.eye(4)
    T[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    T[:3, 3] = [0.1, -0.2, 0.3]
    assert np.allclose(
        pose7_base_to_fixture(pose7, T),
        pose7_batch_to_fixture(pose7, T),
    )


class _FakeRuntime:
    def __init__(self):
        self.seen = None

    def fill_obs_raw(self, obs, robot_id=0):
        self.seen = np.asarray(obs["ts_pose_fb_0"], dtype=np.float64).copy()
        p = self.seen[0, :3]
        normal = np.array([0.0, -1.0, 0.0])
        foot = p - np.array([0.0, -0.001, 0.0])
        return {
            "sd": np.array([0.001]),
            "normal": normal.reshape(1, 3),
            "x_s": foot.reshape(1, 3),
        }


def test_measure_contact_uses_fill_obs_raw_on_the_fixture_pose():
    runtime = _FakeRuntime()
    T = np.eye(4)
    pose7 = rotvec_to_pose7(np.array([0.0, 0.2, 0.4, -np.pi / 2, 0.0, 0.0]))
    sample = measure_contact(runtime, T, pose7)
    assert np.allclose(runtime.seen[0], pose7)
    assert abs(sample["sd_m"] - 0.001) < 1e-12
    assert sample["normal_error_deg"] < 1e-6

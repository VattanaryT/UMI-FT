"""Synthetic checks for the offline comparison driver. No checkpoints."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "offline_compare_driver.py"
SPEC = importlib.util.spec_from_file_location("offline_compare_driver", SCRIPT)
DRV = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["offline_compare_driver"] = DRV
SPEC.loader.exec_module(DRV)

SCRIPTS = Path(__file__).resolve().parents[2] / "UMIFT_Data" / "surface_frames_scripts"
sys.path.insert(0, str(SCRIPTS))
from fixb_field import integrate_fixb  # noqa: E402
from fixb_open_loop_metrics import decode as fixb_decode  # noqa: E402


class FlatField:
    """Closest point is the query. Normal is +Z."""

    def query(self, p):
        p = np.asarray(p, dtype=np.float64)
        eye = np.eye(3, dtype=np.float64)
        return {"x_s": p.copy(), "R": np.broadcast_to(eye, p.shape[:-1] + (3, 3)).copy()}


def test_import_does_not_load_hardware():
    assert not any("hardware_interfaces" in name for name in sys.modules)


def test_query_seed_ignores_other_queries():
    alone = DRV.seeded_randn("hist", 2, (5,))
    DRV.seeded_randn("hist", 0, (5,))
    DRV.seeded_randn("hist", 9, (8,))
    again = DRV.seeded_randn("hist", 2, (5,))
    assert torch.equal(alone, again)
    other = DRV.seeded_randn("r2", 2, (5,))
    assert not torch.equal(alone, other)


def test_seeded_call_matches_a_stock_style_randn():
    def predict():
        return torch.randn(4, 21) + torch.randn(4, 21)

    torch.manual_seed(DRV.query_seed("r1rel", 3))
    reference = predict()
    got = DRV.seeded_call("r1rel", 3, predict)
    assert torch.equal(reference, got)


def test_channels_fail_independently():
    report = DRV.agreement_report(
        time_s=np.array([0.0]),
        pos_m=np.array([0.0, 0.0, 0.0]),
        rot_deg=np.array([1.0]),
        stiffness=np.array([0.0]),
    )
    assert report["channels"]["position_m"]["ok"]
    assert not report["channels"]["rotation_deg"]["ok"]
    assert not report["ok"]
    assert report["channels"]["rotation_deg"]["tol"] == DRV.TOL_ROT_DEG


def test_geodesic_is_degrees():
    quarter = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    ang = DRV.geodesic_deg(np.eye(3), quarter)
    assert ang == pytest.approx(90.0, abs=1e-6)


def test_command_decode_matches_stock_and_keeps_the_tcp():
    from PyriteUtility.spatial_math import spatial_utilities as su

    raw = np.zeros((2, 21), dtype=np.float64)
    pose9 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    raw[:, 0:9] = pose9
    raw[:, 9:18] = pose9
    raw[:, 18] = 400.0
    base = su.pose9_to_SE3(np.array([0.2, -0.1, 0.4, 1, 0, 0, 0, 1, 0]))
    cmd, vt, stiff, _grip = DRV.decode_command_like_stock(raw, [base])
    stock = DRV._load_script(
        "run_policy_open_loop_val_export_ref",
        SCRIPT.parent / "run_policy_open_loop_val_export.py",
    )
    ref = stock.decode_horizon_actions(raw, [base], [0], 21)
    assert np.allclose(cmd, ref[0])
    assert np.allclose(vt, ref[1])
    assert np.allclose(cmd[:, :3], [0.2, -0.1, 0.4])
    assert np.allclose(stiff, ref[2])


def test_fixb_position_matches_integrate_and_existing_decode():
    times = np.array([[0.0, 0.07, 0.13, 0.20]])
    action = np.zeros((1, 4, 21), dtype=np.float64)
    action[0, :, 0] = 0.05
    action[0, :, 2] = 0.01
    p0 = np.array([[0.0, 0.0, 0.3]])
    r0 = np.eye(3)[None]
    r_off = np.eye(3)
    field = FlatField()
    pos, rot = DRV.decode_fixb_like_metrics(
        "r2_world_fixb", action, times, p0, r0, field, r_off
    )
    ref_pos, ref_rot = fixb_decode(
        "r2_world_fixb", action, times, p0, r0, field, r_off
    )
    dec = integrate_fixb(
        field, p0, action[..., 0:2], action[..., 2], DRV.integration_dt(times)
    )
    assert np.allclose(pos, ref_pos)
    assert np.allclose(rot, ref_rot)
    assert np.allclose(pos, dec["x_d"])
    assert not np.allclose(pos[:, 0], pos[:, -1])


def test_r4_rotation_is_not_the_r2_heading():
    times = np.array([[0.0, 0.066, 0.132]])
    action = np.zeros((1, 3, 27), dtype=np.float64)
    action[0, :, 21:27] = [0, -1, 0, 1, 0, 0]
    p0 = np.array([[0.0, 0.0, 0.0]])
    # Measured tool yawed 90 deg about Z, so the r2 heading branch can differ.
    c, s = 0.0, 1.0
    r0 = np.array([[[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]])
    r_off = np.eye(3)
    field = FlatField()
    _, r_r2 = DRV.decode_fixb_like_metrics(
        "r2_world_fixb", action[..., :21], times, p0, r0, field, r_off
    )
    _, r_r4 = DRV.decode_fixb_like_metrics(
        "r4_world_fixb_tilt", action, times, p0, r0, field, r_off
    )
    ang = DRV.geodesic_deg(r_r2[:, 0], r_r4[:, 0])
    assert float(np.asarray(ang).reshape(-1)[0]) == pytest.approx(90.0, abs=1e-4)


def test_prepare_obs_keeps_recorded_depth():
    obs = {"sparse": {"depth_0": np.ones((2, 2), dtype=np.float32), "rgb_0": np.zeros((1,))}}
    kept = DRV.prepare_obs(obs, zero_depth=False)
    cleared = DRV.prepare_obs(obs, zero_depth=True)
    assert kept["sparse"]["depth_0"][0, 0] == 1
    assert cleared["sparse"]["depth_0"][0, 0] == 0
    assert obs["sparse"]["depth_0"][0, 0] == 1


def test_action_targets_follow_rgb_timestamps():
    rgb = np.arange(10, dtype=np.float64) * 0.066
    robot = np.arange(200, dtype=np.float64) * 0.005
    idx, times, n_valid = DRV.action_targets(rgb, robot, rgb_id=2, horizon=4, stride=1)
    assert n_valid == 4
    assert times == pytest.approx(rgb[2:6])
    assert np.all(np.diff(times) > 0)
    assert np.all(idx == np.searchsorted(robot, rgb[2:6]))


def test_smoke_arrays_reject_bad_time_and_nan():
    raw = np.zeros((1, 4, 21))
    times = np.array([[0.0, 0.1, 0.1, 0.2]])
    pos = np.zeros((1, 4, 3))
    rot = np.broadcast_to(np.eye(3), (1, 4, 3, 3)).copy()
    valid = np.ones((1, 4), dtype=bool)
    problems = DRV.check_smoke_arrays(raw, times, pos, rot, valid, 21)
    assert any("increasing" in item for item in problems)
    raw[0, 0, 0] = np.nan
    times = np.array([[0.0, 0.1, 0.2, 0.3]])
    problems = DRV.check_smoke_arrays(raw, times, pos, rot, valid, 21)
    assert any("non-finite" in item for item in problems)


def test_physical_scores_keep_position_and_rotation_apart():
    pos = np.zeros((1, 1, 3))
    ref = np.array([[[0.001, 0.0, 0.0]]])
    eye = np.eye(3)[None, None]
    quarter = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])[None, None]
    scores = DRV.physical_scores(pos, eye, ref, quarter, np.ones((1, 1), dtype=bool))
    assert scores["position_mm_mean"] == pytest.approx(1.0)
    assert scores["rotation_deg_mean"] == pytest.approx(90.0, abs=1e-4)


def test_pose7_matches_the_existing_helper():
    from fixb_labels import pose7_to_Rp

    pose = np.array([0.1, -0.2, 0.3, 0.5, 0.5, 0.5, 0.5])
    rot, pos = DRV.tcp_pose7_to_Rp(pose)
    ref_rot, ref_pos = pose7_to_Rp(pose)
    assert np.allclose(rot, ref_rot)
    assert np.allclose(pos, ref_pos)


def test_missing_container_path_uses_the_local_asset():
    local = DRV.replacement_asset(
        "/opt/umift/UMIFT_Data/surface_frames_scripts/assets/oct_v35_pose_jump_flags.npz"
    )
    assert local is not None and Path(local).is_file()
    assert DRV.replacement_asset(local) is None


def test_fresh_directory_and_record(tmp_path):
    out = tmp_path / "compare"
    assert DRV.main(["--out", str(out)]) == 0
    text = (out / "settings.json").read_text()
    assert "not run" in text
    assert "recorded depth" in text
    assert "validation split" in text
    with pytest.raises(FileExistsError):
        DRV.require_fresh_dir(out)

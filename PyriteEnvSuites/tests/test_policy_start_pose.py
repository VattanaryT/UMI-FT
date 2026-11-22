"""Policy-only episode start: rotation steps, wrist-3 winding, arrival check."""

from pathlib import Path

import numpy as np
import zarr
from scipy.spatial.transform import Rotation

from PyriteEnvSuites.env_runners.policy_start_pose import (
    ANG_TOL_DEG,
    POLICY_LOOKBACK_S,
    POLICY_START_ENV,
    POS_TOL_M,
    WRIST3_LIMIT_RAD,
    allow_normal_gap,
    arrival_error,
    consecutive_rotation_deg,
    descent_alpha_at_surface,
    first_frame_pose7,
    plan_wrist_unwind,
    pose7_to_ur_rotvec,
    position_accepted,
    position_parts,
    read_policy_start_episode,
    rotation_capped_waypoints,
    select_wrist3,
    wrist3_has_room,
    wrist3_joint_path,
    wrist3_seeds,
)


def _pose(xyz, rotvec):
    quat_xyzw = Rotation.from_rotvec(rotvec).as_quat()
    return np.array(
        [*xyz, quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float64,
    )


def test_env_is_opt_in_and_rejects_a_replay(monkeypatch):
    monkeypatch.delenv(POLICY_START_ENV, raising=False)
    monkeypatch.delenv("UMIFT_REPLAY_EPISODE", raising=False)
    assert read_policy_start_episode() is None
    monkeypatch.setenv(POLICY_START_ENV, "40")
    assert read_policy_start_episode() == 40
    monkeypatch.setenv("UMIFT_REPLAY_EPISODE", "40")
    try:
        read_policy_start_episode()
    except RuntimeError as exc:
        assert "GT replay" in str(exc)
    else:
        raise AssertionError("replay should reject the policy start")


def test_bad_episode_numbers_raise():
    try:
        read_policy_start_episode({POLICY_START_ENV: "episode_40"})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
    try:
        read_policy_start_episode({POLICY_START_ENV: "-1"})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_first_frame_is_episode_40(tmp_path):
    path = tmp_path / "acp_replay_buffer.zarr"
    root = zarr.open(str(path), mode="w")
    data = root.create_group("data")
    for name, x in (("episode_0", 0.0), ("episode_40", 0.4)):
        pose = np.zeros((3, 7))
        pose[:, 0] = x
        pose[:, 3] = 1.0
        data.create_dataset(name + "/ts_pose_fb_0", data=pose)
    got = first_frame_pose7(path, 40)
    assert got.shape == (7,)
    assert got[0] == 0.4


def test_rotation_steps_stay_under_5_deg_and_under_the_servo_limit():
    start = _pose([0.0, 0.0, 0.3], [0.0, 0.0, 0.0])
    goal = _pose([0.01, 0.0, 0.3], [0.0, 0.0, np.deg2rad(170.0)])
    waypoints = rotation_capped_waypoints(start, goal, max_step_deg=5.0, min_steps=2)
    steps = consecutive_rotation_deg(waypoints)
    assert len(waypoints) >= 35
    assert np.all(steps <= 5.0 + 1e-6)
    assert np.max(steps) <= np.degrees(0.2)
    assert np.allclose(waypoints[0], start)
    assert np.allclose(waypoints[-1], goal, atol=1e-6)


def test_plus_pi_winding_is_rejected_and_minus_2pi_is_kept():
    q6 = 3.122
    shifted = q6 - 2.0 * np.pi
    assert not wrist3_has_room(q6)
    assert wrist3_has_room(shifted)
    chosen = select_wrist3(q6, [q6, shifted, q6 + 2.0 * np.pi])
    assert abs(chosen - shifted) < 1e-9
    # Already on the good winding: do not add another turn.
    assert abs(select_wrist3(shifted, [q6, shifted]) - shifted) < 1e-9
    # A start at 0 already has room.
    assert abs(select_wrist3(0.0, [0.0, -2.0 * np.pi]) - 0.0) < 1e-9


def test_no_winding_inside_a_half_turn_raises():
    try:
        select_wrist3(3.122, [3.122, 3.122 - 2.0 * np.pi], limit_rad=np.pi)
    except ValueError as exc:
        assert "174" in str(exc)
    else:
        raise AssertionError("±180 deg cannot hold the flip")


def test_unwind_path_is_5_deg_steps_on_wrist_3_only():
    current = np.array([0.1, -0.2, 0.3, -1.0, 0.5, 3.122])
    seeds = wrist3_seeds(current)
    # IK at the target: continuous solution stays near +π, the other is -2π.
    here = current.copy()
    here[5] = 3.0
    other = here.copy()
    other[5] = 3.0 - 2.0 * np.pi
    sols = []
    for seed in seeds:
        sols.append(here if abs(seed[5] - current[5]) < 1e-6 else other)
    path, chosen = plan_wrist_unwind(current, sols)
    assert abs(chosen - (3.0 - 2.0 * np.pi)) < 1e-6
    assert path
    assert abs(path[-1][5] - (current[5] - 2.0 * np.pi)) < 1e-6
    for q in path:
        assert np.allclose(q[:5], current[:5])
    steps = np.diff([current[5], *[q[5] for q in path]])
    assert np.all(np.abs(steps) <= np.deg2rad(5.0) + 1e-9)
    # Already the continuous solution that has room: no joint move.
    good = current.copy()
    good[5] = 0.2
    empty, chosen_good = plan_wrist_unwind(current, [good, good])
    assert empty == []
    assert abs(chosen_good - 0.2) < 1e-9


def test_joint_path_for_a_full_turn_ends_on_the_target():
    current = np.zeros(6)
    current[5] = 3.122
    target = 3.122 - 2.0 * np.pi
    path = wrist3_joint_path(current, target)
    assert abs(path[-1][5] - target) < 1e-9
    assert abs(path[0][5] - current[5]) > 1e-6


def test_ur_rotvec_roundtrip():
    pose = _pose([-0.3, -0.09, 0.04], [0.2, -0.4, 1.1])
    x, y, z, rx, ry, rz = pose7_to_ur_rotvec(pose)
    assert np.allclose([x, y, z], pose[:3])
    assert np.allclose([rx, ry, rz], [0.2, -0.4, 1.1])


def test_arrival_check_names_the_failure():
    assert arrival_error(True, 0.001, 0.0, 1.0, -3.16) is None
    dead = arrival_error(False, 0.0, 0.0, 0.0, 0.0)
    assert "Robot thread is not running" in dead
    far = arrival_error(True, POS_TOL_M + 0.001, 0.0, 0.5, 0.0)
    assert "tangential" in far and "Aborting" in far
    twisted = arrival_error(True, 0.0, 0.0, ANG_TOL_DEG + 0.5, 0.0)
    assert "deg" in twisted
    tight = arrival_error(True, 0.0, 0.0, 0.0, 3.122)
    assert "Wrist 3" in tight
    # A normal gap is accepted only at contact, and only tangentially.
    assert arrival_error(True, 0.001, 0.030, 0.5, -3.16, allow_normal=True) is None
    buried = arrival_error(True, 0.001, 0.030, 0.5, -3.16, allow_normal=False)
    assert "normal" in buried
    assert position_accepted(0.004, 0.0, 2.5, allow_normal=False)
    assert not position_accepted(0.006, 0.0, 0.0, allow_normal=False)
    assert WRIST3_LIMIT_RAD == 2.0 * np.pi
    assert POLICY_LOOKBACK_S == 8.0
    assert POS_TOL_M == 0.005
    assert ANG_TOL_DEG == 3.0


def test_outside_tcp_is_not_cut_and_an_inside_tcp_stops_at_the_surface():
    alphas = np.linspace(0.0, 1.0, 5)
    outside = np.array([0.073, 0.060, 0.050, 0.040, 0.031])
    assert descent_alpha_at_surface(alphas, outside) == 1.0
    # Zero is the third sample. The first negative sample is the next one.
    crossing = np.array([0.020, 0.010, 0.0, -0.010, -0.020])
    assert abs(descent_alpha_at_surface(alphas, crossing) - 0.5) < 1e-9
    try:
        descent_alpha_at_surface([0.0, 1.0], [-0.01, -0.02])
    except ValueError as exc:
        assert "hover" in str(exc).lower()
    else:
        raise AssertionError("an inside hover must raise")


def test_normal_gap_is_allowed_only_when_the_tcp_was_inside():
    tang, normal = position_parts([0.0, 0.0, 0.03], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    assert abs(tang) < 1e-9
    assert abs(normal - 0.03) < 1e-9
    assert not allow_normal_gap(0.031, 0.055, normal)
    assert allow_normal_gap(-0.031, 0.001, normal)
    assert not allow_normal_gap(-0.031, -0.010, normal)
    assert position_accepted(0.001, 0.030, 0.5, allow_normal=True)
    assert not position_accepted(0.001, 0.030, 0.5, allow_normal=False)


def test_runner_checks_before_the_prompt_and_holds_the_lookback():
    src = Path(
        "/home/boeing/umi_ft_ws/worktrees/UMI-FT-smooth_frame/"
        "PyriteEnvSuites/env_runners/umift_env_runner.py"
    ).read_text()
    wait = src.find("                _wait_policy_descent(")
    confirm = src.find("                confirm_policy_start_reached(")
    prompt = src.find("policy_hold = True")
    hold = src.find("if policy_hold:")
    assert wait != -1 and confirm != -1 and prompt != -1 and hold != -1
    assert wait < confirm < prompt < hold
    assert "stop_at_contact=policy_episode is not None" in src
    wait_fn = src.split("def _wait_policy_descent", 1)[1].split("\ndef ", 1)[0]
    assert wait_fn.find("if retried:") < wait_fn.find("raise RuntimeError")
    confirm_fn = src.split("def confirm_policy_start_reached", 1)[1].split("\ndef ", 1)[0]
    assert "Continuing to the 8 s hold." in confirm_fn
    assert "UMIFT_POLICY_START_EPISODE" in src or "read_policy_start_episode" in src
    assert "max_rot_step_deg" in src

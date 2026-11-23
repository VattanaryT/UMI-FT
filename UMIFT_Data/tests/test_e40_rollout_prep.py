"""Offline checks for the e40 rollout. Does not touch the robot."""

from pathlib import Path

import numpy as np
import pytest
import zarr

LOG = Path("/home/boeing/data/control_log/checkpoints_e40_rollout0_1795128531")
ROBOT = Path("/home/boeing/umi_ft_ws/UMI-FT/runs/episode_1795128508/robot_data_0.json")
FIELD = Path(
    "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_field.npz"
)
CAMERA_LAG_S = 0.105
SLOW = 2.0


def _raw_end_gap(pose_t, rgb_t, hardware_now, wall_now):
    from mold_aware_runtime import _on_hardware_clock

    rgb_hw = _on_hardware_clock(rgb_t, hardware_now, wall_now)
    return abs(float(pose_t[-1]) - float(rgb_hw[-1]))


def test_two_clocks_105ms_becomes_one_sample(monkeypatch):
    from mold_aware_runtime import align_obs_to_image

    monkeypatch.setenv("UMIFT_ALIGN_OBS_TO_IMAGE", "1")
    hardware_now = 57.686
    wall_now = 1_795_128_539.929 + CAMERA_LAG_S
    rgb = wall_now - CAMERA_LAG_S - np.arange(8, 0, -1.0 / 15.0)[::-1]
    rgb = np.concatenate([rgb, [wall_now - CAMERA_LAG_S]])
    pose_t = np.arange(hardware_now - 1.0, hardware_now, 0.002)
    wrench_t = pose_t.copy()
    raw = {
        "rgb_time_stamps_0": rgb,
        "ts_pose_fb_0": np.zeros((len(pose_t), 7)),
        "robot_time_stamps_0": pose_t,
        "wrench_left_0": np.zeros((len(wrench_t), 6)),
        "wrench_time_stamps_0": wrench_t,
    }
    before = _raw_end_gap(pose_t, rgb, hardware_now, wall_now)
    report = align_obs_to_image(raw, hardware_now_s=hardware_now, wall_now_s=wall_now)
    assert before == pytest.approx(CAMERA_LAG_S, abs=0.02)
    assert report["pose_ms"] < 10.0
    assert report["wrench_ms"] < 10.0
    assert abs(raw["robot_time_stamps_0"][-1] - raw["rgb_time_stamps_0"][-1]) < 0.004


def _robot_pose():
    """The episode dump is JSON with trailing commas, so scan the lines."""
    times, poses = [], []
    stamp = None
    for line in ROBOT.open():
        if '"robot_time_stamps"' in line:
            stamp = float(line.split(":", 1)[1].strip().rstrip(","))
        elif '"ts_pose_fb"' in line and stamp is not None:
            nums = line[line.index("[") + 1 : line.index("]")]
            poses.append([float(x) for x in nums.split(",")])
            times.append(stamp)
            stamp = None
    return np.asarray(times) / 1000.0, np.asarray(poses)


def test_logged_rollout_alignment(monkeypatch):
    from mold_aware_runtime import align_obs_to_image

    monkeypatch.setenv("UMIFT_ALIGN_OBS_TO_IMAGE", "1")
    log = zarr.open(str(LOG), "r")
    rgb = np.asarray(log["horizon_0"]["obs_inference"]["rgb_time_stamps_0"], np.float64)
    hardware_now = float(log["horizon_0"]["obs_inference"].attrs["hardware_time_s"])
    pose_t, pose = _robot_pose()
    # The server timer and the json stamps are one clock. Keep the second
    # that contains the query, plus the slack the runner now requests.
    lo = hardware_now - 1.2
    keep = (pose_t >= lo) & (pose_t <= hardware_now + 0.01)
    pose_t, pose = pose_t[keep], pose[keep]
    wall_now = float(rgb[-1]) + CAMERA_LAG_S
    before = _raw_end_gap(pose_t, rgb, hardware_now, wall_now)
    raw = {
        "rgb_time_stamps_0": rgb.copy(),
        "ts_pose_fb_0": pose,
        "robot_time_stamps_0": pose_t.copy(),
        "wrench_left_0": np.zeros((len(pose_t), 6)),
        "wrench_time_stamps_0": pose_t.copy(),
    }
    report = align_obs_to_image(raw, hardware_now_s=hardware_now, wall_now_s=wall_now)
    assert before > 0.05
    assert report["pose_ms"] < 10.0
    assert report["wrench_ms"] < 10.0


def test_first_queries_and_stationary_history():
    log = zarr.open(str(LOG), "r")
    speeds = []
    for i in range(5):
        v = np.asarray(log[f"horizon_{i}"]["action"]["v_par"], np.float64)[0]
        speeds.append(float(np.linalg.norm(v)))
    hist = np.asarray(log["horizon_0"]["obs_policy"]["robot0_eef_pos"], np.float64)
    travel = float(np.linalg.norm(hist.max(0) - hist.min(0)))
    # Query 0 is already ~0.11 m/s. The next four are the slow ones.
    assert speeds[0] > 0.08
    assert max(speeds[1:]) < 0.06
    assert travel < 0.005


def _logged_steps():
    log = zarr.open(str(LOG), "r")
    v, dn, dt = [], [], []
    for i in range(25):
        group = log[f"horizon_{i}"]["action"]
        v.append(np.asarray(group["v_par"], np.float64))
        dn.append(np.asarray(group["delta_n"], np.float64).reshape(-1))
        ts = np.asarray(group["timestamps_s"], np.float64).reshape(-1)
        step = np.diff(ts, prepend=ts[0] - (ts[1] - ts[0])) / SLOW
        dt.append(step)
    return np.concatenate(v), np.concatenate(dn), np.concatenate(dt)


def test_geodesic_keeps_small_steps_after_7s(monkeypatch):
    from fixb_smooth_field import SmoothFrameField
    from mold_aware_runtime import integrate_tcp_foot

    v, dn, dt = _logged_steps()
    t = np.cumsum(dt)
    late = t > 7.5
    assert late.any()
    side = FIELD.with_name("fixb_smooth_frames.npz")
    field = SmoothFrameField(FIELD, side)
    field._causal_npz = (FIELD, side)
    start = np.asarray(zarr.open(str(LOG), "r")["horizon_0"]["action"]["foot"][0], np.float64)

    held = np.zeros((len(v), 3))

    def wall_speed(mode):
        monkeypatch.setenv("UMIFT_FOOT_WALK", mode)
        decoded = integrate_tcp_foot(field, start, v, dn, dt, tcp_step=held)
        foot = decoded["x_s"]
        wall = np.maximum(dt[1:] * SLOW, 1e-6)
        return float(np.median(np.linalg.norm(np.diff(foot, axis=0), axis=1)[late[1:]] / wall[late[1:]]))

    legacy = wall_speed("legacy")
    geodesic = wall_speed("geodesic")
    predicted = float(np.median(np.linalg.norm(v[late], axis=1))) / SLOW
    # The measured TCP is held still, which is the stall. Legacy walk-back
    # cancels the step. Geodesic keeps it near the commanded v/2.
    assert legacy < 0.5 * predicted
    assert geodesic > 0.5 * predicted

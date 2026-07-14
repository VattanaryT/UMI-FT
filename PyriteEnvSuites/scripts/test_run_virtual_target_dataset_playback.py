import importlib.util
import pathlib

import numpy as np


SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parent / "run_virtual_target_dataset_playback.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_virtual_target_dataset_playback", SCRIPT_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_validate_episode_keys_accepts_required_schema():
    episode_data = {
        "ts_pose_command_0": np.zeros((10, 7)),
        "ts_pose_virtual_target_0": np.zeros((10, 7)),
        "stiffness_0": np.ones((10,)),
        "robot_time_stamps_0": np.linspace(0.0, 0.9, 10).reshape(-1, 1),
    }
    MODULE.validate_episode_keys(episode_data)


def test_validate_episode_keys_rejects_missing_required_key():
    episode_data = {
        "ts_pose_command_0": np.zeros((10, 7)),
        "stiffness_0": np.ones((10,)),
        "robot_time_stamps_0": np.linspace(0.0, 0.9, 10).reshape(-1, 1),
    }
    try:
        MODULE.validate_episode_keys(episode_data)
        assert False, "Expected ValueError for missing key."
    except ValueError as exc:
        assert "ts_pose_virtual_target_0" in str(exc)


def test_compute_stiffness_matrices_has_expected_shape_and_diagonal():
    # Target at origin, virtual target displaced in +x, so compliance axis is x.
    ts_pose_command = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    ts_pose_virtual_target = np.array(
        [
            [0.01, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.02, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    stiffness = np.array([800.0, 1200.0])
    current_pose = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    mats = MODULE.compute_stiffness_matrices(
        current_pose7=current_pose,
        ts_pose_command=ts_pose_command,
        ts_pose_virtual_target=ts_pose_virtual_target,
        ts_stiffness=stiffness,
        default_stiffness=5000.0,
        default_stiffness_rot=100.0,
    )

    assert mats.shape == (6, 12)
    # First step translational xx stiffness should match provided target stiffness
    step0 = mats[:, :6]
    assert np.isclose(step0[0, 0], 800.0, atol=1e-6)
    assert np.isclose(step0[3, 3], 100.0, atol=1e-6)


def test_compute_replay_timestamps_applies_slowdown():
    timestamps = MODULE.compute_replay_timestamps_s(
        count=4, raw_time_step_s=0.02, slow_down_factor=2.0
    )
    assert np.allclose(timestamps, np.array([0.0, 0.04, 0.08, 0.12]))


def test_select_target_trajectory_uses_virtual_or_nominal():
    ts_pose_command = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    ts_pose_virtual_target = np.array(
        [
            [0.0, 0.2, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.1, 0.2, 0.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )

    nominal = MODULE.select_target_trajectory(
        ts_pose_command=ts_pose_command,
        ts_pose_virtual_target=ts_pose_virtual_target,
        test_nominal_target=True,
    )
    virtual = MODULE.select_target_trajectory(
        ts_pose_command=ts_pose_command,
        ts_pose_virtual_target=ts_pose_virtual_target,
        test_nominal_target=False,
    )

    assert np.allclose(nominal, ts_pose_command)
    assert np.allclose(virtual, ts_pose_virtual_target)


def test_align_trajectories_to_current_pose_anchors_first_command():
    ts_pose_command = np.array(
        [
            [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
            [2.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    ts_pose_virtual_target = np.array(
        [
            [1.0, 2.2, 3.0, 1.0, 0.0, 0.0, 0.0],
            [2.0, 2.2, 3.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    current_pose = np.array([10.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    aligned_cmd, aligned_vt = MODULE.align_trajectories_to_current_pose(
        ts_pose_command=ts_pose_command,
        ts_pose_virtual_target=ts_pose_virtual_target,
        current_pose7=current_pose,
    )

    assert np.allclose(aligned_cmd[0], current_pose)
    # Relative +1 x motion is preserved from first to second step.
    assert np.isclose(aligned_cmd[1, 0] - aligned_cmd[0, 0], 1.0)
    # Virtual target offset from command is preserved.
    assert np.isclose(aligned_vt[0, 1] - aligned_cmd[0, 1], 0.2)

import importlib.util
import pathlib
import tempfile

import numpy as np
import zarr


SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parent / "run_policy_open_loop_val_export.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_policy_open_loop_val_export", SCRIPT_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_compute_sampler_stride_matches_zucchini_defaults():
    # sparse_execution_horizon=4, action_ds=10, query_ds=8 -> 5
    stride = MODULE.compute_sampler_stride(
        sparse_execution_horizon=4,
        sparse_action_down_sample_steps=10,
        sparse_query_frequency_down_sample_steps=8,
    )
    assert stride == 5


def test_compute_sampler_stride_rejects_non_positive_result():
    try:
        MODULE.compute_sampler_stride(
            sparse_execution_horizon=0,
            sparse_action_down_sample_steps=10,
            sparse_query_frequency_down_sample_steps=8,
        )
        assert False, "Expected ValueError for zero execution horizon"
    except ValueError as exc:
        assert "must be >= 1" in str(exc)


def test_compute_sparse_action_timesteps_s():
    ts = MODULE.compute_sparse_action_timesteps_s(
        sparse_action_horizon=3,
        sparse_action_down_sample_steps=10,
        raw_time_step_s=0.02,
        slow_down_factor=1.0,
    )
    assert np.allclose(ts, np.array([0.0, 0.2, 0.4]))


def test_validate_export_episode_keys_accepts_required_schema():
    episode_data = {
        "ts_pose_command_0": np.zeros((10, 7)),
        "ts_pose_virtual_target_0": np.zeros((10, 7)),
        "stiffness_0": np.ones((10,)),
        "robot_time_stamps_0": np.linspace(0.0, 0.9, 10).reshape(-1, 1),
    }
    MODULE.validate_export_episode_keys(episode_data)


def test_validate_export_episode_keys_rejects_missing_and_mismatch():
    episode_data = {
        "ts_pose_command_0": np.zeros((10, 7)),
        "stiffness_0": np.ones((10,)),
        "robot_time_stamps_0": np.linspace(0.0, 0.9, 10).reshape(-1, 1),
    }
    try:
        MODULE.validate_export_episode_keys(episode_data)
        assert False, "Expected ValueError for missing key"
    except ValueError as exc:
        assert "ts_pose_virtual_target_0" in str(exc)

    episode_data = {
        "ts_pose_command_0": np.zeros((10, 7)),
        "ts_pose_virtual_target_0": np.zeros((8, 7)),
        "stiffness_0": np.ones((10,)),
        "robot_time_stamps_0": np.linspace(0.0, 0.9, 10).reshape(-1, 1),
    }
    try:
        MODULE.validate_export_episode_keys(episode_data)
        assert False, "Expected ValueError for length mismatch"
    except ValueError as exc:
        assert "length mismatch" in str(exc)


def test_group_sampler_indices_by_episode():
    indices = [(3, 0), (3, 8), (7, 0), (3, 16)]
    grouped = MODULE.group_sampler_indices_by_episode(indices)
    assert set(grouped.keys()) == {3, 7}
    assert grouped[3] == [(0, 0), (1, 8), (3, 16)]
    assert grouped[7] == [(2, 0)]


def test_decode_horizon_actions_pose9pose9s1a2_shapes():
    # Identity relative action: pose9 = [0,0,0, 1,0,0, 0,1,0]
    identity_pose9 = np.array(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64
    )
    vt_pose9 = np.array(
        [0.01, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64
    )
    horizon = 4
    raw = np.zeros((horizon, 21), dtype=np.float64)
    for i in range(horizon):
        raw[i, 0:9] = identity_pose9
        raw[i, 9:18] = vt_pose9
        raw[i, 18] = 800.0 + i
        raw[i, 19] = 0.05
        raw[i, 20] = 1.0

    base = np.eye(4, dtype=np.float64)
    base[:3, 3] = np.array([1.0, 2.0, 3.0])
    ts_cmd, ts_vt, stiff, grip = MODULE.decode_horizon_actions(
        raw_action=raw,
        base_se3=[base],
        id_list=[0],
        action_dim=21,
    )
    assert ts_cmd.shape == (horizon, 7)
    assert ts_vt.shape == (horizon, 7)
    assert stiff.shape == (horizon,)
    assert grip is not None and grip.shape == (horizon,)
    assert np.allclose(ts_cmd[:, :3], np.array([1.0, 2.0, 3.0]), atol=1e-6)
    assert np.allclose(ts_vt[:, 0], 1.01, atol=1e-6)
    assert np.allclose(stiff, np.array([800.0, 801.0, 802.0, 803.0]))


def test_write_episode_to_zarr_roundtrip_schema():
    episode_data = {
        "ts_pose_command_0": np.zeros((5, 7), dtype=np.float64),
        "ts_pose_virtual_target_0": np.ones((5, 7), dtype=np.float64),
        "stiffness_0": np.full((5,), 900.0, dtype=np.float64),
        "robot_time_stamps_0": np.linspace(0.0, 0.4, 5).reshape(-1, 1),
        "gripper_0": np.zeros((5, 1), dtype=np.float64),
    }
    MODULE.validate_export_episode_keys(episode_data)

    with tempfile.TemporaryDirectory() as tmp:
        out = str(pathlib.Path(tmp) / "policy_open_loop_val.zarr")
        MODULE.write_episode_to_zarr(out, "episode_3_policy", episode_data)
        root = zarr.open(out, mode="r")
        ep = root["data"]["episode_3_policy"]
        for key in MODULE.REQUIRED_KEYS:
            assert key in ep
        assert ep["ts_pose_command_0"].shape == (5, 7)
        assert ep["stiffness_0"].shape == (5,)
        assert "gripper_0" in ep

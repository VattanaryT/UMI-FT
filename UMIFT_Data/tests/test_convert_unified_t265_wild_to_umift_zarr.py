import importlib.util
import pathlib

import numpy as np


SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "convert_unified_t265_wild_to_umift_zarr.py"
SPEC = importlib.util.spec_from_file_location("convert_unified_t265_wild_to_umift_zarr", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_normalize_timestamps_uses_global_first_sample_offset():
    streams = {
        "rgb": np.array([[10.5], [10.8], [11.1]], dtype=np.float64),
        "robot": np.array([[9.0], [9.1]], dtype=np.float64),
        "wrench": np.array([[9.2], [9.5], [10.0]], dtype=np.float64),
    }
    normalized = MODULE.normalize_timestamps(streams)

    assert np.isclose(normalized["robot"][0, 0], 0.0)
    assert np.isclose(normalized["rgb"][0, 0], 1.5)
    assert np.isclose(normalized["wrench"][0, 0], 0.2)


def test_build_ts_pose_fb_converts_world_and_body_frames_before_offset():
    odom = {
        "pos_x": np.array([1.0, 2.0]),
        "pos_y": np.array([3.0, 4.0]),
        "pos_z": np.array([5.0, 6.0]),
        "orient_x": np.array([0.0, 0.0]),
        "orient_y": np.array([0.0, 0.0]),
        "orient_z": np.array([0.0, 0.0]),
        "orient_w": np.array([1.0, 1.0]),
    }
    pose = MODULE.build_ts_pose_fb(
        odom,
        sensor_offset_xyz_m=np.array([0.0, 0.0, -0.02], dtype=np.float64),
    )

    assert pose.shape == (2, 7)
    # FT-world/FT-body -> TCP-world/TCP-body uses the same axis relabeling on
    # both pose frame labels. An identity pose remains identity after both
    # labels are converted, while the absolute position is expressed in TCP
    # world coordinates.
    # sensor_offset_xyz_m is target->sensor, so [0, 0, -0.02] shifts the
    # converted sensor pose 2 cm forward to the TCP origin.
    expected_position = np.array([[-3.0, 1.0, 5.02], [-4.0, 2.0, 6.02]])
    expected_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
    assert np.allclose(pose[:, :3], expected_position)
    assert np.allclose(pose[:, 3:], np.broadcast_to(expected_quat_wxyz, (2, 4)))


def test_build_wrench_keys_populates_coinft_and_concat():
    ft = {
        "force_x": np.array([1.0, 2.0]),
        "force_y": np.array([3.0, 4.0]),
        "force_z": np.array([5.0, 6.0]),
        "torque_x": np.array([7.0, 8.0]),
        "torque_y": np.array([9.0, 10.0]),
        "torque_z": np.array([11.0, 12.0]),
    }
    keys = MODULE.build_wrench_keys(ft)

    expected_left = np.array([[-3.0, 1.0, 5.0, -9.0, 7.0, 11.0], [-4.0, 2.0, 6.0, -10.0, 8.0, 12.0]])
    expected_right = np.zeros_like(expected_left)
    expected_concat = np.hstack((expected_left, expected_right))

    assert np.allclose(keys["wrench_left_0"], expected_left)
    assert np.allclose(keys["wrench_right_0"], expected_right)
    assert np.allclose(keys["wrench_concat_0"], expected_concat)
    assert np.allclose(keys["wrench_left_coinft_0"], expected_left)
    assert np.allclose(keys["wrench_right_coinft_0"], expected_right)
    assert np.allclose(keys["wrench_concat_coinft_0"], expected_concat)


def test_build_wrench_keys_applies_point_offset_to_torque():
    ft = {
        "force_x": np.array([2.0]),
        "force_y": np.array([3.0]),
        "force_z": np.array([4.0]),
        "torque_x": np.array([10.0]),
        "torque_y": np.array([20.0]),
        "torque_z": np.array([30.0]),
    }
    # sensor is 3 cm at -z from target point
    keys = MODULE.build_wrench_keys(ft, sensor_offset_xyz_m=np.array([0.0, 0.0, -0.03]))

    # FT -> TCP rotates force to [-3, 2, 4] and torque to [-20, 10, 30].
    # Then tau_target = tau_sensor + r x F, r=[0,0,-0.03], r x F = [0.06, 0.09, 0.0].
    expected_left = np.array([[-3.0, 2.0, 4.0, -19.94, 10.09, 30.0]])
    assert np.allclose(keys["wrench_left_0"], expected_left)


def test_resolve_tag_traj_path_and_resample(tmp_path):
    folder = tmp_path / "data"
    zarr_dir = folder / "zarr"
    tag_dir = folder / "tag_traj"
    zarr_dir.mkdir(parents=True)
    tag_dir.mkdir(parents=True)
    unified = zarr_dir / "demo_unified.zarr"
    unified.mkdir()
    sidecar = tag_dir / "demo_ee_master.npz"
    ts = np.array([0.0, 1.0, 2.0])
    pose7 = np.zeros((3, 7))
    pose7[:, 0] = [0.0, 1.0, 2.0]
    pose7[:, 3] = 1.0  # qw
    np.savez(sidecar, timestamp_sec_unified=ts, pose7=pose7)

    assert MODULE.resolve_tag_traj_path(unified) == sidecar
    query = np.array([0.5, 1.5])
    out = MODULE.load_tag_traj_pose7(sidecar, query)
    assert out.shape == (2, 7)
    assert np.isclose(out[0, 0], 0.5)
    assert np.isclose(out[1, 0], 1.5)

    depth_mm = np.array([[[0, 500], [2000, 7000]]], dtype=np.uint16)
    depth = MODULE.convert_depth_to_umift_depth(depth_mm, depth_scale=0.001, depth_clip=0.5)

    assert depth.shape == (1, 2, 2, 3)
    assert depth.dtype == np.float16
    # 0mm => 0m
    assert float(depth[0, 0, 0, 0]) == 0.0
    # 500mm => 0.5m (not clipped)
    assert np.isclose(float(depth[0, 0, 1, 0]), 0.5, atol=1e-3)
    # >0.5m clipped
    assert np.isclose(float(depth[0, 1, 0, 0]), 0.5, atol=1e-3)
    assert np.isclose(float(depth[0, 1, 1, 0]), 0.5, atol=1e-3)

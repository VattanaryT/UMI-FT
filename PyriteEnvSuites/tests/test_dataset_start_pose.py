"""Dataset start pose used by the runner's 'd' option."""

import numpy as np
import zarr

from PyriteEnvSuites.env_runners.dataset_start_pose import (
    central_episode_start,
    pose7_change_frame,
    resolve_dataset_zarr,
    tcp_inside_safe_zone,
)


def _write_episode(root, name, xyz):
    pose = np.zeros(7, dtype=np.float64)
    pose[:3] = xyz
    pose[3] = 1.0
    group = root.create_group(name)
    group.create_dataset("ts_pose_fb_0", data=pose.reshape(1, 7))


def test_central_episode_is_closest_to_median(tmp_path):
    path = tmp_path / "acp_replay_buffer.zarr"
    root = zarr.open(str(path), mode="w")
    data = root.create_group("data")
    _write_episode(data, "episode_0", [0.0, 0.0, 0.0])
    _write_episode(data, "episode_1", [0.10, 0.0, 0.0])
    _write_episode(data, "episode_2", [0.02, 0.0, 0.0])

    name, pose, median = central_episode_start(path)
    assert name == "episode_2"
    assert np.allclose(pose[:3], [0.02, 0.0, 0.0])
    assert np.allclose(median, [0.02, 0.0, 0.0])


def test_frame_change_applies_translation():
    pose = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])
    transform = np.eye(4)
    transform[:3, 3] = [10.0, 0.0, -1.0]
    moved = pose7_change_frame(pose, transform)
    assert np.allclose(moved[:3], [11.0, 2.0, 2.0])
    assert np.allclose(moved[3:], pose[3:])


def test_safe_zone_is_inclusive():
    zone = [-0.5, 0.7, -0.9, 0.9, 0.235, 0.8]
    assert tcp_inside_safe_zone([-0.2, 0.78, 0.46], zone)
    assert not tcp_inside_safe_zone([0.8, 0.78, 0.46], zone)


def test_missing_cfg_path_uses_task_name(tmp_path, monkeypatch):
    import PyriteEnvSuites.env_runners.dataset_start_pose as start_pose

    local = tmp_path / "r1.zarr"
    local.mkdir()
    monkeypatch.setitem(
        start_pose._LOCAL_DATASET_BY_TASK,
        "OCT-V35-FIXB-R1-WORLD-WORLD",
        str(local),
    )
    found = resolve_dataset_zarr(
        "/data/acp_replay_buffer.zarr", "OCT-V35-FIXB-R1-WORLD-WORLD"
    )
    assert found == local

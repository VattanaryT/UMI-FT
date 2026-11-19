"""Dataset start pose used by the runner's 'd' option."""

import numpy as np
import zarr

from PyriteEnvSuites.env_runners.dataset_start_pose import (
    _LOCAL_DATASET_BY_TASK,
    central_episode_start,
    dataset_zarr_for_option_d,
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


def test_named_episode_overrides_the_median(tmp_path):
    path = tmp_path / "acp_replay_buffer.zarr"
    root = zarr.open(str(path), mode="w")
    data = root.create_group("data")
    _write_episode(data, "episode_0", [0.0, 0.0, 0.0])
    _write_episode(data, "episode_1", [0.10, 0.0, 0.0])
    _write_episode(data, "episode_2", [0.02, 0.0, 0.0])

    name, pose, _median = central_episode_start(path, episode_name="episode_1")
    assert name == "episode_1"
    assert np.allclose(pose[:3], [0.10, 0.0, 0.0])


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


def test_r1rel_uses_the_october_r1_zarr():
    r1 = (
        "/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb/"
        "r1_world_world/acp_replay_buffer.zarr"
    )
    assert _LOCAL_DATASET_BY_TASK["OCT-V35-FIXB-R1-WORLD-WORLD"] == r1
    assert _LOCAL_DATASET_BY_TASK["OCT-V35-FIXB-R1REL"] == r1
    assert _LOCAL_DATASET_BY_TASK["OCT-V35-FIXB-R1REL-H4S"] == r1


def test_hyak_dataset_name_finds_the_local_zarr(tmp_path, monkeypatch):
    import PyriteEnvSuites.env_runners.dataset_start_pose as start_pose

    root = tmp_path / "mold"
    causal = root / "set" / "r4vt_frame_continuity_causal" / "acp_replay_buffer.zarr"
    smooth = root / "set" / "r4vt_smooth_frame" / "acp_replay_buffer.zarr"
    causal.mkdir(parents=True)
    smooth.mkdir(parents=True)
    monkeypatch.setattr(start_pose, "_LOCAL_MOLD_ROOT", root)
    assert (
        resolve_dataset_zarr(
            "/data/r4vt_frame_continuity_causal/acp_replay_buffer.zarr",
            "OCT-V35-FIXB-R4VT-H4S",
        )
        == causal
    )
    assert (
        resolve_dataset_zarr(
            "/data/r4vt_smooth_frame/acp_replay_buffer.zarr",
            "OCT-V35-FIXB-R4VT-H4S",
        )
        == smooth
    )
    assert resolve_dataset_zarr(str(causal), "OCT-V35-FIXB-R4VT-H4S") == causal


def test_r4vt_replay_option_d_uses_the_replay_zarr(tmp_path, monkeypatch):
    causal = tmp_path / "r4vt_causal_foot" / "acp_replay_buffer.zarr"
    causal.mkdir(parents=True)
    cfg = "/data/r4vt_smooth_tcp_track_clean/acp_replay_buffer.zarr"
    monkeypatch.setenv("UMIFT_REPLAY_EPISODE", "40")
    monkeypatch.setenv("UMIFT_R4VT_ZARR", str(causal))
    found, note = dataset_zarr_for_option_d(cfg, "OCT-V35-FIXB-R4VT-H4S", r4vt=True)
    assert found == causal
    assert note == "r4vt replay labels"
    found, note = dataset_zarr_for_option_d(cfg, "OCT-V35-FIXB-R4VT-H4S", r4vt=False)
    assert found != causal
    assert note == ""
    monkeypatch.delenv("UMIFT_REPLAY_EPISODE")
    found, note = dataset_zarr_for_option_d(cfg, "OCT-V35-FIXB-R4VT-H4S", r4vt=True)
    assert found != causal
    assert note == ""


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

"""Horizon banner and the append-only rollout journal."""

from PyriteEnvSuites.env_runners.rollout_experiment_log import (
    append_rollout_journal,
    format_horizon_banner,
    read_rollout_journal,
)


def test_horizon_banner_states_frames_and_seconds():
    text = format_horizon_banner(
        rgb_horizon=5,
        rgb_ds=8,
        wrench_horizon=32,
        wrench_ds_live=36,
        wrench_hz=500.0,
        action_horizon=24,
        action_executed=16,
        step_wall_s=2.0 / 15.0,
        slow_down=2.0,
        depth_zeroed=True,
    )
    assert "5 frames" in text
    assert "2.13 s wall" in text
    assert "1.07 s demo" in text
    assert "24 frames" in text
    assert "3.07 s wall" in text
    assert "16 frames" in text
    assert "2.00 s wall" in text
    assert "zeroed" in text
    assert "15 Hz" in text


def test_gt_replay_folder_name(monkeypatch):
    monkeypatch.setenv("PYRITE_CHECKPOINT_FOLDERS", "/tmp")
    monkeypatch.setenv("PYRITE_HARDWARE_CONFIG_FOLDERS", "/tmp")
    monkeypatch.setenv("PYRITE_CONTROL_LOG_FOLDERS", "/tmp")
    from PyriteEnvSuites.env_runners.umift_env_runner import gt_replay_rollout_id

    assert gt_replay_rollout_id(0, 1700000000) == "gt_replay_ep0_1700000000"


def test_gt_replay_journal_records_mode(tmp_path):
    path = tmp_path / "rollout_log.jsonl"
    append_rollout_journal(
        path,
        {
            "rollout_id": "gt_replay_ep0_1700000000",
            "mode": "gt_replay",
            "gt_replay_episode": 0,
            "control_log": str(tmp_path / "gt_replay_ep0_1700000000"),
        },
    )
    row = read_rollout_journal(path)[0]
    assert row["mode"] == "gt_replay"
    assert row["gt_replay_episode"] == 0
    assert row["rollout_id"] == "gt_replay_ep0_1700000000"


def test_plot_control_log_loads_demo_overlay(tmp_path):
    import importlib.util
    from pathlib import Path

    import numpy as np
    import zarr

    root = tmp_path / "gt_replay_ep0_1"
    action = zarr.open(str(root), mode="w").create_group("horizon_0").create_group("action")
    action.create_dataset("ts_nominal_targets_0", data=np.zeros((6, 7)))
    action.create_dataset("ts_virtual_targets_0", data=np.ones((6, 7)))
    action.create_dataset("ts_demo_0", data=np.full((6, 7), 0.25))
    path = Path(__file__).resolve().parents[1] / "scripts" / "plot_control_log.py"
    spec = importlib.util.spec_from_file_location("plot_control_log", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    log = zarr.open(str(root), mode="r")
    assert mod.sorted_horizon_keys(log) == ["horizon_0"]
    action_log = log["horizon_0"]["action"]
    nominal = np.asarray(action_log["ts_nominal_targets_0"][:])
    virtual = np.asarray(action_log["ts_virtual_targets_0"][:])
    demo = np.asarray(action_log["ts_demo_0"][:])
    assert nominal.shape == virtual.shape == demo.shape == (6, 7)
    assert "ts_demo_0" in action_log


def test_journal_appends_in_order(tmp_path):
    path = tmp_path / "rollout_log.jsonl"
    append_rollout_journal(path, {"rollout_id": "a", "ckpt_path": "/r1"})
    append_rollout_journal(path, {"rollout_id": "b", "ckpt_path": "/r4"})
    rows = read_rollout_journal(path)
    assert [r["rollout_id"] for r in rows] == ["a", "b"]
    assert rows[0]["ckpt_path"] == "/r1"
    assert "time_utc" in rows[1]

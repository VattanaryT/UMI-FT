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


def test_journal_appends_in_order(tmp_path):
    path = tmp_path / "rollout_log.jsonl"
    append_rollout_journal(path, {"rollout_id": "a", "ckpt_path": "/r1"})
    append_rollout_journal(path, {"rollout_id": "b", "ckpt_path": "/r4"})
    rows = read_rollout_journal(path)
    assert [r["rollout_id"] for r in rows] == ["a", "b"]
    assert rows[0]["ckpt_path"] == "/r1"
    assert "time_utc" in rows[1]

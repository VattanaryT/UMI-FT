"""One append-only journal of rollout order, horizons, and the config that ran."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# D405 collection and the training rgb stream. One frame is 1/15 s of demo time.
CAMERA_HZ = 15.0


def format_horizon_banner(
    *,
    rgb_horizon: int,
    rgb_ds: int,
    wrench_horizon: int,
    wrench_ds_live: int,
    wrench_hz: float,
    action_horizon: int,
    action_executed: int,
    step_wall_s: float,
    slow_down: float,
    depth_zeroed: bool,
    camera_hz: float = CAMERA_HZ,
) -> str:
    """Past and future windows in frames and seconds, demo time and wall time.

    ``rgb_ds`` is the live stride (already multiplied by slow_down). ``step_wall_s``
    is how long one predicted action step lasts on the robot.
    """
    rgb_lookback_wall = (max(rgb_horizon, 1) - 1) * rgb_ds / camera_hz
    rgb_lookback_demo = rgb_lookback_wall / slow_down if slow_down else rgb_lookback_wall
    wrench_lookback_wall = (
        (max(wrench_horizon, 1) - 1) * wrench_ds_live / wrench_hz if wrench_hz else 0.0
    )
    pred_wall = (max(action_horizon, 1) - 1) * step_wall_s
    exe_wall = (max(action_executed, 1) - 1) * step_wall_s
    pred_demo = pred_wall / slow_down if slow_down else pred_wall
    exe_demo = exe_wall / slow_down if slow_down else exe_wall
    depth = "zeroed (not used)" if depth_zeroed else "live depth fed to the policy"
    lines = [
        "[horizons] D405 / training rgb is "
        f"{camera_hz:.0f} Hz (1 frame = {1.0 / camera_hz:.3f} s of demo time). "
        f"Robot runs x{slow_down:g}, so one action step is {step_wall_s:.3f} s wall.",
        (
            f"  past RGB: {rgb_horizon} frames, stride {rgb_ds} live frames "
            f"({rgb_ds / camera_hz:.3f} s). "
            f"Lookback {rgb_lookback_wall:.2f} s wall = {rgb_lookback_demo:.2f} s demo."
        ),
        (
            f"  past wrench: {wrench_horizon} samples, live stride {wrench_ds_live} "
            f"at {wrench_hz:.0f} Hz. Lookback {wrench_lookback_wall:.2f} s wall."
        ),
        (
            f"  future action predicted: {action_horizon} frames, "
            f"last step at {pred_wall:.2f} s wall ({pred_demo:.2f} s demo)."
        ),
        (
            f"  future action executed: {action_executed} frames, "
            f"then replan. Last sent step at {exe_wall:.2f} s wall "
            f"({exe_demo:.2f} s demo)."
        ),
        f"  depth: {depth}",
        f"  cameras: warn if wrist or center drops below {camera_hz:.0f} Hz.",
    ]
    return "\n".join(lines)


def append_rollout_journal(path: Path, record: dict) -> None:
    """Append one JSON object per rollout. File order is experiment order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(record)
    row.setdefault(
        "time_utc", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def read_rollout_journal(path: Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows

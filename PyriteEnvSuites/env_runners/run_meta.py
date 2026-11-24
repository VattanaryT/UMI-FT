"""Write meta.json and an empty notes.md for one rollout directory.

Used by scripts/run_rollout.sh --dry-run and by the runner once the start
pose is known. Importing this module does not start the robot.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

# Sim-only defaults from closedloop_sim/lift_strategies.py. The runner does
# not apply them. meta.json records that so a hardware log cannot imply otherwise.
_OVERRIDE_THRESHOLDS = {
    "enabled": False,
    "wired_in_runner": False,
    "on_abs_fn_n": 5.0,
    "off_abs_fn_n": 3.0,
    "off_hold_s": 0.05,
    "press_dn_m": -0.001,
    "retract_m": 0.01,
    "retract_s": 0.2,
    "release_hold_s": 0.5,
    "release_travel_m": 0.02,
    "timeout_s": 2.0,
    "ramp_s": 0.2,
}


def git_state(root: str) -> dict:
    """Commit, branch, and whether tracked files differ from HEAD."""
    def out(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", root, *args], text=True
        ).strip()

    dirty = (
        subprocess.call(
            ["git", "-C", root, "diff", "--quiet"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        != 0
        or subprocess.call(
            ["git", "-C", root, "diff", "--cached", "--quiet"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        != 0
    )
    return {
        "commit": out("rev-parse", "HEAD"),
        "branch": out("branch", "--show-current"),
        "dirty": dirty,
    }


def write_run_meta(
    run_dir,
    *,
    mode: str,
    ckpt_path: str,
    normalizer_sha: str,
    start_pose,
    start_note: str,
    stiffness_floor,
    slow_down_factor: float,
    command: str,
    repo_root: str | None = None,
) -> Path:
    """Write ``<run_dir>/meta.json``. Create empty ``notes.md`` if it is absent."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    notes = run_dir / "notes.md"
    if not notes.exists():
        notes.write_text("")
    root = repo_root or os.environ.get("UMIFT_ROOT", "")
    pose = None if start_pose is None else [
        float(v) for v in np.asarray(start_pose, dtype=np.float64).reshape(7)
    ]
    meta = {
        "datetime_pt": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(
            timespec="seconds"
        ),
        "git": git_state(root) if root else None,
        "ckpt_path": ckpt_path,
        "normalizer_sha": normalizer_sha,
        "mode": mode,
        "start_pose7": pose,
        "start_note": start_note,
        "stiffness_floor_n_per_m": (
            None if stiffness_floor is None else float(stiffness_floor)
        ),
        "slow_down_factor": float(slow_down_factor),
        "override": dict(_OVERRIDE_THRESHOLDS),
        "command": command,
    }
    path = run_dir / "meta.json"
    path.write_text(json.dumps(meta, indent=2) + "\n")
    return path


def _pose_from_zarr(zarr_path: str, episode: int, index: int):
    import zarr

    root = zarr.open(zarr_path, mode="r")
    name = f"episode_{int(episode)}"
    pose = np.asarray(root["data"][name]["ts_pose_fb_0"], dtype=np.float64)
    return pose[int(index)].reshape(7).copy()


def write_from_env() -> Path:
    """Dry-run entry. Reads the env the shell just exported. Does not move a robot."""
    run_dir = os.environ.get("UMIFT_RUN_DIR", "").strip()
    if not run_dir:
        raise SystemExit("UMIFT_RUN_DIR is unset")
    replay = os.environ.get("UMIFT_REPLAY_EPISODE", "").strip()
    zarr_path = os.environ.get("UMIFT_R4VT_ZARR", "").strip()
    if replay:
        pose = _pose_from_zarr(zarr_path, int(replay), 0) if zarr_path else None
        note = f"recorded episode {replay} index 0, fixture frame"
        mode = "replay"
    else:
        episode = os.environ.get("UMIFT_POLICY_START_EPISODE", "").strip()
        index_raw = os.environ.get("UMIFT_POLICY_START_INDEX", "").strip()
        index = int(index_raw) if index_raw else 0
        if episode and zarr_path:
            pose = _pose_from_zarr(zarr_path, int(episode), index)
            note = f"episode {episode} index {index}, fixture frame"
        else:
            pose = None
            note = "current TCP"
        mode = "policy"
    floor_raw = os.environ.get("UMIFT_STIFFNESS_FLOOR", "").strip()
    floor = float(floor_raw) if floor_raw else None
    return write_run_meta(
        run_dir,
        mode=mode,
        ckpt_path=os.environ.get("UMIFT_CKPT_PATH", ""),
        normalizer_sha=os.environ.get("UMIFT_ACTION_NORMALIZER_SHA", ""),
        start_pose=pose,
        start_note=note,
        stiffness_floor=floor,
        slow_down_factor=float(os.environ.get("UMIFT_SLOW_DOWN_FACTOR", "2")),
        command=os.environ.get("UMIFT_RUN_COMMAND", ""),
    )


if __name__ == "__main__":
    print(write_from_env())

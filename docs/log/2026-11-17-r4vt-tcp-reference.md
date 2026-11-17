---
date: 2026-11-17
stream: frame
branch: frame_continuity
commit: uncommitted
status: done
artifacts: []
---
# r4vt reference pose is the TCP foot

**What changed:** `decoded_to_action_mats` sets the nominal to the foot and the virtual target to `foot + delta_n·n`. Each replan re-projects that foot from the measured TCP. `run_rollout.sh` uses this worktree, the causal zarr, and the orange extrinsics. `--bench` turns the −5 mm floor on.

**Why:** The runner was scheduling `foot + delta_n·n` as the pose, so a −150 mm press became the cartesian target while the tangent axes stayed at 5000 N/m.

**Result / status:** Offline ep0 test passes: the nominal is within 5 mm of the demo TCP's closest point, and VT − nominal lies on `n`. Not run on the robot.

**Repro:** `pytest UMIFT_Data/tests/test_r4vt_tcp_reference.py`. Free-space bench: `scripts/run_rollout.sh r4vt --replay-episode 0 --bench` with the mold removed.

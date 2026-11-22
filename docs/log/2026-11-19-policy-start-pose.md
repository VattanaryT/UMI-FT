---
date: 2026-11-19
stream: infra
branch: runner_causal_foot
commit: uncommitted
status: in-progress
artifacts: []
---
# Policy start at episode 40's first pose

**What changed:** `UMIFT_POLICY_START_EPISODE` (off when unset) makes a policy rollout approach that episode's first TCP pose, position and full orientation, through the existing 5 cm tool-axis hover. Rotation steps are at most 5 deg. Wrist 3 is moved, before the hover, onto the ±360 deg winding that still has 30 deg of room after a +174 deg flip. The runner aborts if the robot thread has stopped or the TCP is outside 2 mm / 2 deg, then holds 8 s before the first query. GT replay does not use it.

**Why:** Epoch 40 was started on the live TCP, about 14 mm off the mold with the tool roll ~180 deg from the demos, so the lookback and the first δn were out of distribution. The +π wrist-3 start also has no room for the return flip.

**Result / status:** Unit tests pass. Not yet run on the robot.

**Repro:** export `UMIFT_POLICY_START_EPISODE=40` and run `scripts/run_rollout.sh r4vt --bench` with the causal-foot epoch-40 checkpoint. Do not set `UMIFT_REPLAY_EPISODE`.

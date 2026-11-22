---
date: 2026-11-19
stream: frame
branch: smooth_frame
commit: 861ecd2
status: done
artifacts:
  - /home/boeing/data/control_log/gt_replay_ep40_1795075079
---
# Hold the label orientation before the flange check

**What changed:** Replay passes the label tool orientation into `decode_chunk`. A step more than 25° from it is replaced with the previous orientation before the D405 flange-plane check. Rollouts do not pass a label.

**Why:** On `gt_replay_ep40_1795075079` the check aborted at demo 7.00 s on a command 29.5° from the label (corner −6.3 mm). The 25° hold would have kept that corner at +14.5 mm, but it ran after the check.

**Result / status:** Unit test `test_replay_holds_orientation_before_the_flange_check`. Not run on the robot.

**Repro:** `bash scripts/run_rollout.sh r4vt --replay-episode 40 --bench`

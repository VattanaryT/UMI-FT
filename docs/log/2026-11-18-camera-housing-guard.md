---
date: 2026-11-18
stream: frame
branch: smooth_frame
commit: uncommitted
status: superseded-by 2026-11-18-camera-table-plane.md
artifacts:
  - /home/boeing/data/control_log/gt_replay_ep40_1795061663
---
# Camera guard uses the D405 housing corners

**What changed:** The table check uses the eight corners of the Intel D405 case (23 × 42 × 42 mm, from `realsense2_description` `_d405.urdf.xacro`), mounted by `T_tcp←d405`. A corner within 15 mm of the mold's lowest vertex holds and hard-stops. The 25° replay orientation hold is unchanged.

**Why:** The optical origin at demo 7.00 s was still 28 mm up. The housing corner was 7.0 mm up.

**Result / status:** On the logged command at that step the guard fires. The label's own closest corner is 13 mm, so the same margin also reaches a faithful demo. Not run on the robot. 26 tests passed.

**Repro:** `bash scripts/run_rollout.sh r4vt --replay-episode 40 --bench` from this worktree.

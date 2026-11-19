---
date: 2026-11-18
stream: frame
branch: smooth_frame
commit: uncommitted
status: done
artifacts:
  - /home/boeing/data/control_log/gt_replay_ep40_1795061663
---
# Camera table is the level flange edge

**What changed:** The camera guard measures housing corners against a gravity-level plane through the lowest scanned flange edge (base Y 881.9 mm), not the mold's lowest vertex. The margin is 0. The 25° replay orientation hold is unchanged.

**Why:** In `1795061663` the 186 N spike is at demo 7.07 s. The corner is 5 mm through that plane and 81 mm off the scan. Demo 7.00 s, the sample previously called the hit, still had the case 40 mm above the old vertex plane and no camera force. The vertex plane is 13 mm below the flange and 5° off level. Episode 40's label stays 0.1 mm above the flange edge, so a positive margin would stop it.

**Result / status:** Not run on the robot.

**Repro:** `bash scripts/run_rollout.sh r4vt --replay-episode 40 --bench`

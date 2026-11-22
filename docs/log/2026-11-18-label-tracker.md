---
date: 2026-11-18
stream: frame
branch: smooth_frame
commit: uncommitted
status: done
artifacts:
  - /home/boeing/data/control_log/gt_replay_ep40_1795057901
---
# Runner foot step is the label tracker

**What changed:** A chunk start on the same face takes the TCP's closest point. Steps after that call `step_foot` / `track_local_feet` with the label query, the 6 mm gate, and that query's tangent. No separate tangent hold.

**Why:** The carried frame turned 343° where the label frame turned 666°, so ΔR was left in the tool.

**Result / status:** `track_feet` on episode 40's stored feet matches the stored frame (max far under 1°). Offline replay of `gt_replay_ep40_1795057901`, 240 steps: frame error median 14.0°, max 54.3°. Commanded tool rotation 669° against the label's 248°. Foot error median/max: crown 45/178 mm, fillet 17/211 mm, wall 35/198 mm. The commanded foot does reach the wall (16 steps, normal 57.9° from up; the label's wall is 12 steps, 60.8°). Not run on the robot.

**Repro:** `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest UMIFT_Data/tests/test_r4vt_tcp_reference.py UMIFT_Data/tests/test_smooth_frame.py` (22 passed).

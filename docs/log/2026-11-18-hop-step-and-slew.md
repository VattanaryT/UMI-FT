---
date: 2026-11-18
stream: frame
branch: smooth_frame
commit: uncommitted
status: done
artifacts:
  - /home/boeing/data/control_log/gt_replay_ep40_1795057901
---
# Same-face hop follows the step, slew rises to 8° after 1 s

**What changed:** A hop whose normal stays within 25° of the carried frame is accepted up to that step's own length. A larger turn of the normal still holds. Commanded orientation stays at 3° per step for the first second, then 8°.

**Why:** The 10 mm hold was rejecting same-face steps the label actually takes, and the 3° slew was chopping the label's turns into a click train.

**Result / status:** Offline replay of `gt_replay_ep40_1795057901`, safety on, logged TCP each chunk, 240 steps. Commanded foot path 796 mm vs label 1198 mm. Foot error median 16.0 mm, max 72 mm. Tilt-from-vertical absolute error median 3.11°, max 27°. Hop hold 39 (38 at chunk start), re-anchor 0, foot cap 0. rd_slew 31/240 (12.9%), all after the first second. Commanded rotation sum 574° vs the label's 248°. Misses the 10 mm / 3° / 10% bar. Not run on the robot.

**Repro:** `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest UMIFT_Data/tests/test_r4vt_tcp_reference.py UMIFT_Data/tests/test_smooth_frame.py` (21 passed).

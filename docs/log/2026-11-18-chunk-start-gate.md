---
date: 2026-11-18
stream: frame
branch: smooth_frame
commit: uncommitted
status: done
artifacts:
  - /home/boeing/data/control_log/gt_replay_ep40_1795060735
---
# Chunk-start snap uses the 6 mm gate

**What changed:** A chunk start takes the TCP closest point only when `step_foot`'s 6 mm gate accepts it. After more than 5 rejected chunk starts, it re-anchors only if the new normal is within 25° of the current frame.

**Why:** The ungated snap in `gt_replay_ep40_1795060735` jumped the virtual target 60 mm onto a steeper patch, and the tool squared itself to that patch.

**Result / status:** Offline replay of that log, safety on, its measured TCPs. 11 gate rejects, 2 fallbacks. Foot error median/max: crown 48/57 mm, fillet 12/48 mm, wall 24/40 mm. Tilt from vertical, command minus label: crown +33°, fillet +12°, wall −1°. The logged TCP is already on the steep patch, so following it does not put the foot back on the label. Not run on the robot.

**Repro:** `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest UMIFT_Data/tests/test_r4vt_tcp_reference.py UMIFT_Data/tests/test_smooth_frame.py` (22 passed).

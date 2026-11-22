---
date: 2026-11-18
stream: frame
branch: smooth_frame
commit: uncommitted
status: done
artifacts:
  - /home/boeing/data/control_log/gt_replay_ep40_1795056196
---
# Start foot is the mesh point, and a hop hold re-anchors

**What changed:** Episode start no longer pulls the foot to 10 mm from the TCP. A cap that does run is projected back onto the mesh. After more than 5 rejected closest-point hops the foot re-anchors to the TCP's closest point and keeps the previous tangent.

**Why:** `gt_replay_ep40_1795056196` capped the first foot off the mesh. The 10 mm hop hold then rejected every later step, so the tool hovered and only turned.

**Result / status:** Offline replay of that log, safety on, logged TCP each chunk. Over 28.0 s (demo 0–16.7 s, 252 steps) the commanded foot travels 598 mm against a label path of 1260 mm. Foot-to-mesh is 0.00 mm. Hop hold 50 steps, foot cap 0, one re-anchor. That re-anchor jumps 101 mm back to the logged TCP, which never left the start. Not run on the robot.

**Repro:** `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest UMIFT_Data/tests/test_r4vt_tcp_reference.py UMIFT_Data/tests/test_smooth_frame.py`

---
date: 2026-11-18
stream: frame
branch: smooth_frame
commit: uncommitted
status: done
artifacts:
  - /home/boeing/data/control_log/gt_replay_ep40_1795065961
---
# Approach descends along the tool axis

**What changed:** The pre-episode hover is 5 cm back along the tool axis, and the descent is that same line. The episode log starts before the move. A D405 corner on the flange refuses the path. The old hover was 5 cm along the smooth-field normal, which points into the mold.

**Why:** `gt_replay_ep40_1795065961` bumped the mold on the way down to the start. That move was not in the log. The replay itself stays on the crown because the commanded foot follows the measured TCP and `v_par` does not slide it.

**Result / status:** Not run on the robot. On episode 40's start, backing up the tool axis stays outside the mesh (signed distance 31 mm to 74 mm). The old normal offset was 19 mm inside.

**Repro:** `bash scripts/run_rollout.sh r4vt --replay-episode 40 --bench`

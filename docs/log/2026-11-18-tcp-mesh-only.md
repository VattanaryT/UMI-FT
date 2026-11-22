---
date: 2026-11-18
stream: frame
branch: smooth_frame
commit: uncommitted
status: done
artifacts: []
---
# TCP guard is the mesh distance only

**What changed:** The TCP check no longer uses a table plane. It holds when the TCP is more than 100 mm from the mesh. The flange-edge plane stays on the D405 corners, with a 0 mm margin.

**Why:** Episode 40's label tip is 20 mm past the flange edge, so that plane aborts a faithful replay. The camera does not.

**Result / status:** Episode 40's camera stays 0.1 mm above the flange. Its tip is 136 mm from the mesh for 0.6 s at demo 24.4 s, so that one stretch trips the mesh guard. The virtual target stays within 15 mm. Not run on the robot.

**Repro:** `bash scripts/run_rollout.sh r4vt --replay-episode 40 --bench`

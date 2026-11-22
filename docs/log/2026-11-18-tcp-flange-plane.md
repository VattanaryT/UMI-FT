---
date: 2026-11-18
stream: frame
branch: smooth_frame
commit: uncommitted
status: superseded-by 2026-11-18-tcp-mesh-only.md
artifacts: []
---
# TCP guard uses the flange-edge plane

**What changed:** The TCP table check uses the same level flange edge as the camera guard. The TCP margin stays 5 mm past that plane. The camera margin stays 0.

**Why:** The two guards were measuring different surfaces. The camera plane is the lowest scanned flange edge. The TCP guard was the mold's lowest vertex, 13 mm below that and 5° off level.

**Result / status:** 28 tests passed. Not run on the robot. Episode 40's label TCP does go past the 5 mm margin, so a faithful replay can trip `table_plane`.

**Repro:** `bash scripts/run_rollout.sh r4vt --replay-episode 40 --bench`

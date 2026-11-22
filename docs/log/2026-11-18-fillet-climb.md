---
date: 2026-11-18
stream: frame
branch: smooth_frame
commit: 861ecd2
status: done
artifacts:
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_smooth_tcp_track_clean/acp_replay_buffer.zarr
---
# Climb back up the fillet in one chord

**What changed:** A normal step is still at most 6 mm. When `v_par` points back up the fillet, the foot takes the shortest of 8, 10, 12, and 15 mm that lands at least 8° less steep, and only if the normal stays within 50° of the current face.

**Why:** On episode 40 a 2 mm projection from the wall closest-points back onto the wall, so the foot stayed there after the label returned to the fillet and the commanded camera went to −9.5 mm.

**Result / status:** Offline through the runner decode on episode 40, bench floor on: commanded camera clearance minimum +6.7 mm at 22.73 s, no step through the plane (822 steps). From 8.87–9.13 s the foot is on the same face as the label. Not run on the robot.

**Repro:** `bash scripts/run_rollout.sh r4vt --replay-episode 40 --bench`

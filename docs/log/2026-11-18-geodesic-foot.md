---
date: 2026-11-18
stream: frame
branch: smooth_frame
commit: uncommitted
status: done
artifacts:
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_smooth_tcp_track_clean/acp_replay_buffer.zarr
---
# Geodesic foot walk, off by default

**What changed:** `UMIFT_FOOT_WALK=geodesic` moves the foot across mesh triangles by the `v_par` step. Unset, the runner keeps the 20 mm lead and the fillet climb.

**Why:** Chord-and-snap from the wall closest-points back onto the wall, so a short step cannot climb the fillet.

**Result / status:** Episode 40, bench floor, 822 steps. Legacy foot error median 17 mm, camera minimum +6.7 mm, none through the plane. Geodesic median 37 mm (worse on 68% of steps), camera minimum +12.6 mm, also none through the plane. It matches the label for the first 6 s and then drifts, because nothing pulls a drifted foot back to the TCP. Not used tonight. Not run on the robot.

**Repro:** leave `UMIFT_FOOT_WALK` unset. `bash scripts/run_rollout.sh r4vt --replay-episode 40 --bench`

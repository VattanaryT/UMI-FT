---
date: 2026-11-19
stream: frame
branch: smooth_frame
commit: uncommitted
status: done
artifacts:
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_smooth_tcp_track_clean/acp_replay_buffer.zarr
---
# Fillet climb off unless UMIFT_FOOT_CLIMB is set

**What changed:** The 8–15 mm fillet chord runs only when `UMIFT_FOOT_CLIMB` is `1`, `true`, `on`, or `yes`. Unset is off. The 20 mm lead is unchanged.

**Why:** On `gt_replay_ep40_1795075079` the climb put the foot 6.3 / 19 mm off the label at demo 4–6 s. The 6 mm gate run stayed at 0.4 / 4.1 mm.

**Result / status:** Offline episode 40, demo TCP, bench floor, climb off. Foot error med/max mm: 0–6 s 0.0/10.3, 6–9 s 19.0/36.2, 9–15 s 25.9/77.3, 15–30 s 18.3/76.8, 30–55 s 15.2/81.2. Camera minimum −25.8 mm, 20 steps through the plane, first at demo 6.80 s. The wall stall is back: 13 steps from 8.13–9.13 s have the foot on the wall while the label is on the fillet. Not run on the robot.

**Repro:** `unset UMIFT_FOOT_CLIMB UMIFT_FOOT_WALK && bash scripts/run_rollout.sh r4vt --replay-episode 40 --bench`

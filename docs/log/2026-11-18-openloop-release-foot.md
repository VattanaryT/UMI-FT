---
date: 2026-11-18
stream: frame
branch: smooth_frame
commit: uncommitted
status: done
artifacts:
  - /mnt/data/umidata/ft/open_loop_exports/r4vt_smooth_e0030_43335222_release
  - /mnt/data/umidata/ft/open_loop_exports/r4vt_causal_e0030_43308311_release
---

# Open-loop queries release the commanded foot

**What changed:** `decode_raw_motion` calls `release_command_foot()` before each r4vt window. `export_episode` resets the runtime at the start of each episode, then the stitched path still carries the foot. Re-exported smooth and causal epoch 30 into new directories. Same 43 train episodes plus val 9, 12, and 28. The old exports were not overwritten.

**Why:** The stitched path left `_cmd_foot` at the end of the episode, and the 10 mm gate kept every raw query there. Step 0 was ~250 mm from the TCP while δn and ΔR, which are action-space, were unaffected.

**Result / status:** Train step-0 median ||x_d − TCP|| is 85.7 mm smooth and 79.7 mm causal (was 253.6 / 228.5). Train 6-step MAE, smooth then causal: position 94.9 / 91.2 mm, δn 37.6 / 46.2 mm, ΔR 20.0 / 29.1°, world tool 18.9 / 28.9°. δn and ΔR match the carried-foot export. Full-chunk position is 125.5 / 119.6 mm, world tool 32.7 / 43.8°. Not committed.

**Repro:** `python UMIFT_Data/surface_frames_scripts/export_openloop.py --role r4vt --ckpt <epoch 30> --all-train --held-out 3 --exec-horizon 6 --out <new dir>`

---
date: 2026-11-16
stream: frame
branch: frame_continuity
commit: uncommitted
status: done
artifacts:
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_frame_continuity/train_episode_frames.npz
---
# Training-episode frame overlay and the switch list

**What changed:** `train_episode_frames.npz` holds old and new frames for the seed-42 train episodes (35 and 44 excluded) and for every other episode that contains a crown-wall switch or a near-cap step. The open-loop viewer lists those 22 switches and the 3631 near-cap steps when `--frames_npz` points at this file. `ep0_ep9_frames.npz` was not written.

**Why:** The overlay file only had episodes 0 and 9, so the 22 switches and the near-cap steps could not be opened from the viewer.

**Result / status:** 22 switches and 3631 near-cap steps, same counts as the slew check. Every episode has at least one near-cap step, so the file contains episodes 0–49. Not submitted.

**Repro:** `python UMIFT_Data/surface_frames_scripts/export_train_frames.py`. Viewer port 8766 with `--frames_npz` set to `train_episode_frames.npz`.

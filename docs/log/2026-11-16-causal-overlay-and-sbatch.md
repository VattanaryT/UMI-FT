---
date: 2026-11-16
stream: frame
branch: frame_continuity
commit: uncommitted
status: done
artifacts:
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_frame_continuity_causal/acp_replay_buffer.zarr
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_frame_continuity_causal/train_episode_frames.npz
---
# Causal overlay and the retrain sbatch

**What changed:** The overlay export defaults to the causal 0.1 s window and reads `r4vt_frame_continuity_causal`. A new sbatch points training at that zarr and keeps `oct_v35_h4s_r4vt_action_normalizer_minmax_exbad.pkl`. The centered overlay and `ep0_ep9_frames.npz` were not rewritten.

**Why:** The viewer and the retrain should use the labels the live query matches.

**Result / status:** Causal overlay has 23 crown-wall switches and 3735 near-cap steps. Kept-row delta_n (583035 rows, episodes 35 and 44 and bad spans dropped) is −0.30143726..0.16871069 m, 4 µm and 8 µm from the pickle. Full-episode max 0.334584 m is inside a dropped span on episode 38. Sbatch is not submitted.

**Repro:** `python UMIFT_Data/surface_frames_scripts/export_train_frames.py`. Sbatch: `sbatch UMIFT_Data/surface_frames_scripts/fixb_train_oct_v35_r4vt_frame_continuity_causal_klone.sbatch` on Hyak after the rsync in that file.

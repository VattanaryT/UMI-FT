---
date: 2026-11-22
stream: frame
branch: surface_frame
commit: uncommitted
status: done
artifacts:
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_heading_tilt_carry/acp_replay_buffer.zarr
  - /home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames/UMIFT_Data/surface_frames_scripts/assets/oct_v35_h4s_r4vt_action_normalizer_minmax_heading_tilt_carry.pkl
---
# Heading-tilt labels with a carried normal

**What changed:** New hardlink of `r4vt_heading_tilt` plus a rewritten `fixb_heading_tilt_0` and `fixb_heading_tilt_keep_0`. New 8-d min/max normalizer. The old heading-tilt zarr and the existing pickles were not modified. No sbatch.

**Why:** 1549 chart steps over 45° were closest-point normal hops. The tool geodesic on those steps stayed under 45°.

**Result / status:** Carry leaves 181 samples masked. Adjacent kept pairs: 0 jumps over 45°. Decode max error 4.3e-12°. Normalizer sha `496ef9f3b343048b8d3b8e29882eed9b3c32715e4c20972238decd3548afe219`, 33906 rows. Task yamls still point at `r4vt_heading_tilt`.

**Repro:** `python UMIFT_Data/surface_frames_scripts/fixb_heading_tilt_carry.py`

---
date: 2026-11-16
stream: frame
branch: frame_continuity
commit: uncommitted
status: done
artifacts:
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_frame_continuity_causal/acp_replay_buffer.zarr
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_frame_continuity_causal/frame_continuity_report.json
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_frame_continuity_causal/ep0_ep9_frames.npz
---
# Causal 0.1 s query smooth

**What changed:** `smooth_positions` is a trailing Savitzky-Golay (poly 2, samples in `[t - 0.1 s, t]`). `fill_obs_raw` applies that same filter to the wrist-tracker position before the face query. October labels were rewritten to `r4vt_frame_continuity_causal`. The previous zarr was not opened for writing. The retrain sbatch was not changed and was not submitted.

**Why:** The label filter was a centered Savitzky-Golay, so it read future samples. The live query used the raw position, so the two paths could pick different faces.

**Result / status:** Replaying episodes 0 and 9 through overlapping live windows versus one label query: face-decision mismatch 0/14016 and 0/11350, frame angle max and median 0°. Logged ΔR mask stays 0% train and val. Accepted crown-to-wall switches are 23 on this zarr (were 22 with the centered filter).

**Repro:** `python UMIFT_Data/surface_frames_scripts/fixb_frame_continuity.py --out /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_frame_continuity_causal`

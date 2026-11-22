---
date: 2026-11-17
stream: frame
branch: smooth_frame
commit: uncommitted
status: done
artifacts:
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_smooth_frame/foot_jumps_gt_20mm.txt
---
# Open-loop ΔR modes and the smooth-frame viewer

**What changed:** `episode_error_table` splits ΔR error by the label's own angle: under 45° and over 135°, with counts. `export_openloop.py` builds the r4 runtime with `MoldAwareRuntime.from_assets`, so the sidecar loads. r1rel still uses `FixBMoldField`. New `view_smooth_frame.py` adds the smooth zarr. `view_oct.py` was not edited.

**Why:** The return stroke is a real yaw about n, and the open-loop check has to score it separately. Decode has to use the same field as the labels.

**Result / status:** Not run on a checkpoint. Label feet jump more than 20 mm in one ~5 ms sample 1446 times, every episode. Median jump 54 mm. List is the artifact above. Not repaired.

**Repro:** `python UMIFT_Data/surface_frames_scripts/view_smooth_frame.py` then http://127.0.0.1:8767

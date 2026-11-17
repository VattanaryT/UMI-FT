---
date: 2026-11-16
stream: frame
branch: frame_continuity
commit: uncommitted
status: done
artifacts:
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_frame_continuity/acp_replay_buffer.zarr
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_frame_continuity/frame_continuity_report.json
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_frame_continuity/ep0_ep9_frames.npz
---
# Face hysteresis and a motion-explained ΔR mask

**What changed:** `FixBMoldField.track_foot` keeps the current face unless the new closest point is more than 5 mm closer or the tool axis has rotated toward the new normal. An accepted face change slews the output frame by at most 25° per step. The ΔR mask is the part of that frame change the tool rotation does not explain. October r4vt labels were rewritten to a new zarr. The source zarr, `fixb_field.npz`, and `*_ee_master.npz` were not written.

**Why:** About 71% of ΔR steps were masked because a closest-point hop of ~1 mm in tool motion became a 70° frame jump, and the loss then stayed off for the rest of the chunk.

**Result / status:** Logged mask (seed 42, lexical val episodes 9, 12, 28, 37, 41), before any episode/span drop: train 70.9% (869254/1225472), of which face 647360, end cap 15011, sign 1151, other 205732. Val 73.4% (91366/124544): face 63220, end cap 1224, sign 0, other 26922. After: 0% on train and val, logged and after dropping episodes 35 and 44. 1459 closer-face accepts, 22 of them with a closest-normal change over 45°. Output frame step on episode 0 maxes at 25.0°. Episode 0 old jumps 45 → 0 unexplained, foot median change 0 mm, frame angle at the old jumps 42.7°. Episode 9: 54 → 0, foot 0 mm, 39.1°. delta_n -0.3014..0.3346 m versus -0.3014..0.3344 m. Retrain sbatch is written and not submitted.

**Repro:** `python UMIFT_Data/surface_frames_scripts/fixb_frame_continuity.py`. Sbatch: `UMIFT_Data/surface_frames_scripts/fixb_train_oct_v35_r4vt_frame_continuity_klone.sbatch`.

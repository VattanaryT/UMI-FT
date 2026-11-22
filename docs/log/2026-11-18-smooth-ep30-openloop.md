---
date: 2026-11-18
stream: frame
branch: smooth_frame
commit: uncommitted
status: done
artifacts:
  - /mnt/data/umidata/ft/training_outputs/oct_v35_h4s_r4vt_smooth_frame/oct-v35-h4s-r4vt-smooth_e100_bs64_valtopk_43335222/checkpoints/epoch=0030-val_loss=0.026.ckpt
  - /mnt/data/umidata/ft/open_loop_exports/r4vt_smooth_e0030_43335222
  - /mnt/data/umidata/ft/open_loop_exports/r4vt_causal_e0030_43308311
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/field_without_smooth_sidecar/fixb_field.npz
---

# Open-loop export of smooth epoch 30

**What changed:** Copied job 43335222 `epoch=0030-val_loss=0.026.ckpt` from Hyak into a new local directory. Exported all 43 training episodes plus val episodes 9, 12, and 28, with `raw_pos` and `raw_R` for the full 64-step chunk. The same export was run for the causal epoch-30 checkpoint on `r4vt_frame_continuity_causal`, decoded with `FixBMoldField` and no smooth sidecar.

**Why:** The 6-step and full-chunk toggle needs the full decoded chunk, and the smooth epoch-30 errors need the same split as the causal epoch-30 model.

**Result / status:** Train MAE, smooth then causal. 6-step: position 275 / 253 mm, δn 38 / 46 mm, ΔR 20.0 / 29.1°. Full chunk: position 261 / 247 mm, δn 65 / 68 mm, ΔR 30.8 / 40.3°. Causal labels have no ΔR under 45°. On return yaw (>135°), 6-step ΔR is 17.0° smooth vs 28.3° causal, and position is 341 vs 257 mm. Step 0 is already ~250 mm from the TCP because `x_d` leaves the demonstrated foot. Not committed.

**Repro:** `python UMIFT_Data/surface_frames_scripts/export_openloop.py --role r4vt --ckpt <smooth epoch 30> --val-loss 0.026 --dataset <r4vt_smooth_frame zarr> --all-train --held-out 3 --out /mnt/data/umidata/ft/open_loop_exports/r4vt_smooth_e0030_43335222`

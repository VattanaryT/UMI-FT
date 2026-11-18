---
date: 2026-11-18
stream: frame
branch: smooth_frame
commit: 859afc5
status: done
artifacts:
  - /mnt/data/umidata/ft/training_outputs/oct_v35_h4s_r4vt_track_clean/oct-v35-h4s-r4vt-track-clean_e100_bs64_valtopk_43351238/checkpoints/epoch=0055-val_loss=0.030.ckpt
  - /mnt/data/umidata/ft/open_loop_exports/r4vt_track_clean_e0055_43351238
  - /mnt/data/umidata/ft/open_loop_exports/r4vt_smooth_e0030_43335222_release_feet
  - /mnt/data/umidata/ft/open_loop_exports/r4vt_causal_e0030_43308311_release_feet
---

# Track-clean epoch 55 open loop, every 4th query

**What changed:** Copied only `epoch=0055-val_loss=0.030.ckpt` from job 43351238. Exported it with `--query-stride 4` to a new directory. The viewer draws the foot and a segment of length δn, and keeps the virtual-target path behind a toggle. Smooth and causal feet were decoded from the saved actions into sibling directories. The existing r1rel export is listed as a tool-pose checkpoint.

**Why:** The full-query export was cancelled so this one would finish in about 10 minutes. The comparison tables use the same every-4th-query cut on the smooth and causal exports.

**Result / status:** Export took 311 s. Train, every 4th query, 6-step MAE: track-clean position 92.9 mm, δn 28.1 mm, ΔR 14.0°. Smooth 95.7 / 37.8 / 21.2. Causal 89.9 / 46.8 / 29.2. Track-clean v2 correlation with the label is 0.64 on the first 6 steps; v1 is 0.35. The viewer keeps the selected checkpoint across episodes, breaks the foot path across query gaps, and draws the label and prediction in separate colors. Training job 43351238 was not touched.

**Repro:** `python UMIFT_Data/surface_frames_scripts/export_openloop.py --role r4vt --ckpt <epoch 55> --val-loss 0.030 --checkpoint-epoch 55 --dataset <r4vt_smooth_tcp_track_clean zarr> --normalizer <track_clean pickle> --all-train --held-out 3 --exec-horizon 6 --query-stride 4 --out /mnt/data/umidata/ft/open_loop_exports/r4vt_track_clean_e0055_43351238`

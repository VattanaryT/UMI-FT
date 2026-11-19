---
date: 2026-11-19
stream: frame
branch: viewer_causal_foot
commit: uncommitted
status: done
artifacts:
  - /mnt/data/umidata/ft/open_loop_exports/causal_foot_43390127_e25_e40/report.json
  - /mnt/data/umidata/ft/training_outputs/oct_v35_h4s_r4vt_causal_foot/oct-v35-h4s-r4vt-causal-foot_e200_bs64_valtopk_43390127/checkpoints/epoch=0025-val_loss=0.058.ckpt
  - /mnt/data/umidata/ft/training_outputs/oct_v35_h4s_r4vt_causal_foot/oct-v35-h4s-r4vt-causal-foot_e200_bs64_valtopk_43390127/checkpoints/epoch=0040-val_loss=0.063.ckpt
  - /mnt/data/umidata/ft/training_outputs/oct_v35_h4s_r4vt_track_clean/oct-v35-h4s-r4vt-track-clean_e100_bs64_valtopk_43351238/checkpoints/latest.ckpt
---
# Causal-foot epoch 25 and 40 open loop

**What changed:** Scored job 43390127 epoch 25 and epoch 40, plus track_clean epoch 95, on the GT-label query list (episodes 9, 12, 28, 37, 41; rgb stride 8; 6 steps). The scorer now also reports v1, v2, and stiffness absolute error. New export only.

**Why:** Pick a checkpoint for hardware from open-loop contact error, not val loss.

**Result / status:** 490 queries, 2940 steps, 0 failed. Median / p90, all steps. Epoch 25: pos 12.4/36.5 mm, rot 11/34 °, δn 9.41/34.8 mm, tilt 9.18/27.7 °, v1 42.4/130 mm/s, v2 50.4/151 mm/s, stiffness 68/1530. Epoch 40: pos 10.3/28.6 mm, rot 8.5/28.3 °, δn 7/26.8 mm, tilt 7.41/24.1 °, v1 38.9/121 mm/s, v2 45.6/135 mm/s, stiffness 44.1/1220. Track_clean e95: pos 19.8/57.8 mm, rot 6.63/21.8 °, δn 18.8/57.2 mm, tilt 5.66/17.5 °, v1 32.2/105 mm/s, v2 39.1/116 mm/s, stiffness 22.3/811. Epoch 40 is better than epoch 25 on every split. Robot was not run.

**Repro:** `python UMIFT_Data/surface_frames_scripts/score_causal_foot.py --device cuda --out /mnt/data/umidata/ft/open_loop_exports/causal_foot_43390127_e25_e40 --ckpt <epoch=0025-val_loss=0.058.ckpt> --ckpt <epoch=0040-val_loss=0.063.ckpt>`

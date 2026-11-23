---
date: 2026-11-23
stream: frame
branch: tilt1
commit: uncommitted
status: done
artifacts:
  - UMIFT_Data/surface_frames_scripts/eval/score_field_v2_add.json
  - /mnt/data/umidata/ft/training_outputs/oct_v35_h4s_r4vt_field_v2_k3000/checkpoints/epoch=0025-val_loss=0.048.ckpt
  - /mnt/data/umidata/ft/training_outputs/oct_v35_h4s_r4vt_field_v2_k3000/checkpoints/latest_1005.ckpt
---
# field_v2 on the sign-flip queries

**What changed:** New scorer `eval/score_field_v2_add.py`. field_v2 e25 and latest_1005 (epoch 140) scored on the same 9/94/180 windows, with reorient and mold-local position, no test-time S flip. Horizon and one-step are both in `score_field_v2_add.json`. v2 and onestep jsons were not rewritten.

**Why:** The thin flip changes mold rot6d and the demo frame only. tilt_a is not negated. Negating it raises thin step-1 rotation (C 87.7 to 105.4, A 94.2 to 99.6).

**Result / status:** field_v2 horizon train rot e25 12.1/27.7/34.9/42.2, e140 6.8/22.4/27.4/37.0. Thin step-1 rot e25 C/A 40.1/53.3, e140 61.8/68.7.

**Repro:** `python UMIFT_Data/surface_frames_scripts/eval/score_field_v2_add.py` from UMI-FT-tilt1, pyrite env.

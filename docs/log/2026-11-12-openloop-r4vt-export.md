---
date: 2026-11-12
stream: frame
branch: surface_frame
commit: uncommitted
status: done
artifacts:
  - /mnt/data/umidata/ft/open_loop_exports/r4vt_e66520fae9ae
---
# Open-loop r4vt export, two episodes

**What changed:** New `export_openloop.py`, `view_openloop.py`, and `openloop_lib.py`. `view_oct.py` was not edited. One r4vt export from the epoch-185 checkpoint, episodes 0 (train) and 9 (val). No viewer server and no r1rel export.

**Why:** The finished 200-epoch runs kept train-loss checkpoints. This export is the recorded-observation prediction the open-loop viewer will read.

**Result / status:** 465 train queries and 364 val queries. Wall time 102.6 s. Export is 18 MB. Safety clamps off. Checkpoint sha256 `e66520fae9ae55809d5cc2f98e70e43d56cdf781181c6b9639b54dc8194080c2`, val loss 0.07917. Per-dim MAE is in `report.json`. Bad-span bins are empty: episodes 0 and 9 have no bad span, and the sampler already drops any query whose action targets hit one.

**Repro:** `python UMIFT_Data/surface_frames_scripts/export_openloop.py --role r4vt --ckpt /mnt/data/umidata/ft/training_outputs/oct_v35_h4s_odom_r4vt/checkpoints/epoch=0185-train_loss=0.011.ckpt --val-loss 0.07917`

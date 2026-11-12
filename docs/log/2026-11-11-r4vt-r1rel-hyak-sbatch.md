---
date: 2026-11-11
stream: frame
branch: surface_frame
commit: uncommitted
status: done
artifacts:
  - UMIFT_Data/surface_frames_scripts/fixb_train_oct_v35_r4vt_odom_klone.sbatch
  - UMIFT_Data/surface_frames_scripts/fixb_train_oct_v35_r1rel_odom_klone.sbatch
  - UMIFT_Data/surface_frames_scripts/config/task/umift_oct_fixb_r1rel_odom_h4s.yaml
  - UMIFT_Data/surface_frames_scripts/fixb_r1rel_dataset.py
  - UMIFT_Data/surface_frames_scripts/assets/oct_v35_h4s_r1rel_odom_action_pose_normalizer_2c20fba56851.pkl
---
# Hyak sbatch for odom-corrected r4vt and r1rel

**What changed:** Two sbatch files, not submitted. Both use the 4.0 s horizons (RGB/depth 16×4, pose and mold 51×16, wrench 118×1, action 64×1), batch 64, 32 workers, prefetch 8, val batch 64, shuffled val, 200 epochs, checkpoint every 5, `checkpoint.topk.k=2`, and `save_last_ckpt` so `latest.ckpt` is overwritten in place. r4vt task `umift_oct_fixb_r4vt_h4s` excludes episodes 35 and 44. r1rel task `umift_oct_fixb_r1rel_odom_h4s` keeps them. Both drop bad spans and `long_dt_gt_100ms.npz`. New r1rel pickle `oct_v35_h4s_r1rel_odom_action_pose_normalizer_2c20fba56851.pkl`, sha256 `2c20fba56851f59ec7d1a652d4e08a979ec9f7a8ea46368737df984475a0cf05`. The old r1rel and r4vt pickles were not opened for writing.

**Why:** Compare r4vt with r1rel on the corrected October labels. The local 4090 job was cancelled. Checkpoints and logs go to `/gscratch/scrubbed/vtevy/`.

**Result / status:** not submitted. r1rel normalizer fit kept 17,577 train queries (from 19,125 after the action-gap filter): 0 episode drops, 140 bad span, 1,344 long-dt, 64 both. Action xyz scale `[5.908, 7.279, 7.730]`, offset `[-0.00694, -0.2834, 0.03491]`.

**Repro:** not submitted. `UMIFT_HOST=/gscratch/scrubbed/vtevy/UMI-FT-surface_frames` (not `/gscratch/weirdlab/vtevy/UMI-FT-fixb`). Bad spans are `bad_span_0` inside each zarr. `sbatch` the two `fixb_train_oct_v35_*_odom_klone.sbatch` files after that checkout, both pickles, both odom zarrs, and `shared/long_dt_gt_100ms.npz` are on Hyak. Output dirs `/gscratch/scrubbed/vtevy/training_outputs/oct_v35_h4s_odom_{r4vt,r1rel}`.

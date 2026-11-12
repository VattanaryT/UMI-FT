---
date: 2026-11-12
stream: frame
branch: surface_frame
commit: uncommitted
status: done
artifacts:
  - UMIFT_Data/surface_frames_scripts/fixb_train_oct_v35_r4vt_odom_valtopk_klone.sbatch
  - UMIFT_Data/surface_frames_scripts/fixb_train_oct_v35_r1rel_odom_valtopk_klone.sbatch
  - /mnt/data/umidata/ft/training_outputs/oct_v35_h4s_odom_r4vt/checkpoints/epoch=0185-train_loss=0.011.ckpt
  - /mnt/data/umidata/ft/training_outputs/oct_v35_h4s_odom_r1rel/checkpoints/latest.ckpt
---
# Val-loss top-k sbatch copies, and the kept Hyak checkpoints

**What changed:** New sbatch copies, not edits of the e200 scripts. 40 epochs, val and checkpoint every 5, `checkpoint.topk.monitor_key=val_loss`, `mode=min`, `k=1`, filename `epoch={epoch:04d}-val_loss={val_loss:.3f}.ckpt`, plus `latest.ckpt`. Output dirs `oct_v35_h4s_odom_{r4vt,r1rel}_valtopk` on scrubbed. Pulled only the best file still on disk from the finished e200 runs: r4vt epoch 185 and r1rel `latest.ckpt` (epoch 195), plus hydra, exclusion stats, logs, and slurm. Not submitted.

**Why:** The e200 top-k tracked train loss, so the lowest-val epochs (r4vt 30, r1rel 25) were deleted. The next runs should keep the best validation checkpoint.

**Result / status:** Lab copies match Hyak sha256. r4vt epoch 185 `e66520fae9ae…` (3048139628 bytes). r1rel `latest.ckpt` `df82d01f0d23…` (3047964972 bytes). Scrubbed still has all three files per run. Not submitted.

**Repro:** `sbatch UMIFT_Data/surface_frames_scripts/fixb_train_oct_v35_r4vt_odom_valtopk_klone.sbatch` and the r1rel twin, from the scrubbed checkout. Do not point them at `/gscratch/weirdlab/vtevy/UMI-FT-fixb`.

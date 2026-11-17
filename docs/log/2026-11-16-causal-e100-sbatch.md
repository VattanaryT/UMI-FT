---
date: 2026-11-16
stream: frame
branch: frame_continuity
commit: uncommitted
status: done
artifacts: []
---
# 100-epoch causal retrain script

**What changed:** Added `fixb_train_oct_v35_r4vt_frame_continuity_causal_e100_klone.sbatch`. Same causal zarr, same normalizer, episodes 35 and 44 excluded, val top-k and a checkpoint every 5 epochs. `training.num_epochs=100` and the run name is `e100`. The 40-epoch script was not edited. Pending job 43307939 was cancelled and is gone from squeue. Not submitted.

**Why:** The causal retrain should run for 100 epochs.

**Result / status:** Not submitted.

**Repro:** `sbatch UMIFT_Data/surface_frames_scripts/fixb_train_oct_v35_r4vt_frame_continuity_causal_e100_klone.sbatch` on Hyak after rsync.

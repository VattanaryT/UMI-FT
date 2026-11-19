---
date: 2026-11-19
stream: bc
branch: causal_foot
commit: 1b0a0d9
status: in-progress
artifacts: []
---
# Causal-foot checkpoints every 5 epochs, then every 20

**What changed:** The 200-epoch sbatch saves epochs 0, 5, 10, 15, 20, 25, 30, 35, 40, then 60, 80, 100, 120, 140, 160, 180, and 199. Validation runs on those epochs. `topk.k=20` keeps the files. `latest.ckpt` is still overwritten, and the last write is epoch 199.

**Why:** Epochs 5–30 are for hardware tests. One interval cannot do both, and `k=1` would delete the early files.

**Result / status:** Not submitted.

**Repro:** `sbatch UMIFT_Data/surface_frames_scripts/fixb_train_oct_v35_r4vt_causal_foot_e200_klone.sbatch`

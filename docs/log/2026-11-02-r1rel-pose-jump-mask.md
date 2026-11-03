---
date: 2026-11-02
stream: frame
branch: surface_frame
commit: uncommitted
status: done
artifacts:
  - UMIFT_Data/surface_frames_scripts/assets/oct_v35_pose_jump_flags.npz
  - UMIFT_Data/surface_frames_scripts/assets/oct_v35_r1rel_action_pose_normalizer_fdec6dfe2027.pkl
---
# Mask tracking jumps before the r1rel retrain

**What changed:** Robot-rate TCP steps faster than 1.0 m/s or 150 deg/s (dt floored at 4 ms) are flagged for the 50 October episodes. r1rel and r4 zero the action loss on every later step in a chunk that contains such a flag. The r1rel normalizer was refit on the unflagged action steps, with range dims clipped to the 0.5th–99.5th percentile. Checkpoints now store `git rev-parse HEAD`.

**Why:** Episode 44 at 31.5 s is a tracking jump (about 2 m/s while the wrist still reads 30–40 N), and the same tail shows up in other episodes. 0.3 m/s is ordinary layup speed (about the 94th percentile) and would flag every episode.

**Result / status:** 1476 steps in 30 of 50 episodes flagged. Pose-jump action-loss mask is identical on r1rel and r4: train 11586/484968 steps (2.4%) over 20207 chunks, val 1472/49320 (3.0%) over 2055 chunks. r4's separate ΔR mask is unchanged (39.3% train). New action range (m): x [-0.103, 0.103], y [-0.092, 0.142], z [-0.124, 0.094]; virtual-target xyz [-0.104, 0.114], [-0.163, 0.168], [-0.176, 0.298]; force [-4.31, 6.00] N. Rot6d scale stays 1. Not submitted.

**Repro:** `python UMIFT_Data/surface_frames_scripts/fixb_pose_jump.py` is not a CLI; flags come from `fixb_pose_jump.save_flags()`. Normalizer: `python UMIFT_Data/surface_frames_scripts/fixb_oct_v35_r1rel_normalizer.py`. Submit later with `RUN=r1rel_world sbatch fixb_train_oct_v35_klone.sbatch` on `surface_frame`.

---
date: 2026-11-02
stream: frame
branch: surface_frame
commit: fb4b1d3
status: done
artifacts:
  - /mnt/data/umidata/ft/training_outputs/oct_v35_fixb/r1_world_world/epoch=0195-train_loss=0.013.ckpt
---
# Execute 6 action steps per replan

**What changed:** `sparse_execution_horizon` in the surface_frames runner is 6. Horizon-24 checkpoints stay at 6. A checkpoint with action horizon >= 64 is still raised to 32.

**Why:** The October r1 epoch-195 rollout tracked for about 21 s, then one replan's absolute pose jumped 122 mm and 14° after ensembling. Executing 16 of 24 steps replans every 2.0 s of wall clock at half speed, so only the previous chunk's far steps overlap the new one. Six steps replans every 0.67 s and lets more chunks vote. This dilutes a bad pose. It does not reject it.

**Result / status:** Not yet run on the robot. The loaded checkpoint is still October r1 epoch 195.

**Repro:** `source /home/boeing/umi_ft_ws/worktrees/activate.sh UMI-FT-surface_frames` then `python PyriteEnvSuites/env_runners/umift_env_runner.py`.

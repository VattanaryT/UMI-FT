---
date: 2026-10-29
stream: frame
branch: surface_frame
commit: uncommitted
status: done
artifacts:
  - /mnt/data/umidata/ft/data/control_log/r2_world_fixb_e35_rollout0_1792708310
  - /home/boeing/data/control_log/r2_world_fixb_e35_rollout0_1793245851
---
# Orientation clamps off for r2 and r4 contact

**What changed:** `SafetyLimits.enable_tilt_clamp`, `enable_rot_rate_clamp`, and `enable_mesh_jump_hold` default False. Speed, step, and δn clamps and the mesh-distance hard stop are unchanged. `mold_aware_dry_run` stays False.

**Why:** Oct 28 r2 contact rewrote orientation (tilt clamp ~42% of steps, commanded ΔR stuck at the 20°/step cap). The Oct 22 contact run (`…1792708310`, dry_run false) decoded before those clamps existed (682e24f, 2026-10-26).

**Result / status:** Not yet run on the robot.

**Repro:** `source /home/boeing/umi_ft_ws/worktrees/activate.sh UMI-FT-surface_frames` then `python PyriteEnvSuites/env_runners/umift_env_runner.py` with ckpt `/scanner_fixb/r2_world_fixb/epoch=0035-train_loss=0.016.ckpt`.

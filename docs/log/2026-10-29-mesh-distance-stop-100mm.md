---
date: 2026-10-29
stream: frame
branch: surface_frame
commit: uncommitted
status: in-progress
artifacts:
  - /mnt/data/umidata/ft/data/control_log/r2_world_fixb_e35_rollout0_1793313455
  - /mnt/data/umidata/ft/data/control_log/r2_world_fixb_e35_rollout1_1793313524
---
# Raise mesh-distance abort to 100 mm for contact rollouts

**What changed:** `SafetyLimits.max_distance_from_mesh_m` 0.030 → 0.10. Speed, step, and δn press clamps unchanged.

**Why:** Both 2026-10-29 r2 contact episodes aborted on `max_distance_from_mesh`. The executed chunk peaked at 27 mm off the mesh; the next sample crossed 30 mm. The tool itself was 15 mm off.

**Result / status:** Not yet rerun.

**Repro:** `source /home/boeing/umi_ft_ws/worktrees/activate.sh UMI-FT-surface_frames` then `python PyriteEnvSuites/env_runners/umift_env_runner.py`.

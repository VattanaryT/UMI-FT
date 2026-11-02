---
date: 2026-11-02
stream: frame
branch: surface_frame
commit: uncommitted
status: in-progress
artifacts:
  - /mnt/data/umidata/ft/training_outputs/oct_v35_fixb/r1_world_world/epoch=0195-train_loss=0.013.ckpt
  - /mnt/data/umidata/ft/data/control_log/r4_world_fixb_tilt_e195_rollout1_1793654379
---
# Next rollout is October r1; viewer fits one window

**What changed:** `pipeline_para["ckpt_path"]` is the October v3.5 r1 epoch-195 checkpoint. The inference HTML viewer is a 2×4 grid (RGB, depth, center camera, 3D on top; pose, force, stiffness, torque below) and the config block starts collapsed. A second slider and the left/right arrow keys step one policy query at a time. A missing center camera is labeled instead of drawn black.

**Why:** The r4 epoch-195 contact rollout (`r4_world_fixb_tilt_e195_rollout1_1793654379`) hit UR protective stop C153A3 on wrist 1. Temporal ensembling was on (`enable_ensemble_chunking` true, decay 0.01). The next comparison is r1. The old viewer was taller than the window.

**Result / status:** r1 epoch-195 was run (`r1_world_world_e195_rollout0_1793655926`). It tracked for about 21 s, then one replan jumped. The follow-up is `2026-11-02-execution-horizon-6.md`. r4 episode 1 logged 105 N at every query, mesh signed distance never under 3 mm, software safety stops 0.

**Repro:** `source /home/boeing/umi_ft_ws/worktrees/activate.sh UMI-FT-surface_frames` then `python PyriteEnvSuites/env_runners/umift_env_runner.py`.

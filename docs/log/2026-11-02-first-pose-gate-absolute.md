---
date: 2026-11-02
stream: frame
branch: surface_frame
commit: bc57c8e
status: done
artifacts: []
---
# First-pose gate also covers absolute r1

**What changed:** The 25 mm / 10° first-pose gate in the surface-frames runner now applies to absolute pose-action checkpoints as well as relative ones. Fix B r2/r3/r4 decode is unchanged.

**Why:** The October r1 epoch-195 rollout spiked on an absolute fixture pose (122 mm, 14°). The gate previously ran only when `pose_frames.action` was relative.

**Result / status:** Not yet run on the robot. The r1rel normalizer was not refit.

**Repro:** `source /home/boeing/umi_ft_ws/worktrees/activate.sh UMI-FT-surface_frames` then `python PyriteEnvSuites/env_runners/umift_env_runner.py`.

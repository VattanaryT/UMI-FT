---
date: 2026-11-02
stream: frame
branch: surface_frame
commit: uncommitted
status: done
artifacts: []
---
# Menu option d moves to a dataset start pose

**What changed:** Before an episode, and on the post-episode menu, `d` moves the right arm to the demo start closest to the median of the checkpoint's dataset. Existing menu letters are unchanged. `r` still returns to the pose captured at the camera preview.

**Why:** The r4 epoch-195 rollout started about 24 cm outside the October v3.5 start cluster. Jogging that pose by hand is not repeatable.

**Result / status:** Used on the r1 epoch-195 rollout. Feedback started at the episode-30 pose, inside the October start cluster. The goal is checked against the cartesian safe zone before the existing lift-travel-descend path is sent.

**Repro:** `source /home/boeing/umi_ft_ws/worktrees/activate.sh UMI-FT-surface_frames` then `python PyriteEnvSuites/env_runners/umift_env_runner.py`. Type `d` at the start prompt.

---
date: 2026-11-04
stream: frame
branch: surface_frame
commit: uncommitted
status: done
artifacts:
  - /home/boeing/data/control_log/r1rel_world_e25_rollout0_1793826797
  - /home/boeing/umi_ft_ws/UMI-FT/runs/episode_1793826797
---
# r1rel dataset start and red/yellow runner failures

**What changed:** Menu `d` for `OCT-V35-FIXB-R1REL` and `OCT-V35-FIXB-R1REL-H4S` uses the October `r1_world_world` zarr (episode closest to the median start). Camera open/read failures, a camera that records 0 frames, dataset-start failures, scan failures, and mold safety aborts print in red. Frame-gap warnings, a low camera rate, first-pose-gate holds, and mold clamps print in yellow.

**Why:** The r1rel h4s epoch-25 rollout printed "no local dataset zarr" on `d`, so the arm started 48 mm and 24° off episode 30. The center Orbbec wrote 0 frames and the failure was easy to miss.

**Result / status:** Not re-run on the robot. The logged rollout is `r1rel_world_e25_rollout0_1793826797` (33 s, wrist 15.0 Hz, Orbbec 0 frames).

**Repro:** `source /home/boeing/umi_ft_ws/worktrees/activate.sh UMI-FT-surface_frames` then `python PyriteEnvSuites/env_runners/umift_env_runner.py`. Press `d` before the episode.

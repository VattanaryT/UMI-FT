---
date: 2026-11-04
stream: frame
branch: surface_frame
commit: uncommitted
status: done
artifacts:
  - /home/boeing/data/control_log/r1rel_world_e25_rollout0_1793830474
---
# Alternate dataset start and Orbbec color node

**What changed:** Menu `d` now goes to October episode 39 instead of the median episode 30. The center recorder skips the Bolt's Y16 depth node (`video-index0`, 1024x512) and opens `video-index1` as MJPEG color.

**Why:** Episode 30 put the TCP about 48 mm off the mesh with the tool 22 deg from straight down. The Firefox center panel was that depth node decoded as a picture: 368 frames, 5 unique images, stuck near 5 Hz.

**Result / status:** Not re-run on the robot. Episode 39 right-base goal is about `[-0.197, 0.803, 0.452]` m, 30 mm lower than episode 30 and 7 deg from straight down.

**Repro:** Restart the runner after `source /home/boeing/umi_ft_ws/worktrees/activate.sh UMI-FT-surface_frames`, then press `d`.

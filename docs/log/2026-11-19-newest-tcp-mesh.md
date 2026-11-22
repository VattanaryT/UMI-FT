---
date: 2026-11-19
stream: frame
branch: smooth_frame
commit: 861ecd2
status: done
artifacts:
  - /home/boeing/data/control_log/checkpoints_e-1_rollout0_1795077439
  - /home/boeing/data/control_log/checkpoints_e-1_rollout1_1795077459
  - /home/boeing/data/control_log/gt_replay_ep40_1795077346
---
# Measured-mesh abort uses the newest TCP

**What changed:** `note_measured_mesh_distance` checks only the newest measured TCP against the 100 mm limit. `scripts/run_rollout.sh` keeps an already-exported `UMIFT_CKPT_PATH`.

**Why:** GT replay `1795077346` and policy episode 1 aborted before any motion. The current TCP was 31 mm and 53 mm from the mesh. The 8 s observation window still held the approach (114 mm) and the menu-d lift (~147 mm). Policy episode 0 was a real miss: the TCP itself was 139 mm off, at right_base y 0.669 instead of the episode-40 start.

**Result / status:** Unit test `test_old_far_pose_in_the_window_does_not_abort`. Not run on the robot.

**Repro:** `UMIFT_ABORT_ON_MEASURED_MESH_DISTANCE=1 bash scripts/run_rollout.sh r4vt --replay-episode 40 --bench`

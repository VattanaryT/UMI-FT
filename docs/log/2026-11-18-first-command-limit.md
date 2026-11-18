---
date: 2026-11-18
stream: frame
branch: smooth_frame
commit: uncommitted
status: done
artifacts:
  - /home/boeing/data/control_log/checkpoints_e30_rollout0_1794958462
---
# Limit the first r4vt command from the measured TCP

**What changed:** The first replan slews `R_d` from the measured tool and keeps the first foot within 10 mm of the TCP. The virtual target ramps in over 1 s, at most 25 mm times elapsed/1 s per step. After that, a step longer than 25 mm repeats the last safe virtual target. On the smooth field the ramp follows the outward normal and the face hold stays off. Option d resolves a Hyak `/data/<name>/acp_replay_buffer.zarr` path to the local mold zarr of that name. The runner prints the ramp and the 25 mm hold, and the rollout script points at track-clean epoch 55.

**Why:** Epoch-30 rollout `checkpoints_e30_rollout0_1794958462` commanded 117° and a virtual target 69 mm from a normal start. The slew, the foot cap, and the 25 mm check had no previous command, so step 0 was not limited.

**Result / status:** Replayed that logged command offline. Orientation 117.0° → 3.00°. Virtual target 68.8 mm → 1.67 mm. Not run on the robot.

**Repro:** `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest UMIFT_Data/tests/test_r4vt_tcp_reference.py PyriteEnvSuites/tests/test_dataset_start_pose.py`.

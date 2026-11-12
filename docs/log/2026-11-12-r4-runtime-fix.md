---
date: 2026-11-12
stream: frame
branch: r4_runtime_fix
commit: uncommitted
status: done
artifacts:
  - /mnt/data/umidata/ft/replay_exports/gt_r4_runtime_fix/
---
# r4vt decode uses the demo step, action blend, and measured-TCP abort

**What changed:** Fix B integration dt is `down_sample_steps * (1/15)` only when the checkpoint action is r4vt. r2, r3, and r1rel keep the previous integral. Waypoint timestamps are unchanged. r4vt blends v, δn, and ΔR, then integrates once from the measured foot. There is no ensemble flag. The −5 mm δn floor is off for r4vt only. Measured TCP penetration below −20 mm aborts. Wrench at or above 90 N holds that replan and aborts if the next replan is still there. Those two thresholds sit in `control_para` and are applied and logged only for r4vt. Every control log records `ensemble_mode` (`pose` or `action_blend`) and `decode_dt_s`. `raise_exec_horizon_64_to_32` defaults off, so a 64-step checkpoint executes 6. r1rel replay calls `blend_pose_chunk`, the same push/pop the live loop calls.

**Why:** Slow-down was stretching the foot step, and pose blending was moving a correct integral. δn is the virtual-target depth, about −150 mm in contact, while the measured TCP stays near −1 mm.

**Result / status:** Not run on the robot. Offline episodes 0 and 9 at dt 1/15: ensemble off and action blend match. Foot median 0.20 / 0.80 mm (episode 0, H 6 / 32) and 0.32 / 2.92 mm (episode 9). Depth 0. Tests: 13 passed.

**Repro:** `pytest UMIFT_Data/tests/test_r4_runtime_fix.py UMIFT_Data/tests/test_replay_gt.py` then `python -c "import replay_gt as rg; rg.write_runtime_replay((0, 9))"`.

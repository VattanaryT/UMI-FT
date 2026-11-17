---
date: 2026-11-16
stream: frame
branch: surface_frame
commit: uncommitted
status: done
artifacts: []
---
# Opt-in measured TCP mesh-distance abort

**What changed:** `SafetyLimits.abort_on_measured_mesh_distance` defaults off. On, the runner skips the commanded `||x_d - x_s||` stop and aborts when the measured TCP is farther than 0.10 m from the surface. The 20 mm measured-penetration abort stays on.

**Why:** GT replay ep0 `gt_replay_ep0_1794877086` aborted at h=19 because the label's commanded `|delta_n|` was 140 mm at stiffness 200. The measured TCP was still on the demo.

**Result / status:** unit test only. Not run on the robot.

**Repro:** `UMIFT_ABORT_ON_MEASURED_MESH_DISTANCE=1` before `scripts/run_rollout.sh r4vt --replay-episode 0`.

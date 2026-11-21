---
date: 2026-11-21
stream: frame
branch: runner_causal_foot
commit: uncommitted
status: done
artifacts:
  - closedloop_sim/out/policy_fk60_fixb_ep40.npz
  - closedloop_sim/out/policy_e40_fixb_ep40.npz
  - closedloop_sim/out/policy_r1rel_fixb_ep40.npz
  - closedloop_sim/out/policy_fixb_ep40.json
  - closedloop_sim/out/demo_ep40.npz
  - closedloop_sim/out/fk60_obs_compare.json
---
# Closed-loop fk60, e40, and r1rel on the clipped spring

**What changed:** Policy rollouts import this worktree and drop the main checkout from `sys.path`. The sim skips the D405 `camera_table` hold; the runner still applies it. Traces now store the virtual target and the contact force. Playback is a mode on the existing port-8082 viewer.

**Why:** fk60 and e40 were aborted by the camera-table check, and fk60's TCP stalled about 86 mm off the demo. The check is a housing-to-table test. This plant has no D405.

**Result / status:** Episode 40, full 58.8 s for fk60 and e40. r1rel stopped on penetration at 54.1 s (150 mm). After the start pad, contact loss is 0% / 0% / 26%. Median penetration 3.0 / 13.3 / 13.3 mm. Median force 9.2 / 40 / 40 N. Median force/|k δn| 0.49 / 0.17 / 0.11. Median speed 0.012 / 0.039 / 0.091 m/s. Median distance to the demo 117 / 175 / 340 mm. fk60 has no elapsed-time or depth input. At t=0 its absolute pose and mold match the demo and the black RGB changes the action from 40 mm/s to 27 mm/s. At t=8 the absolute pose is already 88 mm off the demo.

**Repro:** `python closedloop_sim/policy_rollout.py fk60 e40 r1rel` in pyrite, from this worktree. Viewer: `view_thin_pred.py --port 8082` in the geodude venv, then View → Closed-loop playback.

---
date: 2026-11-21
stream: frame
branch: runner_causal_foot
commit: uncommitted
status: in-progress
artifacts:
  - closedloop_sim/out/gt_fixb_ep40.json
  - closedloop_sim/out/gt_fixb_ep40.npz
---
# compliance_command refactor and fixb virtual-target replay

**What changed:** The runner's inline stiffness assembly is `compliance_command` in `mold_aware_runtime.py`. New Cartesian spring plant under `closedloop_sim/`. No robot, no MuJoCo, no policy rollouts.

**Why:** Closed-loop contact sim has to schedule the same virtual-target pose and 6x6 the runner schedules, then apply an SDF spring instead of a convex-hull mesh contact.

**Result / status:** On the e40 control log (150 steps, floor 3000) the old inline code and `compliance_command` differ by 0. Stored stiffness matches; stored pose differs by 2.2e-16. Episode 40 label replay on the fixb mesh runs 59 s without a diverging stop. After the start pad, contact-force / |k δn| median is 0.96 and contact loss is 2.3%. Isolated command hops still hit 15 m/s for a few samples because the 5000 N/m axes are unclipped on the 0.2 kg admittance. Policies not run.

**Repro:** `python UMIFT_Data/tests/test_compliance_command.py` and `python closedloop_sim/gt_replay.py --episode 40` in the pyrite env with this worktree activated.

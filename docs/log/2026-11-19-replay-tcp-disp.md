---
date: 2026-11-19
stream: frame
branch: runner_causal_foot
commit: uncommitted
status: done
artifacts: []
---
# Replay internal TCP follows the demo step

**What changed:** An r4vt GT replay sets `mold_runtime._tcp_step` to the demo TCP displacement (`demo_tcp_disp`, the sim gate's `tcp_disp`) for the executed window. Policy decode still leaves `_tcp_step` unset, so the internal TCP follows the command.

**Why:** On `gt_replay_ep40_1795119835` the bench floor held δn at −5 mm while the label was −41 mm and v was 0. Contact walked the arm off the fillet. The foot followed that TCP, then later label velocity was integrated from the wrong foot. The tool stayed on the mesh and ended ~180 mm from the demo, on the crown.

**Result / status:** Unit tests cover the displacement array and the carry. Robot not run.

**Repro:** `unset UMIFT_R4VT_ZARR UMIFT_R4VT_NORMALIZER UMIFT_R4VT_NORMALIZER_SHA UMIFT_RESET_TCP_TO_MEASURED; bash scripts/run_rollout.sh r4vt --replay-episode 40 --bench`

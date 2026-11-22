---
date: 2026-11-19
stream: infra
branch: runner_causal_foot
commit: uncommitted
status: in-progress
artifacts: []
---
# Press-axis stiffness floor

**What changed:** `UMIFT_STIFFNESS_FLOOR` (off when unset) raises the scheduled press-axis stiffness to that many N/m. The other two axes stay 5000. Each horizon logs `stiffness_raw_<id>` and `stiffness_floored_<id>`. `UMIFT_BENCH_DN_FLOOR_MM` sets the bench δn floor (default −5). Labels, the checkpoint, and the 40 N spring clip are unchanged.

**Why:** Causal-foot labels sit at the virtual-target floor of 200 N/m once the smoothed force exceeds 8 N. Epoch 40 predicts about 220 N/m from the second chunk, and the replay then loses contact.

**Result / status:** Unit tests pass. Not yet run on the robot.

**Repro:** `UMIFT_STIFFNESS_FLOOR=3000 UMIFT_BENCH_DN_FLOOR_MM=-9 bash scripts/run_rollout.sh r4vt --replay-episode 40 --bench`, then the same flags with `UMIFT_CKPT_PATH` set to the causal-foot epoch-40 checkpoint and no replay episode.

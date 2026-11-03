---
date: 2026-11-03
stream: frame
branch: surface_frame
commit: 120be1e
status: in-progress
artifacts: []
---
# Drop wrist depth from October Fix B training

**What changed:** `task.drop_depth_inputs` removes `depth_0` and `depth_time_stamps_0` from `shape_meta` before the policy is built. The 4 s tasks set the flag, and both October sbatch scripts pass it. RGB, wrench, pose, and `robot0_mold_geom` stay. Chunk indices do not depend on depth.

**Why:** Do not train a depth branch from now on.

**Result / status:** not yet run. The 4 s jobs still need to be submitted from a Hyak login that can answer Duo, on the commit that contains this change.

**Repro:** `pytest PyriteML/tests/test_drop_depth_inputs.py`

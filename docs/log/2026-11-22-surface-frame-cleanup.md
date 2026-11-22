---
date: 2026-11-22
stream: frame
branch: surface_frame
commit: uncommitted
status: done
artifacts: []
---
# Surface-frame path cleanup

**What changed:** Task yamls and the 10-d normalizer test now open pickles in this worktree's `assets/`. The two extra `rot6d_rows_to_R` copies call `mold_aware_runtime`. The hardware-import check runs in a subprocess. The vision log test uses `tmp_path`. `check_nocam_forward.py` is deleted.

**Why:** The causal_foot worktree is gone, so those paths failed. The rot6d copies were the same Gram-Schmidt.

**Result / status:** Full suite 316 passed. Pickles were not modified.

**Repro:** `pytest PyriteML/tests PyriteUtility/tests UMIFT_Data/tests PyriteEnvSuites/tests`

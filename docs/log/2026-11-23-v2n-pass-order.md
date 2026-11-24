---
date: 2026-11-23
stream: frame
branch: v2n_lift
commit: uncommitted
status: done
artifacts:
  - UMIFT_Data/surface_frames_scripts/eval/v2n_pass/numbers.json
  - UMIFT_Data/surface_frames_scripts/eval/v2n_pass/pass_error.png
---
# Two passes per half, and the position is already in the obs

**What changed:** Read-only. Pass counts, obs contents, and checkpoint A open-loop versus arc length. Draft sbatch only.

**Why:** See whether post-contact δn and the half switch are the same failure.

**Result / status:** 37/43 train demos visit each half twice, usually wall base, wall far, wall base, wall far. `robot0_eef_pos` x correlates 0.9996 with arc length. No non-RGB state is shared by a second-pass future and an other-half future. A is a few mm worse within 0.5 s of the midpoint, not a spike. Not submitted.

**Repro:** `python UMIFT_Data/surface_frames_scripts/pass_order.py`

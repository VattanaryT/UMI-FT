---
date: 2026-11-11
stream: frame
branch: surface_frame
commit: uncommitted
status: done
artifacts:
  - UMIFT_Data/surface_frames_scripts/assets/oct_v35_h4s_r4vt_action_normalizer_minmax_exbad.pkl
---
# r4vt min/max normalizer without the flagged bad spans

**What changed:** A third r4vt action normalizer. Min/max of delta_n on October rows, excluding episodes 35 and 44 and the flagged bad spans (episode 38 from recording 50 s through the end, episode 47 from recording 31–47 s). The earlier min/max and percentile pickles, the old r4 pickle, and the long-wait mask were not modified. Presses below −0.25 m stay in the fit.

**Why:** The +334 mm end of the previous min/max was the episode 38 bad span, not a press. The −301 mm end is a real hard press and should set the low side.

**Result / status:** 583,035 rows kept, 3,773 bad-span rows dropped. delta_n min −0.301441 m, max 0.168703 m, scale 4.254014, offset 0.282335. v1 and v2 stay scale 1, offset 0. Not trained.

**Repro:** fit script was a one-off; the pickle is the artifact.

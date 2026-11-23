---
date: 2026-11-23
stream: frame
branch: tilt1
commit: uncommitted
status: done
artifacts:
  - UMIFT_Data/surface_frames_scripts/eval/score_sign_flip_v2.json
  - UMIFT_Data/surface_frames_scripts/eval/score_sign_flip_onestep.json
---
# Two sign-flip tables

**What changed:** Horizon 1/6/16/64 table restored to `score_sign_flip_v2.json` from the pre-one-step run. One-step tangent table copied to `score_sign_flip_onestep.json`. The scorer now writes the one-step file.

**Why:** The one-step rescore had replaced the horizon file.

**Result / status:** Tilt1 training rotation after the axis fix is 12.0/24.5/30.9/42.5 (e25) and 9.5/23.9/29.8/37.4 (e180), not ~86°. Four `test_score_sign_flip` tests passed.

**Repro:** files above. Tests: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest UMIFT_Data/tests/test_score_sign_flip.py`.

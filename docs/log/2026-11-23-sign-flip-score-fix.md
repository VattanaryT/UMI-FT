---
date: 2026-11-23
stream: frame
branch: tilt1
commit: uncommitted
status: superseded-by 2026-11-23-sign-flip-score-tests.md
artifacts:
  - UMIFT_Data/surface_frames_scripts/eval/score_sign_flip.json
---
# Sign-flip score, decoder fix

**What changed:** Scorer only. Surface position now uses `rollout_tool` (step 0 includes δn). Tilt1 training uses that mold's long axis, and a window keeps the side carried in from the episode. r1rel is unchanged.

**Why:** `heldout._walk` writes the anchor before the action and drops δn, so every r4vt checkpoint repeated the GT step-1 position. Training tilt was decoded with the thin-wall axis, 86° from the training axis.

**Result / status:** k3000 and tilt1 position changed at every step. Tilt1 rotation changed (training step 1 85° to 12.0/9.5). k3000 rotation and all r1rel numbers did not. Label 1-tilt round-trip on the 180 training windows is median 4.1°.

**Repro:** `python UMIFT_Data/surface_frames_scripts/eval/score_sign_flip.py` from UMI-FT-tilt1, pyrite env.

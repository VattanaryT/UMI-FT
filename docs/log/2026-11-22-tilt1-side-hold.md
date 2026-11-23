---
date: 2026-11-22
stream: frame
branch: tilt1
commit: uncommitted
status: done
artifacts:
  - /home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames/UMIFT_Data/surface_frames_scripts/assets/oct_v35_h4s_r4vt_action_normalizer_minmax_tilt1.pkl
---
# Hold the 1-tilt side through the crown

**What changed:** The 0°/180° side holds until n·Z crosses ±0.4. An episode starts on its first clear sample. Decode uses the same function, with `previous` for a carried side. Normalizer refit, sha256 `f491603f645bbc1f819efd814f72c31f8e09fe4f6f6f8ee80776d0c3181b0769`. Not submitted.

**Why:** On the crown n·Z chatters through zero, so the side flipped and the pose miss p90 was 149°.

**Result / status:** Crown miss median 10.60°, p90 74.72°. Walls stay at 3.61° and 3.35° median. `test_tilt1.py` 6 passed. Not trained.

**Repro:** `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest UMIFT_Data/tests/test_tilt1.py -q`

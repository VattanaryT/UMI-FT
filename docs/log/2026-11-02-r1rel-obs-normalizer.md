---
date: 2026-11-02
stream: frame
branch: surface_frame
commit: uncommitted
status: done
artifacts:
  - UMIFT_Data/surface_frames_scripts/assets/oct_v35_r1rel_action_pose_normalizer_64e3aa3c37f5.pkl
---
# Unclip the r1rel relative-pose observation range

**What changed:** `robot0_eef_pos` in the r1rel normalizer uses the unclipped train min/max. Action range dims, including dim 20, stay on the 0.5–99.5 percentile scale. The percentile clip never edits the training tensors; it only set the linear scale.

**Why:** The 5×16 pose window's oldest sample has |p| p99 = 11.1 cm. The ±5 cm scale put 1.00% of eef_pos values, and 3.06% of the oldest samples, outside [-1, 1]. Dim 20 is `-wrench_left_x / 2` (right wrench is zero), not the 30–40 N contact norm, so it stays percentile-scaled.

**Result / status:** eef_pos range (m) is [-0.424, -0.325, -0.185] to [0.356, 0.180, 0.142]. Action dim 20 stays [-4.31, 6.00]. Not submitted.

**Repro:** `python UMIFT_Data/surface_frames_scripts/fixb_oct_v35_r1rel_normalizer.py`

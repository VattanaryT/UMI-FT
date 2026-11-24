---
date: 2026-11-23
stream: frame
branch: v2n_v3r_lift
commit: uncommitted
status: abandoned
artifacts:
  - /home/boeing/umi_ft_ws/worktrees/UMI-FT-v2n_v3r_lift/UMIFT_Data/surface_frames_scripts/eval/fuse_clearance/table.json
  - /home/boeing/umi_ft_ws/worktrees/UMI-FT-v2n_v3r_lift/UMIFT_Data/surface_frames_scripts/eval/fuse_clearance/top2_lift.png
---
# Force-anchored clearance offset does not make a quiet latch

**What changed:** Read-only. Firm contact is raw |F·n| > 8 N for at least 0.1 s (1173 strokes). A subtracts the per-stroke median and holds it 1 s after lift-off. B fits a quadratic in arc length plus linear width. C is a 1 mm/s random-walk Kalman because A and B left residual std 9 mm and 13 mm. No zarr. `r4vt_field_v2n_v3r_lift` and sha `743c83c8` were not touched.

**Why:** Clearance during the 1 s force gate has a 16 mm spread, so a clearance latch needs the registration error removed first.

**Result / status:** Firm-contact clearance itself is median −0.3 mm, p10/p90 −14/+14, std 15 mm. A leaves std 9 mm. B leaves std 13 mm, so the error is not a smooth function of position. C leaves std 6 mm. The 50 ms jitter stays about 3.5–3.8 mm. The best latch (A, 2 mm) still calls 49% of 1 s contact steps air, with 37 mid-stroke flips. Not built.

**Repro:** `python UMIFT_Data/surface_frames_scripts/fuse_clearance.py`

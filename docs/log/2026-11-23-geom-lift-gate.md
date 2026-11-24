---
date: 2026-11-23
stream: frame
branch: v2n_v3r_lift
commit: uncommitted
status: abandoned
artifacts:
  - /home/boeing/umi_ft_ws/worktrees/UMI-FT-v2n_v3r_lift/UMIFT_Data/surface_frames_scripts/eval/geom_lift/table.json
  - /home/boeing/umi_ft_ws/worktrees/UMI-FT-v2n_v3r_lift/UMIFT_Data/surface_frames_scripts/eval/geom_lift/top2_lift.png
---
# Clearance latch does not replace the force gate

**What changed:** Read-only score of a clearance latch (on at 0.5, 1, 2 mm while rising over 0.05 s, off below half that) over the 1 s force blend. No new zarr or normalizer. `r4vt_field_v2n_v3r_lift` and sha `743c83c8` were left as they are.

**Why:** The force gate's post-rise δn stays negative because clearance rises while |F·n| is still above 3 N.

**Result / status:** Post-rise median is +4.3 mm, but every threshold calls about half of 1 s contact steps air and has 55–64 mid-stroke flips. Not built. During 1 s contact the clearance median is +1.6 mm and 48% of samples are already above +2 mm; a 50 ms change while both samples are within 3 mm of the surface has std 1.3 mm.

**Repro:** `python UMIFT_Data/surface_frames_scripts/geom_lift_gate.py`

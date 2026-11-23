---
date: 2026-11-23
stream: frame
branch: tilt1
commit: uncommitted
status: done
artifacts:
  - /home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames/UMIFT_Data/surface_frames_scripts/eval/teleop_field_sweep.json
---
# Vector-heat t_spatial sweep

**What changed:** New script `eval/teleop_field_sweep.py`. Main's vector-heat field (`5b09d7f`), t1 flipped onto the mold long axis, no 8 mm snap, no temporal blend. Training body decimated to 15,231 vertices and the field built. Thin body is many sheets; the Laplacian built on a 15,036-vertex wall and the 513-vertex end cap, and rejected the other wall (17,139) plus two cross-section sheets.

**Why:** The full 73k-vertex thin mesh ran out of memory. The sweep asks whether a wider spatial kernel makes the thin field as steady as the training field.

**Result / status:** No factor has thin jump p99 ≤ training jump p99. At 4×mean_edge² the thin p99 is 132° and the training p99 is 49°. The foot frame is steadier on both (thin p99 91°, training p99 43°). Heading flips are 0 except one training flip at 0.25×. Not run on the robot.

**Repro:** `conda activate geodude-teleop && python -u UMIFT_Data/surface_frames_scripts/eval/teleop_field_sweep.py`

---
date: 2026-11-24
stream: frame
branch: lift_deploy
commit: uncommitted
status: done
artifacts:
  - /mnt/data/umidata/ft/training_outputs/oct_v35_h4s_r4vt_field_v3r/latest_final.ckpt
  - /mnt/data/umidata/ft/training_outputs/oct_v35_h4s_odom_r1rel/checkpoints/latest.ckpt
  - /home/boeing/umi_ft_ws/worktrees/UMI-FT-lift_deploy/closedloop_sim/out_v3r_r1rel_intersect/open_loop.json
---
# v3r vs r1rel on the shared frames

**What changed:** `closedloop_sim/compare_v3r_r1rel_intersect.py`. Recorded RGB. Frames kept by both datasets. Horizons 1, 5, and 16. Hold baseline is the measured pose.

**Why:** The earlier table used different kept frames and only the first action row.

**Result / status:** v3r keeps 1121 of the val queries, r1rel 1817, intersection 1121, 1089 have a 16-step future. Contact, model vs hold, median position mm / ΔR deg / δn mm: step 1 v3r 4.4/4.8/1.2 vs hold 11.9/0/9.3, r1rel 3.3/0.7/1.3 vs hold 0/0/0. Step 16 v3r 22.0/5.3/1.8 vs hold 71.8/9.9/16.3, r1rel 22.9/4.8/7.9 vs hold 71.0/9.9/38.4.

**Repro:** `python closedloop_sim/compare_v3r_r1rel_intersect.py` in `UMI-FT-lift_deploy`.

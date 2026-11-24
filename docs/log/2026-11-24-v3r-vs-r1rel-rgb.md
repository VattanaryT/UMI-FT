---
date: 2026-11-24
stream: frame
branch: lift_deploy
commit: uncommitted
status: done
artifacts:
  - /mnt/data/umidata/ft/training_outputs/oct_v35_h4s_r4vt_field_v3r/latest_final.ckpt
  - /mnt/data/umidata/ft/training_outputs/oct_v35_h4s_odom_r1rel/checkpoints/latest.ckpt
  - /home/boeing/umi_ft_ws/worktrees/UMI-FT-lift_deploy/closedloop_sim/out_v3r_r1rel_rgb/open_loop.json
---
# v3r vs original r1rel, recorded RGB

**What changed:** New open-loop scorer `closedloop_sim/compare_v3r_r1rel_rgb.py`. No closed loop. v3r rotation is R_tcp @ ΔR. r1rel is the relative pose composed with the measured TCP.

**Why:** Compare the RGB checkpoints on the same held-out episodes in one world frame.

**Result / status:** v3r `latest_final.ckpt` epoch 199 of 200. r1rel `oct_v35_h4s_odom_r1rel/checkpoints/latest.ckpt` epoch 195 of 200. Contact medians: v3r 4.4 mm / 5.1° (14/179 over 10°) / 1.2 mm δn; r1rel 3.6 mm / 0.7° (0/274 over 10°) / 1.5 mm δn.

**Repro:** `python closedloop_sim/compare_v3r_r1rel_rgb.py` in `UMI-FT-lift_deploy`.

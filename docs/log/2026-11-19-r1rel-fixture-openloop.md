---
date: 2026-11-19
stream: frame
branch: viewer_causal_foot
commit: uncommitted
status: done
artifacts:
  - /mnt/data/umidata/ft/open_loop_exports/causal_foot_r1rel_e40_track/report.json
  - /mnt/data/umidata/ft/training_outputs/oct_v35_h4s_odom_r1rel/checkpoints/latest.ckpt
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r1rel_world/acp_replay_buffer.zarr
---
# r1rel beside causal-foot epoch 40

**What changed:** Scored the odom r1rel `latest.ckpt` on the same 490 queries as the causal-foot label check (episodes 9, 12, 28, 37, 41, rgb stride 8, 6 steps). The relative action is decoded with `action21_postprocess` into a fixture-frame tool pose and compared with the label pose. Epoch 40 and track_clean e95 are the scores already in `causal_foot_43390127_e25_e40`. New export only.

**Why:** Put r1rel, epoch 40, and track_clean on one tool-pose metric. r1rel has no δn or v1/v2.

**Result / status:** 2940 steps, 0 failed. Median / p90, all steps. Position mm: r1rel 10.9/30.2, epoch 40 10.3/28.6, track e95 19.8/57.8. Rotation °: 2.2/5.88, 8.5/28.3, 6.63/21.8. Tilt °: 1.86/5.17, 7.41/24.1, 5.66/17.5. r1rel labels still contain the T265 glitches; the clean was not applied to `r1rel_world`. Robot was not run.

**Repro:** `python UMIFT_Data/surface_frames_scripts/score_causal_foot.py --device cuda --r1rel-ckpt /mnt/data/umidata/ft/training_outputs/oct_v35_h4s_odom_r1rel/checkpoints/latest.ckpt --prior-report /mnt/data/umidata/ft/open_loop_exports/causal_foot_43390127_e25_e40/report.json --out /mnt/data/umidata/ft/open_loop_exports/causal_foot_r1rel_e40_track`

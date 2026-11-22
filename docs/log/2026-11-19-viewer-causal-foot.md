---
date: 2026-11-19
stream: frame
branch: viewer_causal_foot
commit: uncommitted
status: in-progress
artifacts:
  - /mnt/data/umidata/ft/open_loop_exports/causal_foot_gt_labels
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_causal_foot/acp_replay_buffer.zarr
  - UMIFT_Data/surface_frames_scripts/assets/oct_v35_h4s_r4vt_action_normalizer_minmax_causal_foot.pkl
---
# Causal-foot open-loop scorer

**What changed:** New scorer on `viewer_causal_foot` (from `causal_foot` at `835a7f1`). It walks the label with `runner_causal_foot` `decode_chunk` at robot rate, `apply_safety=False`, and writes a new viewer export. Existing viewer, zarr, and pickle files were not modified.

**Why:** A 10-d causal-foot action has to be decoded with the same foot walk the runner uses before region and ΔR-mode errors mean anything.

**Result / status:** Label-only, held-out episodes 9, 12, 28, 37, 41, 2940 steps. Median / p90: position 5.12e-6 / 1.12e-5 mm, rotation 0 / 1.21e-6 °, δn 0 / 0 mm, tilt 0 / 1.21e-6 °. Crown, fillet, wall, near-identity, and return-yaw are the same size. Epoch 40 (job 43390127) is not local, so that comparison was not run.

**Repro:** `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest UMIFT_Data/tests/test_openloop_causal_foot.py -q` then `python UMIFT_Data/surface_frames_scripts/score_causal_foot.py`. Checkpoint compare, not run: `python UMIFT_Data/surface_frames_scripts/score_causal_foot.py --ckpt /path/to/43390127/epoch=0040-....ckpt` (also scores track_clean `latest.ckpt` epoch 95, job 43351238, on the same episodes and query indices).

---
date: 2026-10-21
stream: frame
branch: surface_frame
commit: 31c541a
status: in-progress
artifacts: [/mnt/data/umidata/ft/mold_aware_datasets/stringer_scanner_v2_fixb/, /mnt/data/umidata/ft/training_outputs/scanner_fixb/, /mnt/data/umidata/ft/training_outputs/scanner_fixb/eval/REPORT.md]
---
# Fix B (mold-frame actions) r1/r2/r3: offline eval

**What changed:** three runs on `stringer_scanner_v2_fixb`: r1 world obs + world actions, r2 world obs + Fix B actions (v_par, delta_n),
r3 surface-frame obs + curvature + Fix B actions. Live decode + safety aborts in `mold_aware_runtime.py`, wired into `umift_env_runner.py` (committed in 31c541a).

**Why:** express actions along the mold surface to simplify what the policy must learn.

**Result / status:** epoch-35 open-loop eval (anchored decode, off-edge chunks dropped): r1 best on first-step error (sd_err 6.65 mm vs 9.03 / 10.3) and seam angle (4.2 vs 16 deg).
Fix B does not beat the baseline offline; val shares the mold with train, so this measures fit only. No contact rollout recorded. Hyak jobs hit OOM before epoch 200.
Runner currently points `ckpt_path` at `r2_world_fixb/epoch=0035`; `mold_aware_dry_run` is False in code though the rollout README says True (verify before a robot run).

**Repro:** `UMIFT_Data/surface_frames_scripts/sanity_mold_aware_ep5.py` (must print SANITY PASS), `scanner_fixb_posttrain_eval.py --ckpt epoch0035`.

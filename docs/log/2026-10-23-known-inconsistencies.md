---
date: 2026-10-23
stream: infra
branch: logging
commit: 39f9f0b
status: in-progress
artifacts: []
---
# Known stale/inconsistent items at log start

- `README_layup_pipeline.md` says inference loads `stringer_scanner_v2/epoch=0025` and `periodic_scan_enabled` True; periodic scan auto-disables for no-scan / mold-aware ckpts.
- `umift_zucchini.yaml` `dataset_path` points at an old teleop set; always pass `task.dataset_path`.
- Runner `layup_eval_low_ply` = `scan_20261011_161311`; README names `scan_20260927_151409`.
- `layup_workcell_extrinsics.yaml` + heatmap extent are for the stringer_scanner_v2 mold placement; unknown if stringer_v3/v3.5 used the same.
- Wrist FT is described as custom; code/docs call it FT300-S.
- Worktrees: ~/.bashrc pins PYTHONPATH to the main UMI-FT checkout; use `source worktrees/activate.sh <checkout>`.

**Result / status:** resolve each, then add a superseding entry.

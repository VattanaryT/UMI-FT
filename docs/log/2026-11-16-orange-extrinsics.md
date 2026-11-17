---
date: 2026-11-16
stream: frame
branch: surface_frame
commit: uncommitted
status: done
artifacts:
  - PyriteConfig/tasks/umift/layup_workcell_extrinsics_20261116_orange.yaml
  - /mnt/data/umidata/ft/calibration/20261116/
---
# Orange 6-point extrinsics

**What changed:** `layup_workcell_extrinsics_20261116_orange.yaml` is the 6-point ICP fit (1.63 mm RMS) from the free touches, using the pendant TCP active on 2026-11-16. The runner prints the extrinsics path and sha256. `scripts/run_rollout.sh` takes `--extrinsics` and `--ckpt`. Touch CSVs and HTML stay out of git; copies are in `/mnt/data/umidata/ft/calibration/20261116/`.

**Why:** the October yaml no longer matches the mold. GT replay ep0 tracked the demo at 1.96 mm median.

**Result / status:** mold-calibration tests run before the commit. Robot not run for this commit.

**Repro:** `scripts/run_rollout.sh r4vt --replay-episode 0 --extrinsics PyriteConfig/tasks/umift/layup_workcell_extrinsics_20261116_orange.yaml`

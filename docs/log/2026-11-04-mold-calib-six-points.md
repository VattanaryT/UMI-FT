---
date: 2026-11-04
stream: frame
branch: surface_frame
commit: uncommitted
status: done
artifacts:
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_scanner_v2_fixb/shared/fixb_field.npz
---
# Six-point rigid mold calibration

**What changed:** `mold_calib_points.yaml` names the six touch points and the fixture axes (near = +X, left = +Z, crown outward = -Y). `mold_calib_check.py` prompts those labels, warns if the foot is more than 20 mm from the expected spot, and fits a small rigid offset. `mold_calib_viser.py` shows the mesh, live TCP, and CSV touches. Held-out xyz in the yaml are copies of the train mesh until a separate field exists.

**Why:** Four free-form touches could not separate an along-stringer shift from a yaw. The flange corners give in-plane position.

**Result / status:** Not yet run on the robot. Unit tests cover the rigid fit, the 20 mm spot distance, and that the fixture map matches `pose7_batch_to_fixture`.

**Repro:** `source /home/boeing/umi_ft_ws/worktrees/activate.sh UMI-FT-surface_frames` then `python UMIFT_Data/surface_frames_scripts/mold_calib_check.py --mold train`. Viewer: geodude venv, `python UMIFT_Data/surface_frames_scripts/mold_calib_viser.py --mold train --csv <csv>`, open http://127.0.0.1:8081.

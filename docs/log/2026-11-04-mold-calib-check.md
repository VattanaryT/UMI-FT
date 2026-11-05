---
date: 2026-11-04
stream: frame
branch: surface_frame
commit: fd58008
status: superseded-by 2026-11-04-mold-calib-six-points.md
artifacts:
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_scanner_v2_fixb/shared/fixb_field.npz
---
# Read-only mold touch check

**What changed:** `mold_calib_check.py` reads the live TCP over RTDE receive, maps it with `load_workcell_extrinsics`, and queries the mesh through `MoldAwareRuntime.fill_obs_raw`. `--mold train` is the policy field. `--mold heldout` needs a separate `--field`.

**Why:** A contact eval should not start until a few hand-jogged touches show the fixture frame and the mesh agree with the tool tip.

**Result / status:** Not yet run on the robot. The three `fixb_field.npz` copies (scanner v2, October v3.5, stringer_data_v2_full) are the same sha256 `f841837b937c1d2f0c6e80c1cf273f5d201ba72a89f7466f83199f7cf64664f2`. There is no held-out field on disk.

**Repro:** `source /home/boeing/umi_ft_ws/worktrees/activate.sh UMI-FT-surface_frames` then `python UMIFT_Data/surface_frames_scripts/mold_calib_check.py --mold train`.

---
date: 2026-11-10
stream: frame
branch: surface_frame
commit: uncommitted
status: done
artifacts:
  - /mnt/data/umidata/ft/datasets/stringer_v3.5/tag_traj_odom_corr/
---
# T265 teleport sidecar

**What changed:** `fix_odom_teleport.py` writes `tag_traj_odom_corr/<name>_odom_corr.npz` for all 68 `stringer_v3.5` recordings. `*_ee_master.npz` and the training zarrs are unchanged. The October viewer has an off-by-default `t265 fixed` trace.

**Why:** One-sample T265 teleports (15–51 cm, under 0.5°) pull the measured TCP off the mold. Tag fixes on those steps have normal reprojection.

**Result / status:** 49 steps removed, 0 rejected by the velocity test. Jump-episode tag closures fall to 2–18 cm except episode 40/41/49 (the jumps cancel before the next tag, so the closure was already 5–11 cm) and episode 47 (31–47 s has no later tag and stays flagged). Not used for labels yet.

**Repro:** `~/miniforge3/envs/pyrite/bin/python UMIFT_Data/surface_frames_scripts/fix_odom_teleport.py`

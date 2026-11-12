---
date: 2026-11-10
stream: frame
branch: surface_frame
commit: uncommitted
status: done
artifacts:
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4_world_fixb_tilt/acp_replay_buffer.zarr
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_world_fixb_tilt/acp_replay_buffer.zarr
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r1rel_world/acp_replay_buffer.zarr
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/label_stats.json
---
# October labels from the corrected T265 poses

**What changed:** Episodes 0–49 were relabeled from `tag_traj_odom_corr` into `stringer_v3.5_fixb_odom_corr`. r4 is the tracked TCP foot. r4vt keeps that foot and v1/v2, and sets `delta_n = (p_vt − x_s) · n` with the source virtual-target offset carried onto the corrected TCP. r1rel stores the same TCP. Existing r1, r4, and `stringer_v3.5` were not opened for writing. Bad spans are stored, not filtered: episode 47 recording 31–47 s (3188 samples), episode 38 from 50 s to the end (585 samples).

**Why:** One-sample T265 teleports were removed upstream. The surface labels and the relative-pose arm have to use that same TCP.

**Result / status:** Contact (`||F|| > 2 N`) median of `mold_signed_distance` stays near 0 mm (episode-median 0.0 → 0.5 mm). r4vt `delta_n` contact median is −117 → −113 mm across episodes. Episodes 35 (−12 → −2 mm) and 44 (+72 → +58 mm) were already outside that band. Not filtered and not trained.

**Repro:** `python UMIFT_Data/surface_frames_scripts/fixb_relabel_odom_corr.py`

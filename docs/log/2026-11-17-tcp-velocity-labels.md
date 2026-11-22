---
date: 2026-11-17
stream: frame
branch: smooth_frame
commit: uncommitted
status: done
artifacts:
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_smooth_tcp_track/acp_replay_buffer.zarr
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_smooth_dof/acp_replay_buffer.zarr
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_velocity_options_report.json
---
# TCP velocity labels, tracked foot and diffused field

**What changed:** Two new October zarrs beside `r4vt_smooth_frame`. `track` projects the odom-corrected TCP velocity onto the smooth frame at a foot that can move at most 6 mm per sample. `dof` uses the same velocity and a harmonic off-surface frame (Laplace extension of the smooth frames, the PDE in Bilaloglu et al. 2026). The current smooth zarr, r1rel, and the normalizer pickle are unchanged. The viewer on port 8767 gained `track` and `dof` buttons.

**Why:** v1/v2 in the smooth zarr is the closest-point foot velocity, which hops crown to wall.

**Result / status:** Over 607756 samples, `|v|>0.5` went from 16796 to 3672 (track) and 3735 (dof). Foot steps over 20 mm went from 1446 to 0 and 0. The remaining fast samples are TCP motion (about 30 mm in 50 ms), not foot hops. δn p95 stays about 6 mm because that is the virtual target's own motion; the 154 mm tail is 41 mm on track. One dof sample still moves δn by 148 mm where the field normal turns 48°. Frame change for a TCP moved ±5 mm, p95: closest 20.3°, track 8.4°, dof 9.7°. The max stays near 180° at the tip vortices. Kept-row ΔR median 47.8° (track) and 40.3° (dof), still both modes. v1/v2 exceed the old ±1 pickle range. No new pickle was written. r1rel's trained action does not hop; its `mold_geom` frame does (1388 foot steps over 20 mm, 1938 normal steps over 30°). Not trained.

**Repro:** `python UMIFT_Data/surface_frames_scripts/fixb_tcp_velocity_labels.py` then `python UMIFT_Data/surface_frames_scripts/view_smooth_frame.py --bags-dir /mnt/data/umidata/ft/datasets/stringer_v3.5_base/bags`

---
date: 2026-11-17
stream: frame
branch: smooth_frame
commit: uncommitted
status: done
artifacts:
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_smooth_tcp_track_clean/acp_replay_buffer.zarr
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_smooth_tcp_track_clean/glitches.txt
  - UMIFT_Data/surface_frames_scripts/assets/oct_v35_h4s_r4vt_action_normalizer_minmax_track_clean.pkl
  - UMIFT_Data/surface_frames_scripts/fixb_train_oct_v35_r4vt_track_clean_e100_klone.sbatch
---
# Cleaned TCP track labels, and why dof ep15 turns 48°

**What changed:** Glitch samples in the odom-corrected TCP (step faster than 0.5 m/s, or a fast step that reverses within 200 ms) are listed per episode with RGB times. Spans of at most 200 ms whose endpoints still agree are interpolated into `r4vt_smooth_tcp_track_clean`. One-way jumps and longer spans are left in the pose and written into that zarr's `bad_span_0`, so training windows that overlap them drop. The raw track zarr, r1rel, and the old 21-d pickle are unchanged. `view_smooth_frame.py` has a "track clean" button; red spans and 3D points mark the glitches. `view_oct.py` is unchanged.

**Why:** The |v| > 1 m/s bursts are T265 slips. The tool does not move that fast, and the wrist RGB does not jump. A low-pass would smear the slip into the neighboring labels. A one-way slip cannot be interpolated: the pose does not come back, so the straight line is still the slip (ep5 at 16.87 s is 1.66 m/s between the endpoints).

**Result / status:** 65 spans interpolated, 450 excluded. On the kept rows (34112, episodes 35 and 44 still out) |v| > 0.5 is 0, and v1/v2 are −0.470..0.419 and −0.453..0.410. δn still reaches 0.1872, above the old pickle's 0.1687, so the new pickle was written. sha256 `29a48f69ff51807df631b1e350087d6991091cba9739f0be88b30e2c85cf3bd4`. Old pickle sha is still `ac13d3fd7b647541d093688896ff43f1edb3a025397b6037da01dfffb6707bee`. r1rel is not modified: 7148 of 19418 kept queries have a glitch inside the action-chunk time range (8652 glitch rows). The sbatch is written and not submitted.

Dof ep15, for later: at t=21.298 s (robot index 4242) the diffused-field normal turns 47.6° then 37.2°, from about [−0.02, −0.72, 0.70] on the fillet to [−0.003, −0.998, −0.059] on the crown and then [0.016, −0.76, −0.65]. The foot only moves about 2 mm and the TCP slides 3.6–4.4 mm per sample. The tracked-foot normal stays continuous (z about −0.67 to −0.63). The TCP is about 50 mm off the surface at mid-mold x≈−0.17, so the off-surface blend swings between the crown and the two walls. δn goes from −0.091 m to −0.239 m because that swing hits a long virtual-target offset. Ten samples in the whole dof set have a normal step over 30°. That is why track was kept and dof was not.

**Repro:** `python UMIFT_Data/surface_frames_scripts/fixb_tcp_track_clean.py` (refuses if the clean zarr already exists). Viewer: `python UMIFT_Data/surface_frames_scripts/view_smooth_frame.py --bags-dir /mnt/data/umidata/ft/datasets/stringer_v3.5_base/bags`.

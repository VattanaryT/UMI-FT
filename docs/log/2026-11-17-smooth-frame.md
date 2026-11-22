---
date: 2026-11-17
stream: frame
branch: smooth_frame
commit: uncommitted
status: done
artifacts:
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared/fixb_smooth_frames.npz
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb/shared/fixb_smooth_frames.npz
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_smooth_frame/acp_replay_buffer.zarr
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_smooth_frame/smooth_frame_report.json
---
# Smooth inward mold frame for r4vt

**What changed:** Sidecar `fixb_smooth_frames.npz` (40 mm ball, z = −n). Labels and `integrate_tcp_foot` both read it. r4vt zarr hardlinked from the causal October tree into `r4vt_smooth_frame`. Face hold is off only when this sidecar loads. Foot cap 10 mm and the 3° R_d slew stay. r1rel does not load it.

**Why:** Closest-face frames flip at the crown-to-wall corner, and frame z pointed out while the tool points in, so ΔR sat near 180°.

**Result / status:** Fillet radius 35.4 mm, so the smooth radius is 40 mm. Neighbor t1 max is 179.2° on 37 tip edges; p99 is 11.9°; interior (40 mm off each end) max is 40.6°. Kept-step ΔR (n=40555): median 170.3° → 48.1°, fraction under 45° 0 → 0.482, fraction over 135° 0.858 → 0.421. ±5 mm frame rotation median 8.05° → 3.74°, p95 39.5° → 19.6°. Along episodes 0, 9, 31 the full-frame angle p50 is ~171° (the z flip) and the outward-normal angle p50 is 8.7–9.7°. δn on kept rows is inside `oct_v35_h4s_r4vt_action_normalizer_minmax_exbad.pkl`; no refit. Not trained.

**Repro:** `python UMIFT_Data/surface_frames_scripts/fixb_smooth_frame_labels.py` from that directory, pyrite env. Train with `fixb_train_oct_v35_r4vt_smooth_frame_e100_klone.sbatch` after rsync. Not submitted.

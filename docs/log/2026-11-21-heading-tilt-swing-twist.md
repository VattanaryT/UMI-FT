---
date: 2026-11-21
stream: frame
branch: causal_foot
commit: uncommitted
status: done
artifacts:
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_heading_tilt/acp_replay_buffer.zarr
---
# Swing-twist heading labels

**What changed:** `fixb_heading_tilt_0` was rewritten. ψ is the twist of `R_zero.T @ R_tool` after the swing is removed. `(a, b)` is the full swing rotvec after `Rz(ψ)`. The normalizer, 76.5° clip, and sbatch were left as they were.

**Why:** Measuring ψ from the tilted tool x left an about-z residual, so the 4-number pack did not round-trip.

**Result / status:** GT decode, no clip, stored float32. Episodes 0 / 9 / 31 geodesic median 4.2e-7° / 5.2e-7° / 5.8e-7°, max 3.4e-6° on each. Not submitted.

**Repro:** `python UMIFT_Data/surface_frames_scripts/fixb_heading_tilt_zarr.py --rewrite-heading`

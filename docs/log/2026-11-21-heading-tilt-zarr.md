---
date: 2026-11-21
stream: frame
branch: causal_foot
commit: uncommitted
status: superseded-by 2026-11-21-heading-tilt-swing-twist.md
artifacts:
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_heading_tilt/acp_replay_buffer.zarr
  - /home/boeing/umi_ft_ws/worktrees/UMI-FT-causal_foot/UMIFT_Data/surface_frames_scripts/assets/oct_v35_h4s_r4vt_action_normalizer_minmax_heading_tilt.pkl
---
# Heading-tilt zarr and sbatch

**What changed:** New hardlink of the causal-foot zarr plus `fixb_heading_tilt_0` (sin ψ, cos ψ, a, b). New 8-d normalizer. Task yaml and e200 sbatch. The causal-foot zarr and the `14308b72` pickle were not modified.

**Why:** ψ from the mold long axis, tilt in the frame after Rz(ψ), so the action does not follow the carried t1.

**Result / status:** GT decode, no clip, episodes 0 / 9 / 31: median geodesic 0.55° / 0.53° / 0.65°, p90 2.3° / 3.1° / 2.1°, max 28° / 33° / 24°. The error is the dropped about-z twist. Kept-row tilt p99.5 is 76.5°. e_long carry was unused (0 samples). Sbatch not submitted. Normalizer sha `71bd4cdc7be5a066e786d0b1fb9be526e01d6a6ff1e4ad7dd9c6280ac3709f84`.

**Repro:** `python UMIFT_Data/surface_frames_scripts/fixb_heading_tilt_zarr.py` from the causal_foot checkout. Train with `fixb_train_oct_v35_r4vt_heading_tilt_e200_klone.sbatch` after the user rsyncs.

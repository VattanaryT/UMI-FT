---
date: 2026-11-21
stream: frame
branch: causal_foot
commit: uncommitted
status: done
artifacts:
  - /home/boeing/umi_ft_ws/worktrees/UMI-FT-causal_foot/UMIFT_Data/surface_frames_scripts/assets/oct_v35_h4s_r4vt_action_normalizer_minmax_heading_tilt_swing.pkl
---
# Heading-tilt normalizer refit

**What changed:** New pickle on the swing-twist labels. Sbatch `NORM_FILE` and `NORM_SHA` point at it. The `71bd4cdc` pickle is unchanged.

**Why:** `71bd4cdc` was fit before the swing-twist rewrite.

**Result / status:** sha `557e4d25a584ea026c253c95134f983dec0d5624dd1dd90652b78cd679630215`. Same 34112 kept rows. Tilt p99.5 on these labels is 76.4°. The stored runtime clip stays 76.5°. Not submitted.

**Repro:** min/max scale on kept rows of `r4vt_heading_tilt/acp_replay_buffer.zarr`, episodes 35 and 44 excluded.

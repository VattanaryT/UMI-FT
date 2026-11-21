---
date: 2026-11-21
stream: frame
branch: causal_foot
commit: uncommitted
status: done
artifacts:
  - /home/boeing/umi_ft_ws/worktrees/UMI-FT-causal_foot/UMIFT_Data/surface_frames_scripts/assets/oct_v35_h4s_r4vt_action_normalizer_minmax_heading_tilt_swing_sc.pkl
---
# Heading-tilt sin identity normalizer

**What changed:** New pickle, task yaml, and e200 sbatch. sin ψ scale is 1 and offset is 0. The `557e4d25` pickle and the zarr attr are unchanged.

**Why:** The swing pickle min/max-scaled sin ψ. cos ψ was already identity.

**Result / status:** sha `e93b87561e762c2d7f1549a883eb18d45e6d50d929df463c43fce4817444e2f5`. Not submitted.

**Repro:** `fixb_train_oct_v35_r4vt_heading_tilt_sc_e200_klone.sbatch`

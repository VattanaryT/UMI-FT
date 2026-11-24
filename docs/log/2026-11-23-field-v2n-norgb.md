---
date: 2026-11-23
stream: frame
branch: field_v2n_norgb
commit: uncommitted
status: done
artifacts:
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_field_v2n/acp_replay_buffer.zarr
  - /home/boeing/umi_ft_ws/worktrees/UMI-FT-field_v2n/UMIFT_Data/surface_frames_scripts/assets/oct_v35_h4s_r4vt_action_normalizer_minmax_field_v2n.pkl
---
# field_v2n no-RGB ablation

**What changed:** Task `umift_oct_fixb_r4vt_field_v2n_norgb` drops `rgb_0` and `rgb_time_stamps_0` from policy obs. Same zarr and normalizer sha `4d7c4feb…`. CLIP is not built. Run name `oct_v35_h4s_r4vt_field_v2n_norgb`.

**Why:** Ablate wrist RGB against field_v2n.

**Result / status:** Unit test only. Not transferred, not submitted.

**Repro:** `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest UMIFT_Data/tests/test_norgb_task.py PyriteML/tests/test_drop_rgb_inputs.py`

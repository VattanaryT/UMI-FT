---
date: 2026-11-19
stream: bc
branch: causal_foot
commit: uncommitted
status: in-progress
artifacts:
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_causal_foot/acp_replay_buffer.zarr
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_causal_foot.tar
  - UMIFT_Data/surface_frames_scripts/assets/oct_v35_h4s_r4vt_action_normalizer_minmax_causal_foot.pkl
---
# Train the causal-foot action as 10 dims

**What changed:** `umift_oct_fixb_r4vt_causal_foot` keeps v1, v2, δn, stiffness, and ΔR. The dataset does not SE3-transform that vector. The normalizer replaces all 10 dims. A field jump zeros ΔR dims 4:10. The sbatch is 200 epochs, checkpoint every 20, plus the last. Not submitted.

**Why:** The other 17 action slots are constants, so they do not need a learned range. ΔR does.

**Result / status:** Episode 40 slice matches the stored labels. Normalizer sha256 `14308b72d55600b4359dfa67d10aedab378e42ad26f2bbcac988eed4d97a1ac2`, 34112 kept rows. Training not started. The live runner still decodes 27-d actions.

**Repro:** `sbatch UMIFT_Data/surface_frames_scripts/fixb_train_oct_v35_r4vt_causal_foot_e200_klone.sbatch` on Hyak after the rsync in that file. Do not run it here.

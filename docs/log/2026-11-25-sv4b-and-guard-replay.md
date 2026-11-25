---
date: 2026-11-25
stream: frame
branch: surface_frame
commit: 20965da
status: done
artifacts:
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_v4_fixb/r4vt_field_v2n_v3r_liftos_sv4b/acp_replay_buffer.zarr
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_v4_fixb/r4vt_field_v2n_v3r_liftos_sv4b/frames_pca_r020.npz
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_v4_fixb/shared/long_dt_gt_100ms_sv4b.npz
  - UMIFT_Data/surface_frames_scripts/assets/oct_v35_h4s_r4vt_action_normalizer_minmax_field_v2n_v3r_liftos_sv4b.pkl
  - UMIFT_Data/surface_frames_scripts/assets/lift_isolated_ends_sv4b.npz
  - offline_hw/out/replay_guard_train_frame
---
# sv4b horizon 64 and the pe_guard train-frame replay

**What changed:** Replay of `2026-11-24_liftos_hw_pe_guard` h0–h6 through the runner with training-frame mold_geom (`8d77c6a`). stringer_v4 rebuilt at action horizon 64 as `r4vt_field_v2n_v3r_liftos_sv4b`. A second config upweights δn only and raises lift oversampling to 30%.

**Why:** The hardware divergence at h3 might have been the mold_geom packer. The horizon-24 sv4 set is the wrong chunk length for the liftos recipe.

**Result / status:** The corrected packer does not stop the h3 divergence. h0 (ep9 1220) foot mean 7.7 mm, max 12.9 mm, twist error +0.5°, v2 +0.074 vs label +0.034, δn +10.7 mm vs label −2.3 mm. h2 foot mean 25 mm (max 39). h3 idx 1447 foot mean 41 mm. δn on h0–h3 is clearance while the label is a press.

sv4b normalizer sha `250297835d3330172a58a1a78de4aeda9da0247ec57470b3c323a0591b3ca70d`, 329 lifts, 21372 windows (1,367,808 rows at horizon 64). Train sampler, episodes 7/14/30 excluded:

| sampler | unique lift windows | lift rows per epoch | epoch length |
|---|---:|---:|---:|
| sv4b, 15%, sv4b ends | 736 | 3550 | 23668 |
| sv4b-lift, 30%, same ends | 736 | 8622 | 28740 |
| October liftos, 15% | 348 | 1902 | 12681 |

The October end file has 272 ends. Those episode ids are not stringer_v4 rows, so the 30% sampler does not use them. Batch 64 OOM'd on the 4090 (viewers resident). Batch 32 ran epoch 0 to the last logged step 739/740, loss 0.0905. Not submitted.

**Repro:** `python offline_hw/replay_guard_train_frame.py`. `python UMIFT_Data/surface_frames_scripts/build_liftos_sv4b.py`. Smoke: `task=umift_oct_fixb_r4vt_field_v2n_v3r_liftos_sv4b training.num_epochs=1 dataloader.batch_size=32`. Sbatches are drafts.

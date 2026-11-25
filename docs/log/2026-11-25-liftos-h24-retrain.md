---
date: 2026-11-25
stream: frame
branch: surface_frame
commit: uncommitted
status: in-progress
artifacts:
  - UMIFT_Data/surface_frames_scripts/config/task/umift_oct_fixb_r4vt_field_v2n_v3r_liftos_h24.yaml
  - UMIFT_Data/surface_frames_scripts/climb_h24_dataset.py
  - UMIFT_Data/surface_frames_scripts/fixb_train_oct_v35_r4vt_field_v2n_v3r_liftos_h24_e200_klone.sbatch
  - UMIFT_Data/surface_frames_scripts/eval/climb_labels/ep9_climb_labels.png
  - UMIFT_Data/surface_frames_scripts/eval/climb_labels/ep9_lift_labels.png
---
# Horizon-24 liftos retrain, loss x4 on v2 and δn

**What changed:** New task yaml and dataset class. Same liftos zarr and normalizer sha `ce85fae2…`. Action horizon 24. Loss weight 4 on v2 and δn after the field-jump ΔR mask. No new label array.

**Why:** On ep9 the sampler-index v*dt already matches the climb (z +0.163 vs foot +0.177 from idx 5600 to 6700). The +70 mm δn edge is two action steps, then a 1.5 s plateau. A 4.2 s chunk and uniform normalized MSE bury both.

**Result / status:** Local 1-epoch smoke, batch 16, exit 0. Action `[16, 24, 10]`. train_loss 0.282. val_loss 0.085 on 2 batches. Hyak sbatch is a draft and was not submitted.

**Repro:** `python train.py --config-name=train_umift_workspace task=umift_oct_fixb_r4vt_field_v2n_v3r_liftos_h24` with the liftos zarr and normalizer. Hyak file is `fixb_train_oct_v35_r4vt_field_v2n_v3r_liftos_h24_e200_klone.sbatch`.

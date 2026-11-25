---
date: 2026-11-25
stream: frame
branch: surface_frame
commit: 81b1ce3
status: done
artifacts:
  - /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_field_v2_k3000/frames_pca_r020.npz
  - /mnt/data/umidata/ft/training_outputs/oct_v35_h4s_r4vt_field_v2n_v3r_liftos/latest.ckpt
  - UMIFT_Data/surface_frames_scripts/eval/climb_labels/force_openloop.png
---
# Training-frame mold_geom is the runner default

**What changed:** The runner publishes mold_geom from CausalFoot on `frames_pca_r020.npz`, PCA-reoriented and carried. `--no-mold-geom-train-frame` or `UMIFT_MOLD_GEOM_TRAIN_FRAME=0` keeps the live smooth field. meta.json stores the flag and the orientation-file sha256 `fa220db41eca8ac3…`.

**Why:** The live sidecar `fixb_smooth_frames.npz` is a different orientation field (median 8° at the vertices) and was not reoriented onto `field_v2_axes`. On ep9's own poses the training packer matches the liftos zarr to 0.00° and 0.00 mm.

**Result / status:** Open-loop liftos `latest.ckpt`, contact `|min(|δn|·3000, 25) − |F·n||`: ep9 n=669 median 4.60 N p90 15.17 N; thin stringer C n=57 median 6.67 N p90 10.22 N. All six ep9 lifts (1527, 2350, 4366, 4773, 5201, 6008) predict a positive δn within ±0.3 s. Peaks are 13, 9, 0.3, 5, 5, and 77 mm against label peaks 22, 11, 13, 17, 9, and 70 mm. The 14 hardware horizons stay ~91 mm from those demo TCPs, so this does not by itself restore the climb.

**Repro:** `python UMIFT_Data/surface_frames_scripts/eval/force_openloop.py`

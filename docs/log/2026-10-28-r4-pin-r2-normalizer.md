---
date: 2026-10-28
stream: frame
branch: surface_frame
commit: 6a8d90b
status: done
artifacts: [/mnt/data/umidata/ft/training_outputs/scanner_fixb/r2_world_fixb/sparse_normalizer.pkl, /mnt/data/umidata/ft/training_outputs/scanner_fixb/r4_world_fixb_tilt/sparse_normalizer.pkl, /mnt/data/umidata/ft/mold_aware_datasets/stringer_scanner_v2_fixb/r4_world_fixb_tilt/checks/viz/, /mnt/data/umidata/ft/training_outputs/scanner_fixb/r4_world_fixb_tilt/epoch=0060-train_loss=0.018.ckpt]
---
# r4 pins the r2 action normalizer

**What changed:** `UmiFTDatasetR4.get_normalizer` copies the r2 job-41907426 scale, offset, and input stats onto action dims 0:21 and leaves ΔR (21:27) identity. r4 logs diffusion MSE by group (`v_t`, `delta_n`, stiffness, grip/grasp, `ΔR`) and the masked-ΔR fraction as Python floats, train and val. The r4 sbatch sets `training.log_gradient_norm=true`. Gradient clipping stays off. ΔR figures are under the r4 `checks/viz` directory.

**Why:** Raw δn matches between the r2 and r4 zarrs. Episode 21's 0.989 m peak (t=23.06 s) sits in scan pause [11.44, 26.44] s. Today's sampler drops 41.5% of queries on that filter, and the kept train windows (63557 chunks, 1525368 steps) have δn min/max/mean −0.160913 / 0.317869 / 0.01568, the same numbers as the 2026-10-27 r4 normalizer. Stiffness mean on those windows is 724.7, also the r4 file. The 2026-10-16 r2 normalizer still has δn max 0.988657. Only dimension 2 of scale/offset differs inside 0:21. The ΔR loss mask does not enter the normalizer fit.

**Result / status:** After the pin, max |scale| and |offset| difference vs r2 on dims 0:21 is 0. δn scale 1.73978, offset −0.720047, r2 δn max 0.988657. ΔR scale is 1 and offset is 0. `sanity_mold_aware_ep5.py --run r4_world_fixb_tilt` printed SANITY PASS. Viz episodes: flat 56 (longest on-mesh |k1|<8 run), fillet 14 (longest on-mesh |k1|>30 run), off-edge 57 (longest in-plane >5 mm run). Not submitted to Hyak. Epoch 35 was deleted by top-k. The local checkpoint is epoch 60 and still stores the 2026-10-27 normalizer; the pin is applied the next time training fits a checkpoint.

**Repro:** `python UMIFT_Data/surface_frames_scripts/sanity_mold_aware_ep5.py --run r4_world_fixb_tilt` and `python UMIFT_Data/surface_frames_scripts/fixb_r4_deltaR_viz.py`. r4 train override: `task.dataset.r2_action_normalizer_path` pointing at `assets/r2_sparse_normalizer_41907426.pkl`.

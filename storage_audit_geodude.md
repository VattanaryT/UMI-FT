# geodude-control storage audit

Read-only except this file. No `rm`, `mv`, or `chmod`. Collected 2026-11-23 ~01:15 PT. Owner is `stat` user/uid, then path and mtime. Every path under `/mnt/data/umidata` and `/home/boeing` is uid 1004 `boeing`. `/mnt/data/geodude` is uid 1001 `geodude`.

## Disks

| Mount | Size | Used | Avail | Use |
|---|---|---|---|---|
| `/` (`/dev/nvme0n1p2`) | 1.8T | 1.7T | 26G | 99% |
| `/home/boeing` | same as `/` | | 26G | 99% |
| `/mnt/data` (`/dev/nvme1n1p1`) | 3.6T | 3.2T | 277G | 93% |
| `/boot/efi` | 511M | 6.1M | 505M | 2% |

`/mnt/data` breaks down as umidata 2.0T (ft 1.9T), boeing 938G, geodude 224G. `/home/boeing` is 532G.

## How owner was decided

- **Vat:** path is one of `UMI-FT-surface_frames`, `UMI-FT-compare_r1rel`, `UMI-FT-tilt1`, or an `oct_v35_*` / `mold_aware` run those branches train. Normalizer sha prefixes were hashed and match.
- **Teammate:** `UMI-FT-standard_bc`, `UMI-FT-ffs_depth`, `UMI-FT/` (main), `ffs_depth_proto`, and `stringer_v3.5/hist` dagger checkpoints. Same uid, classified by path.
- **Other:** uid 1001, or a path that is not the UMI-FT tree (`/mnt/data/boeing`, `/mnt/data/geodude`, `/home/boeing/ros2_ws`).
- **Unknown:** the shared `boeing` login caches, and older `datasets/` trees both fronts use.

## Totals for Vat

Safe to drop (duplicate, truncated, or already extracted): **about 7 GB**.

| Path | Size | Why it is safe |
|---|---|---|
| `.../r4vt_causal_foot.tar` | 3.4G (3580805120 bytes) | zarr exists; same bytes also on Hyak |
| heading `epoch=0199-val_loss=0.237.ckpt` | 920M (964689920) | good checkpoints are 3047598380 bytes. Hyak never got this file; it has e25 and e100 only |
| causal `epoch=0120-val_loss=0.148.ckpt` | 2.9G | not in `checkpoints.json`. e25, e40, and latest stay |
| `r4vt_causal_foot_carry` | 78M | pre-k3000 zarr. The live set is `r4vt_causal_foot_carry_k3000` (15M unique here, 5.8G on Hyak) |

`mold_aware` (123G) and `oct_v35_fixb` (20G) stay. Weirdlab has those bytes. `scanner_fixb` (20G) stays too: weirdlab has no checkpoint of that size.

## Vat — keep

| Path | Size | mtime | Status | Reason |
|---|---|---|---|---|
| `assets/oct_v35_h4s_r4vt_action_normalizer_minmax_causal_foot.pkl` | 3.2K | — | KEEP | sha 14308b72 |
| `..._heading_tilt.pkl` | 3.1K | — | KEEP | sha 71bd4cdc |
| `..._heading_tilt_swing.pkl` | 3.1K | — | KEEP | sha 557e4d25 |
| `..._heading_tilt_swing_sc.pkl` | 3.1K | — | KEEP | sha e93b8756 |
| `..._heading_tilt_swing_sc_robust_ab.pkl` | 3.1K | — | KEEP | sha 3f9b16b1 |
| `.../r4vt_causal_foot_carry_k3000` | 15M | Nov 22 20:54 | KEEP | current k3000 zarr. `du` is small because rgb is hardlinked |
| `.../r4vt_heading_tilt` | 23M | Nov 21 | KEEP | labels for the running robust-ab job |
| `.../r4vt_causal_foot` | 100M | Nov 19 | KEEP | labels for tilt1 and the e40 policy |
| `.../shared` | 12M | Nov 17 | KEEP | field sidecar |
| causal epoch 25, epoch 40, latest | 2.9G each, 3047655148 bytes | Nov 19 | KEEP | `checkpoints.json` plus the e40 hardware checkpoint |
| heading epoch 25 and latest | 2.9G each, 3047598380 bytes | Nov 21 14:24 and 16:52 | KEEP | `checkpoints.json`. latest matches Hyak epoch 100 by size and mtime |
| track_clean epoch 55 and latest | 2.9G each, 3048139948 bytes | Nov 18 | KEEP | `run_rollout.sh` default |
| r1rel e25 and `oct_v35_h4s_odom_r1rel/.../latest.ckpt` | 2.9G each | Nov 12 | KEEP | `checkpoints.json` |
| `oct-v35-...-43465423`, `43467749`, `43472139` | 5.7G each | Nov 23 01:05 | KEEP | partial mirrors of the three running jobs. Do not delete while they write |
| `worktrees/UMI-FT-surface_frames` | 224M | Nov 22 | KEEP | current branch |
| `worktrees/UMI-FT-compare_r1rel` | 412M | — | KEEP | live worktree |
| `worktrees/UMI-FT-tilt1` | 116M | — | KEEP | live worktree |

`checkpoints.json` is `UMI-FT-compare_r1rel/.../eval/thin_heldout/checkpoints.json`. Its optional job ids are swapped: it calls 43465423 tilt1 and 43467749 robust-ab. squeue has 43465423 = heading-robustab, 43467749 = tilt1, 43472139 = k3000-carry.

## Vat — candidate or ask

| Path | Size | mtime | Status | Reason |
|---|---|---|---|---|
| `.../r4vt_causal_foot.tar` | 3.4G | Nov 22 15:10 | CANDIDATE | extracted |
| heading `epoch=0199` | 920M | Nov 21 21:21 | CANDIDATE | truncated, 964689920 vs 3047598380 |
| causal `epoch=0120` | 2.9G | Nov 19 14:37 | CANDIDATE | extra pulled checkpoint |
| `r4vt_causal_foot_fixedk3000_nocam_43437525` | ~2.8G | Nov 21 | KEEP | 1496932127 bytes matches every weirdlab nocam epoch |
| `r4vt_causal_foot_carry` | 78M | Nov 22 19:56 | CANDIDATE | superseded by carry_k3000 |
| `training_outputs/mold_aware` | 123G | Oct 14 | KEEP | same 2095201116-byte checkpoints exist on `/gscratch/weirdlab/vtevy`. Drop the weirdlab copy |
| `training_outputs/oct_v35_fixb` | 20G | Nov 3 | KEEP | epoch-195 files match weirdlab byte-for-byte. Drop the weirdlab copies |
| `training_outputs/scanner_fixb` | 20G | Oct 28 | KEEP | September scanner Fix B. No size-matched copy on weirdlab |
| `training_outputs/oct_v35_h4s_r4vt_smooth_frame` | 2.9G | Nov 18 | CANDIDATE | finished, also on Hyak |
| `training_outputs/oct_v35_h4s_r4vt_frame_continuity_causal` | 2.9G | Nov 17 | CANDIDATE | finished |
| `training_outputs/oct_v35_h4s_r4vt_track_clean_step_mask` | 5.7G | Nov 18 | CANDIDATE | finished |
| `training_outputs/oct_v35_h4s_odom_r4vt` and `_valtopk` | 2.9G each | Nov 12 | ASK | superseded by track_clean for rollout, but checkpoints.json does not list them |
| `ft/images/umift-train_v8.sif` | 6.6G | Nov 9 22:21 | KEEP | newest image |
| `ft/images/umift-train_v7.sif` | 6.6G | Nov 9 20:26 | CANDIDATE | same byte count as v8 (7126274048) |
| `ft/images/` v0–v6 and `umift-train.sif` | ~46G | Oct 9–18 | ASK | seven more ~6.6G images. v1 and `umift-train.sif` are both 7116103680 bytes. Confirm v8 is the one sbatch uses before deleting |

## Top directories

Sizes are `du` totals, so a parent includes its children. Do not add these up.

| Path | Size | Owner | mtime | Status | Reason |
|---|---|---|---|---|---|
| `/mnt/data/umidata/ft/datasets` | 1.2T | unknown | — | ASK | shared source data, both fronts |
| `/mnt/data/umidata/ft/training_outputs` | 509G | mixed | Nov 23 | mix | Vat oct_v35 plus older and dagger runs |
| `/mnt/data/boeing/outputs` | 428G | other | — | — | not UMI-FT. uid boeing. diffusion sweeps from summer |
| `/mnt/data/umidata/ft/datasets/stringer_scanner_v2` | 350G | unknown | Sep–Oct | ASK | includes two ~20G tars |
| `/mnt/data/boeing/processed_data_layup` | 235G | other | — | — | not UMI-FT |
| `/mnt/data/geodude` | 224G | other | Jun 16 | — | uid 1001 geodude |
| `/mnt/data/umidata/ft/datasets/stringer_data_v2_full` | 205G | unknown | — | ASK | bags 119G plus zarr 82G |
| `/mnt/data/boeing/data` | 189G | other | — | — | not UMI-FT, bag dumps |
| `/home/boeing/ros2_ws` | 172G | other | — | — | legacy layup stack, not UMI-FT |
| `/mnt/data/umidata/ft/datasets/stringer_v4` | 170G | unknown | — | ASK | bags 98G, zarr 65G |
| `/mnt/data/umidata/ft/datasets/stringer_v3.5_base` | 151G | unknown | — | ASK | bags 86G, zarr 53G, tar 3.2G |
| `/mnt/data/umidata/ft/training_outputs/mold_aware` | 123G | Vat | Oct 14 | KEEP | weirdlab has the same 2095201116-byte files |
| `/mnt/data/geodude/geodude_bags` | 107G | other | — | — | uid 1001 |
| `/home/boeing/.cache` | 95G | unknown | — | ASK | shared login. wandb 41G, huggingface 37G, pip 14G |
| `/mnt/data/umidata/ft/data/control_log` | 83G | unknown | Nov 17 | ASK | rollout logs, several 7–14G dagger sessions |
| `/home/boeing/.apptainer/cache` | 82G | unknown | — | ASK | shared image cache on the 99% root disk |
| `/mnt/data/umidata/ft/datasets/stringer_v3.5_dagger_2` | 79G | teammate | Nov | ASK | standard_bc dagger. tars 5.6G + 4.1G + 5.4G |
| `/mnt/data/umidata/ft/training_outputs/local_policies` | 72G | unknown | Aug | ASK | July–August zucchini runs, before the surface_frame split |
| `/mnt/data/umidata/ft/training_outputs/stringer_v3.5` | 69G | teammate | Nov | ASK | `hist/dagger_*` checkpoints |
| `/mnt/data/umidata/ft/datasets/stringer_v3.5_dagger` | 67G | teammate | Nov | ASK | bags 39G, zarr 25G |
| `/mnt/data/umidata/ft/datasets/stringer_v3.5_dagger_3` | 66G | teammate | Nov | ASK | tar 6.5G plus bags 31G |
| `/home/boeing/ros2_ws/outputs` | 62G | other | Jul | — | not UMI-FT |
| `/mnt/data/geodude/bags_old` | 61G | other | — | — | uid 1001 |
| `/mnt/data/umidata/ft/images` | 60G | Vat | Oct–Nov | ASK | nine ~6.6G sif files. See above |
| `/mnt/data/boeing/scans` | 57G | other | Oct | — | not the UMI-FT ft tree. One session is 11G |
| `/mnt/data/umidata/ft/training_outputs/stringer_scanner_v2` | 51G | unknown | Oct | ASK | og / flatten_crop / abs_flatten_crop |
| `/home/boeing/ros2_ws/inference_logs` | 50G | other | Jul | — | not UMI-FT |
| `/mnt/data/umidata/ft/mold_aware_datasets` | 48G | Vat | Nov | mix | odom_corr is 7.9G unique. The tar is the large duplicate |
| `/mnt/data/geodude/processed_data_test` | 27G | other | — | — | includes a 3.1G zarr zip |
| `/home/boeing/umi_ft_ws/UMI-FT` | 21G | teammate | — | ASK | main checkout. 15G is `.git/objects` |
| `/mnt/data/umidata/ft/training_outputs/oct_v35_fixb` | 20G | Vat | Nov 3 | KEEP | epoch-195 files match weirdlab |
| `/mnt/data/umidata/ft/training_outputs/scanner_fixb` | 20G | Oct 28 | Vat | KEEP | no size-matched copy on weirdlab |
| `/mnt/data/umidata/ft/ffs_depth_proto` | 24G | teammate | Nov 21 | ASK | `UMI-FT-ffs_depth` data, defects/A is 10G |
| `/home/boeing/miniforge3` | 34G | unknown | — | ASK | shared env. envs 20G, pkgs 14G |
| `/home/boeing/data/control_log` | 18G | unknown | — | ASK | `$PYRITE_CONTROL_LOG_FOLDERS` |
| `/mnt/data/umidata/ft/training_outputs/oct_v35_h4s_r4vt_causal_foot` | 12G | Vat | Nov 19 | KEEP | four checkpoints; drop only epoch 120 |
| `/mnt/data/umidata/ft/datasets/stringer_data_v1` | 36G | unknown | — | ASK | bags plus zarr |
| `/mnt/data/umidata/ft/datasets/combined_umi_teleop` | 12G | unknown | Aug | ASK | includes a 5.7G tar |
| `/home/boeing/Desktop/Tool_Scan10_22` | 12G | unknown | Oct 22 | ASK | not an UMI-FT path |
| `/home/boeing/ros2_ws/log` | 5.5G | other | — | — | ROS logs |
| `/home/boeing/Downloads` | 2.4G | unknown | — | ASK | |

## Archives over 500 MB

| Path | Size | Owner | mtime | Status | Reason |
|---|---|---|---|---|---|
| `ft/images/umift-train_v0.sif` | 6.6G | Vat | Oct 9 | ASK | old image |
| `ft/images/umift-train_v1.sif` | 6.6G | Vat | Oct 9 | ASK | same bytes as `umift-train.sif` |
| `ft/images/umift-train.sif` | 6.6G | Vat | Oct 9 | ASK | same bytes as v1 |
| `ft/images/umift-train_v2.sif` | 6.6G | Vat | Oct 9 | ASK | same bytes as v3 |
| `ft/images/umift-train_v3.sif` | 6.6G | Vat | Oct 9 | ASK | same bytes as v2 |
| `ft/images/umift-train_v5.sif` | 6.6G | Vat | Oct 18 | ASK | |
| `ft/images/umift-train_v6.sif` | 6.6G | Vat | Oct 18 | ASK | |
| `ft/images/umift-train_v7.sif` | 6.6G | Vat | Nov 9 | CANDIDATE | same bytes as v8 |
| `ft/images/umift-train_v8.sif` | 6.6G | Vat | Nov 9 | KEEP | newest |
| `datasets/stringer_scanner_v2/dataset.tar` | 19G | unknown | Sep 30 | ASK | zarr dir is 274G, so this may already be extracted |
| `datasets/stringer_scanner_v2/dataset_zero_scan.tar` | 19G | unknown | Oct 5 | ASK | zero-scan zarr is 19G |
| `datasets/stringer_v3.5_dagger_3/dagger_123.tar` | 6.5G | teammate | Nov 12 | ASK | |
| `datasets/stringer_v3.5_dagger_2/all_dataset.tar` | 5.6G | teammate | Nov 9 | ASK | |
| `datasets/stringer_v3.5_dagger_2/all_dataset_filtered.tar` | 5.4G | teammate | Nov 12 | ASK | |
| `datasets/combined_umi_teleop/dataset.tar` | 5.7G | unknown | Aug 24 | ASK | |
| `datasets/stringer_v3.5_dagger_2/2_only_dataset.tar` | 4.1G | teammate | Nov 9 | ASK | |
| `.../r4vt_causal_foot.tar` | 3.4G | Vat | Nov 22 | CANDIDATE | extracted |
| `datasets/stringer_v3.5_base/dataset.tar` | 3.2G | unknown | Nov 12 | ASK | |
| `geodude/handover_data_bags/handover_data.zip` | 3.0G | other | Jun 15 | — | uid 1001 |
| `geodude/processed_data_test/bimanual_train.hdf5.zarr.zip` | 3.1G | other | Apr 30 | — | uid 1001 |
| `datasets/teleop_merged/dataset.tar` | 2.5G | unknown | Aug 24 | ASK | |
| `umidata/data/dataset/new_dp_train_data.zarr.zip` | 866M | unknown | May 20 | ASK | not the ft tree |

Checkpoints over 500 MB are almost all ~2.0–4.1 GB policy files under `training_outputs/`. The ones that matter for this audit are in the Vat tables above. The July–August zucchini files (~2.0–4.2G, under `local_policies/` and `training_outputs/training_outputs/`) are the unknown 72G + 28G block.

## Others, for a message

Not Vat's. Not counted in the 9 GB.

- **geodude (uid 1001), 224G** on `/mnt/data/geodude`: `geodude_bags` 107G, `bags_old` 61G, `processed_data_test` 27G, `bags_peg_in_hole_260519` 12G, `handover_data_bags` 9.5G.
- **`/mnt/data/boeing`, 938G**, uid boeing but not the UMI-FT tree: `outputs` 428G, `processed_data_layup` 235G, `data` 189G, `scans` 57G, `inference_logs` 30G.
- **`/home/boeing/ros2_ws`, 172G**: `outputs` 62G, `inference_logs` 50G, `processed_data_layup` 31G.
- **standard_bc / dagger:** `stringer_v3.5_dagger_2` 79G, `stringer_v3.5_dagger` 67G, `stringer_v3.5_dagger_3` 66G, `training_outputs/stringer_v3.5` 69G. Tars inside those are 5–6G each.
- **ffs_depth:** `ft/ffs_depth_proto` 24G, worktree `UMI-FT-ffs_depth` 115M.
- **main checkout:** `UMI-FT/` 21G, of which 15G is git objects.
- **Shared login, root disk is at 26G free:** `.cache` 95G (wandb 41G, huggingface 37G), `.apptainer/cache` 82G, `miniforge3` 34G.

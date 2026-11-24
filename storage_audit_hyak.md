# Hyak storage audit (vtevy)

Read-only. No files were removed or written on Hyak. Scanned only `/gscratch/scrubbed/vtevy` and `/mmfs1/home/vtevy`. Collected 2026-11-23 ~01:15 PT.

`/gscratch/weirdlab/vtevy` is 285 GB, walked below. Other users on that filesystem were not listed.

## Quota

| Location | Usage | Limit | Notes |
|---|---|---|---|
| `/mmfs1/home/vtevy` | 7 GB (70%), 41054 files | 10 GB, 256000 files | `hyakstorage` |
| `/gscratch/scrubbed` (vtevy) | 214 GB, 304397 files | filesystem 149929 / 153600 GB (98%) | `hyakstorage /gscratch/scrubbed`; `du` total 214G |
| `/gscratch/scrubbed` df | 147 T used, 3.3 T free, 98% | 150 T | whole filesystem, not this user |
| `/gscratch/weirdlab` (vtevy) | 285 GB, 17486 files | filesystem 16826 / 17408 GB (97%) | not scanned |
| `/gscratch/cse` | group 18784 / 19456 GB (97%) | no per-user line | not scanned |

No `slurm-*.out` over 100 MB. `logs/fixb` is 69M.

## Totals

| | Size |
|---|---|
| Scrubbed total | 214 G |
| training_outputs | 151 G |
| umidata (zarr copies + one tar) | 62 G |
| git checkouts | ~1.7 G |
| logs | 69 M |
| Home | 7.4 G / 10 G |

Running jobs (squeue, all RUNNING, WorkDir `/gscratch/scrubbed/vtevy`):

| Job | Name | Command checkout | Data it reads |
|---|---|---|---|
| 43465423 | fixb-r4vt-heading-robustab | `UMI-FT-causal_foot` mem128 sbatch | `r4vt_heading_tilt` zarr |
| 43467749 | fixb-r4vt-tilt1 | `UMI-FT-tilt1` | `r4vt_causal_foot` zarr |
| 43472139 | fixb-r4vt-causal-carry-k3000 | `UMI-FT-causal_foot` | `r4vt_causal_foot_carry_k3000` zarr |

`checkpoints.json` on geodude swaps two of these ids (it labels 43465423 as tilt1 and 43467749 as robust-ab). squeue is the source used here. 43472139 is k3000-carry in both.

Scrubbed CANDIDATE rows are about 117 GB (tar, unused zarr copies, causal-foot epochs other than e25/e40/latest, and finished runs whose file sizes match geodude). ASK rows (odom runs, r1rel zarr, tilt-reweight) are not in that number. Weirdlab adds about 235 GB of CANDIDATE, counted in the section below.

## Tree (depth 2–4)

| Path | Size | mtime | Status | Reason |
|---|---|---|---|---|
| `training_outputs/` | 151G | Nov 22 21:05 | mix | see runs below |
| `umidata/` | 62G | Oct 13 dir | mix | full zarr copies, little hardlink sharing (9 × 5.8G plus the old tree fit in 56G) |
| `logs/fixb` | 69M | Nov 11 | KEEP | live jobs write here |
| `UMI-FT-causal_foot` | 279M | Nov 19 | KEEP | sbatch for 43465423 and 43472139 |
| `UMI-FT-tilt1` | 203M | Nov 22 | KEEP | sbatch for 43467749 |
| `UMI-FT-step_mask` | 275M | Nov 17 | CANDIDATE | stale checkout, no running job |
| `UMI-FT-smooth_frame` | 272M | Nov 17 | CANDIDATE | stale checkout |
| `UMI-FT-frame_continuity` | 269M | Nov 16 | CANDIDATE | stale checkout |
| `UMI-FT-tilt_reweight` | 180M | Nov 18 | CANDIDATE | stale checkout |
| `UMI-FT-surface_frames` | 197M | Nov 10 | CANDIDATE | older than the geodude surface_frame checkout; no running job uses it |
| `~/.vscode-server` | 4.8G | — | ASK | 4.7G is `cli`; home is already 70% of 10 GB |
| `~/.cursor-server` | 1.5G | — | ASK | editor cache |
| `~/.cache` | 678M | — | CANDIDATE | huggingface 578M, jax, wandb |
| `~/.local` | 560M | — | ASK | |

## Live runs (KEEP the directory while the job is running)

Each checkpoint file below is ~2.84 GB (over 100 MB). `latest.ckpt` is a second file of the same size, not a symlink.

| Path | Size | mtime | Status | Reason |
|---|---|---|---|---|
| `.../oct_v35_h4s_r4vt_heading_tilt_sc_robustab_mem128/...43465423` | 8.6G | writing | KEEP | job 43465423. Files: epoch 20, 25, latest (3047598572 bytes) |
| `.../oct_v35_h4s_r4vt_tilt1/...43467749` | 8.6G | writing | KEEP | job 43467749. Files: epoch 20, 25, latest (3047511660 bytes) |
| `.../oct_v35_h4s_r4vt_causal_foot_carry_k3000/...43472139` | 8.6G | writing | KEEP | job 43472139. Files: epoch 20, 25, latest (3047655404 bytes) |
| `.../r4vt_heading_tilt/acp_replay_buffer.zarr` | 5.8G | Nov 21 10:55 | KEEP | dataset for 43465423 |
| `.../r4vt_causal_foot/acp_replay_buffer.zarr` | 5.8G | Nov 19 03:55 | KEEP | dataset for 43467749 |
| `.../r4vt_causal_foot_carry_k3000/acp_replay_buffer.zarr` | 5.8G | Nov 22 20:54 | KEEP | the k3000 zarr the running job reads |
| `.../stringer_v3.5_fixb_odom_corr/shared` | 13M | — | KEEP | all three sbatches require `fixb_smooth_frames.npz` and `long_dt_gt_100ms.npz` |

Failed stubs next to the live tilt1 run: `43467751` and `43467490` (1.1M each). CANDIDATE, empty starts.

## Checkpoints over 100 MB

Byte sizes are the file size. "Match" means the geodude file has the same byte count. mtime is Hyak local time.

### Causal foot 43390127 (finished) — dir 52G

Every file is 3047655148 bytes. Geodude has four of them, same size and same timestamps: epoch 25, epoch 40, epoch 120, latest.

| File | mtime | Status | Reason |
|---|---|---|---|
| epoch 25, epoch 40, latest | Nov 19 | KEEP | checkpoints.json (e25, latest) and tomorrow's e40 run. Same bytes already on geodude |
| epoch 120 | Nov 19 14:37 | CANDIDATE | same bytes already on geodude; not in checkpoints.json |
| epoch 0, 5, 10, 15, 20, 30, 35, 60, 80, 100, 140, 160, 180, 199 | Nov 19 | CANDIDATE | finished-run extras, not on geodude. 14 × 2.84 GB ≈ 40 GB |

### Heading 43448962 (finished) — dir 5.7G

| File | Bytes | mtime | Status | Reason |
|---|---|---|---|---|
| epoch 25 | 3047598380 | Nov 21 14:24 | CANDIDATE | same size and mtime as the geodude file |
| epoch 100 | 3047598380 | Nov 21 16:52 | CANDIDATE | same size; geodude `latest.ckpt` has this mtime and this size. Hyak has no e199 |

The truncated e199 (920 MB) is only on geodude. See that report.

### Other finished runs

| Path | Size | Status | Reason |
|---|---|---|---|
| `oct_v35_h4s_r4vt_track_clean/...43351238` | 5.7G | CANDIDATE | epoch 55 and latest are 3048139948 bytes, same as geodude. Rollout default stays on geodude |
| `oct_v35_h4s_r4vt_track_clean_step_mask/...43363017` | 5.7G | CANDIDATE | finished; two ckpts (epoch 30, latest), 3048140140 bytes |
| `oct_v35_h4s_r4vt_smooth_frame/...43335222` | 5.7G | CANDIDATE | finished; epoch 30 and latest, 3048139948 bytes. Local dir is also 2.9–5.7G |
| `oct_v35_h4s_r4vt_frame_continuity_causal/...43308311` | 5.7G | CANDIDATE | finished; epoch 30 and latest. A 1.1M failed sibling `43307939` is also CANDIDATE |
| `oct_v35_h4s_r4vt_heading_tilt_sc_robustab/...43454035` | 5.7G | CANDIDATE | epoch 0 and latest only (3047598572). Superseded by the running mem128 job 43465423 |
| `oct_v35_h4s_r4vt_tilt_reweight/` | 12G | ASK | two finished jobs (43376205, 43376291), epoch 40 and latest each, 3048140204 bytes. Not checked file-by-file against geodude |
| `oct_v35_h4s_odom_r1rel/...43155836` | 8.6G | ASK | e185, e195, latest all 3047964972 bytes. Geodude `latest.ckpt` is that size but the local dir is only 2.9G, so one file was pulled, not three. checkpoints.json uses the geodude latest |
| `oct_v35_h4s_odom_r1rel_valtopk/...43190601` | 5.7G | ASK | e25 and latest are 3047965100, matching the geodude e25 file. Local dir is 2.9G |
| `oct_v35_h4s_odom_r4vt/...43155835` | 8.6G | ASK | e185, e195, latest, 3048139628 bytes. Local dir 2.9G |
| `oct_v35_h4s_odom_r4vt_valtopk/...43190600` | 5.7G | ASK | e30 and latest, 3048139756 bytes. Local dir 2.9G |
| `r4vt_causal_foot_fixedk3000_cam` and `_nocam` | ~3M | CANDIDATE | failed starts (43436610, 43436834, 43436862), no real checkpoint |

## Zarr directories and the tar

| Path | Size | mtime | Status | Reason |
|---|---|---|---|---|
| `.../r4vt_causal_foot.tar` | 3.34G (3580805120 bytes) | Nov 22 15:10 | CANDIDATE | already extracted. Same byte size as the geodude tar. The zarr is present |
| `.../r4vt_causal_foot_fixedk3000` | 5.8G | Nov 20 16:10 | CANDIDATE | superseded by `r4vt_causal_foot_carry_k3000`, which is the zarr job 43472139 reads |
| `.../r4vt_world_fixb_tilt` | 5.8G | Nov 10 | CANDIDATE | not used by the three running jobs |
| `.../r1rel_world` | 5.8G | Nov 10 | ASK | compare_r1rel training is finished; say if that zarr is still needed on Hyak |
| `.../r4vt_smooth_tcp_track_clean` | 5.8G | Nov 17 | CANDIDATE | finished experiment, full copy |
| `.../r4vt_frame_continuity_causal` | 5.9G | Nov 16 | CANDIDATE | finished experiment, full copy |
| `.../r4vt_smooth_frame` | 5.8G | Nov 17 | CANDIDATE | finished experiment, full copy |
| `.../stringer_v3.5_fixb/` (pre odom_corr) | 6.1G | Oct 28 | CANDIDATE | superseded by `stringer_v3.5_fixb_odom_corr` |
| `.../stringer_scanner_v2_fixb/` | 29M | Nov 18 | CANDIDATE | small, old scanner fixb, not in the live jobs |

## `/gscratch/weirdlab/vtevy` (285G)

Owner `vtevy`. No zarrs. No running job uses this tree (the three live jobs read `/gscratch/scrubbed/vtevy`). `du -hd 2` plus every `.ckpt` and `.sif` over 100 MB.

| Path | Size | mtime | Status | Reason |
|---|---|---|---|---|
| `training_outputs/mold_aware_world_surface` | 41G | Oct 13 | CANDIDATE | checkpoints are 2095201116 bytes, same as geodude `mold_aware` |
| `training_outputs/mold_aware_world_world` | 41G | Oct 13 | CANDIDATE | same 2095201116-byte files as geodude |
| `training_outputs/mold_aware_surface_surface` | 41G | Oct 13 | CANDIDATE | same 2095201116-byte files as geodude |
| `training_outputs/oct_v35_fixb_r1_world_world/...42581383` | 7.9G | Oct 29 | CANDIDATE | epoch 195 and latest are 4190341037 bytes, same as geodude |
| `training_outputs/oct_v35_fixb_r2_world_fixb/...42747614` | 7.9G | Nov 3 | CANDIDATE | epoch 195 and latest are 4190341101 bytes, same as geodude |
| `training_outputs/oct_v35_fixb_r1rel_world/...42747616` | 7.9G | Nov 3 | CANDIDATE | epoch 195 and latest are 4190341421 bytes, same as geodude |
| `training_outputs/oct_v35_fixb_r4_world_fixb_tilt/...42747615` | 7.9G | Nov 3 | CANDIDATE | epoch 195 and latest are 4190515885 bytes, same as geodude |
| `training_outputs/oct_v35_fixb_r4_world_fixb_tilt/...42581384` | 7.9G | Oct 30 | CANDIDATE | epoch 195 and latest are 4190515629 bytes, same as geodude `epoch=0195...ckpt.pre_jumpmask` |
| `training_outputs/oct_v35_fixb_r2_world_fixb/...42739516` | 7.9G | Nov 2 | ASK | epoch 115 only. Not on geodude |
| `training_outputs/oct_v35_h4s_fixb_r1_world_world/...42739879` | 8.6G | Nov 2 | ASK | epoch 35 and latest, 4598705949 bytes. Only copy |
| `training_outputs/oct_v35_h4s_fixb_r2_world_fixb/...42739880` | 8.6G | Nov 2 | ASK | epoch 25 and latest, 4598705949 bytes. Only copy |
| `training_outputs/oct_v35_h4s_fixb_r4_world_fixb_tilt/...42739878` | 8.6G | Nov 2 | ASK | epoch 35 and latest, 4598880477 bytes. Only copy |
| `training_outputs/oct_v35_h4s_fixb_r4_world_fixb_tilt/...42787134` | 5.8G | Nov 4 | ASK | epoch 195 and latest, 3048139244 bytes. Only copy |
| `training_outputs/oct_v35_h4s_fixb_r2_world_fixb/...42787133` | 5.8G | Nov 4 | ASK | epoch 195 and latest, 3047964460 bytes. Only copy |
| `training_outputs/oct_v35_h4s_fixb_r1rel_world/...42787132` | 5.8G | Nov 4 | ASK | epoch 195 and latest, 3047964716 bytes. Only copy |
| `training_outputs/r4vt_causal_foot_fixedk3000_cam/...43437517` | 40G | Nov 21 | CANDIDATE | superseded by carry_k3000. Files are 3047654380 bytes. Geodude keeps epoch 25 and latest at that size; the other epochs are only here |
| `training_outputs/r4vt_causal_foot_fixedk3000_nocam/...43437525` | 20G | Nov 21 | CANDIDATE | superseded by carry_k3000. Every file is 1496932127 bytes, which is the nocam size. Geodude keeps epoch 25 and latest |
| `images/umift-train_v3.sif` | 6.7G | Oct 9 | CANDIDATE | 7116107776 bytes, same as geodude v3 |
| `UMI-FT` | 3.4G | — | CANDIDATE | stale checkout. 3.3G is `runs/` |
| `UMI-FT-fixb.bak.20261029133219` | 2.3G | Oct 29 | CANDIDATE | stale backup. 2.1G is `UMIFT_Data` |
| `UMI-FT-fixb` | 357M | — | CANDIDATE | stale checkout. Live jobs use the scrubbed `UMI-FT-causal_foot` and `UMI-FT-tilt1` trees |
| `logs` | 70M | — | CANDIDATE | old fixb and mold_aware logs. The live jobs log under scrubbed |

Weirdlab CANDIDATE total is about 235 GB. The ASK rows are about 50 GB and are not in `cleanup_hyak.sh`.

# LeRobot Teleop → UMIFT Training Zarr

## Goal

Convert LeRobot teleop data into the UMIFT episode-zarr schema, with **all modalities on a 10 Hz grid**, and retune training/inference configs so asymmetric-buffer `down_sample_steps` no longer assume ~15 Hz RGB / ~200 Hz pose / ~28 Hz wrench. Prefer offloading resize into preprocessing. Do not modify original bags.

## Source / sink

| Role | Path |
|---|---|
| Source | `/mnt/data/umidata/ft/teleop/lerobot_dataset_08_03_stringer_bimanual_10hz_vention_frame_obsrot6d_actdelta_desired_combined` |
| Output | `/mnt/data/umidata/ft/teleop_umift_08_03_right/acp_replay_buffer.zarr` |
| Untouched | Original bags under `stringer_data_v2_full/bags/` and raw LeRobot trees |

## Decisions

- **Approach:** dedicated LeRobot → UMIFT converter, then reuse existing virtual-target postprocess (same as stringer).
- **Arm / cameras:** right arm only; `wrist_cam_0` → `rgb_0`; drop center cam and `wrist_cam_1` / left arm.
- **Depth:** keep `depth_0` in schema; fill with zeros (same HxWx3 float16 layout as stringer). Confirmed `wrench_right_0` in stringer is all zeros (not a left duplicate) — mirror that.
- **Image size:** write `rgb_0` / `depth_0` / `ultrawide_0` at **224×224** in converters. Train/infer configs already use 224.
- **Rate:** teleop converter always writes **10 Hz for every stream** (source is locked). Unified→UMIFT can optionally `--resample-10hz` for matching 10 Hz training zarrs; default leaves native multi-rate. No 15 Hz teleop set exists.
- **Train/infer retune (required):** because the stack uses index-based asymmetric buffers, zucchini + env-runner must be updated for 10 Hz data. **Most `down_sample_steps` → 1**; horizons retuned so wall-clock history / action chunks stay reasonable (see below). Inference `raw_time_step_s` and layup `*_downsample_correction` updated for 10 Hz training vs live RTDE rates.
- **Poses:** LeRobot has feedback in `observation.state` and desired as `actdelta`. For UMIFT labels we match stringer:
  - `ts_pose_fb_0` = feedback (rot6d → pose7)
  - `ts_pose_command_0` = feedback (nominal)
  - `ts_pose_virtual_target_0` + `stiffness_0` from **existing** `postprocessing_add_virtual_target_label_umift.py` (fb + wrench)
  - LeRobot desired/`actdelta` is **not** used for VT
- **Query-relative:** unchanged via existing `umift_type_conversions`.
- **Stringer path:** only `convert_unified_t265_wild_to_umift_zarr.py` output side. Always resize to 224; **10 Hz resampling is behind a CLI flag** (default off for backward-compatible wild rates; pass the flag when producing 10 Hz training zarrs). Never rewrite bags or input `*_unified.zarr`.

## Pipeline

```
LeRobot 08_03 combined
  → convert_lerobot_teleop_to_umift_zarr.py
  → acp_replay_buffer.zarr  (geometry, zero depth/ultrawide, 224, 10 Hz all streams)
  → postprocessing_add_virtual_target_label_umift.py  (unchanged)
  → train with retuned umift_zucchini.yaml

Unified wild recordings (future / regen)
  → convert_unified_t265_wild_to_umift_zarr.py  (+224; optional --resample-10hz)
  → same VT postprocess
  → bags / unified.zarr left untouched
```

## Field mapping (teleop converter)

| UMIFT key | Source |
|---|---|
| `rgb_0` | decode `wrist_cam_0`; resize 224×224 |
| `depth_0` | zeros `(T,224,224,3)` float16 |
| `ultrawide_0` | zeros `(T,224,224,3)` uint8 (schema placeholder) |
| `map_to_uw_idx_0` / `map_to_d_idx_0` | identity |
| `ts_pose_fb_0` | right obs pos3 + rot6d → pose7 |
| `ts_pose_command_0` | copy of `ts_pose_fb_0` |
| `wrench_left_0` | right obs wrench6 |
| `wrench_right_0` | zeros |
| `gripper_0` | zeros |
| all `*_time_stamps_0` | same 10 Hz episode clock |
| VT / stiffness | shared postprocess |

Episode packaging: `data/episode_k/` + cumulative `meta/episode_*_len`. Optional coinft-named wrench mirrors.

## Unified → UMIFT converter changes

In `convert_unified_t265_wild_to_umift_zarr.py` only (output side):

1. Resize rgb / depth / ultrawide to 224×224 before write (always).
2. **Optional** `--resample-10hz` (default **false**): when set, resample onto a 10 Hz grid — nearest for images (+ rebuild `map_to_*_idx`); interpolate pose7; nearest wrench; shared timestamps. When unset, keep native multi-rate streams (legacy wild behavior aside from 224 resize).
3. Never modify bags or input unified zarr in place.

Teleop converter always writes 10 Hz (source is already locked); no equivalent off-switch needed there.

## Train / infer config retune (10 Hz)

Target: most `down_sample_steps = 1`. Horizons chosen for wall-clock, not legacy index counts.

### Proposed `umift_zucchini.yaml` (policy rate = 10 Hz)

| Param | Current | Proposed | Wall-clock @ 10 Hz, ds=1 |
|---|---|---|---|
| `sparse_obs_rgb_down_sample_steps` | 8 | **1** | — |
| `sparse_obs_rgb_horizon` | 10 | **16** | 1.6 s history |
| `sparse_obs_depth_*` | same as rgb | **ds=1, h=16** | 1.6 s |
| `sparse_obs_low_dim_down_sample_steps` | 2 | **1** | — |
| `sparse_obs_low_dim_horizon` | 32 | **16** | 1.6 s pose history |
| `sparse_obs_wrench_down_sample_steps` | 1 | **1** | — |
| `sparse_obs_wrench_horizon` | 32 | **16** | 1.6 s wrench history |
| `sparse_action_down_sample_steps` | 2 | **1** | — |
| `sparse_action_horizon` | 32 | **32** | 3.2 s action chunk |
| `sparse_query_frequency_down_sample_steps` | 2 | **1** | query @ 10 Hz |
| `training_duration_per_sparse_query` | 5.0 | keep / align to action span | — |

Rationale: at 10 Hz with `ds=1`, histories are denser than the old sparsely strided buffers, so **horizon 16** (~1.6 s) is enough for rgb/depth/pose/wrench. Action chunk stays **32** (3.2 s). Query every step (10 Hz).

### Inference (`umift_env_runner.py` / layup path)

- `raw_time_step_s`: **1/10** (was 1/15 for layup).
- Live buffers remain fast (RTDE ~500 Hz, cam may still be ~15 Hz). Keep correction factors so **strided sampling matches 10 Hz training wall-clock**:
  - `robot_downsample_correction` ≈ `live_robot_hz / 10`
  - `wrench_downsample_correction` ≈ `live_wrench_hz / 10`
  - depth/rgb corrections updated if live camera ≠ 10 Hz
- Effective: training `ds=1` on 10 Hz data ↔ inference `ds ≈ live_hz/10` on high-rate buffers.

### Touch points

- `PyriteML/diffusion_policy/config/task/umift_zucchini.yaml`
- `train_umift_workspace.yaml` only if it overrides horizons/ds
- `PyriteEnvSuites/env_runners/umift_env_runner.py` (`raw_time_step_s`, corrections)
- Any playback / open-loop val scripts that hardcode old ds or 15 Hz assumptions
- Docs/comments that describe AH3-DS8 physical times

## Validation

- Episode count matches LeRobot meta (47).
- `wrench_right_0` all-zero; `depth_0` all-zero; images `(T,224,224,3)`; all stream lengths equal (10 Hz).
- After VT: stiffness in ~[200,3000]; VT present.
- Smoke `UmiFTDataset` batch with retuned zucchini.
- Inference dry-check: corrected `down_sample_steps` yield intended wall-clock strides.

## Non-goals

- Using LeRobot desired pose as virtual target.
- Train-time LeRobot adapter.
- Editing original bags or source LeRobot trees.
- Changing action dim / query-relative math.
- Collapsing live hardware to 10 Hz publish rates (only training store + sampling math).
- Precomputing action21 / pose9 / float-CHW on disk (resize is the expensive sample-time step; leave the rest to existing dataset init / sample path).

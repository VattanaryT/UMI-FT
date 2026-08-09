# LeRobot Teleop → UMIFT Training Zarr

## Goal

Convert LeRobot teleop data into the same UMIFT episode-zarr schema used by `stringer_data_v2_full`, so it can train a UMIFT policy with query-relative kinematics unchanged. Prefer offloading resize and temporal alignment into preprocessing so the training loader does less work. Do not modify original bags.

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
- **Image size:** write `rgb_0` / `depth_0` / `ultrawide_0` at **224×224** in converters (not 244). Train/infer configs already use 224.
- **Rate / wall-clock:** all teleop streams are **10 Hz** (no 15 Hz teleop set). Sampler horizons are index-based, so stream Hz must match train↔infer. Canonicalize to a **10 Hz policy grid**; inference runs at 10 Hz with the same `down_sample_steps` / horizons.
- **Poses:** LeRobot has feedback pose in `observation.state` and desired as `actdelta`. For UMIFT labels we match stringer:
  - `ts_pose_fb_0` = feedback (rot6d → pose7)
  - `ts_pose_command_0` = feedback (nominal)
  - `ts_pose_virtual_target_0` + `stiffness_0` from **existing** `postprocessing_add_virtual_target_label_umift.py` (fb + wrench)
  - LeRobot desired/`actdelta` is **not** used for VT (keeps one shared postprocess path)
- **Query-relative:** unchanged at train/inference via existing `umift_type_conversions`.
- **Stringer path changes:** only in `convert_unified_t265_wild_to_umift_zarr.py` (resize to 224 + 10 Hz vision/low-dim grid on **output** zarr). Do **not** rewrite bags or source `*_unified.zarr`. Regenerating `acp_replay_buffer.zarr` from unified zarr is optional and out-of-band.

## Pipeline

```
LeRobot 08_03 combined
  → convert_lerobot_teleop_to_umift_zarr.py
  → acp_replay_buffer.zarr  (geometry, zero depth, 224, 10 Hz)
  → postprocessing_add_virtual_target_label_umift.py  (unchanged)
  → train via UmiFTDataset / umift_zucchini.yaml

Unified wild recordings (future / regen)
  → convert_unified_t265_wild_to_umift_zarr.py  (+224 resize, +10 Hz grid)
  → same VT postprocess
  → bags / unified.zarr left untouched
```

## Field mapping (teleop converter)

| UMIFT key | Source |
|---|---|
| `rgb_0` | decode `wrist_cam_0` video; resize 224×224 |
| `depth_0` | zeros `(T,224,224,3)` float16 |
| `ultrawide_0` | copy of `rgb_0` |
| `map_to_uw_idx_0` / `map_to_d_idx_0` | identity |
| `ts_pose_fb_0` | right obs pos3 + rot6d → pose7 `(x,y,z,qw,qx,qy,qz)` |
| `ts_pose_command_0` | copy of `ts_pose_fb_0` |
| `wrench_left_0` | right obs wrench6 |
| `wrench_right_0` | zeros |
| `gripper_0` | zeros |
| `*_time_stamps_0` | LeRobot `timestamp` (episode-relative, 10 Hz) |
| VT / stiffness | filled later by shared postprocess |

Episode packaging: `data/episode_k/` + cumulative `meta/episode_*_len` (stringer style). Optional coinft-named wrench mirrors for tooling compatibility.

## Unified → UMIFT converter changes

In `convert_unified_t265_wild_to_umift_zarr.py` only (output side):

1. Resize rgb / depth / ultrawide to 224×224 before write.
2. Resample onto a 10 Hz policy grid before write: nearest-neighbor for rgb/depth/ultrawide (+ rebuild `map_to_*_idx`); interpolate pose7 for robot streams; nearest wrench samples. Same grid timestamps on all `*_time_stamps_0` so zucchini `ds`/`horizon` match teleop and 10 Hz inference.
3. Never modify bags or input unified zarr in place.

## Training / inference consistency

- Point `task.dataset_path` at the new teleop UMIFT zarr (or a regenerated stringer acp buffer from the updated converter).
- Keep query-relative kinematics.
- Run inference observation buffering at **10 Hz** so `horizon × down_sample_steps` matches training wall-clock.

## Validation

- Episode count matches LeRobot meta (47 for source A).
- `wrench_right_0` all-zero; `depth_0` all-zero; images `(T,224,224,3)`.
- After VT postprocess: `stiffness_0` in expected band (~200–3000); `ts_pose_virtual_target_0` present.
- Smoke: construct `UmiFTDataset` and draw one batch without loader resize dependence on native 480×640.

## Non-goals

- Using LeRobot desired pose as virtual target.
- Train-time LeRobot adapter (no on-the-fly parquet/mp4 in `UmiFTDataset`).
- Editing original bags or source LeRobot trees.
- Changing action dim / query-relative math.

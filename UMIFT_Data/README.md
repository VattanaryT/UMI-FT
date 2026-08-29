# 🍎 UMI-FT Data 🍎

## Setup ⚙️

```bash
cd <path/to/umiFT>
pip install -e .
```

---

## 🍇 Data Collection Instructions

1. **Rename the session name on the iPhone app**
  Use the naming convention:  
   `task-YYMMDD-location-demonstrator-<other-info>`
2. **Time Sync** ⌛
  - Open the time sync page:
  - A browser pop-up will display a QR code for time sync. 
  - Sync the GoPro with the checker on, after seeing the check on gopro, turn the checker off.
  - Mount the gopro on the robot, then select [time] on the iphone to start record a timesync video on iphone.
3. **Record Gripper Calibration Video**
  - select [gripper] on the iphone to start record the gripper calibration video (open and closing the gripper for > 5 times)
4. **Record Demos**
  - Set script constants for coinft data collection
  Set the `$raw_umi_data_dir`, `$session_name` and `$time`in the script, `bash/wired_collect_data.sh`
  - `$time`: the duration in seconds for the data collection, make sure to be longer than demo.
  - `$session_name`: Should match the session name set on the UMI_day app.
  - `$raw_umi_data_dir`: Specifies the data path relative to `<UMIFT_REPO_ROOT>` where the multimodal data is stored.
  - Start the FT recording script:
    ```bash
    conda activate wired_coinft_i2c
    bash bash/wired_collect_data.sh
    ```
    - The FT data will be saved to:  
     `<UMIFT_REPO_ROOT>/data/umift_data/<session_name>/coinft`
  - Record GoPro demos using the "Start Recording" button on the UMI_day iPhone app. 🎥
  - Stop the GoPro recording after the demo is complete.
  - Wait for coinFT data collection to finish.
5. **Export Demos from SD Card**
  Save the data to the following folders:
  - **GoPro Demos**:  
  `<SD_CARD_DIR>/DCIM/100GOPRO` → `<UMIFT_REPO_ROOT>/data/umift_data/<session_name>/DCIM/100GOPRO`
  - **ARKit Demos**:  
  `<SD_CARD_DIR>/UMI_iPhone/export_<SESSION_TIME>` → `<UMIFT_REPO_ROOT>/data/umift_data/<session_name>/UMI_iPhone`

### Recommended Folder Structure

```
<raw_umi_data_dir>/ (:=<UMIFT_REPO_ROOT>/data/umift_data)
├── <session_name>/
│   ├── UMI_iphone/
│   │   ├── export_<YYYY-MM-DD_TIME>/
│   │   │   ├── <TIME>_<side>.json
│   │   │   ├── ...
│   ├── DCIM/
│   │   ├── 100GOPRO/
│   │   │   ├── <GOPRO_FILE>.MP4
│   │   │   ├──  ...
│   ├── coinft/
│   │   ├── YYYY-MM-DD/
│   │   │   ├── <COINFT_FILE>_LF.csv
│   │   │   ├── <COINFT_FILE>_RF.csv
│   │   │   ├── ...
```

---

## 🔬 Data Postprocessing Instructions

### 1. Set Constants in Postprocessing Scripts

Set the `$raw_umi_data_dir` and `$session_name` in the following scripts for your collected data session:
    - `bash/data_post_process_gopro_iphone.sh`
    - `bash/data_post_process_multimodal.sh`

- `$session_name`: Should match the session name set on the UMI_day app.
- `$raw_umi_data_dir`: Specifies the data path relative to `<UMIFT_REPO_ROOT>` where the multimodal data is stored.

### 2. Folder Assumptions 📁

```
<raw_umi_data_dir>/ (:=<UMIFT_REPO_ROOT>/data/umift_data)
├── <session_name>/
│   ├── UMI_iphone/
│   │   ├── export_<YYYY-MM-DD_TIME>/
│   │   │   ├── <TIME>_<side>.json
│   │   │   ├── ...
│   ├── DCIM/
│   │   ├── 100GOPRO/
│   │   │   ├── <GOPRO_FILE>.MP4
│   │   │   ├──  ...
│   ├── coinft/
│   │   ├── YYYY-MM-DD/
│   │   │   ├── <COINFT_FILE>_LF.csv
│   │   │   ├── <COINFT_FILE>_RF.csv
│   │   │   ├── ...
│   ├── processed_data/ (output from data processing)
│   │   ├── gopro_iphone/
│   │   ├── all/
```

---

### 3. Run Postprocessing Commands

#### Postprocess the GoPro + iPhone Data

```bash
conda activate umi_day
bash bash/data_post_process_gopro_iphone.sh
```

#### Postprocess All Other Data + Time Sync + Visualization

```bash
conda activate umift
bash bash/data_post_process_multimodal.sh
```

### 4. Output Zarr Data Format 📦

Example zarr data tree for two demos:

```
data.tree()
/
 ├── data
 │   ├── episode_0
 │   │   ├── gripper_0 (580, 1) float64
 │   │   ├── gripper_time_stamps_0 (580, 1) float64
 │   │   ├── rgb_0 (580, 202, 270, 3) uint8
 │   │   ├── rgb_time_stamps_0 (580, 1) float64
 │   │   ├── robot_time_stamps_0 (580, 1) float64
 │   │   ├── ts_pose_fb_0 (580, 7) float64
 │   │   ├── wrench_concat_0 (3600, 12) float64
 │   │   ├── wrench_concat_coinft_0 (3600, 12) float64
 │   │   ├── wrench_left_0 (3600, 6) float64
 │   │   ├── wrench_left_coinft_0 (3600, 6) float64
 │   │   ├── wrench_right_0 (3600, 6) float64
 │   │   ├── wrench_right_coinft_0 (3600, 6) float64
 │   │   ├── wrench_time_stamps_left_0 (3600, 1) float64
 │   │   └── wrench_time_stamps_right_0 (3600, 1) float64
 │   └── episode_1
 │       ├── gripper_0 (903, 1) float64
 │       ├── gripper_time_stamps_0 (903, 1) float64
 │       ├── rgb_0 (903, 202, 270, 3) uint8
 │       ├── rgb_time_stamps_0 (903, 1) float64
 │       ├── robot_time_stamps_0 (903, 1) float64
 │       ├── ts_pose_fb_0 (903, 7) float64
 │       ├── wrench_concat_0 (5404, 12) float64
 │       ├── wrench_concat_coinft_0 (5404, 12) float64
 │       ├── wrench_left_0 (5404, 6) float64
 │       ├── wrench_left_coinft_0 (5404, 6) float64
 │       ├── wrench_right_0 (5404, 6) float64
 │       ├── wrench_right_coinft_0 (5404, 6) float64
 │       ├── wrench_time_stamps_left_0 (5404, 1) float64
 │       └── wrench_time_stamps_right_0 (5404, 1) float64
 └── meta
     ├── episode_gripper0_len (2,) int64
     ├── episode_rgb0_len (2,) int64
     ├── episode_robot0_len (2,) int64
     └── episode_wrench0_len (2,) int64
```

Wrench data order assumption: [Fx, Fy, Fz, Mx, My, Mz]

Wrench body frame and world frame transformation reference from [Modern Robotics, Lynch & Park](https://hades.mech.northwestern.edu/images/7/7f/MR.pdf#page=126.68) 

---

## T265 Wild Dataset Adapter

Use this path when data comes from:

- T265 odometry (pose)
- single wrist FT sensor (no coinFT pair)
- D405 color/depth camera
- no gripper stream

### 1) Convert unified recordings to UMIFT schema

```bash
python UMIFT_Data/scripts/convert_unified_t265_wild_to_umift_zarr.py \
  --input /path/to/straight_convex_data/zarr \
  --output /path/to/output/acp_replay_buffer_gripper_t265_wild.zarr
```

Notes:

- `--input` can be either one `*_unified.zarr` or a directory containing many.
- Output is episode-structured (`data/episode_*`) with UMIFT-compatible keys.
- Adapter behavior:
  - `ts_pose_fb_0` from T265 odom (`[x, y, z, qw, qx, qy, qz]`)
  - `gripper_0` and `gripper_time_stamps_0` synthesized as zeros
  - single wrist FT mapped to `wrench_left_0`, right side filled with zeros
  - `ultrawide_0` filled with zeros (schema placeholder; not used in zucchini training)
  - `depth_0` converted to clipped float16 meters, expanded to 3 channels

### 2) Run virtual-target/stiffness postprocessing

```bash
python PyriteUtility/PyriteUtility/data_pipeline/postprocessing_add_virtual_target_label_umift.py
```

Before running, set the config constants at the top of
`PyriteUtility/PyriteUtility/data_pipeline/postprocessing_add_virtual_target_label_umift.py`:

- `dataset_path` to your converted zarr output
- `apply_mask = False` for D405-converted data
- `clip_depth = True` (keep `depth_clip` as needed, e.g. `0.5`)
- `flag_real = True`

### 3) Validate expected outputs

After postprocessing, each episode should contain:

- `ts_pose_virtual_target_0`
- `stiffness_0`
- `ts_pose_command_0` (auto-filled from `ts_pose_fb_0` if missing)

Also verify compatibility keys exist:

- `rgb_0`, `ultrawide_0`, `depth_0`
- `wrench_left_0`, `wrench_right_0`, `wrench_concat_0`
- `wrench_left_coinft_0`, `wrench_right_coinft_0`, `wrench_concat_coinft_0`
- timestamp keys and `map_to_uw_idx_0` / `map_to_d_idx_0`

### 4) Replay Recorded Virtual-Target Actions (Inference-Path Validation)

Use this to replay recorded `ts_pose_virtual_target_0` + `stiffness_0` from a
postprocessed dataset through the same stiffness scheduling path used during
UMIFT inference (without running policy inference).

Script:
- `PyriteEnvSuites/scripts/run_virtual_target_dataset_playback.py`

Required keys in each replayed episode:
- `ts_pose_command_0`
- `ts_pose_virtual_target_0`
- `stiffness_0`
- `robot_time_stamps_0`

Dry-run (offline, no hardware scheduling):
```bash
PYTHONPATH=/path/to/UMI-FT:/path/to/UMI-FT/PyriteUtility \
python PyriteEnvSuites/scripts/run_virtual_target_dataset_playback.py \
  --dataset_path /path/to/acp_replay_buffer.zarr \
  --episode episode_0 \
  --plot_trajectory \
  --dry_run
```

Hardware replay:
```bash
PYTHONPATH=/path/to/UMI-FT:/path/to/UMI-FT/PyriteUtility \
python PyriteEnvSuites/scripts/run_virtual_target_dataset_playback.py \
  --dataset_path /path/to/acp_replay_buffer.zarr \
  --episode episode_0 \
  --hardware_config_path /path/to/hardware.yaml \
  --execution_horizon 12 \
  --slow_down_factor 1.0
```

Notes:
- `--test_nominal_target` replays `ts_pose_command_0` instead of virtual target.
- `--test_nominal_target_stiffness` (>0) overrides translational stiffness.
- `--plot_trajectory` shows a matplotlib 3D preview of command, virtual-target,
  and replay-selected trajectory before execution.
- If your converted dataset has not yet been postprocessed, this script will
  fail schema validation until `ts_pose_virtual_target_0` and `stiffness_0`
  are generated.

### 5) Open-Loop Val Policy Export + Replay

Use this to evaluate a trained policy open-loop on the same validation episodes
used during training, record decoded virtual-target actions, then replay them
with the virtual-target playback script.

Script:
- `PyriteEnvSuites/scripts/run_policy_open_loop_val_export.py`

The export script:
1. Loads the checkpoint (including Hydra cfg `val_ratio` / `seed`)
2. Builds the validation split via `UmiFTDataset.get_validation_dataset()`
3. Runs MPC-style open-loop inference on recorded val observations
4. Writes replay-compatible episodes (`episode_<id>_policy`) with:
   - `ts_pose_command_0`
   - `ts_pose_virtual_target_0`
   - `stiffness_0`
   - `robot_time_stamps_0`
   - optional `gripper_0`

Export only:
```bash
PYTHONPATH=/path/to/UMI-FT:/path/to/UMI-FT/PyriteUtility:/path/to/UMI-FT/PyriteML:/path/to/UMI-FT/PyriteConfig \
python PyriteEnvSuites/scripts/run_policy_open_loop_val_export.py \
  --checkpoint /path/to/run_or_latest.ckpt \
  --device cuda \
  --sparse_execution_horizon 4
```

Export a subset of val episodes and dry-run playback:
```bash
PYTHONPATH=/path/to/UMI-FT:/path/to/UMI-FT/PyriteUtility:/path/to/UMI-FT/PyriteML:/path/to/UMI-FT/PyriteConfig \
python PyriteEnvSuites/scripts/run_policy_open_loop_val_export.py \
  --checkpoint /path/to/run_or_latest.ckpt \
  --episodes episode_3 \
  --run_playback \
  --playback_dry_run
```

Or export first, then replay manually:
```bash
PYTHONPATH=/path/to/UMI-FT:/path/to/UMI-FT/PyriteUtility \
python PyriteEnvSuites/scripts/run_virtual_target_dataset_playback.py \
  --dataset_path /path/to/policy_open_loop_val.zarr \
  --episode episode_3_policy \
  --plot_trajectory \
  --dry_run
```

Notes:
- Validation episode selection matches training when using the checkpoint cfg
  `seed` / `val_ratio` and the same source zarr (`--dataset_path` optional override).
- Default output is `<checkpoint_dir>/policy_open_loop_val.zarr`.
- Per-episode action MSE metrics are written next to the zarr as
  `policy_open_loop_val_metrics.json`.
- `--dry_run_export` validates rollout/decode without writing zarr.


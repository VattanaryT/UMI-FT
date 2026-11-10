# Surface-frames checkout

Recorded 2026-11-05 from the code in this worktree. Commands below were run except where marked.

## What this checkout is

Fix B front of the carbon-fiber layup stack. A diffusion policy (DDIM, epsilon prediction) is trained on handheld demos. Actions for r2/r3/r4 are tangential speed and normal offset on a mold mesh, not fixture-frame xyz. r1 keeps fixture-frame poses and is the control.

Supports dataset build, training, offline checks, and real-robot deployment. No layup simulator. “Dry run” still commands the arm.

| | Starts | Hardware | Writes |
|---|---|---|---|
| `UMIFT_Data/surface_frames_scripts/fixb_build_mold_field.py`, `fixb_build_datasets.py`, `fixb_build_r4_dataset.py`, `fixb_oct_v35_prepare.py` | CPU label jobs | no | zarrs under `/mnt/data/umidata/ft/mold_aware_datasets/` |
| `PyriteML/train.py` via `fixb_train_*klone.sbatch` | one GPU, `accelerate` | no | checkpoints under `PYRITE_CHECKPOINT_FOLDERS` |
| `sanity_mold_aware_ep5.py` | CPU obs + decode check | no | nothing |
| `PyriteEnvSuites/env_runners/umift_env_runner.py` | right UR5e RTDE, wrist FT, D405, ManipServer admittance | yes | control logs, rollout video, optional scan |

Training entry: `accelerate launch train.py --config-name=train_umift_workspace` with `hydra.searchpath` pointed at `UMIFT_Data/surface_frames_scripts/config` and `task=<yaml>`. Workspace default task is `umift_zucchini`; every Fix B sbatch overrides it. Sbatch files: `fixb_train_klone.sbatch` (old `stringer_data_v2_full_fixb`), `fixb_train_scanner_klone.sbatch`, `fixb_train_oct_v35_klone.sbatch`, `fixb_train_oct_v35_h4s_klone.sbatch`.

Deployment entry: `python PyriteEnvSuites/env_runners/umift_env_runner.py` after `source /home/boeing/umi_ft_ws/worktrees/activate.sh UMI-FT-surface_frames`. Config is the two dicts in `main()`, not CLI args. `~/.bashrc` pins `PYTHONPATH` at the other checkout; without `activate.sh` the runner loads the wrong code.

Robot drivers are not in this repo. UR RTDE and admittance live in `/home/boeing/umi_ft_ws/hardware_interfaces` (`ur5e_wrist_ft.yaml`). Wrist camera is pyrealsense2 inside `ManipServerLayupEnv`. Keyence scans are `/home/boeing/keyence_scan_ws` and are forced off when the checkpoint has `robot0_mold_geom`.

Safe example, run 2026-11-05, no GPU, no robot, exit 0, printed `SANITY PASS`:

```bash
source /home/boeing/umi_ft_ws/worktrees/activate.sh UMI-FT-surface_frames
~/miniforge3/envs/pyrite/bin/python UMIFT_Data/surface_frames_scripts/sanity_mold_aware_ep5.py
```

Reads `stringer_scanner_v2_fixb` episode 5, `shared/fixb_field.npz`, and the r2 epoch-0035 open-loop export `episode_5.npz`. Max obs error ~1e-11 m. Decode error 0.

## Baseline

| | |
|---|---|
| Commit | `7ebd7129a288aade29bf302478595c77121721aa` (`surface_frame`, 2026-11-04, “Point inference at the 4s r1rel checkpoint”) |
| Tree | dirty. Runner `ckpt_path` on disk is **not** the committed one (below). `README_surface_frames.md` is new. |
| Python | `/home/boeing/miniforge3/envs/pyrite/bin/python` 3.12.13. `PyriteConfig` resolved to this worktree. |
| Dataset touched by the baseline | `/mnt/data/umidata/ft/mold_aware_datasets/stringer_scanner_v2_fixb/` (read only) |
| Checkpoint | none. Sanity does not load a policy. |
| Hardware config | none |

Committed runner checkpoint (`HEAD`): `/oct_v35_h4s/r1rel_world/epoch=0025-train_loss=0.029.ckpt`.

Uncommitted runner checkpoint (what a launch today loads): `/oct_v35_h4s/r4_world_fixb_tilt/epoch=0195-train_loss=0.011.ckpt`.

Both have `mold_aware_dry_run: False` and `mold_aware_abort_on_safety: True`. `README_mold_aware_rollout.md` still says dry-run defaults True and the checkpoint is September r2. The runner file wins.

Tests, same python, `PYTHONPATH` includes `UMIFT_Data/surface_frames_scripts`, 14 passed in 3.05 s:

`UMIFT_Data/tests/test_pose_jump_mask.py`, `UMIFT_Data/tests/test_mold_calib_check.py`, `PyriteML/tests/test_drop_depth_inputs.py`.

Those tests cover the pose-jump mask, mold-calib geometry, and depth-key deletion. They do not check policy outputs or closed-loop contact.

Raw zarrs and checkpoints were not modified.

## Training path

`train.py` → `TrainDiffusionUnetImageWorkspace.run`.

| Stage | Owner | In → out | State |
|---|---|---|---|
| Config | Hydra. Task yaml from `surface_frames_scripts/config/task/`. `shape_meta` is what the checkpoint later stores. | CLI `task=` overrides the workspace default | writes `.hydra/` in the run dir |
| Dataset | `hydra.utils.instantiate(cfg.task.dataset)` | zarr → in-memory copy | does not write the zarr |
| Window | `SequenceSamplerUmiFT.sample_sequence` | RGB index, stride `sparse_query_frequency_down_sample_steps: 2` | drops scan pauses, action-timestamp gaps (`action_gap_threshold_mult: 3`), `\|pos\| > 2 m` |
| Sample | `obs_to_obs_sample`, `action21_to_action_sample` | pose7 → pose9 (xyz + rot6d). `pose_frames` applied here | — |
| Batch | DataLoader | `{"obs": {"sparse": ...}, "action": {"sparse": ...}}` | r4 also `action_loss_weight` |
| Normalize | `DiffusionUnetTimmMod1Policy.compute_loss` | `sparse_normalizer` fit in `UmiFTDataset.get_normalizer`, then October subclasses overwrite from a pickle | `sparse_normalizer.pkl` plus the same stats inside the ckpt |
| Policy | `TimmObsEncoderWithForce` → `ConditionalUnet1D` | normalized obs embedding conditions the denoiser | — |
| Loss | MSE vs the noise (`prediction_type: epsilon`). r4 zeros loss on dims 21:27 after a field-frame jump > 30° inside the chunk | scalar | — |
| Step | AdamW 3e-4, cosine, 2000 warmup | — | optimizer state in the ckpt |
| Save | `BaseWorkspace.save_checkpoint` | top-k by `train_loss`, plus `latest.ckpt` | under the Hydra run dir (`/checkpoints/${RUN_TITLE}` on Hyak) |

There is no `n_obs_steps` / `n_action_steps` on this policy. Horizons are per key in `shape_meta.sample`.

Scheduler in `train_umift_workspace.yaml`: `DDIMScheduler`, 50 train timesteps, `squaredcos_cap_v2`, `beta` 1e-4 → 0.02, `prediction_type: epsilon`, `num_inference_steps: 16`. Training adds `input_pertub: 0.1` on the sampled noise. Inference does not.

Variants (yaml `name`, dataset dir):

| Run | Obs | Action | Dataset class |
|---|---|---|---|
| r1 | fixture TCP, absolute | fixture pose9 + real virtual target | `UmiFTDataset` or October `UmiFTDatasetOctNorm` role `r1` |
| r1rel | same zarr as October r1, poses relative to the newest TCP | same, both pose9 blocks relative | `UmiFTDatasetOctNorm` role `r1rel` |
| r2 | fixture TCP, absolute | `[v_t1, v_t2, delta_n]` in dims 0:3, virtual target unused | October role `r2` |
| r3 | pose stored already in the surface frame, plus `robot0_mold_curv` | same Fix B label as r2 | `UmiFTDatasetFixB` |
| r4 | same obs as r2 | r2 action plus `ΔR` rot6d at 21:27 | `UmiFTDatasetR4` / `UmiFTDatasetOctR4` |
| h4s | same, longer windows | action horizon 64 | same classes, different normalizer pickle |

No October r3 yaml. Scanner and October tasks do not feed `scan_heatmap`.

Short horizons (`umift_mold_aware.yaml`): rgb/depth 5 stride 4; eef and mold 5 stride 16; wrench 32 stride 1; action 24 stride 1. h4s: 16 / 16 / 51 / 51 / 118 / 64, same strides. Action stride 1 is one RGB frame (~15 Hz), not one 200 Hz robot sample.

`pose_frames` on these yamls is `absolute` except r1rel (`relative`). Absolute means “as stored.” Training passes no fixture transform, so absolute actions are not rebased. Relative actions are `inv(T_newest) @ T`.

## Deployment path

`umift_env_runner.main`. Process start calls `set_high_level_maintain_position` (hold the current Cartesian pose, clear the waypoint buffer). It does not home. Menu `d` later moves to a dataset start pose. Menu `r` returns to the pose captured when the camera window was confirmed. Both are motion.

| Stage | Owner | In → out | State |
|---|---|---|---|
| Sensors | `ManipServerLayupEnv.get_observation_from_buffer` | RTDE pose @ 500 Hz, UR wrist wrench, D405 RGB. `wrench_right` is zeros. Depth is not opened while `set_depth_to_zero` is true | ring buffers |
| History | runner builds query lengths from `(horizon-1)*stride+1`, then `ModelPredictiveController.sample_sparse_obs` | robot strides × `500/200`, wrench × `500/28`, then × `slow_down_factor` (2) because `scale_obs_windows_by_slow_down` is true | no cross-stream interpolation. Newest sample is index −1 |
| Fixture map | `pose7_batch_to_fixture` with `T_fixture_from_right_base` | live TCP is UR base; mold queries need fixture | base pose is written back before eef packing, then `sparse_obs_to_obs_sample` applies the extrinsics again for absolute pose obs |
| Mold obs | `MoldAwareRuntime.fill_obs_raw` if `robot0_mold_geom` is in the checkpoint `shape_meta` | tip → signed distance, normal, field rot6d, optional curvature | `_prev_foot`, `_prev_n` |
| Preprocess | `raw_to_obs_inference` → `sparse_obs_to_obs_sample` | pose7 → pose9, wrench, mold channels. Depth filled with zeros | — |
| Policy | `compute_one_horizon_action` → `predict_action` | normalized obs → full action chunk | `policy.eval()` in `load_policy`; `torch.no_grad()` in the MPC call |
| Unnormalize | `sparse_normalizer["action"].unnormalize` inside `predict_action` | stats are the ones stored in the checkpoint, not a refit | `last_raw_action` set here, before Fix B decode |
| Decode | r1: `action21_postprocess` (absolute pose9, base = `inv(T_fixture_from_right_base)`). r2/r3/r4: that result is discarded; `decode_chunk` integrates | see below | — |
| Queue | `PoseChunkEnsembler` if `enable_ensemble_chunking` (true, decay 0.01) | full chunk pushed; `sparse_execution_horizon` steps popped | new ensembler object each episode |
| Stiffness | runner, tool frame | virtual target − nominal → 6×6. Defaults 5000 N/m on the two non-compliance axes, 100 on rotation | — |
| Robot | `env.schedule_controls` → `ManipServer::schedule_waypoints` | pose7 in the UR base, timestamps in ms. Admittance (`dt` 0.002, servoL) consumes the buffer | replaces buffer entries newer than the first future waypoint |

Run type (`infer_run_type`, first match): action dim 27 or name contains `R4` → r4; `R3` or `robot0_mold_curv` → r3; `R2` → r2; `R1` → r1. The h4s r4 task name is `OCT-V35-FIXB-R4-H4S`.

r2/r3 position: `integrate_fixb` in the fixture frame. `x_s(0)` is the closest point of the measured tip. `x_s(k+1)` is the closest point of `x_s + R[:, :2] @ v_par * dt`. `x_d = x_s + delta_n * n`. `dt` is `action_down_sample_steps * (1/15) * slow_down_factor` (2), constant for the chunk. Field normals point outward; positive `delta_n` is off the mold. r2/r3 heading is fixed for the chunk: `R(x_s) @ R_off` or the 180° in-plane flip, whichever is closer to the measured tool (`R_off` from `fixb_meta.json`). r4 heading is `R(x_s) @ ΔR` from dims 21:27. Nominal target is `x_d`. Virtual target is `x_d − 0.002 * n`. Then fixture → UR base.

r3 training stored `ts_pose_fb` in the surface frame (`p_loc = R_fᵀ (p − x_s)`, `R_loc = R_fᵀ R_tool`). The runner does not do that rewrite. A live r3 checkpoint would see fixture-frame TCP plus curvature. r3 was not the checkpoint on disk.

`mold_aware_dry_run: True` overwrites every `delta_n` with +0.015 m and skips the mesh-distance hard stop. Speed scaling and the step flag still run. r4 still applies predicted `ΔR`. It still calls `schedule_controls`. That is a hardware hover, not a command-disabled dry run. The file on disk has it False.

Safety when `apply_safety` (live default) in `SafetyLimits`:

| Limit | Default | Effect |
|---|---|---|
| tangential speed | 0.25 m/s | scales `v_par` down |
| step | 0.025 m | flag only, pose unchanged |
| press | `delta_n >= −0.005 m` | clamp, unless dry run |
| mesh distance | 0.10 m | hard stop, skipped in dry run |
| normal flip | on | hard stop, before the policy |
| tilt 45°, rot rate 20°/step, mesh-jump hold 30° | flags exist | **off** (`enable_*` False) |
| `max_force_n` 80, `max_foot_hop_m` 0.020 | stored | not read |

`mold_aware_abort_on_safety` breaks the episode before `schedule_controls` of that chunk. Waypoints already queued keep running. The abort does not clear the C++ buffer. `env.reset()` is empty, so a new episode also does not clear it. `mold_runtime.reset()` and a new ensembler do run. `last_raw_action` is only overwritten by the next `predict_action`. `clear_cmd_buffer` runs at process start, on RTDE resume, and on C++ keyboard takeover (the layup runner does not enable takeover).

First-pose gate (25 mm / 10°) is skipped for r2/r3/r4.

## Horizons, noise, eval mode

Checkpoint `shape_meta` sets the predicted length (`sparse_action_horizon`, 24 or 64). The runner executes `sparse_execution_horizon` steps of that chunk, then replans. Config value is 6. If the loaded horizon is ≥ 64 and the config value is ≤ 16, the runner raises execution to 32. The h4s r4 checkpoint takes that branch: predict 64, execute 32, ensemble the overlap. Wall time between loop starts is `(exec − 1) * (1/15) * slow_down_factor`. Step 0 is retimed to `current_hardware_time_s` immediately before it is scheduled.

Train diffusion steps: 50. Inference steps: 16, from `cfg.policy.num_inference_steps` applied in `load_policy`. Prediction target epsilon. The whole action is denoised (`cond_mask` all false).

Noise is `torch.randn(...)` in `conditional_sample`. `predict_action` does not pass `generator`, so it is the global RNG. Fix that seed (and pass a generator) before comparing two implementations. Nothing in the runner does.

`load_policy` calls `policy.eval()`. The MPC wraps `predict_action` in `torch.no_grad()`. Unnormalization uses the checkpoint normalizer. October r4 pins action dims 0:21 to the shared r2 pickle and leaves 21:27 at scale 1, offset 0; that pin is already inside the checkpoint.

## Data contracts

Pose7 everywhere on this path is `[x, y, z, qw, qx, qy, qz]` (`spatial_utilities.pose7_to_SE3`). `fixb_labels.pose7_to_Rp` reorders to scipy xyzw. Rot6d is the first two rows of R. `T_fixture_from_right_base` maps UR-base coordinates into the fixture. `R_f` columns are `[t1, t2, n]` in the fixture, so `R_f` maps surface coordinates to fixture. Gravity in the workcell convention is `[0, +9.8, 0]`.

Label code (`fixb_labels.py`): for each robot sample, `v_par = Bᵀ (c_next − c) / dt` with `B = [t1 t2]`, `dt` = time until the next RGB-aligned robot sample. Units m/s. Clipped to ±1. `dt ≤ 1e-6` → 0. `delta_n = (p_smooth − x_s) · n` in meters, the offset at that sample, not a per-step delta. Default mode is foot velocity and feedback `delta_n` (Savitzky–Golay, 0.1 s).

Packed into `ts_pose_command_0` as `[v_t1, v_t2, delta_n, qw=1, 0, 0, 0]`. After `raw_to_action21` the policy vector is:

| Dims | r2 / r3 | r1 / r1rel | r4 |
|---|---|---|---|
| 0:3 | `v_t1`, `v_t2` (m/s), `delta_n` (m) | fixture xyz (m), or relative xyz for r1rel | same as r2 |
| 3:9 | identity rot6d | command rot6d | identity rot6d |
| 9:18 | virtual target at 0, identity rot6d | wrench-derived virtual target | same as r2 |
| 18 | stiffness scalar, about 200–3000 N/m from the VT labeller | same | same |
| 19:21 | gripper 0, grasp force `(−Fx_left + Fx_right)/2` | same | same |
| 21:27 | absent | absent | `ΔR = R(x_s)ᵀ R_tool` as rot6d: first two rows of that matrix, dimensionless, not degrees. Decode `R_d = R(x_s) @ ΔR` |

Observations (`shape_meta`, batch axis added by the loader; inference adds a leading 1):

| Key | Shape into the net | Units | Frame | Norm |
|---|---|---|---|---|
| `rgb_0` | `(H, 3, 224, 224)` float32, from HWC uint8 | `/255` → `[0, 1]`. No channel swap in the converter | D405, as stored | identity, then ImageNet inside the encoder (`imagenet_norm: True`) |
| `depth_0` | `(H, 3, 224, 224)` float32, 3 identical channels | meters, clip 0.5, then `/0.5`. h4s yamls set `drop_depth_inputs: true`. Runner `set_depth_to_zero` stays true for mold checkpoints, so live depth is zeros | camera | identity |
| `robot0_eef_pos` | `(H, 3)` | m | fixture TCP, except r3 zarr (surface) and r1rel (newest TCP) | min/max → `[−1, 1]` |
| `robot0_eef_rot_axis_angle` | `(H, 6)` | rot6d, not axis-angle | same as position | identity |
| `robot0_eef_wrench_left` | `(H, 6)` | N and N·m, TCP. Right wrench is zeros | TCP | range (`normalize_wrench: True`) |
| `robot0_mold_geom` | `(H, 10)` | `[sd_m, n_xyz, R_field rot6d]`. `sd = (p − x_s)·n` | fixture. Not rewritten for r3 | range, all 10 |
| `robot0_mold_curv` | `(H, 4)`, r3 only | `[k1, k2, a cos2θ, a sin2θ]`, 1/m, convex > 0 | tangent basis of `R_f` | range |

Normalization forward is `x * scale + offset`, reverse `(x − offset) / scale`. Online fit is range on position-like and wrench channels, identity on rot6d and on r4 dims 21:27. October then overwrites from a pickle: r2 all 21 dims, r4 dims 0:21, r1 only 18:21, r1rel action plus `eef_pos` and the rot6d obs (percentile clip). h4s uses `assets/oct_v35_h4s_r2r4_action_normalizer_3b0863c14453.pkl`. Short October runs use `oct_v35_r2r4_action_normalizer_1ddfa2b4331f.pkl`. The runner never reads those files; it uses the checkpoint copy.

Validity: sampler drops a window if `|eef xyz|` or `|action[:, :3]|` > 2 m. On r2/r4 that second test is `|v|` or `|delta_n|`, and clipped `v_par` cannot trip it. Depth preprocess asserts float16 in `[0, 0.5]` and does not replace NaNs. Missing `fixb_delta_R_rot6d_0` leaves the action at 21; r4 decode raises if the vector is shorter than 27. No max-age check on sensors beyond the history window. Stale C++ waypoints are the failure mode, not a dropped observation.

Dataset defaults:

| Yaml family | Zarr |
|---|---|
| `umift_fixb_r*` | `.../stringer_data_v2_full_fixb/{r1_world_world,r2_world_fixb,r3_surface_fixb}/` |
| `umift_scanner_fixb_r*` | `.../stringer_scanner_v2_fixb/{r1_world_world,r2_world_fixb,r3_surface_fixb,r4_world_fixb_tilt}/` |
| `umift_oct_fixb_*` and `*_h4s` | `/mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb/{r1_world_world,r2_world_fixb,r4_world_fixb_tilt}/acp_replay_buffer.zarr` (r1rel reads the r1 zarr) |

October source, before the mold copy: `/mnt/data/umidata/ft/datasets/stringer_v3.5/acp_replay_buffer.zarr`. The Fix B copy used for training has 50 episodes (about 60 s median, fixture frame, 5 s start pad, no scan). The source zarr on disk now has 68. RGB and depth gaps longer than 0.5 s are on episodes 13, 25, 29, and 42 (5 gaps, max 0.62 s) in `stringer_v3.5_fixb/audit/audit.json`. Robot gaps stay under 47 ms. About 93% of tag samples see zero tags, so most of the pose is T265 fill.

Field the runner loads: `.../stringer_scanner_v2_fixb/shared/fixb_field.npz` (same bytes as `stringer_v3.5_fixb/shared/fixb_field.npz`). Valid only while the mold is in the pose that yaml was calibrated for (`layup_workcell_extrinsics.yaml`, 2026-10-08).

## What has been run

| Stage | Status |
|---|---|
| Episode inspection | `python UMIFT_Data/surface_frames_scripts/view_oct.py` then http://127.0.0.1:8765. Original and mold on one page. |
| Train until the objective drops | Not run here. Hyak checkpoints already exist, including the h4s r4 epoch-195 file named in the dirty runner. Diffusion loss is not expected to hit zero. |
| Reload in a fresh process with fixed noise | Not run. |
| Offline decode | This session: `sanity_mold_aware_ep5.py` on scanner r2 episode 5, `SANITY PASS`, decode error 0 against the stored export. That is recorded-action agreement, not closed-loop success. 2026-10-21 log (epoch-35 scanner open loop, not re-run): r1 first-step surface error 6.65 mm, r2 9.03, r3 10.3. |
| Command-disabled dry run | No such mode. `mold_aware_dry_run` still publishes waypoints. |
| Hardware | Not run. Launching the runner connects to 192.168.1.2, calibrates the wrist wrench, and will servo as soon as the episode starts. With the file as it is, that is contact-capable r4 (`dry_run` false), executing 32 of 64 steps, depth zeroed, scan off. |

## October viewer

```bash
source /home/boeing/umi_ft_ws/worktrees/activate.sh UMI-FT-surface_frames
python UMIFT_Data/surface_frames_scripts/view_oct.py
```

http://127.0.0.1:8765. Drag the 3D view to rotate. Left and right arrows step the wrist frame and the marker on the path. Up and down arrows change episode.

**r1rel** reads `/mnt/data/umidata/ft/datasets/stringer_v3.5/acp_replay_buffer.zarr` (68 episodes). The time plot is the one-step motion standard BC and r1rel learn: where the tool is at the next RGB frame, written in the current tool frame (`dx, dy, dz` in meters, and the rotation of that step in degrees). The 3D path is still the fixture trajectory, so it can be rotated.

**r4** reads `stringer_v3.5_fixb/r4_world_fixb_tilt` (50 episodes). The time plot is `v1`, `v2` (m/s along the mold) and `n` (`delta_n`, meters). The path is colored by tilt, the angle between tool +z and the inward normal. A red tick at the playhead is tool +z. `F·n` stays on the force plot.

Episodes 13, 25, 29, and 42 are marked when the 50-episode audit recorded an RGB/depth gap over 0.5 s. The page does not write an HTML file.

## Training configs

October Fix B yamls under `UMIFT_Data/surface_frames_scripts/config/task/`. Short runs inherit `umift_mold_aware` (rgb/depth horizon 5 stride 4, pose and mold 5 stride 16, wrench 32 stride 1, action 24 stride 1). `*_h4s` keeps those strides and uses rgb/depth 16, pose and mold 51, wrench 118, action 64 (about 4.0 s of history, 4.2 s of action). h4s r2, r4, and r1rel set `drop_depth_inputs: true`.

| Yaml | Zarr | Action |
|---|---|---|
| `umift_oct_fixb_r1` / `_h4s` | `r1_world_world` | 21, fixture pose and wrench-derived virtual target. `UmiFTDatasetOctNorm` role `r1` |
| `umift_oct_fixb_r1rel` / `_h4s` | same r1 zarr | 21, those poses relative to the newest TCP. Role `r1rel` |
| `umift_oct_fixb_r2` / `_h4s` | `r2_world_fixb` | 21, `v_t1`, `v_t2`, `delta_n`. Role `r2` |
| `umift_oct_fixb_r4` / `_h4s` | `r4_world_fixb_tilt` | 27, r2 plus `ΔR` rot6d. `UmiFTDatasetOctR4` |

`standard_bc` task `umift_ss_v2` is not this table. It trains a source zarr with relative command and virtual-target poses, 21 dims, and no mold channels.

Normalizer pickles: short runs `assets/oct_v35_r2r4_action_normalizer_1ddfa2b4331f.pkl`; h4s `assets/oct_v35_h4s_r2r4_action_normalizer_3b0863c14453.pkl`. r1rel has its own pickle, named in `umift_oct_fixb_r1rel.yaml`.

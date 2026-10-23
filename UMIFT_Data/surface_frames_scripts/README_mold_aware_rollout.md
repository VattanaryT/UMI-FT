# Mold-aware rollout (r1 / r2 / r3)

Branch: `surface_frame`. Reuses the September stringer Fix B field
(`stringer_scanner_v2_fixb/shared/fixb_field.npz`). Full mold-package builder
and registration checks come after today’s session.

## Preflight (offline, ~1 min)

```bash
cd /home/boeing/umi_ft_ws/UMI-FT/UMIFT_Data/surface_frames_scripts
/usr/bin/python3 sanity_mold_aware_ep5.py
```

Must print `SANITY PASS` before any robot motion. Needs the ep5 open-loop
export already under `training_outputs/scanner_fixb/eval/exports/`.

## Hardware dry run (no contact)

`mold_aware_dry_run` defaults to `True` (forces δn = +15 mm hover).
Default ckpt is r2 epoch-35. Real depth is used; periodic Keyence scan is off
for these checkpoints (no `scan_heatmap`).

```bash
conda activate pyrite
# open3d required for FixBMoldField
python -c "import open3d" || pip install open3d

export PYRITE_CHECKPOINT_FOLDERS=/mnt/data/umidata/ft/training_outputs
export PYRITE_HARDWARE_CONFIG_FOLDERS=/home/boeing/umi_ft_ws/hardware_interfaces/workcell/table_top_manip/config
export PYRITE_CONTROL_LOG_FOLDERS=$HOME/data/control_log
cd /home/boeing/umi_ft_ws
python UMI-FT/PyriteEnvSuites/scripts/kill_ports.py
python UMI-FT/PyriteEnvSuites/env_runners/umift_env_runner.py
```

Jog UR to start pose, remote mode, confirm video (`q`), then menu.

### Switch policy

Edit `pipeline_para["ckpt_path"]` in `umift_env_runner.py`:

| run | path |
|-----|------|
| r1 | `/scanner_fixb/r1_world_world/epoch=0035-train_loss=0.018.ckpt` |
| r2 | `/scanner_fixb/r2_world_fixb/epoch=0035-train_loss=0.016.ckpt` |
| r3 | `/scanner_fixb/r3_surface_fixb/epoch=0035-train_loss=0.016.ckpt` |

Contact later: set `control_para["mold_aware_dry_run"] = False`.

## What the runner does

1. Detects `robot0_mold_geom` → loads Fix B field, infers r1/r2/r3 from task name.
2. Each step: fills `mold_*` from tip pose (tracked foot) → `raw_to_obs_inference`.
3. r1: absolute fixture poses via existing `action21_postprocess`.
4. r2/r3: anchored Fix B decode (`integrate_fixb` + heading) from `MPC.last_raw_action`.
5. Safety triggers (press depth, speed, step, mesh distance, normal flip) abort if
   `mold_aware_abort_on_safety` is True.

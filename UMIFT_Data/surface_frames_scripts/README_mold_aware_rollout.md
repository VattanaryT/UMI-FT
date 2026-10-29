# Mold-aware rollout (r1 / r2 / r3 / r4)

Branch: `surface_frame`. Reuses the September stringer Fix B field
(`stringer_scanner_v2_fixb/shared/fixb_field.npz`). Full mold-package builder
and registration checks come after today’s session.

## Preflight (offline, ~1 min)

```bash
source /home/boeing/umi_ft_ws/worktrees/activate.sh UMI-FT-surface_frames
python UMIFT_Data/surface_frames_scripts/sanity_mold_aware_ep5.py
```

Must print `SANITY PASS` before any robot motion. Needs the ep5 open-loop
export already under `training_outputs/scanner_fixb/eval/exports/`.

## Hardware dry run (no contact)

`mold_aware_dry_run` defaults to `True` (forces δn = +15 mm hover).
Default ckpt is r2 epoch-35. Real depth is used; periodic Keyence scan is off
for these checkpoints (no `scan_heatmap`).

`~/.bashrc` pins `PYTHONPATH` at the main `UMI-FT/` checkout. `source` the surface-frames worktree first, and run that worktree's runner. `activate.sh` changes directory there. Do not `cd` back to `umi_ft_ws` and do not launch `UMI-FT/PyriteEnvSuites/...`: that file is the other checkout, and the runner loads `mold_aware_runtime` from the directory next to itself.

```bash
conda activate pyrite
source /home/boeing/umi_ft_ws/worktrees/activate.sh UMI-FT-surface_frames
python -c "import PyriteConfig; print(PyriteConfig.__file__)"
# must print .../worktrees/UMI-FT-surface_frames/PyriteConfig/...

export PYRITE_CHECKPOINT_FOLDERS=/mnt/data/umidata/ft/training_outputs
export PYRITE_HARDWARE_CONFIG_FOLDERS=/home/boeing/umi_ft_ws/hardware_interfaces/workcell/table_top_manip/config
export PYRITE_CONTROL_LOG_FOLDERS=$HOME/data/control_log
python PyriteEnvSuites/scripts/kill_ports.py
python PyriteEnvSuites/env_runners/umift_env_runner.py
```

Jog UR to start pose, remote mode, confirm video (`q`), then menu.

### Switch policy

Edit `pipeline_para["ckpt_path"]` in `umift_env_runner.py`:

| run | path |
|-----|------|
| r1 | `/scanner_fixb/r1_world_world/epoch=0035-train_loss=0.018.ckpt` |
| r2 | `/scanner_fixb/r2_world_fixb/epoch=0035-train_loss=0.016.ckpt` |
| r3 | `/scanner_fixb/r3_surface_fixb/epoch=0035-train_loss=0.016.ckpt` |
| r4 | `/scanner_fixb/r4_world_fixb_tilt/epoch=0060-train_loss=0.018.ckpt` (epoch-35 pruned by top-k) |

Contact later: set `control_para["mold_aware_dry_run"] = False`.

## What the runner does

1. Detects `robot0_mold_geom` → loads Fix B field, infers r1/r2/r3/r4 from task name.
2. Each step: fills `mold_*` from tip pose (tracked foot) → `raw_to_obs_inference`.
3. r1: absolute fixture poses via existing `action21_postprocess`.
4. r2/r3: anchored Fix B decode (`integrate_fixb` + heading) from `MPC.last_raw_action`.
5. r4: same as r2 plus predicted tilt ΔR from action dims 21:27.
6. Speed, step, and δn clamps stay on. Tilt (45°), rotation-rate (20°/step), and
   r4 mesh-jump hold are off (`SafetyLimits.enable_*`, default False) so contact
   matches the Oct 22 dry_run=false decode. Hard stops (mesh distance, normal flip)
   still abort if `mold_aware_abort_on_safety` is True.

## Open-loop compare (offline)

```bash
# export (r4 only if r1–r3 already done):
bash UMIFT_Data/surface_frames_scripts/run_scanner_fixb_exports.sh r4_world_fixb_tilt

# score matched = r1–r3@35 + r4@60:
cd UMIFT_Data/surface_frames_scripts
python scanner_fixb_posttrain_eval.py --ckpt matched
```

Artifacts: `training_outputs/scanner_fixb/eval/matched/`.

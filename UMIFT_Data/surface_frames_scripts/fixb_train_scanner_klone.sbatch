#!/bin/bash
#SBATCH --job-name=fixb-s2
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=320G
#SBATCH --time=24:00:00
# checkpoint_every=5 → epoch 35 is written after that epoch finishes (before 36).
# Extra RAM vs prior OOMs; keep batch_size=160. Lower workers if still OOM.
#SBATCH --chdir=/gscratch/scrubbed/vtevy
#SBATCH --output=/gscratch/weirdlab/vtevy/logs/fixb/%j.out
#SBATCH --error=/gscratch/weirdlab/vtevy/logs/fixb/%j.err

# Fix B retrain on stringer_scanner_v2 (September, fixb_v4_tracked feet).
# One GPU job per run. weirdlab L40S.
#
#   RUN=r1_world_world      sbatch fixb_train_scanner_klone.sbatch
#   RUN=r2_world_fixb       sbatch fixb_train_scanner_klone.sbatch
#   RUN=r3_surface_fixb     sbatch fixb_train_scanner_klone.sbatch
#   RUN=r4_world_fixb_tilt  sbatch fixb_train_scanner_klone.sbatch

set -euo pipefail

RUN="${RUN:?set RUN=r1_world_world|r2_world_fixb|r3_surface_fixb|r4_world_fixb_tilt}"
case "${RUN}" in
  r1_world_world) TASK=umift_scanner_fixb_r1 ;;
  r2_world_fixb) TASK=umift_scanner_fixb_r2 ;;
  r3_surface_fixb) TASK=umift_scanner_fixb_r3 ;;
  r4_world_fixb_tilt) TASK=umift_scanner_fixb_r4 ;;
  *) echo "bad RUN=${RUN}"; exit 1 ;;
esac

SIF="${SIF:-/gscratch/weirdlab/vtevy/images/umift-train_v3.sif}"
UMIFT_HOST="${UMIFT_HOST:-/gscratch/weirdlab/vtevy/UMI-FT-fixb}"
DATA_ROOT="${DATA_ROOT:-/gscratch/scrubbed/vtevy/umidata/mold_aware_datasets/stringer_scanner_v2_fixb}"
DATA_HOST="${DATA_ROOT}/${RUN}"
CKPT_HOST="${CKPT_HOST:-/gscratch/weirdlab/vtevy/training_outputs/scanner_fixb_${RUN}}"
WANDB_ENV="${WANDB_ENV:-/gscratch/weirdlab/vtevy/wandb.env}"
SCRIPTS="/opt/umift/UMIFT_Data/surface_frames_scripts"

RUN_TITLE="scanner-fixb-${RUN}_v4tracked_e200_bs160_${SLURM_JOB_ID}"

mkdir -p "${CKPT_HOST}" /gscratch/weirdlab/vtevy/logs/fixb

module load apptainer

apptainer exec --nv "${SIF}" python -c \
  'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'

APPTAINER_ARGS=(
  --nv
  --pwd /opt/umift/PyriteML
  --bind "${UMIFT_HOST}:/opt/umift"
  --bind "${DATA_HOST}:/data"
  --bind "${CKPT_HOST}:/checkpoints"
  --env UMIFT_ROOT=/opt/umift
  --env PYRITE_DATASET_FOLDERS=/data
  --env PYRITE_CHECKPOINT_FOLDERS=/checkpoints
  --env PYTHONPATH="${SCRIPTS}:/opt/umift:/opt/umift/PyriteConfig:/opt/umift/PyriteUtility:/opt/umift/PyriteML"
  --env HYDRA_FULL_ERROR=1
)

if [[ -f "${WANDB_ENV}" ]]; then
  APPTAINER_ARGS+=(--env-file "${WANDB_ENV}")
else
  echo "WARNING: ${WANDB_ENV} not found; wandb may be offline."
fi

EXTRA_OVERRIDES=()
if [[ "${RUN}" == "r4_world_fixb_tilt" ]]; then
  # Writable mask-stats dump (dataset zarr mount is read-only).
  EXTRA_OVERRIDES+=(
    "task.dataset.filter_stats_path=/checkpoints/${RUN_TITLE}/r4_delta_R_mask_stats.json"
    # Pin action dims 0:21 to the vendored r2 normalizer (same bytes as job 41907426).
    "task.dataset.r2_action_normalizer_path=/opt/umift/UMIFT_Data/surface_frames_scripts/assets/r2_sparse_normalizer_41907426.pkl"
    # Log grad norm. Clip stays off (training.sparse_gradient_norm_limit=-1).
    "training.log_gradient_norm=true"
  )
fi

apptainer exec "${APPTAINER_ARGS[@]}" "${SIF}" \
  accelerate launch --mixed_precision=bf16 --num-processes=1 \
    /opt/umift/PyriteML/train.py --config-name=train_umift_workspace \
    "hydra.searchpath=[file://${SCRIPTS}/config]" \
    task="${TASK}" \
    task.dataset_path=/data/acp_replay_buffer.zarr \
    dataloader.batch_size=160 \
    dataloader.prefetch_factor=4 \
    dataloader.num_workers=16 \
    val_dataloader.batch_size=64 \
    training.num_epochs=200 \
    training.checkpoint_every=5 \
    logging.name="${RUN_TITLE}" \
    logging.group="scanner_fixb_v4tracked_scan20261008" \
    "logging.tags=[scanner_fixb,${RUN},fixb_v4_tracked,stringer_scanner_v2,scan_20261008,weirdlab,l40s,e200,bs160]" \
    hydra.run.dir="/checkpoints/${RUN_TITLE}" \
    "${EXTRA_OVERRIDES[@]}"

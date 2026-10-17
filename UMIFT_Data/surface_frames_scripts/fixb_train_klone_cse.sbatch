#!/bin/bash
#SBATCH --job-name=fixb-cse
#SBATCH --account=cse
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --chdir=/gscratch/scrubbed/vtevy
#SBATCH --output=/gscratch/weirdlab/vtevy/logs/fixb/%j.out
#SBATCH --error=/gscratch/weirdlab/vtevy/logs/fixb/%j.err

# Same Fix B training as fixb_train_klone.sbatch, on the CSE partition (L40S).
#   RUN=r1_world_world  sbatch fixb_train_klone_cse.sbatch
#   RUN=r2_world_fixb   sbatch fixb_train_klone_cse.sbatch
#   RUN=r3_surface_fixb sbatch fixb_train_klone_cse.sbatch
#
# The copy already on Hyak does not have this file yet. Same effect:
#   RUN=r1_world_world sbatch --account=cse --partition=gpu-l40s --job-name=fixb-cse fixb_train_klone.sbatch

set -euo pipefail

RUN="${RUN:?set RUN=r1_world_world|r2_world_fixb|r3_surface_fixb}"
case "${RUN}" in
  r1_world_world) TASK=umift_fixb_r1 ;;
  r2_world_fixb) TASK=umift_fixb_r2 ;;
  r3_surface_fixb) TASK=umift_fixb_r3 ;;
  *) echo "bad RUN=${RUN}"; exit 1 ;;
esac

SIF="${SIF:-/gscratch/weirdlab/vtevy/images/umift-train_v3.sif}"
UMIFT_HOST="${UMIFT_HOST:-/gscratch/weirdlab/vtevy/UMI-FT-fixb}"
DATA_ROOT="${DATA_ROOT:-/gscratch/scrubbed/vtevy/umidata/mold_aware_datasets/stringer_data_v2_full_fixb}"
DATA_HOST="${DATA_ROOT}/${RUN}"
CKPT_HOST="${CKPT_HOST:-/gscratch/weirdlab/vtevy/training_outputs/fixb_${RUN}}"
WANDB_ENV="${WANDB_ENV:-/gscratch/weirdlab/vtevy/wandb.env}"
SCRIPTS="/opt/umift/UMIFT_Data/surface_frames_scripts"

RUN_TITLE="fixb-${RUN}_cse-l40s_stringer-v2-full_scan20261008_e100_bs160_${SLURM_JOB_ID}"

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

apptainer exec "${APPTAINER_ARGS[@]}" "${SIF}" \
  accelerate launch --mixed_precision=bf16 --num_processes=1 \
    /opt/umift/PyriteML/train.py --config-name=train_umift_workspace \
    "hydra.searchpath=[file://${SCRIPTS}/config]" \
    task="${TASK}" \
    task.dataset_path=/data/acp_replay_buffer.zarr \
    dataloader.batch_size=160 \
    dataloader.prefetch_factor=8 \
    dataloader.num_workers=32 \
    val_dataloader.batch_size=64 \
    training.num_epochs=100 \
    training.checkpoint_every=5 \
    logging.name="${RUN_TITLE}" \
    logging.group="fixb_stringer_v2_full_scan20261008" \
    "logging.tags=[fixb,${RUN},cse,l40s,stringer_v2_full,scan_20261008,e100,bs160]" \
    hydra.run.dir="/checkpoints/${RUN_TITLE}"

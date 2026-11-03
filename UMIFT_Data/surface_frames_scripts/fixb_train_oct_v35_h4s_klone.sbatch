#!/bin/bash
#SBATCH --job-name=fixb-oct-h4s
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=320G
#SBATCH --time=24:00:00
#SBATCH --chdir=/gscratch/scrubbed/vtevy
#SBATCH --output=/gscratch/weirdlab/vtevy/logs/fixb/%j.out
#SBATCH --error=/gscratch/weirdlab/vtevy/logs/fixb/%j.err

# October stringer_v3.5 Fix B with a 4.0 s lookback and action horizon 64.
# Depth is not an input (task.drop_depth_inputs). RGB and mold geom stay.
# Same zarrs as fixb_train_oct_v35_klone.sbatch. New task yaml and normalizer.
# Start at batch 32 (one optimizer step, no accumulation). Raise the batch
# after the first epoch fits. Do not point this at the short-horizon ckpt dirs.
#
#   RUN=r1rel_world        sbatch fixb_train_oct_v35_h4s_klone.sbatch
#   RUN=r2_world_fixb      sbatch fixb_train_oct_v35_h4s_klone.sbatch
#   RUN=r4_world_fixb_tilt sbatch fixb_train_oct_v35_h4s_klone.sbatch
# r1rel uses the r1_world_world zarr. Skip r1_world_world (absolute fixture pose).
#
# CSE has 256G nodes. The file default is weirdlab 320G.

set -euo pipefail

RUN="${RUN:?set RUN=r1rel_world|r2_world_fixb|r4_world_fixb_tilt|r1_world_world}"
case "${RUN}" in
  r1_world_world) TASK=umift_oct_fixb_r1_h4s ;;
  r2_world_fixb) TASK=umift_oct_fixb_r2_h4s ;;
  r4_world_fixb_tilt) TASK=umift_oct_fixb_r4_h4s ;;
  r1rel_world)
    TASK=umift_oct_fixb_r1rel_h4s
    DATA_HOST_OVERRIDE="${DATA_ROOT:-/gscratch/scrubbed/vtevy/umidata/mold_aware_datasets/stringer_v3.5_fixb}/r1_world_world"
    NORM_FILE=oct_v35_h4s_r1rel_action_pose_normalizer_5505c6604004.pkl
    ;;
  *) echo "bad RUN=${RUN}"; exit 1 ;;
esac

SIF="${SIF:-/gscratch/weirdlab/vtevy/images/umift-train_v3.sif}"
UMIFT_HOST="${UMIFT_HOST:-/gscratch/weirdlab/vtevy/UMI-FT-fixb}"
DATA_ROOT="${DATA_ROOT:-/gscratch/scrubbed/vtevy/umidata/mold_aware_datasets/stringer_v3.5_fixb}"
DATA_HOST="${DATA_HOST_OVERRIDE:-${DATA_ROOT}/${RUN}}"
CKPT_HOST="${CKPT_HOST:-/gscratch/weirdlab/vtevy/training_outputs/oct_v35_h4s_fixb_${RUN}}"
WANDB_ENV="${WANDB_ENV:-/gscratch/weirdlab/vtevy/wandb.env}"
SCRIPTS="/opt/umift/UMIFT_Data/surface_frames_scripts"
NORM_FILE="${NORM_FILE:-oct_v35_h4s_r2r4_action_normalizer_3b0863c14453.pkl}"

RUN_TITLE="oct-v35-h4s-${RUN}_e200_bs32_${SLURM_JOB_ID}"

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

EXTRA_OVERRIDES=(
  "task.dataset.action_normalizer_path=/opt/umift/UMIFT_Data/surface_frames_scripts/assets/${NORM_FILE}"
  "training.log_gradient_norm=true"
  "training.log_per_dim_action_loss=true"
  "checkpoint.topk.k=1000"
  "task.drop_depth_inputs=true"
)
if [[ "${RUN}" == "r4_world_fixb_tilt" ]]; then
  EXTRA_OVERRIDES+=(
    "task.dataset.filter_stats_path=/checkpoints/${RUN_TITLE}/r4_delta_R_mask_stats.json"
  )
fi
if [[ "${RUN}" == "r1rel_world" || "${RUN}" == "r4_world_fixb_tilt" ]]; then
  EXTRA_OVERRIDES+=(
    "task.dataset.pose_jump_flag_path=/opt/umift/UMIFT_Data/surface_frames_scripts/assets/oct_v35_pose_jump_flags.npz"
  )
fi

apptainer exec "${APPTAINER_ARGS[@]}" "${SIF}" \
  accelerate launch --mixed_precision=bf16 --num-processes=1 \
    /opt/umift/PyriteML/train.py --config-name=train_umift_workspace \
    "hydra.searchpath=[file://${SCRIPTS}/config]" \
    task="${TASK}" \
    task.dataset_path=/data/acp_replay_buffer.zarr \
    dataloader.batch_size=32 \
    dataloader.prefetch_factor=4 \
    dataloader.num_workers=16 \
    val_dataloader.batch_size=32 \
    training.num_epochs=200 \
    training.checkpoint_every=5 \
    logging.name="${RUN_TITLE}" \
    logging.group="oct_v35_h4s_scan20261008" \
    "logging.tags=[oct_v35_h4s,${RUN},fixb_v4_tracked,stringer_v35,scan_20261008,l40s,e200,bs32,no_depth]" \
    hydra.run.dir="/checkpoints/${RUN_TITLE}" \
    "${EXTRA_OVERRIDES[@]}"

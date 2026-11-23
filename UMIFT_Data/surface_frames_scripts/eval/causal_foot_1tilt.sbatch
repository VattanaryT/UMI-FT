#!/bin/bash
#SBATCH --job-name=fixb-r4vt-1tilt
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=96:00:00
#SBATCH --chdir=/gscratch/scrubbed/vtevy
#SBATCH --output=/gscratch/scrubbed/vtevy/logs/fixb/%j.out
#SBATCH --error=/gscratch/scrubbed/vtevy/logs/fixb/%j.err

# PROPOSAL ONLY. Do not submit.
# Causal-foot 1-tilt: action is [v1, v2, δn, stiffness, a].
# Heading ψ comes from the field frame. b is fixed at 0.
# Same zarr, same horizon, same obs as job 43390127.
# v1, v2, δn, stiffness min/max are copied from the 10-d causal-foot pickle.
# a is fit on the same training rows. The 10-d pickle itself is not loaded.

set -euo pipefail

SIF="${SIF:-/gscratch/weirdlab/vtevy/images/umift-train_v3.sif}"
UMIFT_HOST="${UMIFT_HOST:-/gscratch/scrubbed/vtevy/UMI-FT-causal_foot}"
DATA_ROOT="${DATA_ROOT:-/gscratch/scrubbed/vtevy/umidata/mold_aware_datasets/stringer_v3.5_fixb_odom_corr}"
CKPT_HOST="${CKPT_HOST:-/gscratch/scrubbed/vtevy/training_outputs/oct_v35_h4s_r4vt_causal_foot_1tilt}"
WANDB_ENV="${WANDB_ENV:-/gscratch/weirdlab/vtevy/wandb.env}"
SCRIPTS="/opt/umift/UMIFT_Data/surface_frames_scripts"
NORM_FILE="oct_v35_h4s_r4vt_action_normalizer_minmax_causal_foot_1tilt.pkl"
RUN_TITLE="oct-v35-h4s-r4vt-causal-foot-1tilt_e200_bs64_valtopk_${SLURM_JOB_ID}"

test -f "${SIF}"
test -f "${UMIFT_HOST}/UMIFT_Data/surface_frames_scripts/config/task/umift_oct_fixb_r4vt_causal_foot_1tilt.yaml"
test -d "${DATA_ROOT}/r4vt_causal_foot/acp_replay_buffer.zarr"
test -f "${DATA_ROOT}/shared/fixb_smooth_frames.npz"
test -f "${DATA_ROOT}/shared/long_dt_gt_100ms.npz"

mkdir -p "${CKPT_HOST}" /gscratch/scrubbed/vtevy/logs/fixb
module load apptainer

APPTAINER_ARGS=(
  --nv
  --pwd /opt/umift/PyriteML
  --bind "${UMIFT_HOST}:/opt/umift"
  --bind "${DATA_ROOT}:/data"
  --bind "${CKPT_HOST}:/checkpoints"
  --env UMIFT_ROOT=/opt/umift
  --env PYRITE_DATASET_FOLDERS=/data
  --env PYRITE_CHECKPOINT_FOLDERS=/checkpoints
  --env PYTHONPATH="${SCRIPTS}:/opt/umift:/opt/umift/PyriteConfig:/opt/umift/PyriteUtility:/opt/umift/PyriteML"
  --env HYDRA_FULL_ERROR=1
)
if [[ -f "${WANDB_ENV}" ]]; then
  APPTAINER_ARGS+=(--env-file "${WANDB_ENV}")
fi

# Hydra override sketch. task= the new yaml, which inherits the causal-foot task
# and changes only the action shape to 5.
apptainer exec "${APPTAINER_ARGS[@]}" "${SIF}" \
  accelerate launch --mixed_precision=bf16 --num-processes=1 \
    /opt/umift/PyriteML/train.py --config-name=train_umift_workspace \
    "hydra.searchpath=[file://${SCRIPTS}/config]" \
    task=umift_oct_fixb_r4vt_causal_foot_1tilt \
    task.dataset_path=/data/r4vt_causal_foot/acp_replay_buffer.zarr \
    "task.dataset.action_normalizer_path=${SCRIPTS}/assets/${NORM_FILE}" \
    task.dataset.reduced_r4=true \
    task.dataset.long_dt_mask_path=/data/shared/long_dt_gt_100ms.npz \
    "task.dataset.exclusion_stats_path=/checkpoints/${RUN_TITLE}/exclusion_stats.json" \
    task.dataset.pose_jump_flag_path=null \
    "task.dataset.exclude_episodes=[35,44]" \
    dataloader.batch_size=64 \
    dataloader.num_workers=4 \
    dataloader.prefetch_factor=2 \
    val_dataloader.batch_size=64 \
    val_dataloader.num_workers=4 \
    training.num_epochs=200 \
    'training.val_at=[0,5,10,15,20,25,30,35,40,60,80,100,120,140,160,180,199]' \
    'training.checkpoint_at=[0,5,10,15,20,25,30,35,40,60,80,100,120,140,160,180,199]' \
    checkpoint.topk.monitor_key=val_loss \
    checkpoint.topk.mode=min \
    checkpoint.topk.k=20 \
    'checkpoint.topk.format_str="epoch={epoch:04d}-val_loss={val_loss:.3f}.ckpt"' \
    checkpoint.save_last_ckpt=true \
    task.drop_depth_inputs=true \
    task.sparse_action_horizon=64 \
    task.sparse_action_down_sample_steps=1 \
    logging.name="${RUN_TITLE}" \
    logging.group=oct_v35_h4s_causal_foot_1tilt \
    "logging.tags=[oct_v35_h4s,r4vt,causal_foot,one_tilt,b0,e200,bs64]" \
    "hydra.run.dir=/checkpoints/${RUN_TITLE}"

#!/bin/bash
#SBATCH --job-name=fixb-r4vt-field-v2n
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=320G
#SBATCH --time=24:00:00
#SBATCH --chdir=/gscratch/scrubbed/vtevy
#SBATCH --output=/gscratch/scrubbed/vtevy/logs/fixb/%j.out
#SBATCH --error=/gscratch/scrubbed/vtevy/logs/fixb/%j.err

# Field v2n: field_v2 labels, air δn is TCP clearance. Do not submit from this file.
# 20 mm smooth field, 30° frame carry, stiffness 3000, PCA t1.
# robot0_eef_pos is the mold PCA position.

set -euo pipefail

SIF="${SIF:-/gscratch/weirdlab/vtevy/images/umift-train_v3.sif}"
UMIFT_HOST="${UMIFT_HOST:-/gscratch/scrubbed/vtevy/UMI-FT-causal_foot}"
DATA_ROOT="${DATA_ROOT:-/gscratch/scrubbed/vtevy/umidata/mold_aware_datasets/stringer_v3.5_fixb_odom_corr}"
CKPT_HOST="${CKPT_HOST:-/gscratch/scrubbed/vtevy/training_outputs/oct_v35_h4s_r4vt_field_v2n}"
WANDB_ENV="${WANDB_ENV:-/gscratch/weirdlab/vtevy/wandb.env}"
SCRIPTS="/opt/umift/UMIFT_Data/surface_frames_scripts"
NORM_FILE="oct_v35_h4s_r4vt_action_normalizer_minmax_field_v2n.pkl"
NORM_SHA="4d7c4febbafb93293cc2c470960ddeec6dc02731cebd90ea4ee0b11785f73ac2"
RUN_TITLE="oct_v35_h4s_r4vt_field_v2n_${SLURM_JOB_ID}"

test -f "${SIF}"
test -f "${UMIFT_HOST}/UMIFT_Data/surface_frames_scripts/fixb_r4vt_dataset.py"
test -f "${UMIFT_HOST}/UMIFT_Data/surface_frames_scripts/field_v2_dataset.py"
test -f "${UMIFT_HOST}/UMIFT_Data/surface_frames_scripts/field_v2n_dataset.py"
test -f "${UMIFT_HOST}/UMIFT_Data/surface_frames_scripts/config/task/umift_oct_fixb_r4vt_field_v2n.yaml"
test -f "${UMIFT_HOST}/UMIFT_Data/surface_frames_scripts/assets/${NORM_FILE}"
test -d "${DATA_ROOT}/r4vt_field_v2n/acp_replay_buffer.zarr"
test -f "${DATA_ROOT}/shared/long_dt_gt_100ms.npz"

mkdir -p "${CKPT_HOST}" /gscratch/scrubbed/vtevy/logs/fixb

module load apptainer

apptainer exec --nv "${SIF}" python -c \
  'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'

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
else
  echo "WARNING: ${WANDB_ENV} not found; wandb may be offline."
fi

apptainer exec "${APPTAINER_ARGS[@]}" "${SIF}" \
  accelerate launch --mixed_precision=bf16 --num-processes=1 \
    /opt/umift/PyriteML/train.py --config-name=train_umift_workspace \
    "hydra.searchpath=[file://${SCRIPTS}/config]" \
    task=umift_oct_fixb_r4vt_field_v2n \
    task.dataset_path=/data/r4vt_field_v2n/acp_replay_buffer.zarr \
    "task.dataset.action_normalizer_path=${SCRIPTS}/assets/${NORM_FILE}" \
    task.dataset.action_normalizer_sha256="${NORM_SHA}" \
    task.dataset.reduced_r4=true \
    task.dataset.long_dt_mask_path=/data/shared/long_dt_gt_100ms.npz \
    "task.dataset.exclusion_stats_path=/checkpoints/${RUN_TITLE}/exclusion_stats.json" \
    task.dataset.pose_jump_flag_path=null \
    "task.dataset.exclude_episodes=[35,44]" \
    dataloader.batch_size=64 \
    dataloader.prefetch_factor=8 \
    dataloader.num_workers=32 \
    val_dataloader.batch_size=64 \
    val_dataloader.num_workers=4 \
    val_dataloader.shuffle=True \
    training.num_epochs=200 \
    'training.val_at=[0,5,10,15,20,25,30,35,40,60,80,100,120,140,160,180,199]' \
    'training.checkpoint_at=[0,5,10,15,20,25,30,35,40,60,80,100,120,140,160,180,199]' \
    training.log_gradient_norm=true \
    training.log_per_dim_action_loss=true \
    checkpoint.topk.monitor_key=val_loss \
    checkpoint.topk.mode=min \
    checkpoint.topk.k=2 \
    'checkpoint.topk.format_str="epoch={epoch:04d}-val_loss={val_loss:.3f}.ckpt"' \
    checkpoint.save_last_ckpt=true \
    task.drop_depth_inputs=true \
    task.sparse_obs_rgb_down_sample_steps=4 \
    task.sparse_obs_rgb_horizon=16 \
    task.sparse_obs_depth_down_sample_steps=4 \
    task.sparse_obs_depth_horizon=16 \
    task.sparse_obs_low_dim_down_sample_steps=16 \
    task.sparse_obs_low_dim_horizon=51 \
    task.sparse_obs_mold_down_sample_steps=16 \
    task.sparse_obs_mold_horizon=51 \
    task.sparse_obs_wrench_down_sample_steps=1 \
    task.sparse_obs_wrench_horizon=118 \
    task.sparse_action_down_sample_steps=1 \
    task.sparse_action_horizon=64 \
    task.sparse_obs_gripper_down_sample_steps=1 \
    task.sparse_obs_gripper_horizon=1 \
    task.sparse_obs_ultrawide_down_sample_steps=5 \
    task.sparse_obs_ultrawide_horizon=4 \
    logging.name=oct_v35_h4s_r4vt_field_v2n \
    logging.group=oct_v35_h4s_field_v2n \
    "logging.tags=[oct_v35_h4s,r4vt,field_v2n,k3000,r4_10,october_only,exbad,no_depth,l40s,e200,bs64,valtopk]" \
    "hydra.run.dir=/checkpoints/${RUN_TITLE}"

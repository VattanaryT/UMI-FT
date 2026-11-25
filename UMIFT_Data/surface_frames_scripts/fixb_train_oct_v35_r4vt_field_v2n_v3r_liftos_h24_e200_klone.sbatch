#!/bin/bash
#SBATCH --job-name=fixb-r4vt-liftos-h24
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

# DRAFT. Do not submit from this file.
# Same liftos zarr and normalizer. Action horizon 24. Loss x4 on v2 and δn.
# topk keeps the 2 lowest val_loss checkpoints. save_last_ckpt writes latest.ckpt.
# Account alternative if weirdlab is closed: #SBATCH --account=cse
# Not submitted.

set -euo pipefail

SIF="${SIF:-/gscratch/weirdlab/vtevy/images/umift-train_v3.sif}"
UMIFT_HOST="${UMIFT_HOST:-/gscratch/scrubbed/vtevy/UMI-FT-liftos-h24}"
DATA_ROOT="${DATA_ROOT:-/gscratch/scrubbed/vtevy/umidata/mold_aware_datasets/stringer_v3.5_fixb_odom_corr}"
CKPT_HOST="${CKPT_HOST:-/gscratch/scrubbed/vtevy/training_outputs/oct_v35_h4s_r4vt_field_v2n_v3r_liftos_h24}"
WANDB_ENV="${WANDB_ENV:-/gscratch/weirdlab/vtevy/wandb.env}"
SCRIPTS="/opt/umift/UMIFT_Data/surface_frames_scripts"
NORM_FILE="oct_v35_h4s_r4vt_action_normalizer_minmax_field_v2n_v3r_liftos.pkl"
NORM_SHA="ce85fae23a90d3d7fc29d2bc357693b17254fbadea1af716ba2f10af9be394af"
RUN_TITLE="oct_v35_h4s_r4vt_field_v2n_v3r_liftos_h24_${SLURM_JOB_ID}"

case "${NORM_SHA}" in
  4d7c4febbafb93293cc2c470960ddeec6dc02731cebd90ea4ee0b11785f73ac2|\
  55a99fbffed1c47e018a4638e044208021bf833b92c10f98d22c957e5eae9f30|\
  743c83c80bcb7d5a85bb3441d80e4e4adc8b3010070f84378de40ba3c828040f)
    echo "refusing source normalizer ${NORM_SHA}" >&2
    exit 1
    ;;
esac

test -f "${SIF}"
test -f "${UMIFT_HOST}/UMIFT_Data/surface_frames_scripts/climb_h24_dataset.py"
test -f "${UMIFT_HOST}/UMIFT_Data/surface_frames_scripts/field_v2n_v3r_liftos_dataset.py"
test -f "${UMIFT_HOST}/UMIFT_Data/surface_frames_scripts/config/task/umift_oct_fixb_r4vt_field_v2n_v3r_liftos_h24.yaml"
test -f "${UMIFT_HOST}/UMIFT_Data/surface_frames_scripts/assets/${NORM_FILE}"
test -f "${UMIFT_HOST}/UMIFT_Data/surface_frames_scripts/assets/lift_isolated_ends.npz"
test -d "${DATA_ROOT}/r4vt_field_v2n_v3r_liftos/acp_replay_buffer.zarr"
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
    task=umift_oct_fixb_r4vt_field_v2n_v3r_liftos_h24 \
    task.dataset_path=/data/r4vt_field_v2n_v3r_liftos/acp_replay_buffer.zarr \
    "task.dataset.action_normalizer_path=${SCRIPTS}/assets/${NORM_FILE}" \
    task.dataset.action_normalizer_sha256="${NORM_SHA}" \
    "task.dataset.lift_index_path=${SCRIPTS}/assets/lift_isolated_ends.npz" \
    task.dataset.lift_batch_fraction=0.15 \
    task.dataset.v2_loss_weight=4.0 \
    task.dataset.dn_loss_weight=4.0 \
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
    task.sparse_action_horizon=24 \
    task.sparse_obs_gripper_down_sample_steps=1 \
    task.sparse_obs_gripper_horizon=1 \
    task.sparse_obs_ultrawide_down_sample_steps=5 \
    task.sparse_obs_ultrawide_horizon=4 \
    logging.name=oct_v35_h4s_r4vt_field_v2n_v3r_liftos_h24 \
    logging.group=oct_v35_h4s_field_v2n_v3r_liftos_h24 \
    "logging.tags=[oct_v35_h4s,r4vt,field_v2n,field_v3r,liftos,h24,v2_dn_x4,r4_10,october_only,exbad,no_depth,l40s,e200,bs64,valtopk]" \
    "hydra.run.dir=/checkpoints/${RUN_TITLE}"

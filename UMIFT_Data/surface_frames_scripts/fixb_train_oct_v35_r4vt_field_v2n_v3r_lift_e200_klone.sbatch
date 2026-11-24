#!/bin/bash
#SBATCH --job-name=fixb-r4vt-v2n-v3r-lift
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
# field_v2n obs + mold position, TCP-relative ΔR, 0.5 s / 6 N on / 3 N off δn.

set -euo pipefail

SIF="${SIF:-/gscratch/weirdlab/vtevy/images/umift-train_v3.sif}"
UMIFT_HOST="${UMIFT_HOST:-/gscratch/scrubbed/vtevy/UMI-FT-causal_foot}"
DATA_ROOT="${DATA_ROOT:-/gscratch/scrubbed/vtevy/umidata/mold_aware_datasets/stringer_v3.5_fixb_odom_corr}"
CKPT_HOST="${CKPT_HOST:-/gscratch/scrubbed/vtevy/training_outputs/oct_v35_h4s_r4vt_field_v2n_v3r_lift}"
WANDB_ENV="${WANDB_ENV:-/gscratch/weirdlab/vtevy/wandb.env}"
SCRIPTS="/opt/umift/UMIFT_Data/surface_frames_scripts"
NORM_FILE="oct_v35_h4s_r4vt_action_normalizer_minmax_field_v2n_v3r_lift.pkl"
NORM_SHA="743c83c80bcb7d5a85bb3441d80e4e4adc8b3010070f84378de40ba3c828040f"
RUN_TITLE="oct_v35_h4s_r4vt_field_v2n_v3r_lift_${SLURM_JOB_ID}"

case "${NORM_SHA}" in
  4d7c4febbafb93293cc2c470960ddeec6dc02731cebd90ea4ee0b11785f73ac2|\
  14308b72d55600b4359dfa67d10aedab378e42ad26f2bbcac988eed4d97a1ac2|\
  71bd4cdc7be5a066e786d0b1fb9be526e01d6a6ff1e4ad7dd9c6280ac3709f84|\
  557e4d25a584ea026c253c95134f983dec0d5624dd1dd90652b78cd679630215|\
  e93b87561e762c2d7f1549a883eb18d45e6d50d929df463c43fce4817444e2f5|\
  3f9b16b1b06e4e2c27cba06053da3152c5f12322b35cd5cb94ff744f478d1c41|\
  55a99fbffed1c47e018a4638e044208021bf833b92c10f98d22c957e5eae9f30)
    echo "refusing protected normalizer ${NORM_SHA}" >&2
    exit 1
    ;;
esac

test -f "${SIF}"
test -f "${UMIFT_HOST}/UMIFT_Data/surface_frames_scripts/fixb_r4vt_dataset.py"
test -f "${UMIFT_HOST}/UMIFT_Data/surface_frames_scripts/field_v2_dataset.py"
test -f "${UMIFT_HOST}/UMIFT_Data/surface_frames_scripts/field_v2n_v3r_lift_dataset.py"
test -f "${UMIFT_HOST}/UMIFT_Data/surface_frames_scripts/rot6d.py"
test -f "${UMIFT_HOST}/UMIFT_Data/surface_frames_scripts/config/task/umift_oct_fixb_r4vt_field_v2n_v3r_lift.yaml"
test -f "${UMIFT_HOST}/UMIFT_Data/surface_frames_scripts/assets/${NORM_FILE}"
test -d "${DATA_ROOT}/r4vt_field_v2n_v3r_lift/acp_replay_buffer.zarr"
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
    task=umift_oct_fixb_r4vt_field_v2n_v3r_lift \
    task.dataset_path=/data/r4vt_field_v2n_v3r_lift/acp_replay_buffer.zarr \
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
    logging.name=oct_v35_h4s_r4vt_field_v2n_v3r_lift \
    logging.group=oct_v35_h4s_field_v2n_v3r_lift \
    "logging.tags=[oct_v35_h4s,r4vt,field_v2n,field_v3r,lift,r4_10,october_only,exbad,no_depth,l40s,e200,bs64,valtopk]" \
    "hydra.run.dir=/checkpoints/${RUN_TITLE}"

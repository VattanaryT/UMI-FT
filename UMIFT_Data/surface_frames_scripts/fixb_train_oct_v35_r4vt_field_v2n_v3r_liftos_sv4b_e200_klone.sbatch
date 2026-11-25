#!/bin/bash
#SBATCH --job-name=fixb-r4vt-sv4b
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
# stringer_v4, liftos labels, horizon 64, lift chunks 15%.
# From geodude, not submitted here:
#   tar -C /mnt/data/umidata/ft/mold_aware_datasets/stringer_v4_fixb -cf \
#     /mnt/data/umidata/ft/mold_aware_datasets/stringer_v4_fixb/r4vt_field_v2n_v3r_liftos_sv4b.tar \
#     r4vt_field_v2n_v3r_liftos_sv4b shared/long_dt_gt_100ms_sv4b.npz
#   rsync -a -e "ssh -o ControlMaster=auto -o ControlPersist=8h" \
#     /mnt/data/umidata/ft/mold_aware_datasets/stringer_v4_fixb/r4vt_field_v2n_v3r_liftos_sv4b.tar \
#     vtevy@klone.hyak.uw.edu:/gscratch/scrubbed/vtevy/umidata/mold_aware_datasets/stringer_v4_fixb/
#   ssh -o ControlMaster=auto -o ControlPersist=8h vtevy@klone.hyak.uw.edu \
#     'mkdir -p /gscratch/scrubbed/vtevy/umidata/mold_aware_datasets/stringer_v4_fixb && tar -xf /gscratch/scrubbed/vtevy/umidata/mold_aware_datasets/stringer_v4_fixb/r4vt_field_v2n_v3r_liftos_sv4b.tar -C /gscratch/scrubbed/vtevy/umidata/mold_aware_datasets/stringer_v4_fixb'
# The normalizer, lift npz, yaml, and this sbatch live in the code checkout
# (UMIFT_HOST). Sync those with the code; do not use the horizon-24 sv4 files.

set -euo pipefail

SIF="${SIF:-/gscratch/weirdlab/vtevy/images/umift-train_v3.sif}"
UMIFT_HOST="${UMIFT_HOST:-/gscratch/scrubbed/vtevy/UMI-FT-causal_foot}"
DATA_ROOT="${DATA_ROOT:-/gscratch/scrubbed/vtevy/umidata/mold_aware_datasets/stringer_v4_fixb}"
CKPT_HOST="${CKPT_HOST:-/gscratch/scrubbed/vtevy/training_outputs/oct_v35_h4s_r4vt_field_v2n_v3r_liftos_sv4b}"
WANDB_ENV="${WANDB_ENV:-/gscratch/weirdlab/vtevy/wandb.env}"
SCRIPTS="/opt/umift/UMIFT_Data/surface_frames_scripts"
NORM_FILE="oct_v35_h4s_r4vt_action_normalizer_minmax_field_v2n_v3r_liftos_sv4b.pkl"
NORM_SHA="250297835d3330172a58a1a78de4aeda9da0247ec57470b3c323a0591b3ca70d"
RUN_TITLE="oct_v35_h4s_r4vt_field_v2n_v3r_liftos_sv4b_${SLURM_JOB_ID}"

case "${NORM_SHA}" in
  ce85fae23a90d3d7fc29d2bc357693b17254fbadea1af716ba2f10af9be394af|\
  c93f000b9616cb2409a1c6cc8dfeac43db231824f951776c14e239a83645cdb4|\
  4d7c4febbafb93293cc2c470960ddeec6dc02731cebd90ea4ee0b11785f73ac2|\
  55a99fbffed1c47e018a4638e044208021bf833b92c10f98d22c957e5eae9f30|\
  743c83c80bcb7d5a85bb3441d80e4e4adc8b3010070f84378de40ba3c828040f)
    echo "refusing normalizer ${NORM_SHA}" >&2
    exit 1
    ;;
esac

test -f "${SIF}"
test -f "${UMIFT_HOST}/UMIFT_Data/surface_frames_scripts/fixb_r4vt_dataset.py"
test -f "${UMIFT_HOST}/UMIFT_Data/surface_frames_scripts/field_v2_dataset.py"
test -f "${UMIFT_HOST}/UMIFT_Data/surface_frames_scripts/field_v2n_v3r_liftos_dataset.py"
test -f "${UMIFT_HOST}/UMIFT_Data/surface_frames_scripts/rot6d.py"
test -f "${UMIFT_HOST}/UMIFT_Data/surface_frames_scripts/config/task/umift_oct_fixb_r4vt_field_v2n_v3r_liftos_sv4b.yaml"
test -f "${UMIFT_HOST}/UMIFT_Data/surface_frames_scripts/assets/${NORM_FILE}"
test -f "${UMIFT_HOST}/UMIFT_Data/surface_frames_scripts/assets/lift_isolated_ends_sv4b.npz"
test -d "${DATA_ROOT}/r4vt_field_v2n_v3r_liftos_sv4b/acp_replay_buffer.zarr"
test -f "${DATA_ROOT}/shared/long_dt_gt_100ms_sv4b.npz"

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
    task=umift_oct_fixb_r4vt_field_v2n_v3r_liftos_sv4b \
    task.dataset_path=/data/r4vt_field_v2n_v3r_liftos_sv4b/acp_replay_buffer.zarr \
    "task.dataset.action_normalizer_path=${SCRIPTS}/assets/${NORM_FILE}" \
    task.dataset.action_normalizer_sha256="${NORM_SHA}" \
    "task.dataset.lift_index_path=${SCRIPTS}/assets/lift_isolated_ends_sv4b.npz" \
    task.dataset.lift_batch_fraction=0.15 \
    task.dataset.reduced_r4=true \
    task.dataset.long_dt_mask_path=/data/shared/long_dt_gt_100ms_sv4b.npz \
    "task.dataset.exclusion_stats_path=/checkpoints/${RUN_TITLE}/exclusion_stats.json" \
    task.dataset.pose_jump_flag_path=null \
    "task.dataset.exclude_episodes=[7,14,30]" \
    task.dataset.max_episode=56 \
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
    logging.name=oct_v35_h4s_r4vt_field_v2n_v3r_liftos_sv4b \
    logging.group=oct_v35_h4s_field_v2n_v3r_liftos_sv4b \
    "logging.tags=[oct_v35_h4s,r4vt,field_v2n,field_v3r,liftos,sv4b,r4_10,stringer_v4,exbad,no_depth,l40s,e200,bs64,valtopk]" \
    "hydra.run.dir=/checkpoints/${RUN_TITLE}"

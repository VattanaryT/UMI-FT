#!/bin/bash
#SBATCH --job-name=fixb-r4vt-causal-foot
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=320G
#SBATCH --time=96:00:00
#SBATCH --chdir=/gscratch/scrubbed/vtevy
#SBATCH --output=/gscratch/scrubbed/vtevy/logs/fixb/%j.out
#SBATCH --error=/gscratch/scrubbed/vtevy/logs/fixb/%j.err

# Causal-foot r4vt, 10 action dims, 200 epochs. Checkpoint every 20, and the last.
# Episodes 35 and 44 stay excluded. The 17 constant slots are not in the action.
# ΔR (dims 4:10) is masked after a field jump and is normalized, not left at scale 1.
# Do not point UMIFT_HOST at UMI-FT-surface_frames, UMI-FT-fixb, or UMI-FT-frame_continuity.
# Not submitted.
#
# From geodude, after an interactive Hyak login. Do not run from this file.
#   rsync -a -e "ssh -o ControlMaster=auto -o ControlPersist=8h" \
#     /home/boeing/umi_ft_ws/worktrees/UMI-FT-causal_foot/ \
#     vtevy@klone.hyak.uw.edu:/gscratch/scrubbed/vtevy/UMI-FT-causal_foot/
#   rsync -a -e "ssh -o ControlMaster=auto -o ControlPersist=8h" \
#     /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_causal_foot.tar \
#     /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared \
#     vtevy@klone.hyak.uw.edu:/gscratch/scrubbed/vtevy/umidata/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/
#   ssh vtevy@klone.hyak.uw.edu 'tar -xf /gscratch/scrubbed/vtevy/umidata/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_causal_foot.tar -C /gscratch/scrubbed/vtevy/umidata/mold_aware_datasets/stringer_v3.5_fixb_odom_corr'
#   sbatch /gscratch/scrubbed/vtevy/UMI-FT-causal_foot/UMIFT_Data/surface_frames_scripts/fixb_train_oct_v35_r4vt_causal_foot_e200_klone.sbatch

set -euo pipefail

SIF="${SIF:-/gscratch/weirdlab/vtevy/images/umift-train_v3.sif}"
UMIFT_HOST="${UMIFT_HOST:-/gscratch/scrubbed/vtevy/UMI-FT-causal_foot}"
DATA_ROOT="${DATA_ROOT:-/gscratch/scrubbed/vtevy/umidata/mold_aware_datasets/stringer_v3.5_fixb_odom_corr}"
CKPT_HOST="${CKPT_HOST:-/gscratch/scrubbed/vtevy/training_outputs/oct_v35_h4s_r4vt_causal_foot}"
WANDB_ENV="${WANDB_ENV:-/gscratch/weirdlab/vtevy/wandb.env}"
SCRIPTS="/opt/umift/UMIFT_Data/surface_frames_scripts"
NORM_FILE="oct_v35_h4s_r4vt_action_normalizer_minmax_causal_foot.pkl"
NORM_SHA="14308b72d55600b4359dfa67d10aedab378e42ad26f2bbcac988eed4d97a1ac2"
RUN_TITLE="oct-v35-h4s-r4vt-causal-foot_e200_bs64_valtopk_${SLURM_JOB_ID}"

test -f "${SIF}"
test -f "${UMIFT_HOST}/UMIFT_Data/surface_frames_scripts/fixb_r4vt_dataset.py"
test -f "${UMIFT_HOST}/UMIFT_Data/surface_frames_scripts/config/task/umift_oct_fixb_r4vt_causal_foot.yaml"
test -f "${UMIFT_HOST}/UMIFT_Data/surface_frames_scripts/assets/${NORM_FILE}"
test -d "${DATA_ROOT}/r4vt_causal_foot/acp_replay_buffer.zarr"
test -f "${DATA_ROOT}/shared/fixb_smooth_frames.npz"
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
    task=umift_oct_fixb_r4vt_causal_foot \
    task.dataset_path=/data/r4vt_causal_foot/acp_replay_buffer.zarr \
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
    training.val_every=20 \
    training.checkpoint_every=20 \
    training.log_gradient_norm=true \
    training.log_per_dim_action_loss=true \
    checkpoint.topk.monitor_key=val_loss \
    checkpoint.topk.mode=min \
    checkpoint.topk.k=1 \
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
    logging.name="${RUN_TITLE}" \
    logging.group=oct_v35_h4s_causal_foot \
    "logging.tags=[oct_v35_h4s,r4vt,causal_foot,r4_10,october_only,exbad,no_depth,l40s,e200,bs64,valtopk]" \
    "hydra.run.dir=/checkpoints/${RUN_TITLE}"

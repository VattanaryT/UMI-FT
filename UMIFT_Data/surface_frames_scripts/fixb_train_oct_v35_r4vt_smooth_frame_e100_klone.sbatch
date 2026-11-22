#!/bin/bash
#SBATCH --job-name=fixb-r4vt-smooth
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=320G
#SBATCH --time=48:00:00
#SBATCH --chdir=/gscratch/scrubbed/vtevy
#SBATCH --output=/gscratch/scrubbed/vtevy/logs/fixb/%j.out
#SBATCH --error=/gscratch/scrubbed/vtevy/logs/fixb/%j.err

# October r4vt on the smooth inward frame. Same 100-epoch val-topk setup as the causal script.
# Episodes 0-49 only. Episodes 35 and 44 stay excluded. November is not in this tree.
# Action normalizer stays oct_v35_h4s_r4vt_action_normalizer_minmax_exbad.pkl.
# Kept training rows (40555): v1,v2 still [-1, 1], delta_n -0.29413..0.16732 m.
# The pickle is -0.30144..0.16870 m, so no refit. ΔR dims 21:27 stay scale 1.
# The sidecar fixb_smooth_frames.npz must sit next to fixb_field.npz. Training reads the zarr.
# Do not point UMIFT_HOST at UMI-FT-surface_frames, UMI-FT-fixb, or UMI-FT-frame_continuity.
# Not submitted.
#
# Rsync, from geodude, after an interactive Hyak login. Do not run from this file.
#   rsync -a -e "ssh -o ControlMaster=auto -o ControlPersist=8h" \
#     /home/boeing/umi_ft_ws/worktrees/UMI-FT-smooth_frame/ \
#     vtevy@klone.hyak.uw.edu:/gscratch/scrubbed/vtevy/UMI-FT-smooth_frame/
#   rsync -a -e "ssh -o ControlMaster=auto -o ControlPersist=8h" \
#     /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_smooth_frame \
#     /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared \
#     vtevy@klone.hyak.uw.edu:/gscratch/scrubbed/vtevy/umidata/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/
#
#   sbatch fixb_train_oct_v35_r4vt_smooth_frame_e100_klone.sbatch

set -euo pipefail

SIF="${SIF:-/gscratch/weirdlab/vtevy/images/umift-train_v3.sif}"
UMIFT_HOST="${UMIFT_HOST:-/gscratch/scrubbed/vtevy/UMI-FT-smooth_frame}"
DATA_ROOT="${DATA_ROOT:-/gscratch/scrubbed/vtevy/umidata/mold_aware_datasets/stringer_v3.5_fixb_odom_corr}"
CKPT_HOST="${CKPT_HOST:-/gscratch/scrubbed/vtevy/training_outputs/oct_v35_h4s_r4vt_smooth_frame}"
WANDB_ENV="${WANDB_ENV:-/gscratch/weirdlab/vtevy/wandb.env}"
SCRIPTS="/opt/umift/UMIFT_Data/surface_frames_scripts"
NORM_FILE="oct_v35_h4s_r4vt_action_normalizer_minmax_exbad.pkl"
RUN_TITLE="oct-v35-h4s-r4vt-smooth_e100_bs64_valtopk_${SLURM_JOB_ID}"

test -f "${SIF}"
test -f "${UMIFT_HOST}/UMIFT_Data/surface_frames_scripts/fixb_r4vt_dataset.py"
test -f "${UMIFT_HOST}/UMIFT_Data/surface_frames_scripts/config/task/umift_oct_fixb_r4vt_h4s.yaml"
test -f "${UMIFT_HOST}/UMIFT_Data/surface_frames_scripts/assets/${NORM_FILE}"
test -d "${DATA_ROOT}/r4vt_smooth_frame/acp_replay_buffer.zarr"
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
    task=umift_oct_fixb_r4vt_h4s \
    task.dataset_path=/data/r4vt_smooth_frame/acp_replay_buffer.zarr \
    "task.dataset.action_normalizer_path=${SCRIPTS}/assets/${NORM_FILE}" \
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
    training.num_epochs=100 \
    training.val_every=5 \
    training.checkpoint_every=5 \
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
    logging.group=oct_v35_h4s_smooth_frame \
    "logging.tags=[oct_v35_h4s,r4vt,smooth_frame,october_only,exbad,no_depth,l40s,e100,bs64,valtopk]" \
    "hydra.run.dir=/checkpoints/${RUN_TITLE}"

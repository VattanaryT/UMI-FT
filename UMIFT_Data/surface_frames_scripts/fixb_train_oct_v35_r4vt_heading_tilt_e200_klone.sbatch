#!/bin/bash
#SBATCH --job-name=fixb-r4vt-heading
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

# Heading-tilt r4vt, 8 action dims, 200 epochs.
# Action is v1, v2, δn, stiffness, sin ψ, cos ψ, a, b.
# (a, b) is the tilt rotvec after Rz(ψ). ψ is the mold long axis, not t1.
# Labels are unclipped. The runtime clip is the p99.5 tilt magnitude in the zarr
# attrs, then the existing 3° slew. This script does not start the robot.
# Checkpoints at epochs 0,5,10,15,20,25,30,35,40, then 60,80,100,120,140,160,180, and the last (199).
# topk.k=20 keeps every one of those files. k=1 would delete them. latest.ckpt is still overwritten.
# Episodes 35 and 44 stay excluded. Field-jump ΔR masking is off.
# Do not point UMIFT_HOST at UMI-FT-surface_frames, UMI-FT-fixb, or UMI-FT-frame_continuity.
# Not submitted. The user submits this.
#
# From geodude, after an interactive Hyak login. Do not run from this file.
# The new zarr hardlinks the causal-foot chunks. rsync sends the file bytes.
#   rsync -a -e "ssh -o ControlMaster=auto -o ControlPersist=8h" \
#     /home/boeing/umi_ft_ws/worktrees/UMI-FT-causal_foot/ \
#     vtevy@klone.hyak.uw.edu:/gscratch/scrubbed/vtevy/UMI-FT-causal_foot/
#   rsync -a -e "ssh -o ControlMaster=auto -o ControlPersist=8h" \
#     /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/r4vt_heading_tilt \
#     /mnt/data/umidata/ft/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/shared \
#     vtevy@klone.hyak.uw.edu:/gscratch/scrubbed/vtevy/umidata/mold_aware_datasets/stringer_v3.5_fixb_odom_corr/
#   sbatch /gscratch/scrubbed/vtevy/UMI-FT-causal_foot/UMIFT_Data/surface_frames_scripts/fixb_train_oct_v35_r4vt_heading_tilt_e200_klone.sbatch

set -euo pipefail

SIF="${SIF:-/gscratch/weirdlab/vtevy/images/umift-train_v3.sif}"
UMIFT_HOST="${UMIFT_HOST:-/gscratch/scrubbed/vtevy/UMI-FT-causal_foot}"
DATA_ROOT="${DATA_ROOT:-/gscratch/scrubbed/vtevy/umidata/mold_aware_datasets/stringer_v3.5_fixb_odom_corr}"
CKPT_HOST="${CKPT_HOST:-/gscratch/scrubbed/vtevy/training_outputs/oct_v35_h4s_r4vt_heading_tilt}"
WANDB_ENV="${WANDB_ENV:-/gscratch/weirdlab/vtevy/wandb.env}"
SCRIPTS="/opt/umift/UMIFT_Data/surface_frames_scripts"
NORM_FILE="oct_v35_h4s_r4vt_action_normalizer_minmax_heading_tilt_swing.pkl"
NORM_SHA="557e4d25a584ea026c253c95134f983dec0d5624dd1dd90652b78cd679630215"
RUN_TITLE="oct-v35-h4s-r4vt-heading-tilt_e200_bs64_valtopk_${SLURM_JOB_ID}"

test -f "${SIF}"
test -f "${UMIFT_HOST}/UMIFT_Data/surface_frames_scripts/fixb_heading_tilt_dataset.py"
test -f "${UMIFT_HOST}/UMIFT_Data/surface_frames_scripts/heading_tilt.py"
test -f "${UMIFT_HOST}/UMIFT_Data/surface_frames_scripts/config/task/umift_oct_fixb_r4vt_heading_tilt.yaml"
test -f "${UMIFT_HOST}/UMIFT_Data/surface_frames_scripts/assets/${NORM_FILE}"
test -d "${DATA_ROOT}/r4vt_heading_tilt/acp_replay_buffer.zarr"
test -f "${DATA_ROOT}/shared/fixb_smooth_frames.npz"
test -f "${DATA_ROOT}/shared/long_dt_gt_100ms.npz"
test "${NORM_SHA}" != "PENDING"

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
    task=umift_oct_fixb_r4vt_heading_tilt \
    task.dataset_path=/data/r4vt_heading_tilt/acp_replay_buffer.zarr \
    "task.dataset.action_normalizer_path=${SCRIPTS}/assets/${NORM_FILE}" \
    task.dataset.action_normalizer_sha256="${NORM_SHA}" \
    task.dataset.reduced_r4=false \
    task.dataset.mask_delta_R_after_field_jumps=false \
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
    checkpoint.topk.k=20 \
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
    "logging.tags=[oct_v35_h4s,r4vt,heading_tilt,october_only,exbad,no_depth,l40s,e200,bs64,valtopk]" \
    "hydra.run.dir=/checkpoints/${RUN_TITLE}"

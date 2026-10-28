#!/bin/bash
# Export open-loop chunks for scanner Fix B r1/r2/r3 (epoch0035 + epoch0020)
# and r4 (epoch0060 only — earlier epochs pruned by top-k).
set -eo pipefail
source /home/boeing/miniforge3/etc/profile.d/conda.sh
conda activate pyrite
export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-LP64}"

WT=/home/boeing/umi_ft_ws/worktrees/UMI-FT-surface_frames
SCRIPTS="${WT}/UMIFT_Data/surface_frames_scripts"
EXPORT="${WT}/PyriteEnvSuites/scripts"
OUT=/mnt/data/umidata/ft/training_outputs/scanner_fixb/eval/exports
CK=/mnt/data/umidata/ft/training_outputs/scanner_fixb
DATA=/mnt/data/umidata/ft/mold_aware_datasets/stringer_scanner_v2_fixb
export PYTHONPATH="${SCRIPTS}:${WT}:${WT}/PyriteUtility:${WT}/PyriteML:${WT}/PyriteConfig:${EXPORT}"
cd "${EXPORT}"

declare -A CKPT35=(
  [r1_world_world]=epoch=0035-train_loss=0.018.ckpt
  [r2_world_fixb]=epoch=0035-train_loss=0.016.ckpt
  [r3_surface_fixb]=epoch=0035-train_loss=0.016.ckpt
)
declare -A CKPT20=(
  [r1_world_world]=epoch=0020-train_loss=0.020.ckpt
  [r2_world_fixb]=epoch=0020-train_loss=0.017.ckpt
  [r3_surface_fixb]=epoch=0020-train_loss=0.018.ckpt
)
CKPT60_R4=epoch=0060-train_loss=0.018.ckpt

RUNS_ONLY="${1:-}"  # optional: r4_world_fixb_tilt | r1_world_world | ...

export_run() {
  local run="$1"; shift
  echo "======== EXPORT ${run} ========"
  python -u mold_aware_open_loop_export.py \
    "$@" \
    --dataset_path "${DATA}/${run}/acp_replay_buffer.zarr" \
    --out_dir "${OUT}/${run}" \
    --device cuda
}

for run in r1_world_world r2_world_fixb r3_surface_fixb; do
  if [[ -n "${RUNS_ONLY}" && "${RUNS_ONLY}" != "${run}" ]]; then
    continue
  fi
  export_run "${run}" \
    --checkpoint "epoch0035=${CK}/${run}/${CKPT35[$run]}" \
    --checkpoint "epoch0020=${CK}/${run}/${CKPT20[$run]}"
done

if [[ -z "${RUNS_ONLY}" || "${RUNS_ONLY}" == "r4_world_fixb_tilt" ]]; then
  export_run r4_world_fixb_tilt \
    --checkpoint "epoch0060=${CK}/r4_world_fixb_tilt/${CKPT60_R4}"
fi

echo DONE_ALL_EXPORTS

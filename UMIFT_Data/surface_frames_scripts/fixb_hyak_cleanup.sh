#!/bin/bash
# Run ON Hyak (after ssh). Cancels old Fix B jobs and deletes the July
# wrong-label dataset + their checkpoints. Leaves scanner data, code, SIF, wandb.env.
#
#   bash fixb_hyak_cleanup.sh

set -euo pipefail

echo "=== current jobs ==="
squeue -u "$USER" -o "%.18i %.9P %.30j %.8T %.10M %R" || true

# Cancel any still-running / queued jobs that used the old July Fix B names
mapfile -t OLD_JOBS < <(squeue -u "$USER" -h -o "%i %j" | awk '
  $2 ~ /^fixb-/ || $2 ~ /^fixb_/ || $2 ~ /stringer-v2-full/ { print $1 }
')
if ((${#OLD_JOBS[@]})); then
  echo "scancel old Fix B jobs: ${OLD_JOBS[*]}"
  scancel "${OLD_JOBS[@]}"
else
  echo "no old Fix B jobs in queue"
fi

OLD_DATA="/gscratch/scrubbed/vtevy/umidata/mold_aware_datasets/stringer_data_v2_full_fixb"
NEW_DATA="/gscratch/scrubbed/vtevy/umidata/mold_aware_datasets/stringer_scanner_v2_fixb"

echo
echo "=== sizes before ==="
du -sh "$OLD_DATA" 2>/dev/null || echo "(no old July Fix B data)"
du -sh /gscratch/weirdlab/vtevy/training_outputs/fixb_* 2>/dev/null || echo "(no old fixb_* ckpts)"
du -sh "$NEW_DATA" 2>/dev/null || echo "(scanner data not uploaded yet)"

echo
echo "=== delete July wrong-label dataset ==="
if [[ -d "$OLD_DATA" ]]; then
  rm -rf "$OLD_DATA"
  echo "removed $OLD_DATA"
else
  echo "already gone: $OLD_DATA"
fi

echo
echo "=== delete old Fix B checkpoints (wrong labels) ==="
for d in /gscratch/weirdlab/vtevy/training_outputs/fixb_r1_world_world \
         /gscratch/weirdlab/vtevy/training_outputs/fixb_r2_world_fixb \
         /gscratch/weirdlab/vtevy/training_outputs/fixb_r3_surface_fixb; do
  if [[ -e "$d" ]]; then
    rm -rf "$d"
    echo "removed $d"
  else
    echo "already gone: $d"
  fi
done

# Old July W&B-named run dirs under training_outputs (if any loose copies)
shopt -s nullglob
for d in /gscratch/weirdlab/vtevy/training_outputs/fixb-*; do
  # keep scanner_* dirs; only remove legacy fixb-* job titles from July train
  case "$d" in
    */scanner_*) continue ;;
    */scanner-*) continue ;;
  esac
  rm -rf "$d"
  echo "removed $d"
done
shopt -u nullglob

echo
echo "=== kept (do not touch) ==="
ls -ld /gscratch/weirdlab/vtevy/UMI-FT-fixb \
       /gscratch/weirdlab/vtevy/images/umift-train_v3.sif \
       /gscratch/weirdlab/vtevy/wandb.env \
       "$NEW_DATA" 2>/dev/null || true
ls -ld /gscratch/weirdlab/vtevy/training_outputs/scanner_fixb_* 2>/dev/null || true

echo
echo "=== mold_aware_datasets remaining ==="
du -sh /gscratch/scrubbed/vtevy/umidata/mold_aware_datasets/* 2>/dev/null || true

echo
echo "done. upload scanner data if missing, then sbatch fixb_train_scanner_klone.sbatch"

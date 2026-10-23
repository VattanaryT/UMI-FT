#!/bin/bash
# Pull final scanner Fix B checkpoints from Hyak → local /mnt/data.
# Run on geodude (no sudo). First ssh prompts password + Duo; rsync reuses it.
#
#   bash fixb_hyak_pull_ckpts.sh
#
# Optional:
#   PULL_ALL=1 bash fixb_hyak_pull_ckpts.sh   # entire run dir (large)
#   DRY_RUN=1  bash fixb_hyak_pull_ckpts.sh   # list only
#   ALSO_MAX_EPOCH=1 bash ...                 # also pull highest epoch=*.ckpt

set -euo pipefail

if [[ "$(id -u)" -eq 0 ]]; then
  echo "Do not run with sudo. Root has no Hyak host key or login."
  echo "Run: bash $0"
  exit 1
fi

HYAK="${HYAK:-vtevy@klone.hyak.uw.edu}"
REMOTE_ROOT="${REMOTE_ROOT:-/gscratch/weirdlab/vtevy/training_outputs}"
LOCAL_ROOT="${LOCAL_ROOT:-/mnt/data/umidata/ft/training_outputs/scanner_fixb}"
PULL_ALL="${PULL_ALL:-0}"
DRY_RUN="${DRY_RUN:-0}"
ALSO_MAX_EPOCH="${ALSO_MAX_EPOCH:-0}"

RUNS=(r1_world_world r2_world_fixb r3_surface_fixb)

mkdir -p "${HOME}/.ssh" "${LOCAL_ROOT}"
CTRL="${HOME}/.ssh/cm-%r@%h:%p"
SSH=(ssh -o ControlMaster=auto -o ControlPersist=8h -o "ControlPath=${CTRL}")
RSYNC_SSH="ssh -o ControlMaster=auto -o ControlPersist=8h -o ControlPath=${CTRL}"

echo "Connecting to ${HYAK} (password + Duo if needed)…"
"${SSH[@]}" "${HYAK}" "echo connected"

echo
echo "=== remote scanner_fixb runs ==="
"${SSH[@]}" "${HYAK}" bash -s <<'REMOTE'
set -euo pipefail
ROOT=/gscratch/weirdlab/vtevy/training_outputs
for run in r1_world_world r2_world_fixb r3_surface_fixb; do
  base="${ROOT}/scanner_fixb_${run}"
  echo
  echo "## ${base}"
  if [[ ! -d "${base}" ]]; then
    echo "  MISSING"
    continue
  fi
  rundir=$(ls -1dt "${base}"/*/ 2>/dev/null | head -1 || true)
  if [[ -z "${rundir}" ]]; then
    echo "  no run subdir"
    ls -la "${base}" | head -20
    continue
  fi
  echo "  rundir: ${rundir}"
  ls -lh "${rundir}"latest.ckpt 2>/dev/null || echo "  no latest.ckpt"
  ls -1 "${rundir}"epoch=*.ckpt 2>/dev/null | sort | tail -3 || true
  du -sh "${rundir}" 2>/dev/null || true
done
REMOTE

if [[ "${DRY_RUN}" == "1" ]]; then
  echo
  echo "DRY_RUN=1 — not copying. Re-run without DRY_RUN to pull."
  exit 0
fi

echo
echo "=== pull → ${LOCAL_ROOT} (~2GB/run for latest.ckpt) ==="
for run in "${RUNS[@]}"; do
  remote_base="${REMOTE_ROOT}/scanner_fixb_${run}"
  local_run="${LOCAL_ROOT}/${run}"
  mkdir -p "${local_run}"

  remote_rundir=$("${SSH[@]}" "${HYAK}" \
    "ls -1dt '${remote_base}'/*/ 2>/dev/null | head -1")
  remote_rundir="${remote_rundir%/}"
  if [[ -z "${remote_rundir}" ]]; then
    echo "SKIP ${run}: no remote run dir under ${remote_base}"
    continue
  fi

  echo
  echo "→ ${run}"
  echo "  from ${remote_rundir}/"
  echo "  to   ${local_run}/"

  if [[ "${PULL_ALL}" == "1" ]]; then
    rsync -avh --progress -e "${RSYNC_SSH}" \
      "${HYAK}:${remote_rundir}/" "${local_run}/"
  else
    # sidecars + latest
    rsync -avh --progress -e "${RSYNC_SSH}" \
      --include='latest.ckpt' \
      --include='sparse_normalizer.pkl' \
      --include='logs.json.txt' \
      --include='train.log' \
      --include='wandb_run_id.txt' \
      --include='.hydra/' \
      --include='.hydra/***' \
      --exclude='*' \
      "${HYAK}:${remote_rundir}/" "${local_run}/"

    if [[ "${ALSO_MAX_EPOCH}" == "1" ]]; then
      max_epoch=$("${SSH[@]}" "${HYAK}" \
        "ls -1 '${remote_rundir}'/epoch=*.ckpt 2>/dev/null | sort | tail -1")
      if [[ -n "${max_epoch}" ]]; then
        echo "  also: ${max_epoch##*/}"
        rsync -avh --progress -e "${RSYNC_SSH}" \
          "${HYAK}:${max_epoch}" "${local_run}/"
      fi
    fi
  fi

  ls -lh "${local_run}"/latest.ckpt 2>/dev/null || echo "  WARN: no latest.ckpt locally"
done

echo
echo "Done. Local tree:"
du -sh "${LOCAL_ROOT}"/* 2>/dev/null || true
ls -lh "${LOCAL_ROOT}"/*/latest.ckpt 2>/dev/null || true
echo
echo "Layout matches mold_aware / fixb_open_loop_metrics_scanner.py defaults:"
echo "  ${LOCAL_ROOT}/{r1_world_world,r2_world_fixb,r3_surface_fixb}/latest.ckpt"

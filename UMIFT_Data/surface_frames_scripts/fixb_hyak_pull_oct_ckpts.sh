#!/bin/bash
# Pull October Fix B r1 and r4 checkpoints from Hyak.
# Best = lowest train_loss in the epoch=*.ckpt name. Final = highest epoch.
# Lands next to the September tree:
#   /mnt/data/umidata/ft/training_outputs/oct_v35_fixb/<run>/
#
#   bash fixb_hyak_pull_oct_ckpts.sh
#   DRY_RUN=1 bash fixb_hyak_pull_oct_ckpts.sh

set -euo pipefail

if [[ "$(id -u)" -eq 0 ]]; then
  echo "Do not run with sudo. Root has no Hyak host key or login."
  exit 1
fi

HYAK="${HYAK:-vtevy@klone.hyak.uw.edu}"
REMOTE_ROOT="${REMOTE_ROOT:-/gscratch/weirdlab/vtevy/training_outputs}"
LOCAL_ROOT="${LOCAL_ROOT:-/mnt/data/umidata/ft/training_outputs/oct_v35_fixb}"
DRY_RUN="${DRY_RUN:-0}"
RUNS=(r1_world_world r4_world_fixb_tilt)

mkdir -p "${HOME}/.ssh" "${LOCAL_ROOT}"
CTRL="${HOME}/.ssh/cm-%r@%h:%p"
SSH=(ssh -o ControlMaster=auto -o ControlPersist=8h -o "ControlPath=${CTRL}")
RSYNC_SSH="ssh -o ControlMaster=auto -o ControlPersist=8h -o ControlPath=${CTRL}"

echo "Connecting to ${HYAK} (password + Duo if needed)…"
"${SSH[@]}" "${HYAK}" "echo connected"

pick_py='
import os, re, sys
root = sys.argv[1]
pat = re.compile(r"epoch=(\d+)-train_loss=([0-9.]+)\.ckpt$")
best = None
final = None
n = 0
for dirpath, _dirs, files in os.walk(root):
    for name in files:
        m = pat.search(name)
        if not m:
            continue
        n += 1
        epoch = int(m.group(1))
        loss = float(m.group(2))
        path = os.path.join(dirpath, name)
        row = (loss, -epoch, path)
        if best is None or row < best:
            best = row
        if final is None or epoch > final[0]:
            final = (epoch, path)
print(n)
print(best[2] if best else "")
print(final[1] if final else "")
'

echo
echo "=== select best (min train_loss) and final (max epoch) ==="
for run in "${RUNS[@]}"; do
  remote_base="${REMOTE_ROOT}/oct_v35_fixb_${run}"
  local_run="${LOCAL_ROOT}/${run}"
  mkdir -p "${local_run}"
  mapfile -t picked < <("${SSH[@]}" "${HYAK}" "python3 -c $(printf '%q' "${pick_py}") $(printf '%q' "${remote_base}")")
  n="${picked[0]:-0}"
  best="${picked[1]:-}"
  final="${picked[2]:-}"
  echo
  echo "## ${run}  ckpts=${n}"
  if [[ -z "${best}" || -z "${final}" ]]; then
    echo "  SKIP: no epoch=*.ckpt under ${remote_base}"
    continue
  fi
  echo "  best:  ${best}"
  echo "  final: ${final}"
  rundir=$(dirname "${best}")
  if [[ "${DRY_RUN}" == "1" ]]; then
    continue
  fi
  rsync -avh --progress -e "${RSYNC_SSH}" \
    "${HYAK}:${best}" "${HYAK}:${final}" "${local_run}/"
  # sidecars from the best rundir (same job as the kept ckpts)
  rsync -avh -e "${RSYNC_SSH}" \
    --include='sparse_normalizer.pkl' \
    --include='logs.json.txt' \
    --include='train.log' \
    --exclude='*' \
    "${HYAK}:${rundir}/" "${local_run}/" || true
  ls -lh "${local_run}"/epoch=*.ckpt
done

echo
echo "Done."
du -sh "${LOCAL_ROOT}"/* 2>/dev/null || true

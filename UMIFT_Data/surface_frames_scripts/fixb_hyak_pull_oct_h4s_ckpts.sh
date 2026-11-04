#!/bin/bash
# Pull the October 4s Fix B checkpoints from Hyak.
# Best = saved epoch with the lowest val_loss in logs.json.txt.
# Final = highest epoch=*.ckpt. Filenames stay epoch=XXXX-train_loss=Y.ckpt.
# Lands next to the short-horizon tree:
#   /mnt/data/umidata/ft/training_outputs/oct_v35_h4s/<run>/
#
#   bash fixb_hyak_pull_oct_h4s_ckpts.sh

set -euo pipefail

if [[ "$(id -u)" -eq 0 ]]; then
  echo "Do not run with sudo. Root has no Hyak host key or login."
  exit 1
fi

HYAK="${HYAK:-vtevy@klone.hyak.uw.edu}"
REMOTE_ROOT="${REMOTE_ROOT:-/gscratch/weirdlab/vtevy/training_outputs}"
LOCAL_ROOT="${LOCAL_ROOT:-/mnt/data/umidata/ft/training_outputs/oct_v35_h4s}"
DRY_RUN="${DRY_RUN:-0}"
RUNS=(r1rel_world r2_world_fixb r4_world_fixb_tilt)

mkdir -p "${HOME}/.ssh" "${LOCAL_ROOT}"
CTRL="${HOME}/.ssh/cm-%r@%h:%p"
SSH=(ssh -o ControlMaster=auto -o ControlPersist=8h -o "ControlPath=${CTRL}")
RSYNC_SSH="ssh -o ControlMaster=auto -o ControlPersist=8h -o ControlPath=${CTRL}"

echo "Connecting to ${HYAK} (password + Duo if needed)…"
"${SSH[@]}" "${HYAK}" "echo connected"

pick_py='
import json, os, re, sys
root = sys.argv[1]
pat = re.compile(r"epoch=(\d+)-train_loss=([0-9.]+)\.ckpt$")
newest = None
if os.path.isdir(root):
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isdir(path):
            stamp = os.path.getmtime(path)
            if newest is None or stamp > newest[0]:
                newest = (stamp, path)
if newest is None:
    print("0")
    sys.exit(0)
rundir = newest[1]
by_epoch = {}
for dirpath, _dirs, files in os.walk(rundir):
    for name in files:
        m = pat.search(name)
        if m:
            by_epoch[int(m.group(1))] = os.path.join(dirpath, name)
best = None
log_path = os.path.join(rundir, "logs.json.txt")
if os.path.isfile(log_path):
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "val_loss" not in row or "epoch" not in row:
                continue
            epoch = int(row["epoch"])
            if epoch not in by_epoch:
                continue
            val = float(row["val_loss"])
            if best is None or val < best[0]:
                best = (val, epoch)
final_epoch = max(by_epoch) if by_epoch else None
print(len(by_epoch))
print(rundir)
if best is None:
    print("")
    print("")
    print("")
else:
    print(by_epoch[best[1]])
    print(best[1])
    print(f"{best[0]:.8g}")
if final_epoch is None:
    print("")
    print("")
else:
    print(by_epoch[final_epoch])
    print(final_epoch)
'

echo
echo "=== best = min val_loss among saved epochs; final = max epoch ==="
for run in "${RUNS[@]}"; do
  remote_base="${REMOTE_ROOT}/oct_v35_h4s_fixb_${run}"
  local_run="${LOCAL_ROOT}/${run}"
  mkdir -p "${local_run}"
  mapfile -t picked < <("${SSH[@]}" "${HYAK}" "python3 -c $(printf '%q' "${pick_py}") $(printf '%q' "${remote_base}")")
  n="${picked[0]:-0}"
  rundir="${picked[1]:-}"
  best="${picked[2]:-}"
  best_epoch="${picked[3]:-}"
  best_val="${picked[4]:-}"
  final="${picked[5]:-}"
  final_epoch="${picked[6]:-}"
  echo
  echo "## ${run}  ckpts=${n}"
  if [[ -z "${best}" || -z "${final}" ]]; then
    echo "  SKIP: no val_loss checkpoint under ${remote_base}"
    continue
  fi
  echo "  best:  epoch ${best_epoch}  val_loss ${best_val}"
  echo "         ${best}"
  echo "  final: epoch ${final_epoch}"
  echo "         ${final}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    continue
  fi
  rsync -avh --progress -e "${RSYNC_SSH}" \
    "${HYAK}:${best}" "${HYAK}:${final}" "${local_run}/"
  rsync -avh -e "${RSYNC_SSH}" \
    --include='sparse_normalizer.pkl' \
    --include='logs.json.txt' \
    --include='train.log' \
    --exclude='*' \
    "${HYAK}:${rundir}/" "${local_run}/" || true
  cat > "${local_run}/SELECTED.txt" <<EOF
best_epoch=${best_epoch}
best_val_loss=${best_val}
best_file=$(basename "${best}")
final_epoch=${final_epoch}
final_file=$(basename "${final}")
remote=${rundir}
EOF
  ls -lh "${local_run}"/epoch=*.ckpt
done

echo
echo "Done."
du -sh "${LOCAL_ROOT}"/* 2>/dev/null || true

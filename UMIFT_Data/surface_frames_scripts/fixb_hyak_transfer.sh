#!/bin/bash
# Transfer scanner Fix B scripts + datasets to Hyak, then print sbatch lines.
# Run on geodude after interactive login works, e.g.:
#   ssh -o ControlMaster=auto -o ControlPersist=8h vtevy@klone.hyak.uw.edu
#   bash fixb_hyak_transfer.sh
#
# Do not use sudo. First ssh prompts (password + Duo); rsync reuses that connection.

set -euo pipefail

if [[ "$(id -u)" -eq 0 ]]; then
  echo "Do not run this with sudo. Root has no Hyak host key or login."
  echo "Run: bash $0"
  exit 1
fi

HYAK="${HYAK:-vtevy@klone.hyak.uw.edu}"
LOCAL_SCRIPTS="${LOCAL_SCRIPTS:-/home/boeing/umi_ft_ws/UMI-FT/UMIFT_Data/surface_frames_scripts}"
LOCAL_DATA="${LOCAL_DATA:-/mnt/data/umidata/ft/mold_aware_datasets/stringer_scanner_v2_fixb}"
REMOTE_CODE="${REMOTE_CODE:-/gscratch/weirdlab/vtevy/UMI-FT-fixb}"
REMOTE_DATA="${REMOTE_DATA:-/gscratch/scrubbed/vtevy/umidata/mold_aware_datasets/stringer_scanner_v2_fixb}"
# Optional: hardlink identical RGB/depth from prior scanner (non-fixb) copy if present.
LINK_DEST="${LINK_DEST:-/gscratch/scrubbed/vtevy/umidata/mold_aware_datasets/stringer_scanner_v2}"

mkdir -p "${HOME}/.ssh"
CTRL="${HOME}/.ssh/cm-%r@%h:%p"
SSH=(ssh -o ControlMaster=auto -o ControlPersist=8h -o "ControlPath=${CTRL}")

"${SSH[@]}" "${HYAK}" "echo connected; mkdir -p '${REMOTE_CODE}/UMIFT_Data/surface_frames_scripts' '${REMOTE_DATA}' /gscratch/weirdlab/vtevy/logs/fixb /gscratch/weirdlab/vtevy/training_outputs"

RSYNC_SSH="ssh -o ControlMaster=auto -o ControlPersist=8h -o ControlPath=${CTRL}"

rsync -a -e "${RSYNC_SSH}" --info=stats2 \
  "${LOCAL_SCRIPTS}/" "${HYAK}:${REMOTE_CODE}/UMIFT_Data/surface_frames_scripts/"

# -H preserves hardlinks; --link-dest saves space if non-fixb scanner copy exists
LINK_ARGS=()
if "${SSH[@]}" "${HYAK}" "test -d '${LINK_DEST}'"; then
  LINK_ARGS=(--link-dest="${LINK_DEST}")
fi
rsync -aH -e "${RSYNC_SSH}" --info=stats2 \
  "${LINK_ARGS[@]}" \
  "${LOCAL_DATA}/" "${HYAK}:${REMOTE_DATA}/"

echo
echo "On Hyak — cleanup old July data, then train:"
echo "  bash ${REMOTE_CODE}/UMIFT_Data/surface_frames_scripts/fixb_hyak_cleanup.sh"
echo "  cd ${REMOTE_CODE}/UMIFT_Data/surface_frames_scripts"
echo "  RUN=r1_world_world  sbatch fixb_train_scanner_klone.sbatch"
echo "  RUN=r2_world_fixb   sbatch fixb_train_scanner_klone.sbatch"
echo "  RUN=r3_surface_fixb sbatch fixb_train_scanner_klone.sbatch"

#!/usr/bin/env bash
#
# Manually start the PC-side live camera fan-out receiver for one collection
# session. Replaces start_ingest_server.sh -- the device no longer records
# locally and rsyncs after the fact, it streams frames live (see
# Robotiq-portable-real2sim's plan doc / camera_stream_writer.py).
#
# frame_ingest_server.py lives in Robotiq-portable-real2sim (not here)
# because it's self-contained there -- protocol.py is a sibling file, no
# cross-repo import needed, unlike ingest_server.py. This script is just a
# thin wrapper so you can run it from the same place/habit as before.
#
# Ctrl+C to stop -- intentionally not a systemd service, so it's only
# listening while you're actually collecting data.
#
# Usage:
#   bash UMI-FT/UMIFT_Data/bash/start_frame_ingest_server.sh <folder_name> [-- extra frame_ingest_server.py args]
#
# Example:
#   bash UMI-FT/UMIFT_Data/bash/start_frame_ingest_server.sh straight_convex_data
#
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $(basename "$0") <folder_name> [-- extra frame_ingest_server.py args]" >&2
    exit 1
fi

FOLDER_NAME="$1"
shift
if [[ "${1:-}" == "--" ]]; then
    shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_pipeline_common.sh"

FRAME_INGEST_SERVER_SCRIPT="${WS_ROOT}/Robotiq-portable-real2sim/python/live_ingest/frame_ingest_server.py"
CONTROL_PORT="${UMIFT_FRAME_INGEST_CONTROL_PORT:-6789}"
DATA_PORT="${UMIFT_FRAME_INGEST_DATA_PORT:-6791}"

log "Starting frame ingest server for folder_name=${FOLDER_NAME} (control=${CONTROL_PORT}, data=${DATA_PORT}) ..."
log "Data will land in ${LOCAL_FT_ROOT}/${FOLDER_NAME}/"
python3 "${FRAME_INGEST_SERVER_SCRIPT}" \
    --folder-name "${FOLDER_NAME}" \
    --local-ft-root "${LOCAL_FT_ROOT}" \
    --control-port "${CONTROL_PORT}" \
    --data-port "${DATA_PORT}" \
    "$@"

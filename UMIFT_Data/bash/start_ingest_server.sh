#!/usr/bin/env bash
#
# Manually start the PC-side live ingest server for one collection session.
# Start this BEFORE running live_ingest/uplink_daemon.py on the data
# collection device. Ctrl+C to stop -- intentionally not a systemd service,
# so it's only listening while you're actually collecting data.
#
# Usage:
#   bash UMI-FT/UMIFT_Data/bash/start_ingest_server.sh <folder_name> [-- extra ingest_server.py args]
#
# Example:
#   bash UMI-FT/UMIFT_Data/bash/start_ingest_server.sh straight_convex_data
#
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $(basename "$0") <folder_name> [-- extra ingest_server.py args]" >&2
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

INGEST_SERVER_SCRIPT="${SCRIPT_DIR}/../live_ingest/ingest_server.py"
INGEST_PORT="${UMIFT_INGEST_PORT:-6789}"

log "Starting ingest server for folder_name=${FOLDER_NAME} on port ${INGEST_PORT} ..."
log "Data will land in ${LOCAL_FT_ROOT}/${FOLDER_NAME}/"
python3 "${INGEST_SERVER_SCRIPT}" \
    --folder-name "${FOLDER_NAME}" \
    --local-ft-root "${LOCAL_FT_ROOT}" \
    --port "${INGEST_PORT}" \
    "$@"

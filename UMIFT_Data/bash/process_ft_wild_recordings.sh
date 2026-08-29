#!/usr/bin/env bash
#
# Pull wild T265+FT recordings from geoduderp5, then run the per-trajectory
# processing stages and the whole-dataset combine stage against everything
# just pulled. This is the manual/offline counterpart to the live ingest
# system (UMI-FT/UMIFT_Data/live_ingest/) -- both ultimately call the same
# process_local_recordings.sh / combine_umift_dataset.sh.
#
# Usage:
#   bash UMIFT_Data/bash/process_ft_wild_recordings.sh <folder_name>
#   bash UMIFT_Data/bash/process_ft_wild_recordings.sh tests --skip-existing
#   bash UMIFT_Data/bash/process_ft_wild_recordings.sh tests --skip-scp
#   bash UMIFT_Data/bash/process_ft_wild_recordings.sh tests --skip-scp --skip-batch
#
# Example:
#   bash UMIFT_Data/bash/process_ft_wild_recordings.sh straight_convex_data
#
set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") <folder_name> [options]

  folder_name  Subfolder under geoduderp5 robotiq_ws/recordings/ to copy and process.
               Data lands in /mnt/data/umidata/ft/<folder_name>/.

Options:
  --skip-existing  Incremental mode: rsync only recordings without a unified zarr,
                   and skip batch conversion for zarr outputs that already exist.
  --skip-scp       Skip remote copy (reuse data already in /mnt/data/umidata/ft/<folder_name>/).
  --skip-batch     Skip Docker batch conversion (reuse existing zarr/ outputs;
                   AprilTag transform still runs against them).
  --pose-source {auto,t265,tag}
                   Forwarded to combine_umift_dataset.sh (default: auto).
  -h, --help       Show this help.
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

FOLDER_NAME="$1"
SKIP_SCP=0
SKIP_BATCH=0
SKIP_EXISTING=0
POSE_SOURCE="auto"
shift
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-existing) SKIP_EXISTING=1 ;;
        --skip-scp) SKIP_SCP=1 ;;
        --skip-batch) SKIP_BATCH=1 ;;
        --pose-source)
            if [[ $# -lt 2 ]]; then
                echo "Error: --pose-source requires a value (auto|t265|tag)" >&2
                exit 1
            fi
            POSE_SOURCE="$2"
            shift
            ;;
        --pose-source=*)
            POSE_SOURCE="${1#*=}"
            ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
    shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_pipeline_common.sh"

# --- configurable paths ---
REMOTE_USER_HOST="${REMOTE_USER_HOST:-geoduderp5@192.168.1.74}"
REMOTE_RECORDINGS="${REMOTE_RECORDINGS:-/home/geoduderp5/robotiq_ws/recordings}"

LOCAL_DIR="${LOCAL_FT_ROOT}/${FOLDER_NAME}"
UNIFIED_ZARR_DIR="${LOCAL_DIR}/zarr"
UMIFT_ZARR="${LOCAL_DIR}/acp_replay_buffer.zarr"

sync_new_remote_recordings() {
    local remote_dir="${REMOTE_RECORDINGS}/${FOLDER_NAME}"
    local copied=0
    local skipped=0

    mkdir -p "${LOCAL_DIR}" "${UNIFIED_ZARR_DIR}"

    log "Listing remote recordings in ${REMOTE_USER_HOST}:${remote_dir} ..."
    mapfile -t remote_bags < <(
        ssh "${REMOTE_USER_HOST}" \
            "for f in '${remote_dir}'/*_camera.bag; do [ -f \"\${f}\" ] && basename \"\${f}\"; done" \
            2>/dev/null || true
    )

    if [[ ${#remote_bags[@]} -eq 0 ]]; then
        log "No remote camera bags found under ${remote_dir}"
        return 0
    fi

    for bag_name in "${remote_bags[@]}"; do
        [[ -z "${bag_name}" ]] && continue
        local prefix="${bag_name%_camera.bag}"
        local unified_zarr="${UNIFIED_ZARR_DIR}/${prefix}_unified.zarr"

        if [[ -d "${unified_zarr}" ]]; then
            log "  [SKIP] ${prefix} (unified zarr already exists)"
            skipped=$((skipped + 1))
            continue
        fi

        log "  Copying ${prefix} ..."
        rsync -a \
            "${REMOTE_USER_HOST}:${remote_dir}/${prefix}_camera.bag" \
            "${LOCAL_DIR}/"
        rsync -a \
            "${REMOTE_USER_HOST}:${remote_dir}/${prefix}_sensors/" \
            "${LOCAL_DIR}/${prefix}_sensors/"
        copied=$((copied + 1))
    done

    log "Remote sync done: ${copied} new pair(s), ${skipped} already processed"
}

# --- 1. copy from geoduderp5 ---
if [[ "${SKIP_SCP}" -eq 0 ]]; then
    mkdir -p "${LOCAL_FT_ROOT}"
    if [[ "${SKIP_EXISTING}" -eq 1 ]]; then
        sync_new_remote_recordings
    else
        log "Copying ${REMOTE_USER_HOST}:${REMOTE_RECORDINGS}/${FOLDER_NAME} -> ${LOCAL_FT_ROOT}/"
        scp -r "${REMOTE_USER_HOST}:${REMOTE_RECORDINGS}/${FOLDER_NAME}" "${LOCAL_FT_ROOT}/"
    fi
else
    log "Skipping remote copy (--skip-scp)"
fi

if [[ ! -d "${LOCAL_DIR}" ]]; then
    echo "Error: local data directory not found: ${LOCAL_DIR}" >&2
    exit 1
fi

# --- 2/2b. per-trajectory stages: batch convert + AprilTag transform ---
LOCAL_ARGS=()
if [[ "${SKIP_EXISTING}" -eq 1 ]]; then
    LOCAL_ARGS+=(--skip-existing)
fi
if [[ "${SKIP_BATCH}" -eq 1 ]]; then
    LOCAL_ARGS+=(--skip-docker)
fi
bash "${SCRIPT_DIR}/process_local_recordings.sh" "${FOLDER_NAME}" "${LOCAL_ARGS[@]}"

# --- 3/4. whole-dataset stages: combine + virtual-target postprocessing ---
bash "${SCRIPT_DIR}/combine_umift_dataset.sh" "${FOLDER_NAME}" --pose-source "${POSE_SOURCE}"

log "Done."
log "  Raw bags:       ${LOCAL_DIR}/bags/ (if moved by batch converter)"
log "  Unified zarr:   ${UNIFIED_ZARR_DIR}/"
log "  Tag traj:       ${LOCAL_DIR}/tag_traj/"
log "  UMIFT zarr:     ${UMIFT_ZARR}"

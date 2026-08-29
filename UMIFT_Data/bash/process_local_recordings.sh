#!/usr/bin/env bash
#
# Per-trajectory processing stages ONLY: batch-convert recording pairs to
# unified zarr (Docker, ROS Jazzy), then compute AprilTag master-tag TCP
# trajectories (ee_tag_traj) for whichever unified zarrs don't have one yet.
#
# Both stages are idempotent / skip-existing, so this is safe to call
# repeatedly as new trajectories arrive -- each call only does real work on
# what's new (batch_convert_recordings_to_zarr.py moves each converted
# pair's raw bags and optional *_scan_markers.json into <folder>/bags/, so
# they drop out of the next scan; ee_tag_traj is run with --skip-existing).
# Live ingest passes --prefix so an in-flight rsync of the next trajectory
# is not picked up mid-transfer. This is what live_ingest/ingest_server.py
# runs automatically after each trajectory is verified on the PC.
#
# This does NOT combine anything into the final UMIFT dataset or run
# virtual-target postprocessing -- those are whole-dataset stages you run
# manually once collection is done: see combine_umift_dataset.sh.
#
# Usage:
#   bash UMI-FT/UMIFT_Data/bash/process_local_recordings.sh <folder_name> [options]
#
set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") <folder_name> [options]

  folder_name  Subfolder under \${LOCAL_FT_ROOT} (default: /mnt/data/umidata/ft)
               containing *_camera.bag / *_sensors/ recording pairs.

Options:
  --skip-existing  Skip Docker batch conversion for pairs whose unified zarr
                   already exists (harmless to omit -- it just re-checks).
  --skip-docker    Skip the Docker batch-conversion stage entirely and reuse
                   whatever unified zarr outputs already exist. AprilTag
                   transform still runs against them.
  --prefix NAME    Convert only this recording prefix (used by live ingest so
                   an in-flight rsync of the next trajectory is ignored).
  -h, --help       Show this help.
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

FOLDER_NAME="$1"
shift
SKIP_EXISTING=0
SKIP_DOCKER=0
PREFIX=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-existing) SKIP_EXISTING=1 ;;
        --skip-docker) SKIP_DOCKER=1 ;;
        --prefix)
            if [[ $# -lt 2 || "$2" == -* ]]; then
                echo "Error: --prefix requires a value" >&2
                usage
                exit 1
            fi
            PREFIX="$2"
            shift
            ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
    shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_pipeline_common.sh"

LOCAL_DIR="${LOCAL_FT_ROOT}/${FOLDER_NAME}"
UNIFIED_ZARR_DIR="${LOCAL_DIR}/zarr"

if [[ ! -d "${LOCAL_DIR}" ]]; then
    echo "Error: local data directory not found: ${LOCAL_DIR}" >&2
    exit 1
fi

# --- batch convert new pairs in the persistent Docker container (zarr v2) ---
if [[ "${SKIP_DOCKER}" -eq 0 ]]; then
    ensure_docker_container

    log "Running batch zarr conversion in Docker (zarr v2) ..."
    BATCH_EXTRA_ARGS=()
    if [[ "${SKIP_EXISTING}" -eq 1 ]]; then
        BATCH_EXTRA_ARGS+=(--skip-existing)
    fi
    if [[ -n "${PREFIX}" ]]; then
        BATCH_EXTRA_ARGS+=(--prefix "${PREFIX}")
    fi
    docker exec "${DOCKER_CONTAINER}" bash -lc "
        set -eo pipefail
        # ROS setup.bash references unset vars; avoid bash -u (nounset) here.
        source /opt/ros/jazzy/setup.bash
        export ZARR_DEFAULT_ZARR_FORMAT=2
        python3 '${BATCH_SCRIPT_CONTAINER_PATH}' \
            '${LOCAL_DIR}' ${BATCH_EXTRA_ARGS[*]@Q}
    "
else
    log "Skipping Docker batch conversion (--skip-docker)"
fi

if [[ -n "${PREFIX}" ]]; then
    PREFIX_ZARR="${UNIFIED_ZARR_DIR}/${PREFIX}_unified.zarr"
    if [[ ! -e "${PREFIX_ZARR}" ]]; then
        echo "Error: no unified zarr for prefix ${PREFIX}: ${PREFIX_ZARR}" >&2
        exit 1
    fi
elif [[ ! -d "${UNIFIED_ZARR_DIR}" ]] || [[ -z "$(ls -A "${UNIFIED_ZARR_DIR}" 2>/dev/null || true)" ]]; then
    echo "Error: no unified zarr output found under ${UNIFIED_ZARR_DIR}" >&2
    exit 1
fi

# --- AprilTag master-tag TCP trajectories (lightweight npz sidecars) ---
# Best-effort by design (matches the prior combined script): a trajectory
# without a usable tag detection falls back to T265 odom at the combine
# step (combine_umift_dataset.sh --pose-source …, default auto)
# rather than permanently blocking that trajectory's processing.
TAG_TRAJ_DIR="${LOCAL_DIR}/tag_traj"
if [[ -f "${TAG_TRAJ_SCRIPT}" && -f "${TAG_TRAJ_CONFIG}" ]]; then
    log "Computing master-tag TCP trajectories -> ${TAG_TRAJ_DIR}/ ..."
    activate_conda
    export PYTHONPATH="${WS_ROOT}/ee_tag_traj:${PYTHONPATH:-}"
    TAG_TRAJ_ARGS=(
        --config "${TAG_TRAJ_CONFIG}"
        --output-dir "${TAG_TRAJ_DIR}"
        --skip-existing
    )
    if [[ -n "${PREFIX}" ]]; then
        TAG_TRAJ_ARGS+=(--zarr "${UNIFIED_ZARR_DIR}/${PREFIX}_unified.zarr")
    else
        TAG_TRAJ_ARGS+=(--zarr-dir "${UNIFIED_ZARR_DIR}")
    fi
    python "${TAG_TRAJ_SCRIPT}" "${TAG_TRAJ_ARGS[@]}" \
        || log "WARNING: ee_tag_traj failed for some demos; they will fall back to T265 pose at combine time"
else
    log "Skipping ee_tag_traj (script/config missing at ${TAG_TRAJ_SCRIPT} / ${TAG_TRAJ_CONFIG})"
fi

log "Done."
log "  Unified zarr: ${UNIFIED_ZARR_DIR}/"
log "  Tag traj:     ${TAG_TRAJ_DIR}/"

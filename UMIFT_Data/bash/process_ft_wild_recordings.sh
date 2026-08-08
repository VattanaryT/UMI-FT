#!/usr/bin/env bash
#
# Pull wild T265+FT recordings from geoduderp5, convert to unified zarr (v2),
# build UMIFT episode zarr, and run virtual-target postprocessing.
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
  --skip-batch     Skip Docker batch conversion (reuse existing zarr/ outputs).
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
shift
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-existing) SKIP_EXISTING=1 ;;
        --skip-scp) SKIP_SCP=1 ;;
        --skip-batch) SKIP_BATCH=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
    shift
done

# --- configurable paths ---
REMOTE_USER_HOST="${REMOTE_USER_HOST:-geoduderp5@192.168.1.74}"
REMOTE_RECORDINGS="${REMOTE_RECORDINGS:-/home/geoduderp5/robotiq_ws/recordings}"
LOCAL_FT_ROOT="${LOCAL_FT_ROOT:-/mnt/data/umidata/ft}"
WS_ROOT="${WS_ROOT:-/home/boeing/umi_ft_ws}"
CONDA_ENV="${UMIFT_CONDA_ENV:-pyrite}"
DOCKER_IMAGE="${DOCKER_IMAGE:-umift-zarr-convert:jazzy}"
DOCKER_CONTAINER="${DOCKER_CONTAINER:-umift-zarr-convert}"

LOCAL_DIR="${LOCAL_FT_ROOT}/${FOLDER_NAME}"
UNIFIED_ZARR_DIR="${LOCAL_DIR}/zarr"
UMIFT_ZARR="${LOCAL_DIR}/acp_replay_buffer.zarr"

BATCH_SCRIPT="${WS_ROOT}/Robotiq-portable-real2sim/src/batch_convert_recordings_to_zarr.py"
CONVERT_UMIFT_SCRIPT="${WS_ROOT}/UMI-FT/UMIFT_Data/scripts/convert_unified_t265_wild_to_umift_zarr.py"
POSTPROCESS_SCRIPT="${WS_ROOT}/UMI-FT/PyriteUtility/PyriteUtility/data_pipeline/postprocessing_add_virtual_target_label_umift.py"
DOCKERFILE="${WS_ROOT}/Robotiq-portable-real2sim/docker/Dockerfile.zarr-convert"

log() { echo "[$(basename "$0")] $*"; }

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

# --- 2. batch convert in persistent Docker container (ROS Jazzy, zarr v2) ---
ensure_docker_container() {
    if ! docker image inspect "${DOCKER_IMAGE}" >/dev/null 2>&1; then
        log "Building Docker image ${DOCKER_IMAGE} ..."
        docker build -t "${DOCKER_IMAGE}" -f "${DOCKERFILE}" \
            "${WS_ROOT}/Robotiq-portable-real2sim/docker"
    fi

    if ! docker ps -a --format '{{.Names}}' | grep -qx "${DOCKER_CONTAINER}"; then
        log "Creating persistent container ${DOCKER_CONTAINER} ..."
        docker create \
            --name "${DOCKER_CONTAINER}" \
            -v "${WS_ROOT}:/ws" \
            -v "/mnt/data/umidata:/mnt/data/umidata" \
            -e ZARR_DEFAULT_ZARR_FORMAT=2 \
            "${DOCKER_IMAGE}" \
            sleep infinity
    fi

    if ! docker ps --format '{{.Names}}' | grep -qx "${DOCKER_CONTAINER}"; then
        log "Starting container ${DOCKER_CONTAINER} ..."
        docker start "${DOCKER_CONTAINER}" >/dev/null
    fi
}

if [[ "${SKIP_BATCH}" -eq 0 ]]; then
    ensure_docker_container

    log "Running batch zarr conversion in Docker (zarr v2) ..."
    BATCH_EXTRA_ARGS=()
    if [[ "${SKIP_EXISTING}" -eq 1 ]]; then
        BATCH_EXTRA_ARGS+=(--skip-existing)
    fi
    docker exec "${DOCKER_CONTAINER}" bash -lc "
        set -eo pipefail
        # ROS setup.bash references unset vars; avoid bash -u (nounset) here.
        source /opt/ros/jazzy/setup.bash
        export ZARR_DEFAULT_ZARR_FORMAT=2
        python3 /ws/Robotiq-portable-real2sim/src/batch_convert_recordings_to_zarr.py \
            '${LOCAL_DIR}' ${BATCH_EXTRA_ARGS[*]}
    "
else
    log "Skipping Docker batch conversion (--skip-batch)"
fi

if [[ ! -d "${UNIFIED_ZARR_DIR}" ]] || [[ -z "$(ls -A "${UNIFIED_ZARR_DIR}" 2>/dev/null || true)" ]]; then
    echo "Error: no unified zarr output found under ${UNIFIED_ZARR_DIR}" >&2
    exit 1
fi

activate_conda() {
    if [[ -n "${CONDA_PREFIX:-}" ]] && [[ "$(basename "${CONDA_PREFIX}")" == "${CONDA_ENV}" ]]; then
        return 0
    fi
    for candidate in \
        "${HOME}/miniforge3/etc/profile.d/conda.sh" \
        "${HOME}/miniconda3/etc/profile.d/conda.sh" \
        "${HOME}/anaconda3/etc/profile.d/conda.sh" \
        "/opt/conda/etc/profile.d/conda.sh"
    do
        if [[ -f "${candidate}" ]]; then
            # Conda/MKL activate hooks reference unset vars; disable nounset briefly.
            set +u
            # shellcheck disable=SC1090
            source "${candidate}"
            conda activate "${CONDA_ENV}"
            set -u
            return 0
        fi
    done
    echo "Error: could not find conda and activate env '${CONDA_ENV}'" >&2
    exit 1
}

# --- 2b. optional master-tag EE trajectories (lightweight npz sidecars) ---
TAG_TRAJ_SCRIPT="${WS_ROOT}/ee_tag_traj/scripts/run_trajectory.py"
TAG_TRAJ_CONFIG="${TAG_TRAJ_CONFIG:-${WS_ROOT}/ee_tag_traj/configs/example_stringer.yaml}"
TAG_TRAJ_DIR="${LOCAL_DIR}/tag_traj"
if [[ -f "${TAG_TRAJ_SCRIPT}" && -f "${TAG_TRAJ_CONFIG}" ]]; then
    log "Computing master-tag TCP trajectories -> ${TAG_TRAJ_DIR}/ ..."
    activate_conda
    export PYTHONPATH="${WS_ROOT}/ee_tag_traj:${PYTHONPATH:-}"
    python "${TAG_TRAJ_SCRIPT}" \
        --zarr-dir "${UNIFIED_ZARR_DIR}" \
        --config "${TAG_TRAJ_CONFIG}" \
        --output-dir "${TAG_TRAJ_DIR}" \
        --skip-existing \
        || log "ee_tag_traj failed for some demos; convert will fall back to T265 where sidecars are missing"
else
    log "Skipping ee_tag_traj (script/config missing); convert will use T265 poses"
fi

# --- 3. convert unified zarr -> UMIFT episode zarr (host conda env, zarr v2) ---
log "Converting unified zarr -> UMIFT schema (conda env: ${CONDA_ENV}) ..."
activate_conda
export PYTHONPATH="${WS_ROOT}/UMI-FT/PyriteUtility:${PYTHONPATH:-}"

python "${CONVERT_UMIFT_SCRIPT}" \
    --input "${UNIFIED_ZARR_DIR}" \
    --output "${UMIFT_ZARR}" \
    --pose-source auto

# --- 4. virtual target postprocessing ---
log "Running virtual-target postprocessing on ${UMIFT_ZARR} ..."
export UMIFT_DATASET_PATH="${UMIFT_ZARR}"
export PYRITE_DATASET_FOLDERS="${LOCAL_DIR}"

python "${POSTPROCESS_SCRIPT}"

log "Done."
log "  Raw bags:       ${LOCAL_DIR}/bags/ (if moved by batch converter)"
log "  Unified zarr:   ${UNIFIED_ZARR_DIR}/"
log "  Tag traj:       ${TAG_TRAJ_DIR}/"
log "  UMIFT zarr:     ${UMIFT_ZARR}"

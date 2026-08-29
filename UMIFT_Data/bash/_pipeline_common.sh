#!/usr/bin/env bash
#
# Shared config + helpers for the UMIFT recording pipeline scripts:
# process_local_recordings.sh, combine_umift_dataset.sh, and
# process_ft_wild_recordings.sh. Not meant to be run directly -- source it.
#
set -euo pipefail

WS_ROOT="${WS_ROOT:-/home/boeing/umi_ft_ws}"
CONDA_ENV="${UMIFT_CONDA_ENV:-pyrite}"
DOCKER_IMAGE="${DOCKER_IMAGE:-umift-zarr-convert:jazzy}"
DOCKER_CONTAINER="${DOCKER_CONTAINER:-umift-zarr-convert}"
LOCAL_FT_ROOT="${LOCAL_FT_ROOT:-/mnt/data/umidata/ft/datasets}"

# Path to batch_convert_recordings_to_zarr.py as seen *inside* the Docker
# container, where the whole workspace is bind-mounted at /ws.
BATCH_SCRIPT_CONTAINER_PATH="/ws/Robotiq-portable-real2sim/python/batch_convert_recordings_to_zarr.py"
CONVERT_UMIFT_SCRIPT="${WS_ROOT}/UMI-FT/UMIFT_Data/scripts/convert_unified_t265_wild_to_umift_zarr.py"
POSTPROCESS_SCRIPT="${WS_ROOT}/UMI-FT/PyriteUtility/PyriteUtility/data_pipeline/postprocessing_add_virtual_target_label_umift.py"
DOCKERFILE="${WS_ROOT}/Robotiq-portable-real2sim/docker/Dockerfile.zarr-convert"
TAG_TRAJ_SCRIPT="${WS_ROOT}/ee_tag_traj/scripts/run_trajectory.py"
TAG_TRAJ_CONFIG="${TAG_TRAJ_CONFIG:-${WS_ROOT}/ee_tag_traj/configs/example_stringer.yaml}"

log() { echo "[$(basename "$0")] $*"; }

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
            -v "/mnt/data/boeing:/mnt/data/boeing" \
            -e ZARR_DEFAULT_ZARR_FORMAT=2 \
            "${DOCKER_IMAGE}" \
            sleep infinity
    fi

    if ! docker ps --format '{{.Names}}' | grep -qx "${DOCKER_CONTAINER}"; then
        log "Starting container ${DOCKER_CONTAINER} ..."
        docker start "${DOCKER_CONTAINER}" >/dev/null
    fi
    # Container is never stopped/removed after use, so this is a no-op /
    # `docker exec` on every subsequent call -- one persistent container
    # reused across every trajectory, not restarted per call.
}

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

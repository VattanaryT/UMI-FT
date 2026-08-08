#!/usr/bin/env bash
# Local smoke tests for the umift-train image.
# Usage:
#   ./docker/run_local_smoke.sh imports     # no GPU contention risk beyond a tiny alloc
#   ./docker/run_local_smoke.sh cuda        # needs free GPU
#   ./docker/run_local_smoke.sh train-short # needs free GPU; Hydra override for 1 epoch
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-umift-train:local}"
DATA_HOST="${DATA_HOST:-/mnt/data/umidata/ft/stringer_data_v2_full}"
CKPT_HOST="${CKPT_HOST:-${ROOT}/runs/docker_smoke}"
MODE="${1:-imports}"

mkdir -p "${CKPT_HOST}"

run_gpu() {
  docker run --rm --gpus all \
    -e UMIFT_ROOT=/opt/umift \
    -e PYRITE_DATASET_FOLDERS=/data \
    -e PYRITE_CHECKPOINT_FOLDERS=/checkpoints \
    -e PYTHONPATH=/opt/umift/PyriteML:/opt/umift/PyriteML/multimodal_representation/multimodal:/opt/umift \
    -v "${DATA_HOST}:/data:ro" \
    -v "${CKPT_HOST}:/checkpoints" \
    "${IMAGE}" \
    "$@"
}

case "${MODE}" in
  imports)
    docker run --rm "${IMAGE}" python -c "
import sys, torch, timm, diffusers, accelerate, hydra, zarr, einops, cv2, spatialmath, imagecodecs, vit_pytorch
print('imports_ok')
print('python', sys.version.split()[0])
print('torch', torch.__version__, 'cuda_build', torch.version.cuda)
print('zarr', zarr.__version__)
"
    ;;
  cuda)
    run_gpu python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available in container'
x = torch.randn(1024, 1024, device='cuda')
y = x @ x
print('cuda_ok', torch.cuda.get_device_name(0), float(y.mean()))
"
    ;;
  train-short)
    run_gpu accelerate launch --mixed_precision=bf16 --num_processes=1 \
      train.py --config-name=train_umift_workspace \
      task.dataset_path=/data/acp_replay_buffer.zarr \
      training.num_epochs=1 \
      training.max_train_steps=2 \
      dataloader.batch_size=2 \
      dataloader.num_workers=1 \
      val_dataloader.batch_size=2 \
      val_dataloader.num_workers=1 \
      logging.mode=offline
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Use: imports | cuda | train-short" >&2
    exit 1
    ;;
esac

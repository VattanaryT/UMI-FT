# UMI-FT training image — mirrors the README install sequence:
#   mamba create -n pyrite python=3.12
#   pip install torch torchvision torchaudio  (CUDA-matched)
#   pip install timm diffusers accelerate
#   mamba env update --file conda_environment.yaml  (torch lines omitted here)
#   pip install v4l2py toppra atomics vit-pytorch imagecodecs
#
# Build from the UMI-FT repo root:
#   docker build -f docker/Dockerfile -t umift-train:local .
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG GIT_SHA=unknown
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
LABEL org.opencontainers.image.revision="${GIT_SHA}"
LABEL org.opencontainers.image.title="umift-train"

ENV DEBIAN_FRONTEND=noninteractive \
    MAMBA_ROOT_PREFIX=/opt/conda \
    UMIFT_ROOT=/opt/umift \
    PYRITE_DATASET_FOLDERS=/data \
    PYRITE_CHECKPOINT_FOLDERS=/checkpoints \
    HYDRA_FULL_ERROR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        bzip2 \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Micromamba (drop-in for the README's mamba workflow)
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
      | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba \
    && micromamba shell init -s bash -r ${MAMBA_ROOT_PREFIX}

# README: mamba create -n pyrite python=3.12
RUN micromamba create -y -n pyrite python=3.12 && micromamba clean -a -y

# Prefer the env on PATH for subsequent RUN / CMD
ENV PATH=/opt/conda/envs/pyrite/bin:/usr/local/bin:${PATH} \
    PYTHONPATH=/opt/umift/PyriteML:/opt/umift/PyriteML/multimodal_representation/multimodal:/opt/umift

# README: install bulky packages first (CUDA-matched torch), then timm/diffusers/accelerate
RUN pip install --no-cache-dir \
      torch torchvision torchaudio --index-url ${TORCH_INDEX_URL} \
    && pip install --no-cache-dir timm diffusers accelerate

# README: mamba env update --file PyriteML/conda_environment.yaml
# (docker yaml omits pytorch* so pip CUDA wheels are not replaced)
COPY docker/conda_environment.docker.yaml /tmp/conda_environment.docker.yaml
RUN micromamba install -y -n pyrite -f /tmp/conda_environment.docker.yaml \
    && micromamba clean -y -a -f

# Reinstall CUDA torch after conda packages (solver/clean can disturb pip wheels),
# then README pip extras + headless OpenCV.
RUN pip install --no-cache-dir --force-reinstall \
      torch torchvision torchaudio --index-url ${TORCH_INDEX_URL} \
    && pip install --no-cache-dir \
      timm diffusers accelerate \
      opencv-python-headless \
      v4l2py \
      toppra \
      atomics \
      'vit-pytorch>=1.7.12' \
      'imagecodecs>=2023.9.18'

WORKDIR /opt/umift
COPY PyriteUtility /opt/umift/PyriteUtility
COPY PyriteML /opt/umift/PyriteML
COPY PyriteConfig /opt/umift/PyriteConfig

RUN pip install --no-cache-dir -e /opt/umift/PyriteUtility

WORKDIR /opt/umift/PyriteML

# Bind at runtime:
#   -v <zarr_parent_or_PYRITE_DATASET_FOLDERS>:/data:ro
#   -v <runs_dir>:/checkpoints
CMD ["accelerate", "launch", "--mixed_precision=bf16", "--num_processes=1", \
     "train.py", "--config-name=train_umift_workspace", \
     "task.dataset_path=/data/acp_replay_buffer.zarr"]

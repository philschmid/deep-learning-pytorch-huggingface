FROM nvcr.io/nvidia/pytorch:22.12-py3

# Versions
ARG PYTORCH='2.0.1'
ARG CUDA='cu118' # used in the base container: https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-22-12.html
ARG TRANSFORMERS='4.30.2'
ARG DATASETS='2.13.0'
ARG ACCLERATE='0.20.3'
ARG DEEPSPEED='0.9.5'

LABEL maintainer="Philipp Schmid"
ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 


RUN apt-get update \
    && apt-get -y upgrade --only-upgrade systemd openssl cryptsetup \
    && apt-get install -y \
    bzip2 \
    curl \
    git \
    git-lfs \
    tar \
    gcc \
    g++ \
    libaio-dev \
    # audio
    libsndfile1-dev \
    ffmpeg \
    && apt-get clean autoremove --yes \
    && rm -rf /var/lib/{apt,dpkg,cache,log}

# update pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

# remove old torch and 
# Install latest release PyTorch (PyTorch must be installed before any DeepSpeed c++/cuda ops.)
RUN python3 -m pip uninstall -y torch torchvision torchaudio torch-tensorrt \
 && python3 -m pip install --no-cache-dir -U torch==${PYTORCH} torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/${CUDA}

# Install DeepSpeed
RUN python3 -m pip install --no-cache-dir -U deepspeed==${DEEPSPEED}

# Install Hugging Face Libraries
RUN python3 -m pip install --upgrade --no-cache-dir -U \
  transformers[sklearn,sentencepiece,vision]==${TRANSFORMERS} \
  datasets==${DATASETS} \
  accelerate==${ACCLERATE} \
  evaluate \
  tensorboard

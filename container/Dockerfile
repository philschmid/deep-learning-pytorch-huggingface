FROM nvcr.io/nvidia/pytorch:22.12-py3 as script

# Defaults 
LABEL maintainer="Philipp Schmid"
ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 
WORKDIR /workspace

# Versions
ARG PYTORCH='2.1.0'
ARG CUDA='cu118' # used in the base container: https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-22-12.html
ARG TRANSFORMERS='4.35.0'
ARG DATASETS='2.14.6'
ARG ACCELERATE='0.24.1'
ARG TRL='0.7.2'
ARG PEFT='0.6.0'
ARG EVALUATE='0.4.1'

# extra libraries
ARG FLASH_ATTENTION='2.3.3'
ARG BNB='0.41.1'

# Distrubted libraries
ARG DEEPSPEED='0.12.2'


# Update and install dependencies
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
  python-is-python3 \
  libaio-dev \
  # audio
  libsndfile1-dev \
  ffmpeg \
  && apt-get clean autoremove --yes \
  && rm -rf /var/lib/{apt,dpkg,cache,log}

# update pip
RUN python -m pip install --no-cache-dir --upgrade pip

# remove old torch and 
# Install latest release PyTorch (PyTorch must be installed before any DeepSpeed c++/cuda ops.)
RUN pip uninstall -y torch torchvision torchaudio torch-tensorrt \
  && pip install --no-cache-dir -U torch==${PYTORCH} torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/${CUDA}

# Install DeepSpeed
RUN pip install --no-cache-dir -U deepspeed==${DEEPSPEED}

# Install Flash Attention
RUN pip install --no-cache-dir ninja packaging
RUN pip install flash-attn==${FLASH_ATTENTION} --no-build-isolation

# Install Hugging Face Libraries
RUN pip install --upgrade --no-cache-dir -U \
  transformers[sklearn,sentencepiece,vision]==${TRANSFORMERS} \
  datasets==${DATASETS} \
  accelerate==${ACCELERATE} \
  trl==${TRL} \
  peft==${PEFT} \
  evaluate==${EVALUATE} \
  tensorboard \
  bitsandbytes \
  einops \
  loralib \
  bitsandbytes==${BNB}


############################################
#     Jupyter Notebook Container Setup     #
############################################
FROM script as notebook

# Install Jupyter environment
RUN pip install jupyter

# Expose the Jupyter Notebook port
EXPOSE 8888

# Start Jupyter Notebook on container startup
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
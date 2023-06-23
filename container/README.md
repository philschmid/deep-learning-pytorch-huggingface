# PyTorch Deep Learning Containers

This folder contains Dockerfiles for PyTorch Deep Learning Containers including Hugging Face libraries and/or Deepspeed. 

## Dockerfiles

| Container                         | Versions                                             | URI         |
| --------------------------------- | ---------------------------------------------------- | ----------- |
| [Pytorch Deepspeed](./Dockerfile) | torch==2.0.1, transformers==4.30.2, deepspeed==0.9.5 | `philschmi` |

## Getting Started

### Build the Docker image

```bash
docker build -t philschmi/huggingface-pytorch:2.0.1-transformers4.30.2-deepspeed0.9.5-cuda11.8 -t philschmi/huggingface-pytorch:latest  -f Dockerfile .
```

### Run the Docker image

```bash
docker run --gpus all -it --rm philschmi/huggingface-pytorch:latest
```

### Pull the Docker image

```bash
docker pull philschmi/huggingface-pytorch:latest
```

### Push the Docker image

```bash
docker login 
```


push 
```bash
docker push philschmi/huggingface-pytorch:2.0.1-transformers4.30.2-deepspeed0.9.5-cuda11.8
docker push philschmi/huggingface-pytorch:latest
```



## Run PyTorch Scripts

```bash
docker run --rm -it --init \
  --gpus=all \
  --ipc=host \
  --user="$(id -u):$(id -g)" \
  --volume="$PWD:/workspace" \
  philschmi/huggingface-pytorch:latest python train.py --foo bar
```

* `--gpus=all`: Enables GPU support. If you have multiple GPUs, you can use
  `--gpus=0,1,2` to specify which ones to use.
* `--ipc=host`: Required if using multiprocessing, as explained at
  https://github.com/pytorch/pytorch#docker-image.
* `--volume="$PWD:/app"`: Mounts the current working directory into the container.
  The default working directory inside the container is `/workspace`. Optional.
* `--user="$(id -u):$(id -g)"`: Sets the user inside the container to match your
  user and group ID. Optional, but is useful for writing files with correct
  ownership.


## Deriving your own images

The recommended way of adding additional dependencies to an image is to create
your own Dockerfile this project as a base.

```dockerfile
FROM philschmi/huggingface-pytorch:2.0.1-transformers4.30.2-deepspeed0.9.5-cuda11.8

# Install system libraries required by OpenCV.
RUN sudo apt-get update \
 && sudo apt-get install -y libgl1-mesa-glx libgtk2.0-0 libsm6 libxext6 \
 && sudo rm -rf /var/lib/apt/lists/*

# Install OpenCV from PyPI.
RUN pip install opencv-python==4.5.1.48
```
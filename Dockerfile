FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        git \
        python3-pip \
        python3-dev \
        python3-opencv \
        libglib2.0-0 \
        ffmpeg \
        libsm6 \ 
        libxext6 \ 
	wget \
	unzip  \
        libegl1 \
        libgl1 \
        libglib2.0-0 \
        libgl1 \
        graphviz \
        libgomp1 && \
        rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch3d takes long time 
RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Install Chamfer Distance  pytorch implementation 
RUN pip install git+'https://github.com/otaheri/chamfer_distance'

# Install opencv
RUN pip install opencv-python

# Install Open3d kitti and YAML
RUN pip install open3d pykitti pyyaml torchview torchinfo ipykernel wandb ipywidgets graphbiz

# Set the working directory
WORKDIR /workspace

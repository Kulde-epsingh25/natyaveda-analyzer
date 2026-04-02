FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip python3.10-venv \
    ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    git wget curl build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

WORKDIR /app

# Install PyTorch with CUDA 11.8
RUN pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Install MMPose ecosystem
RUN pip install openmim && \
    mim install mmengine mmcv mmdet mmpose

# Install TensorFlow (for MoveNet)
RUN pip install tensorflow==2.15.0 tensorflow-hub

# Copy requirements and install
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy project
COPY . .
RUN pip install -e ".[dev]"

# Pre-download model weights at build time (optional, comment out to save image size)
# RUN python scripts/download_weights.py

# Volumes
VOLUME ["/app/data", "/app/weights", "/app/outputs"]

# Default command
CMD ["python", "scripts/verify_setup.py"]

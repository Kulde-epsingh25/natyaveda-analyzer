# ============================================================
# NatyaVeda Analyzer v2 — Dockerfile
# Base: NVIDIA CUDA 12.1 + Python 3.11 (intentional — avoids
#       Python 3.12 + setuptools conflict for MMPose)
#
# Build:  docker build -t natyaveda .
# Run GPU: docker run --gpus all -v $(pwd)/data:/app/data natyaveda
# Run CPU: docker run -v $(pwd)/data:/app/data natyaveda
# ============================================================

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Use Python 3.11 — avoids the Python 3.12 + setuptools/MMPose issue
# while still being fully modern
ARG PYTHON_VERSION=3.11

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TF_CPP_MIN_LOG_LEVEL=3

# ── System dependencies ─────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
    python3-pip \
    # Video processing
    ffmpeg \
    # OpenCV headless requirements (libGL not needed with headless)
    libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    # MMPose build dependencies
    ninja-build gcc g++ \
    # Git, curl
    git wget curl \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
    && python -m pip install --upgrade pip

# ── CRITICAL: Pin setuptools FIRST ──────────────────────────────
RUN pip install "setuptools>=69.5.1,<81" wheel

# ── PyTorch with CUDA 12.1 ──────────────────────────────────────
RUN pip install torch==2.2.0 torchvision==0.17.0 \
    --index-url https://download.pytorch.org/whl/cu121

# ── MMCV via miropsota prebuilt (Python 3.11, PyTorch 2.2, CUDA 12.1) ─
RUN pip install mmengine>=0.10.0
RUN pip install mmcv==2.2.0+pt2.2.0cu121 \
    --extra-index-url https://miropsota.github.io/torch_packages_builder \
    || pip install mmcv-lite>=2.1.0
RUN pip install mmdet==3.3.0 mmpose==1.3.1

# ── HuggingFace (VitPose-Plus, VideoMAE, DETR) ──────────────────
RUN pip install transformers>=4.40.0 huggingface-hub>=0.22.0 \
    timm>=1.0.0 accelerate>=0.27.0 einops>=0.7.0

# ── TensorFlow CPU (MoveNet fallback) ───────────────────────────
RUN pip install tensorflow-cpu>=2.15.0 tensorflow-hub>=0.16.0

# ── MediaPipe + OpenCV HEADLESS (no libGL) ──────────────────────
RUN pip install mediapipe>=0.10.0 opencv-python-headless>=4.9.0 \
    Pillow>=10.0.0 imageio>=2.31.0 albumentations>=1.3.0

# ── Remaining requirements ───────────────────────────────────────
COPY requirements.txt .
RUN pip install -r requirements.txt --no-deps 2>/dev/null || true

# ── Copy project ─────────────────────────────────────────────────
WORKDIR /app
COPY . .
RUN pip install -e ".[dev]" --no-deps

# ── Create data directories ──────────────────────────────────────
RUN mkdir -p data/raw data/refined data/processed data/splits weights outputs

# ── Download RTMW-x weights ──────────────────────────────────────
RUN python -m mim download mmpose \
    --config td-hm_rtmw-x_8xb2-270e_coco-wholebody-384x288 \
    --dest weights/ || echo "Weights will download on first use"

VOLUME ["/app/data", "/app/weights", "/app/outputs"]

EXPOSE 6006 8888

CMD ["python", "scripts/verify_setup.py"]

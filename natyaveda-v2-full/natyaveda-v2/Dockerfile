FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

ARG NATYAVEDA_INSTALL_MODE=cpu
ARG NATYAVEDA_CUDA_TAG=121

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-venv \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender1 \
    libglib2.0-0 \
    libgl1 \
    libgl1-mesa-glx \
    git \
    ninja-build \
    gcc \
    g++ \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/local/bin/python && \
    ln -sf /usr/bin/pip3 /usr/local/bin/pip

WORKDIR /app

COPY . .

# Use the project's installer so dependency ordering and pinning stay consistent.
RUN if [ "$NATYAVEDA_INSTALL_MODE" = "cpu" ]; then \
      python scripts/install.py --skip-system --cpu; \
    elif [ "$NATYAVEDA_INSTALL_MODE" = "cuda" ]; then \
      python scripts/install.py --skip-system --cuda ${NATYAVEDA_CUDA_TAG}; \
    else \
      python scripts/install.py --skip-system; \
    fi

# Default shell entrypoint: override with any project command in docker run/compose.
CMD ["bash"]

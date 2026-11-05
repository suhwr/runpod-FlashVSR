# ==============================================================================
# ⚡ FlashVSR+ High-Performance Serverless Worker
# Architecture: One-Step Streaming Video Super-Resolution via Hybrid Attention
# Universal GPU Support: Ampere (sm_80/86) -> Ada (sm_89) -> Hopper (sm_90) -> Blackwell (sm_120)
# ==============================================================================
ARG CUDA_VERSION="12.8.0"
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_PREFER_BINARY=1 \
    CUDA_MODULE_LOADING=LAZY \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;12.0+PTX" \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    PYTHONPATH="/app:${PYTHONPATH}" \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# 1. Install system tools, ffmpeg, build packages, and Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    curl \
    git \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

# 2. Install PyTorch with CUDA 12.8 explicitly (Blackwell sm_120 + Ada RTX 4090 support)
RUN python3 -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple \
    torch torchvision torchaudio

# 3. Install worker dependencies
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    -r /app/requirements.txt

# 4. Copy FlashVSR+ framework code and install package
COPY pyproject.toml /app/pyproject.toml
COPY flashvsr_plus /app/flashvsr_plus
COPY inference /app/inference
COPY model_io /app/model_io
COPY models /app/models
COPY operators /app/operators
COPY prompt_tensor /app/prompt_tensor
RUN python3 -m pip install --no-cache-dir -e /app --no-deps

# 5. Pre-bake model weights during build (0ms cold-start latency)
COPY download_models.py /app/download_models.py
RUN python3 /app/download_models.py

# 6. Copy handler & docs
COPY handler.py /app/handler.py
COPY README.md /app/README.md

RUN python3 -m compileall -q /app || true

# 7. Direct entrypoint for RunPod serverless
CMD ["python3", "-u", "/app/handler.py"]

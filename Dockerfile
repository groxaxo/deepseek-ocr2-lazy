# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/root/.local/bin:$PATH" \
    DS_OCR2_SNAPSHOT_DIR="/data/models" \
    DS_OCR2_OUTPUT_ROOT="/data/output"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Alias python to python3.11
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch (CUDA 12.1 compatible)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Unsloth (No-deps method to prevent torch downgrade)
RUN pip install "unsloth[cu121-torch211] @ git+https://github.com/unslothai/unsloth.git" || \
    pip install --no-deps unsloth unsloth_zoo

# Install Server Dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Create app directory
WORKDIR /app
COPY . /app

# Create volumes for persistence
VOLUME /data

# Expose port
EXPOSE 8012

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8012/health || exit 1

# Start server
CMD ["python", "deepseek_ocr2_lazy_server.py"]

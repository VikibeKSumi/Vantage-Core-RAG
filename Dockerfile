FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH

WORKDIR /app

# System dependencies (fixed using update-alternatives — no symlink conflict)
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements first (better layer caching)
COPY requirements.txt .

# Install PyTorch with CUDA (critical for performance)
RUN pip install --no-cache-dir --default-timeout=1000 torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install rest of dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create persistent directories
RUN mkdir -p data/qdrant_storage config

# Make scripts executable
RUN chmod +x run.py ingestion.py run_eval.py

# Expose Streamlit port (for future app.py)
EXPOSE 8501

# Default command (CLI engine)
CMD ["python", "run.py"]
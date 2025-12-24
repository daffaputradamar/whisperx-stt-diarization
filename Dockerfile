# WhisperX Transcription API - Docker Image
# Uses NVIDIA CUDA base image for GPU acceleration

FROM nvidia/cuda:13.1.0-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Python settings
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA support first (for better caching)
RUN pip install --no-cache-dir \
    torch>=2.8.0 \
    torchaudio>=2.8.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user for security
RUN useradd -m -u 1000 appuser

# Copy application code
COPY app/ ./app/
COPY client_example.py .

# Create directories and set permissions
RUN mkdir -p /app/uploads \
    && mkdir -p /home/appuser/.cache/huggingface \
    && mkdir -p /home/appuser/.cache/matplotlib \
    && chown -R appuser:appuser /app \
    && chown -R appuser:appuser /home/appuser/.cache

# Set cache environment variables
ENV HF_HOME=/home/appuser/.cache/huggingface
ENV MPLCONFIGDIR=/home/appuser/.cache/matplotlib

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" || exit 1

# Default command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

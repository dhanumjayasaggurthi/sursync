# ── SurSync RunPod Worker ─────────────────────────────────────────────────────
#
# Uses nvidia/cuda runtime (~3 GB) instead of runpod/pytorch devel (~20 GB).
# Model weights are NOT baked in — they download on first pod start and are
# cached on a RunPod Network Volume so subsequent starts take only ~30 seconds.
#
# Build time:  ~10-15 min (was 2+ hours with the old base image)
# Image size:  ~5-6 GB    (was 20+ GB)
#
# Build + push:
#   docker build -t youruser/sursync-worker:latest .
#   docker push  youruser/sursync-worker:latest
# ─────────────────────────────────────────────────────────────────────────────

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Python 3.10 (Ubuntu 22.04 default) + ffmpeg
# libcublas-12-1 is required by CTranslate2 / faster-whisper for CUDA inference.
# It is normally included in the runtime image but we pin it explicitly to be safe.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg curl \
        libcublas-12-1 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Pre-create the cache directory so it exists even if the Network Volume
# is not mounted (the worker will fall back to this location).
RUN mkdir -p /runpod-volume/hf-cache

# Python packages
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
        "faster-whisper>=1.0.3" \
        "fastapi>=0.109.0" \
        "uvicorn[standard]>=0.27.0" \
        "python-multipart>=0.0.9" \
        "requests>=2.31.0"

# Application
WORKDIR /app
COPY runpod_worker.py /app/runpod_worker.py

# Model name — override with --build-arg WHISPER_MODEL=medium for a lighter model
ARG  WHISPER_MODEL=large-v3
ENV  WHISPER_MODEL=${WHISPER_MODEL}

# Whisper model cache dir — mount a RunPod Network Volume here to avoid
# re-downloading the 3 GB model on every pod start.
ENV  HF_HOME=/runpod-volume/hf-cache

EXPOSE 8000
CMD ["uvicorn", "runpod_worker:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]

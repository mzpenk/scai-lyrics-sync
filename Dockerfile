FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/app/models/torch
ENV XDG_CACHE_HOME=/app/models/cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Pre-download HTDemucs, Whisper large-v3-turbo, and Silero VAD into the image
# so RunPod Serverless cold starts never wait for model downloads
RUN mkdir -p /app/models/torch /app/models/cache && python3 -c "\
import torch, stable_whisper; \
from demucs.pretrained import get_model; \
print('Downloading HTDemucs...'); \
get_model('htdemucs'); \
print('Downloading Whisper large-v3-turbo...'); \
stable_whisper.load_model('turbo', device='cpu'); \
print('Downloading Silero VAD...'); \
torch.hub.load('snakers4/silero-vad', 'silero_vad', onnx=True, trust_repo=True); \
print('All models cached successfully!')"

COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "vllm==0.7.3" \
    "transformers<5" \
    "orpheus-speech" \
    "soundfile"

COPY run_orpheus.py /app/run_orpheus.py

ENTRYPOINT ["python", "/app/run_orpheus.py"]

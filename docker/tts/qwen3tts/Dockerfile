FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        sox \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    qwen-tts \
    soundfile \
    sox

COPY run_qwen3tts.py /app/run_qwen3tts.py

ENTRYPOINT ["python", "/app/run_qwen3tts.py"]

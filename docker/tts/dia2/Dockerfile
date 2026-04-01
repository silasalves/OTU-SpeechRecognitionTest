FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

ENV PYTHONUNBUFFERED=1
ENV DIA2_SRC=/opt/dia2
ENV PYTHONPATH=/opt/dia2

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/nari-labs/dia2 /opt/dia2

RUN pip install --no-cache-dir \
    "torch>=2.8.0" \
    "numpy>=2.1.0,<3.0" \
    "transformers>=4.55.3" \
    "safetensors==0.5.3" \
    "huggingface-hub>=0.24.7" \
    "sphn>=0.2.0" \
    "soundfile>=0.12.1" \
    "whisper-timestamped>=1.14.2" \
    "gradio>=4.44.1"

COPY run_dia2.py /app/run_dia2.py

ENTRYPOINT ["python", "/app/run_dia2.py"]

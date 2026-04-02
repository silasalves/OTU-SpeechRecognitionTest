FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/opt/CosyVoice

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        g++ \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --recursive --depth 1 --shallow-submodules https://github.com/FunAudioLLM/CosyVoice.git /opt/CosyVoice

RUN grep -v '^openai-whisper==20231117$' /opt/CosyVoice/requirements.txt > /tmp/cosyvoice-requirements.txt \
    && pip install --no-cache-dir "setuptools<81" \
    && pip install --no-cache-dir -r /tmp/cosyvoice-requirements.txt \
    && pip install --no-cache-dir --no-build-isolation openai-whisper==20231117 \
    && pip install --no-cache-dir huggingface_hub soundfile

COPY run_cosyvoice2.py /app/run_cosyvoice2.py

ENTRYPOINT ["python", "/app/run_cosyvoice2.py"]

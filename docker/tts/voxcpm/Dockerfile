FROM nvidia/cuda:13.0.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        ffmpeg \
        libsndfile1 \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --break-system-packages --no-cache-dir \
    voxcpm \
    soundfile

COPY run_voxcpm.py /app/run_voxcpm.py

ENTRYPOINT ["python3", "/app/run_voxcpm.py"]

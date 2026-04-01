FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

ENV PYTHONUNBUFFERED=1
ENV GLMTTS_SRC=/opt/GLM-TTS
ENV PYTHONPATH=/opt/GLM-TTS

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/zai-org/GLM-TTS /opt/GLM-TTS

RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.15.3 \
    sentencepiece==0.1.99 \
    soundfile==0.12.1 \
    tiktoken==0.6.0 \
    tqdm==4.66.4 \
    transformers==4.57.3 \
    WeTextProcessing==1.0.3 \
    x_transformers==1.42.26 \
    zhconv==1.4.3 \
    inflect==7.3.1 \
    onnxruntime-gpu==1.19.0 \
    contractions==0.1.73 \
    peft==0.7.0 \
    pronouncing==0.2.0 \
    pydub==0.25.1 \
    pypinyin==0.44.0 \
    PyYAML==6.0.3 \
    regex==2023.12.25 \
    requests==2.32.5 \
    safetensors==0.7.0 \
    hyperpyyaml==1.2.2 \
    ruamel.yaml==0.17.40 \
    jieba==0.42.1 \
    emoji==2.14.1 \
    librosa==0.9.2 \
    soxr==0.5.0.post1 \
    huggingface_hub

COPY run_glmtts.py /app/run_glmtts.py

ENTRYPOINT ["python", "/app/run_glmtts.py"]

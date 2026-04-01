from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from voxcpm import VoxCPM


def _soundfile_load(path: str, *args: object, **kwargs: object) -> tuple[torch.Tensor, int]:
    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), sample_rate


def main() -> None:
    model_id = os.environ["MODEL_ID"]
    text = os.environ["TEXT"]
    output_path = Path(os.environ["OUTPUT_PATH"])
    reference_audio_path = os.environ["REFERENCE_AUDIO_PATH"]
    reference_text = os.environ["REFERENCE_TEXT"]
    hf_cache_dir = os.environ.get("HF_HOME")

    # VoxCPM currently routes prompt audio loading through torchaudio's
    # torchcodec-backed loader. On this host that stack is not stable in Docker,
    # so we load WAV prompts through soundfile instead.
    torchaudio.load = _soundfile_load

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = VoxCPM.from_pretrained(
        model_id,
        cache_dir=hf_cache_dir,
        load_denoiser=False,
        optimize=False,
    )
    audio = model.generate(
        text=text,
        prompt_wav_path=reference_audio_path,
        prompt_text=reference_text,
    )
    sf.write(
        str(output_path),
        np.asarray(audio, dtype=np.float32),
        model.tts_model.sample_rate,
    )


if __name__ == "__main__":
    main()

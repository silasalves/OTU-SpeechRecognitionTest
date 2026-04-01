from __future__ import annotations

import os
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


def main() -> None:
    model_id = os.environ["MODEL_ID"]
    text = os.environ["TEXT"]
    output_path = Path(os.environ["OUTPUT_PATH"])
    reference_audio_path = os.environ["REFERENCE_AUDIO_PATH"]
    reference_text = os.environ["REFERENCE_TEXT"]
    language = os.environ.get("LANGUAGE", "French")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="sdpa",
    )
    wavs, sample_rate = model.generate_voice_clone(
        text=text,
        language=language,
        ref_audio=reference_audio_path,
        ref_text=reference_text,
    )
    sf.write(str(output_path), wavs[0], sample_rate)


if __name__ == "__main__":
    main()

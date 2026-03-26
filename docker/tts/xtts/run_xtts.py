from __future__ import annotations

import os

from TTS.api import TTS


def main() -> None:
    model_id = os.environ["MODEL_ID"]
    text = os.environ["TEXT"]
    output_path = os.environ["OUTPUT_PATH"]
    reference_audio_path = os.environ["REFERENCE_AUDIO_PATH"]
    language = os.environ.get("LANGUAGE", "fr")
    device = os.environ.get("DEVICE", "cuda")

    model = TTS(model_name=model_id, progress_bar=False, gpu=device == "cuda")
    model.tts_to_file(
        text=text,
        speaker_wav=reference_audio_path,
        language=language,
        file_path=output_path,
    )


if __name__ == "__main__":
    main()

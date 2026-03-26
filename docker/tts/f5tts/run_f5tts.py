from __future__ import annotations

import os

import tqdm

from f5_tts.api import F5TTS


def main() -> None:
    model_id = os.environ["MODEL_ID"]
    text = os.environ["TEXT"]
    output_path = os.environ["OUTPUT_PATH"]
    reference_audio_path = os.environ["REFERENCE_AUDIO_PATH"]
    reference_text = os.environ["REFERENCE_TEXT"]
    device = os.environ.get("DEVICE", "cuda")

    model = F5TTS(
        model=model_id,
        device=device,
        hf_cache_dir=os.environ.get("HF_HOME"),
    )
    model.infer(
        ref_file=reference_audio_path,
        ref_text=reference_text,
        gen_text=text,
        show_info=lambda *_args, **_kwargs: None,
        progress=tqdm,
        file_wave=output_path,
    )


if __name__ == "__main__":
    main()

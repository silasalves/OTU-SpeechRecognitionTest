from __future__ import annotations

import os

import soundfile as sf


def main() -> None:
    model_id = os.environ["MODEL_ID"]
    text = os.environ["TEXT"]
    output_path = os.environ["OUTPUT_PATH"]
    reference_audio_path = os.environ["REFERENCE_AUDIO_PATH"]
    language = os.environ.get("LANGUAGE", "fr")
    device = "cuda"

    if model_id == "ResembleAI/chatterbox-multilingual":
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        model = ChatterboxMultilingualTTS.from_pretrained(device)
        wav = model.generate(
            text,
            language_id=language,
            audio_prompt_path=reference_audio_path,
        )
    elif model_id == "ResembleAI/chatterbox-turbo":
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        model = ChatterboxTurboTTS.from_pretrained(device)
        wav = model.generate(text, audio_prompt_path=reference_audio_path)
    elif model_id == "ResembleAI/chatterbox":
        from chatterbox.tts import ChatterboxTTS

        model = ChatterboxTTS.from_pretrained(device)
        wav = model.generate(text, audio_prompt_path=reference_audio_path)
    else:
        raise SystemExit(f"Unsupported chatterbox model: {model_id}")

    audio = wav.detach().cpu().numpy().squeeze(0)
    sf.write(output_path, audio, model.sr)


if __name__ == "__main__":
    main()

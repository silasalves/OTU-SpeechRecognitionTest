from __future__ import annotations

import copy
import importlib.util
import os
from pathlib import Path

import torch
from transformers.utils import logging
from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_streaming_processor import (
    VibeVoiceStreamingProcessor,
)


logging.set_verbosity_error()

_DEFAULT_SPEAKERS = {
    "fr": "fr-Spk1_woman",
    "fr-ca": "fr-Spk1_woman",
    "en": "en-Emma_woman",
}


def main() -> None:
    repo_root = Path(os.environ["REPO_ROOT"]).resolve()
    model_id = os.environ.get("MODEL_ID", "microsoft/VibeVoice-Realtime-0.5B")
    text = os.environ["TEXT"]
    output_path = Path(os.environ["OUTPUT_PATH"]).resolve()
    locale = os.environ.get("LOCALE", "").strip().lower()
    language = os.environ.get("LANGUAGE", "").strip().lower()
    requested_speaker = os.environ.get("SPEAKER_NAME", "").strip()
    cfg_scale = float(os.environ.get("CFG_SCALE", "1.5"))
    ddpm_steps = int(os.environ.get("DDPM_STEPS", "5"))

    target_device = _resolve_device(os.environ.get("DEVICE", "auto"))
    speaker_name = requested_speaker or _default_speaker(locale, language)
    voices_dir = repo_root / ".tmp" / "VibeVoice-official" / "demo" / "voices" / "streaming_model"
    voice_sample = _resolve_voice_sample(voices_dir, speaker_name)

    processor = VibeVoiceStreamingProcessor.from_pretrained(model_id)
    model = _load_model(model_id, target_device)
    model.eval()
    model.set_ddpm_inference_steps(num_steps=ddpm_steps)

    cached_prompt = torch.load(
        voice_sample,
        map_location=target_device,
        weights_only=False,
    )
    inputs = processor.process_input_with_cached_prompt(
        text=text.replace("’", "'").replace("“", '"').replace("”", '"'),
        cached_prompt=cached_prompt,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    for key, value in inputs.items():
        if torch.is_tensor(value):
            inputs[key] = value.to(target_device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=None,
        cfg_scale=cfg_scale,
        tokenizer=processor.tokenizer,
        generation_config={"do_sample": False},
        verbose=False,
        all_prefilled_outputs=copy.deepcopy(cached_prompt),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processor.save_audio(outputs.speech_outputs[0], output_path=str(output_path))
    print(f"saved={output_path}")
    print(f"speaker={speaker_name}")
    print(f"device={target_device}")


def _load_model(
    model_id: str,
    target_device: str,
) -> VibeVoiceStreamingForConditionalGenerationInference:
    if target_device == "cuda":
        load_dtype = torch.bfloat16
    else:
        load_dtype = torch.float32

    attn_implementation = _attn_implementation(target_device)
    try:
        if target_device == "cuda":
            return VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                model_id,
                torch_dtype=load_dtype,
                device_map="cuda",
                attn_implementation=attn_implementation,
            )
        return VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            model_id,
            torch_dtype=load_dtype,
            device_map="cpu",
            attn_implementation=attn_implementation,
        )
    except Exception:
        if attn_implementation != "flash_attention_2":
            raise
        if target_device != "cuda":
            raise
        return VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            model_id,
            torch_dtype=load_dtype,
            device_map="cuda",
            attn_implementation="sdpa",
        )


def _resolve_device(device: str) -> str:
    normalized = device.strip().lower()
    if normalized == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if normalized == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _attn_implementation(target_device: str) -> str:
    if target_device == "cuda" and importlib.util.find_spec("flash_attn") is not None:
        return "flash_attention_2"
    return "sdpa"


def _default_speaker(locale: str, language: str) -> str:
    if locale in _DEFAULT_SPEAKERS:
        return _DEFAULT_SPEAKERS[locale]
    if language in _DEFAULT_SPEAKERS:
        return _DEFAULT_SPEAKERS[language]
    return _DEFAULT_SPEAKERS["en"]


def _resolve_voice_sample(voices_dir: Path, speaker_name: str) -> Path:
    voice_map: dict[str, Path] = {}
    for path in voices_dir.rglob("*.pt"):
        voice_map[path.stem.lower()] = path.resolve()
    if not voice_map:
        raise RuntimeError(f"No VibeVoice voice presets were found under {voices_dir}")
    key = speaker_name.strip().lower()
    if key in voice_map:
        return voice_map[key]
    matched = [
        path
        for name, path in voice_map.items()
        if key in name or name in key
    ]
    if len(matched) == 1:
        return matched[0]
    if len(matched) > 1:
        raise RuntimeError(
            "Multiple VibeVoice speaker presets matched "
            f"'{speaker_name}': {', '.join(sorted(path.stem for path in matched))}"
        )
    raise RuntimeError(
        f"Requested VibeVoice speaker '{speaker_name}' was not found. "
        f"Available speakers: {', '.join(sorted(voice_map))}"
    )


if __name__ == "__main__":
    main()

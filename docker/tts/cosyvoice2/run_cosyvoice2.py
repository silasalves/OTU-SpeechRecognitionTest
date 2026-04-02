from __future__ import annotations

import os
from pathlib import Path
import sys
from urllib.request import urlretrieve

from huggingface_hub import snapshot_download
import numpy as np
import soundfile as sf


_PREFERRED_SPEAKERS = ("longxiaoxia", "xiaohe", "001")


def main() -> None:
    model_id = os.environ["MODEL_ID"]
    text = os.environ["TEXT"]
    output_path = Path(os.environ["OUTPUT_PATH"])
    requested_speaker = os.environ.get("SPEAKER_ID", "").strip()
    spk2info_url = os.environ["SPK2INFO_URL"]
    text_frontend = os.environ.get("TEXT_FRONTEND", "false").strip().lower() == "true"

    checkpoint_dir = _resolve_checkpoint(model_id)
    _ensure_spk2info(checkpoint_dir, spk2info_url)

    sys.path.append("/opt/CosyVoice/third_party/Matcha-TTS")
    from cosyvoice.cli.cosyvoice import AutoModel

    model = AutoModel(model_dir=str(checkpoint_dir))
    speaker_id = _resolve_speaker(model.list_available_spks(), requested_speaker)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    chunks: list[np.ndarray] = []
    for result in model.inference_sft(
        text,
        speaker_id,
        stream=False,
        text_frontend=text_frontend,
    ):
        audio = result["tts_speech"].detach().cpu().numpy().reshape(-1)
        chunks.append(np.asarray(audio, dtype=np.float32))
    if not chunks:
        raise RuntimeError("CosyVoice2 produced no audio chunks.")
    sf.write(str(output_path), np.concatenate(chunks), model.sample_rate)


def _resolve_checkpoint(model_id: str) -> Path:
    hf_home = Path(os.environ.get("HF_HOME", "/workspace/.cache/huggingface"))
    checkpoint_dir = hf_home / "cosyvoice2" / _slug(model_id)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if not (checkpoint_dir / "cosyvoice2.yaml").exists():
        snapshot_download(repo_id=model_id, local_dir=str(checkpoint_dir))
    return checkpoint_dir


def _ensure_spk2info(checkpoint_dir: Path, url: str) -> None:
    target = checkpoint_dir / "spk2info.pt"
    if target.exists():
        return
    urlretrieve(url, target)


def _resolve_speaker(available_speakers: list[str], requested_speaker: str) -> str:
    if not available_speakers:
        raise RuntimeError(
            "CosyVoice2 fixed-speaker cache did not expose any available speakers."
        )
    if requested_speaker:
        if requested_speaker not in available_speakers:
            raise RuntimeError(
                f"Requested CosyVoice2 speaker '{requested_speaker}' was not found. "
                f"Available speakers: {', '.join(available_speakers)}"
            )
        return requested_speaker
    for speaker_id in _PREFERRED_SPEAKERS:
        if speaker_id in available_speakers:
            return speaker_id
    return available_speakers[0]


def _slug(value: str) -> str:
    return (
        value.replace("\\", "-")
        .replace("/", "-")
        .replace(":", "-")
        .replace(" ", "-")
    )


if __name__ == "__main__":
    main()

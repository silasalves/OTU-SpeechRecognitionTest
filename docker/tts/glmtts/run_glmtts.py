from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download
import torch
import torchaudio


def _slug(value: str) -> str:
    return (
        value.replace("\\", "-")
        .replace("/", "-")
        .replace(":", "-")
        .replace(" ", "-")
    )


def _prepare_checkpoint(model_id: str) -> Path:
    hf_home = Path(os.environ.get("HF_HOME", "/workspace/.cache/huggingface"))
    checkpoint_dir = hf_home / "glmtts" / _slug(model_id)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(model_id, local_dir=str(checkpoint_dir))
    return checkpoint_dir


def _link_checkpoint(repo_root: Path, checkpoint_dir: Path) -> None:
    target = repo_root / "ckpt"
    if target.is_symlink():
        if target.resolve() == checkpoint_dir.resolve():
            return
        target.unlink()
    elif target.exists():
        raise RuntimeError(f"GLM-TTS repo already contains a real ckpt directory at {target}")
    target.symlink_to(checkpoint_dir, target_is_directory=True)


def main() -> None:
    model_id = os.environ["MODEL_ID"]
    text = os.environ["TEXT"]
    output_path = Path(os.environ["OUTPUT_PATH"])
    reference_audio_path = os.environ["REFERENCE_AUDIO_PATH"]
    reference_text = os.environ["REFERENCE_TEXT"]
    sample_rate = int(os.environ.get("SAMPLE_RATE", "24000"))
    repo_root = Path(os.environ.get("GLMTTS_SRC", "/opt/GLM-TTS"))

    checkpoint_dir = _prepare_checkpoint(model_id)
    _link_checkpoint(repo_root, checkpoint_dir)
    os.chdir(repo_root)

    import glmtts_inference as upstream

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frontend, text_frontend, _speech_tokenizer, llm, flow = upstream.load_models(
        sample_rate=sample_rate
    )
    prompt_text = text_frontend.text_normalize(reference_text)
    synth_text = text_frontend.text_normalize(text)
    if not prompt_text:
        raise RuntimeError("GLM-TTS prompt text normalization produced no usable text.")
    if not synth_text:
        raise RuntimeError("GLM-TTS synthesis text normalization produced no usable text.")

    prompt_text_token = frontend._extract_text_token(prompt_text + " ")
    prompt_speech_token = frontend._extract_speech_token([reference_audio_path])
    speech_feat = frontend._extract_speech_feat(reference_audio_path, sample_rate=sample_rate)
    embedding = frontend._extract_spk_embedding(reference_audio_path)
    cache_speech_token = [prompt_speech_token.squeeze().tolist()]
    flow_prompt_token = torch.tensor(cache_speech_token, dtype=torch.int32).to(
        upstream.DEVICE
    )
    cache = {
        "cache_text": [prompt_text],
        "cache_text_token": [prompt_text_token],
        "cache_speech_token": cache_speech_token,
        "use_cache": True,
    }

    tts_speech, _tts_mel, _token_list, _text_info = upstream.generate_long(
        frontend=frontend,
        text_frontend=text_frontend,
        llm=llm,
        flow=flow,
        text_info=["otu", synth_text],
        cache=cache,
        embedding=embedding,
        seed=0,
        flow_prompt_token=flow_prompt_token,
        speech_feat=speech_feat,
        device=upstream.DEVICE,
        use_phoneme=False,
    )
    torchaudio.save(str(output_path), tts_speech, sample_rate)


if __name__ == "__main__":
    main()

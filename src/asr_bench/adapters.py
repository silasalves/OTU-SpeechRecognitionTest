from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import signal
from typing import Any

import numpy as np
import soundfile as sf


@dataclass
class BenchmarkContext:
    device: str
    language: str


class AdapterError(RuntimeError):
    pass


class BaseAdapter:
    def __init__(self, model_id: str, context: BenchmarkContext) -> None:
        self.model_id = model_id
        self.context = context

    def transcribe(self, audio_path: Path) -> str:
        raise NotImplementedError


class WhisperAdapter(BaseAdapter):
    def __init__(self, model_id: str, context: BenchmarkContext) -> None:
        super().__init__(model_id, context)
        try:
            import whisper
        except ImportError as exc:
            raise AdapterError(_missing_dependency("whisper")) from exc
        device = None if context.device == "auto" else context.device
        download_root = Path(
            os.environ.get("WHISPER_CACHE_DIR", str(Path.cwd() / ".cache" / "whisper"))
        )
        download_root.mkdir(parents=True, exist_ok=True)
        self._model = whisper.load_model(
            model_id,
            device=device,
            download_root=str(download_root),
        )

    def transcribe(self, audio_path: Path) -> str:
        audio = _load_audio_for_whisper(audio_path)
        result = self._model.transcribe(
            audio,
            language=self.context.language,
            task="transcribe",
            fp16=self.context.device.startswith("cuda"),
        )
        return str(result["text"]).strip()


class MoonshineAdapter(BaseAdapter):
    def __init__(self, model_id: str, context: BenchmarkContext) -> None:
        super().__init__(model_id, context)
        try:
            from huggingface_hub import hf_hub_download
            from moonshine_onnx import MoonshineOnnxModel
            import onnxruntime as ort
        except ImportError as exc:
            raise AdapterError(_missing_dependency("moonshine")) from exc
        model_name = model_id.split("/")[-1]
        models_root = Path.cwd() / ".cache" / "moonshine" / model_name
        models_root.mkdir(parents=True, exist_ok=True)
        for filename in ("encoder_model.onnx", "decoder_model_merged.onnx"):
            hf_hub_download(
                repo_id="UsefulSensors/moonshine",
                filename=filename,
                subfolder=f"onnx/merged/{model_name}/float",
                local_dir=models_root,
            )
        available_providers = set(ort.get_available_providers())
        providers = ["CPUExecutionProvider"]
        if self.context.device.startswith("cuda"):
            if "CUDAExecutionProvider" not in available_providers:
                raise AdapterError(
                    "Moonshine GPU requested, but ONNX Runtime CUDA provider is unavailable."
                )
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._model = MoonshineOnnxModel(
            models_dir=str(models_root / "onnx" / "merged" / model_name / "float"),
            model_name=model_name,
            model_precision="float",
        )
        self._model.encoder.set_providers(providers)
        self._model.decoder.set_providers(providers)

    def transcribe(self, audio_path: Path) -> str:
        from moonshine_onnx import transcribe

        result = transcribe(str(audio_path), self._model)
        if isinstance(result, list):
            return " ".join(str(item).strip() for item in result if str(item).strip())
        return str(result).strip()


class ParakeetAdapter(BaseAdapter):
    def __init__(self, model_id: str, context: BenchmarkContext) -> None:
        super().__init__(model_id, context)
        _ensure_windows_signal_compat()
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError as exc:
            raise AdapterError(_missing_dependency("nemo")) from exc
        checkpoint_path = _download_hf_repo_artifact(model_id, ".nemo")
        self._model = nemo_asr.models.ASRModel.restore_from(str(checkpoint_path))
        if context.device == "cuda":
            self._model = self._model.cuda()
        self._model = self._model.eval()

    def transcribe(self, audio_path: Path) -> str:
        prepared_audio = _prepare_mono_16khz_wav(audio_path)
        outputs = self._model.transcribe(
            [str(prepared_audio)],
            batch_size=1,
            num_workers=0,
            verbose=False,
        )
        return _extract_nemo_text(outputs[0])


class CanaryAdapter(BaseAdapter):
    def __init__(self, model_id: str, context: BenchmarkContext) -> None:
        super().__init__(model_id, context)
        _ensure_windows_signal_compat()
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError as exc:
            raise AdapterError(_missing_dependency("nemo")) from exc
        checkpoint_path = _download_hf_repo_artifact(model_id, ".nemo")
        self._model = nemo_asr.models.ASRModel.restore_from(str(checkpoint_path))
        if context.device == "cuda":
            self._model = self._model.cuda()
        self._model = self._model.eval()

    def transcribe(self, audio_path: Path) -> str:
        prepared_audio = _prepare_mono_16khz_wav(audio_path)
        kwargs = {
            "audio": [str(prepared_audio)],
            "batch_size": 1,
            "num_workers": 0,
            "verbose": False,
            "source_lang": self.context.language,
            "target_lang": self.context.language,
            "taskname": "asr",
            "pnc": "yes",
        }
        try:
            outputs = self._model.transcribe(**kwargs)
        except TypeError:
            kwargs.pop("taskname", None)
            kwargs.pop("pnc", None)
            outputs = self._model.transcribe(**kwargs)
        return _extract_nemo_text(outputs[0])


class OwsmCtcAdapter(BaseAdapter):
    def __init__(self, model_id: str, context: BenchmarkContext) -> None:
        super().__init__(model_id, context)
        try:
            from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise AdapterError(_missing_dependency("espnet")) from exc
        repo_name = model_id.split("/")[-1]
        local_dir = Path.cwd() / ".cache" / "espnet" / repo_name
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
        self._model = Speech2TextGreedySearch(
            s2t_train_config=str(_rewrite_owsm_config(local_dir)),
            s2t_model_file=str(local_dir / _owsm_model_relpath()),
            device=context.device if context.device in {"cpu", "cuda"} else "cpu",
            use_flash_attn=False,
            generate_interctc_outputs=False,
            lang_sym=_espnet_lang_symbol(context.language),
            task_sym="<asr>",
        )

    def transcribe(self, audio_path: Path) -> str:
        speech = _load_audio_mono_16khz(audio_path)
        result = self._model.batch_decode(speech, batch_size=1, context_len_in_secs=4)
        if isinstance(result, list):
            return " ".join(str(item).strip() for item in result if str(item).strip())
        return str(result).strip()


class FunAsrAdapter(BaseAdapter):
    def __init__(self, model_id: str, context: BenchmarkContext) -> None:
        super().__init__(model_id, context)
        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise AdapterError(_missing_dependency("funasr")) from exc
        device = "cpu" if context.device == "auto" else context.device
        resolved_model_id = model_id
        hub = "hf"
        if model_id == "iic/SenseVoiceSmall":
            resolved_model_id = "FunAudioLLM/SenseVoiceSmall"
        self._model = AutoModel(
            model=resolved_model_id,
            device=device,
            hub=hub,
            trust_remote_code=False,
            disable_update=True,
            vad_model=None,
            punc_model=None,
            spk_model=None,
        )

    def transcribe(self, audio_path: Path) -> str:
        prepared_audio = _load_audio_mono_16khz(audio_path)
        result = self._model.generate(
            input=prepared_audio,
            batch_size_s=0,
            language=self.context.language,
        )
        if isinstance(result, list):
            if not result:
                return ""
            item = result[0]
            if isinstance(item, dict):
                return _clean_funasr_text(str(item.get("text", "")).strip())
            return _clean_funasr_text(str(item).strip())
        if isinstance(result, dict):
            return _clean_funasr_text(str(result.get("text", "")).strip())
        return _clean_funasr_text(str(result).strip())


def create_adapter(engine: str, model_id: str, context: BenchmarkContext) -> BaseAdapter:
    factories = {
        "whisper": WhisperAdapter,
        "moonshine": MoonshineAdapter,
        "parakeet": ParakeetAdapter,
        "canary": CanaryAdapter,
        "owsm-ctc": OwsmCtcAdapter,
        "funasr": FunAsrAdapter,
    }
    try:
        adapter_cls = factories[engine]
    except KeyError as exc:
        raise AdapterError(f"Unknown ASR engine: {engine}") from exc
    return adapter_cls(model_id, context)


def _extract_nemo_text(value: Any) -> str:
    if hasattr(value, "text"):
        return str(value.text).strip()
    if isinstance(value, dict) and "text" in value:
        return str(value["text"]).strip()
    return str(value).strip()


def _load_audio_for_whisper(audio_path: Path) -> np.ndarray:
    return _load_audio_mono_16khz(audio_path)


def _prepare_mono_16khz_wav(audio_path: Path) -> Path:
    output_dir = Path.cwd() / ".cache" / "prepared-audio"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{audio_path.parent.name}-{audio_path.stem}-mono16k.wav"
    if output_path.exists():
        return output_path

    audio = _load_audio_mono_16khz(audio_path)

    sf.write(str(output_path), audio, 16000)
    return output_path


def _load_audio_mono_16khz(audio_path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(str(audio_path), always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        target_length = max(1, round(len(audio) * 16000 / sample_rate))
        original_positions = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        target_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
        audio = np.interp(target_positions, original_positions, audio).astype(np.float32)
    return audio


def _download_hf_repo_artifact(repo_id: str, suffix: str) -> Path:
    from huggingface_hub import hf_hub_download, list_repo_files

    repo_name = repo_id.split("/")[-1]
    local_dir = Path.cwd() / ".cache" / "huggingface-local" / repo_name
    local_dir.mkdir(parents=True, exist_ok=True)
    candidate = next((name for name in list_repo_files(repo_id) if name.endswith(suffix)), None)
    if candidate is None:
        raise AdapterError(f"No {suffix} artifact found in {repo_id}")
    path = hf_hub_download(repo_id=repo_id, filename=candidate, local_dir=local_dir)
    return Path(path)


def _ensure_windows_signal_compat() -> None:
    if not hasattr(signal, "SIGKILL"):
        signal.SIGKILL = signal.SIGTERM


def _missing_dependency(extra_name: str) -> str:
    return (
        f"Required dependencies for '{extra_name}' are not installed. "
        f"Install them with: uv sync --extra {extra_name}"
    )


def _espnet_lang_symbol(language: str) -> str:
    normalized = language.strip().lower()
    mapping = {
        "fr": "<fra>",
        "fr-ca": "<fra>",
    }
    return mapping.get(normalized, "<nolang>")


def _owsm_config_relpath() -> Path:
    return Path(
        "exp/s2t_train_s2t_multitask-ctc_ebf27_conv2d8_size1024_raw_bpe50000/config.yaml"
    )


def _owsm_model_relpath() -> Path:
    return Path(
        "exp/s2t_train_s2t_multitask-ctc_ebf27_conv2d8_size1024_raw_bpe50000/valid.total_count.ave_5best.till40epoch.pth"
    )


def _rewrite_owsm_config(model_root: Path) -> Path:
    import yaml
    from espnet2.main_funcs.pack_funcs import find_path_and_change_it_recursive

    source_config = model_root / _owsm_config_relpath()
    rewritten_config = model_root / ".asr_bench_s2t_config.yaml"
    if rewritten_config.exists():
        return rewritten_config

    with source_config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    for path in model_root.glob("**/*"):
        if not path.is_file():
            continue
        relative_path = path.relative_to(model_root)
        config = find_path_and_change_it_recursive(config, str(relative_path), str(path))

    with rewritten_config.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return rewritten_config


def _clean_funasr_text(value: str) -> str:
    return re.sub(r"<\|[^>]+\|>", "", value).strip()


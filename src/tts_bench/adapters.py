from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess

import numpy as np
import soundfile as sf

from tts_bench.docker_runtime import container_path, ensure_image, run_container


@dataclass(frozen=True)
class ReferencePrompt:
    locale: str
    audio_path: Path
    transcript: str


@dataclass(frozen=True)
class BenchmarkContext:
    device: str
    language: str
    reference_prompts: dict[str, ReferencePrompt]


@dataclass(frozen=True)
class SynthesisRequest:
    locale: str
    text: str
    output_path: Path
    reference_audio_path: Path | None
    reference_text: str


class AdapterError(RuntimeError):
    pass


class BaseAdapter:
    def __init__(self, model_id: str, context: BenchmarkContext) -> None:
        self.model_id = model_id
        self.context = context

    def synthesize(self, request: SynthesisRequest) -> None:
        raise NotImplementedError


class KokoroAdapter(BaseAdapter):
    def __init__(self, model_id: str, context: BenchmarkContext) -> None:
        super().__init__(model_id, context)
        try:
            from kokoro import KPipeline
        except ImportError as exc:
            raise AdapterError(_missing_dependency("kokoro")) from exc
        self._sample_rate = 24000
        self._voice = os.environ.get("TTS_BENCH_KOKORO_VOICE", "ff_siwis")
        self._pipeline = KPipeline(lang_code="f")

    def synthesize(self, request: SynthesisRequest) -> None:
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        chunks: list[np.ndarray] = []
        for _, _, audio in self._pipeline(
            request.text,
            voice=self._voice,
            speed=1.0,
            split_pattern=r"\n+",
        ):
            chunks.append(np.asarray(audio, dtype=np.float32))
        if not chunks:
            raise AdapterError("Kokoro produced no audio chunks.")
        sf.write(str(request.output_path), np.concatenate(chunks), self._sample_rate)


class ChatterboxAdapter(BaseAdapter):
    def __init__(self, model_id: str, context: BenchmarkContext) -> None:
        super().__init__(model_id, context)
        try:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        except ImportError as exc:
            raise AdapterError(_missing_dependency("chatterbox")) from exc
        self._model = ChatterboxMultilingualTTS.from_pretrained(
            device=_resolve_device(context.device)
        )

    def synthesize(self, request: SynthesisRequest) -> None:
        if request.reference_audio_path is None:
            raise AdapterError("Chatterbox requires a reference audio prompt.")
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        wav = self._model.generate(
            request.text,
            language_id=_chatterbox_language(self.context.language, request.locale),
            audio_prompt_path=str(request.reference_audio_path),
        )
        audio = wav.detach().cpu().numpy().squeeze(0)
        sf.write(str(request.output_path), audio, self._model.sr)


class KittenTtsAdapter(BaseAdapter):
    def __init__(self, model_id: str, context: BenchmarkContext) -> None:
        super().__init__(model_id, context)
        try:
            from kittentts import KittenTTS
        except ImportError as exc:
            raise AdapterError(_missing_dependency("kittentts")) from exc
        cache_dir = str(Path.cwd() / ".cache" / "huggingface")
        self._model = KittenTTS(model_id, cache_dir=cache_dir)
        voices = list(getattr(self._model, "available_voices", []) or [])
        requested_voice = os.environ.get("TTS_BENCH_KITTEN_VOICE", "Jasper")
        self._voice = requested_voice if requested_voice in voices else (voices[0] if voices else requested_voice)

    def synthesize(self, request: SynthesisRequest) -> None:
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        audio = self._model.generate(
            request.text,
            voice=self._voice,
            speed=1.0,
            clean_text=True,
        )
        sf.write(str(request.output_path), np.asarray(audio, dtype=np.float32), 24000)


class DockerBackedChatterboxAdapter(BaseAdapter):
    def __init__(self, model_id: str, context: BenchmarkContext) -> None:
        super().__init__(model_id, context)
        docker_dir = Path.cwd() / "docker" / "tts" / "chatterbox"
        ensure_image("otu-tts-bench-chatterbox:latest", docker_dir)

    def synthesize(self, request: SynthesisRequest) -> None:
        if request.reference_audio_path is None:
            raise AdapterError("Chatterbox requires a reference audio prompt.")
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        run_container(
            image_tag="otu-tts-bench-chatterbox:latest",
            wants_gpu=self.context.device == "cuda",
            env={
                "MODEL_ID": self.model_id,
                "TEXT": request.text,
                "OUTPUT_PATH": container_path(request.output_path),
                "REFERENCE_AUDIO_PATH": container_path(request.reference_audio_path),
                "LANGUAGE": _chatterbox_language(self.context.language, request.locale),
            },
        )


class F5TtsAdapter(BaseAdapter):
    def __init__(self, model_id: str, context: BenchmarkContext) -> None:
        super().__init__(model_id, context)
        try:
            from f5_tts.api import F5TTS
        except ImportError as exc:
            raise AdapterError(_missing_dependency("f5tts")) from exc
        device = None if context.device == "auto" else context.device
        self._model = F5TTS(
            model=model_id,
            device=device,
            hf_cache_dir=str(Path.cwd() / ".cache" / "huggingface"),
        )

    def synthesize(self, request: SynthesisRequest) -> None:
        if request.reference_audio_path is None:
            raise AdapterError("F5-TTS requires a reference audio prompt.")
        import tqdm

        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._model.infer(
            ref_file=str(request.reference_audio_path),
            ref_text=request.reference_text,
            gen_text=request.text,
            show_info=lambda *_args, **_kwargs: None,
            progress=tqdm,
            file_wave=str(request.output_path),
        )


class DockerBackedF5TtsAdapter(BaseAdapter):
    def __init__(self, model_id: str, context: BenchmarkContext) -> None:
        super().__init__(model_id, context)

    def synthesize(self, request: SynthesisRequest) -> None:
        if request.reference_audio_path is None:
            raise AdapterError("F5-TTS requires a reference audio prompt.")
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        run_container(
            image_tag="ghcr.io/swivid/f5-tts:main",
            wants_gpu=self.context.device == "cuda",
            mount_target="/workspace-host",
            workdir="/workspace/F5-TTS",
            entrypoint="bash",
            env={
                "MODEL_ID": self.model_id,
                "TEXT": request.text,
                "OUTPUT_PATH": container_path(request.output_path, "/workspace-host"),
                "REFERENCE_AUDIO_PATH": container_path(
                    request.reference_audio_path,
                    "/workspace-host",
                ),
                "REFERENCE_TEXT": request.reference_text,
                "DEVICE": _resolve_device(self.context.device),
            },
            command_args=[
                "-lc",
                (
                    "export PYTHONPATH=/workspace/F5-TTS/src; "
                    'python src/f5_tts/infer/infer_cli.py '
                    '--model "$MODEL_ID" '
                    '--ref_audio "$REFERENCE_AUDIO_PATH" '
                    '--ref_text "$REFERENCE_TEXT" '
                    '--gen_text "$TEXT" '
                    '--output_dir "$(dirname "$OUTPUT_PATH")" '
                    '--output_file "$(basename "$OUTPUT_PATH")" '
                    '--device "$DEVICE"'
                ),
            ],
        )


class DockerBackedVoxCpmAdapter(BaseAdapter):
    def __init__(self, model_id: str, context: BenchmarkContext) -> None:
        super().__init__(model_id, context)
        docker_dir = Path.cwd() / "docker" / "tts" / "voxcpm"
        ensure_image("otu-tts-bench-voxcpm:latest", docker_dir)

    def synthesize(self, request: SynthesisRequest) -> None:
        if request.reference_audio_path is None:
            raise AdapterError("VoxCPM requires a reference audio prompt.")
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        run_container(
            image_tag="otu-tts-bench-voxcpm:latest",
            wants_gpu=self.context.device == "cuda",
            env={
                "MODEL_ID": self.model_id,
                "TEXT": request.text,
                "OUTPUT_PATH": container_path(request.output_path),
                "REFERENCE_AUDIO_PATH": container_path(request.reference_audio_path),
                "REFERENCE_TEXT": request.reference_text,
                "MODELSCOPE_CACHE": "/workspace/.cache/modelscope",
            },
        )


class DockerBackedQwen3TtsAdapter(BaseAdapter):
    def __init__(self, model_id: str, context: BenchmarkContext) -> None:
        super().__init__(model_id, context)
        docker_dir = Path.cwd() / "docker" / "tts" / "qwen3tts"
        ensure_image("otu-tts-bench-qwen3tts:latest", docker_dir)

    def synthesize(self, request: SynthesisRequest) -> None:
        if request.reference_audio_path is None:
            raise AdapterError("Qwen3-TTS requires a reference audio prompt.")
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        run_container(
            image_tag="otu-tts-bench-qwen3tts:latest",
            wants_gpu=self.context.device == "cuda",
            env={
                "MODEL_ID": self.model_id,
                "TEXT": request.text,
                "OUTPUT_PATH": container_path(request.output_path),
                "REFERENCE_AUDIO_PATH": container_path(request.reference_audio_path),
                "REFERENCE_TEXT": request.reference_text,
                "LANGUAGE": _qwen3_language(self.context.language, request.locale),
            },
        )


class DockerBackedCosyVoice2Adapter(BaseAdapter):
    def __init__(self, model_id: str, context: BenchmarkContext) -> None:
        super().__init__(model_id, context)
        docker_dir = Path.cwd() / "docker" / "tts" / "cosyvoice2"
        ensure_image("otu-tts-bench-cosyvoice2:latest", docker_dir)

    def synthesize(self, request: SynthesisRequest) -> None:
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        run_container(
            image_tag="otu-tts-bench-cosyvoice2:latest",
            wants_gpu=self.context.device == "cuda",
            env={
                "MODEL_ID": self.model_id,
                "TEXT": request.text,
                "OUTPUT_PATH": container_path(request.output_path),
                "SPEAKER_ID": os.environ.get(
                    "TTS_BENCH_COSYVOICE2_SPEAKER_ID",
                    "longxiaoxia",
                ),
                "SPK2INFO_URL": os.environ.get(
                    "TTS_BENCH_COSYVOICE2_SPK2INFO_URL",
                    "https://raw.githubusercontent.com/qi-hua/async_cosyvoice/main/CosyVoice2-0.5B/spk2info.pt",
                ),
                "TEXT_FRONTEND": os.environ.get(
                    "TTS_BENCH_COSYVOICE2_TEXT_FRONTEND",
                    "false",
                ),
            },
        )


class LocalVenvVibeVoiceAdapter(BaseAdapter):
    def __init__(self, model_id: str, context: BenchmarkContext) -> None:
        super().__init__(model_id, context)
        self._venv_python = _resolve_vibevoice_python()
        self._repo_path = Path.cwd() / ".tmp" / "VibeVoice-official"
        self._runner_path = Path.cwd() / "tools" / "vibevoice" / "run_vibevoice.py"
        if not self._venv_python.exists():
            raise AdapterError(
                "VibeVoice venv is not installed. Run "
                ".\\scripts\\install-tts-backends.ps1 -Engine vibevoice first."
            )
        if not self._repo_path.exists():
            raise AdapterError(
                "Official VibeVoice repo is missing at .tmp/VibeVoice-official. "
                "Run .\\scripts\\install-tts-backends.ps1 -Engine vibevoice first."
            )
        if not self._runner_path.exists():
            raise AdapterError(f"VibeVoice runner script is missing: {self._runner_path}")

    def synthesize(self, request: SynthesisRequest) -> None:
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(
            {
                "MODEL_ID": self.model_id,
                "TEXT": request.text,
                "OUTPUT_PATH": str(request.output_path),
                "LOCALE": request.locale,
                "LANGUAGE": self.context.language,
                "DEVICE": self.context.device,
                "REPO_ROOT": str(Path.cwd()),
                "SPEAKER_NAME": os.environ.get("TTS_BENCH_VIBEVOICE_SPEAKER_NAME", ""),
                "CFG_SCALE": os.environ.get("TTS_BENCH_VIBEVOICE_CFG_SCALE", "1.5"),
                "DDPM_STEPS": os.environ.get("TTS_BENCH_VIBEVOICE_DDPM_STEPS", "5"),
                "PYTHONIOENCODING": "utf-8",
            }
        )
        result = subprocess.run(
            [str(self._venv_python), str(self._runner_path)],
            cwd=Path.cwd(),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode != 0:
            raise AdapterError(
                "VibeVoice local runner failed: "
                f"{_tail_text(result.stderr or result.stdout)}"
            )


class DockerBackedDia2Adapter(BaseAdapter):
    def __init__(self, model_id: str, context: BenchmarkContext) -> None:
        super().__init__(model_id, context)
        docker_dir = Path.cwd() / "docker" / "tts" / "dia2"
        ensure_image("otu-tts-bench-dia2:latest", docker_dir)

    def synthesize(self, request: SynthesisRequest) -> None:
        if request.reference_audio_path is None:
            raise AdapterError("Dia2 requires a reference audio prompt.")
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        run_container(
            image_tag="otu-tts-bench-dia2:latest",
            wants_gpu=self.context.device == "cuda",
            env={
                "MODEL_ID": self.model_id,
                "TEXT": request.text,
                "OUTPUT_PATH": container_path(request.output_path),
                "REFERENCE_AUDIO_PATH": container_path(request.reference_audio_path),
            },
        )


class DockerBackedGlmTtsAdapter(BaseAdapter):
    def __init__(self, model_id: str, context: BenchmarkContext) -> None:
        super().__init__(model_id, context)
        docker_dir = Path.cwd() / "docker" / "tts" / "glmtts"
        ensure_image("otu-tts-bench-glmtts:latest", docker_dir)

    def synthesize(self, request: SynthesisRequest) -> None:
        if request.reference_audio_path is None:
            raise AdapterError("GLM-TTS requires a reference audio prompt.")
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        run_container(
            image_tag="otu-tts-bench-glmtts:latest",
            wants_gpu=self.context.device == "cuda",
            env={
                "MODEL_ID": self.model_id,
                "TEXT": request.text,
                "OUTPUT_PATH": container_path(request.output_path),
                "REFERENCE_AUDIO_PATH": container_path(request.reference_audio_path),
                "REFERENCE_TEXT": request.reference_text,
            },
        )


class DockerBackedOrpheusTtsAdapter(BaseAdapter):
    def __init__(self, model_id: str, context: BenchmarkContext) -> None:
        super().__init__(model_id, context)
        docker_dir = Path.cwd() / "docker" / "tts" / "orpheus"
        ensure_image("otu-tts-bench-orpheus:latest", docker_dir)

    def synthesize(self, request: SynthesisRequest) -> None:
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        run_container(
            image_tag="otu-tts-bench-orpheus:latest",
            wants_gpu=self.context.device == "cuda",
            env={
                "MODEL_ID": self.model_id,
                "TEXT": request.text,
                "OUTPUT_PATH": container_path(request.output_path),
            },
        )


class XttsV2Adapter(BaseAdapter):
    def __init__(self, model_id: str, context: BenchmarkContext) -> None:
        super().__init__(model_id, context)
        try:
            from TTS.api import TTS
        except ImportError as exc:
            raise AdapterError(_missing_dependency("xtts")) from exc
        wants_cuda = _resolve_device(context.device) == "cuda"
        self._model = TTS(model_name=model_id, progress_bar=False, gpu=wants_cuda)

    def synthesize(self, request: SynthesisRequest) -> None:
        if request.reference_audio_path is None:
            raise AdapterError("XTTS v2 requires a reference audio prompt.")
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._model.tts_to_file(
            text=request.text,
            speaker_wav=str(request.reference_audio_path),
            language=_xtts_language(self.context.language, request.locale),
            file_path=str(request.output_path),
        )


class DockerBackedXttsV2Adapter(BaseAdapter):
    def __init__(self, model_id: str, context: BenchmarkContext) -> None:
        super().__init__(model_id, context)
        docker_dir = Path.cwd() / "docker" / "tts" / "xtts"
        ensure_image("otu-tts-bench-xtts:latest", docker_dir)

    def synthesize(self, request: SynthesisRequest) -> None:
        if request.reference_audio_path is None:
            raise AdapterError("XTTS v2 requires a reference audio prompt.")
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        run_container(
            image_tag="otu-tts-bench-xtts:latest",
            wants_gpu=self.context.device == "cuda",
            env={
                "MODEL_ID": self.model_id,
                "TEXT": request.text,
                "OUTPUT_PATH": container_path(request.output_path),
                "REFERENCE_AUDIO_PATH": container_path(request.reference_audio_path),
                "LANGUAGE": _xtts_language(self.context.language, request.locale),
                "DEVICE": _resolve_device(self.context.device),
                "COQUI_TOS_AGREED": "1",
            },
        )


def create_adapter(engine: str, model_id: str, context: BenchmarkContext) -> BaseAdapter:
    if engine == "kokoro":
        return KokoroAdapter(model_id, context)
    if engine == "kitten-tts":
        return KittenTtsAdapter(model_id, context)
    if engine == "chatterbox":
        if context.device == "cuda" or model_id != "ResembleAI/chatterbox-multilingual":
            return DockerBackedChatterboxAdapter(model_id, context)
        return ChatterboxAdapter(model_id, context)
    if engine == "f5-tts":
        return DockerBackedF5TtsAdapter(model_id, context)
    if engine == "voxcpm":
        return DockerBackedVoxCpmAdapter(model_id, context)
    if engine == "qwen3-tts":
        return DockerBackedQwen3TtsAdapter(model_id, context)
    if engine == "cosyvoice2":
        return DockerBackedCosyVoice2Adapter(model_id, context)
    if engine == "vibevoice":
        return LocalVenvVibeVoiceAdapter(model_id, context)
    if engine == "dia2":
        return DockerBackedDia2Adapter(model_id, context)
    if engine == "glm-tts":
        return DockerBackedGlmTtsAdapter(model_id, context)
    if engine == "orpheus-tts":
        return DockerBackedOrpheusTtsAdapter(model_id, context)
    if engine == "xtts-v2":
        return DockerBackedXttsV2Adapter(model_id, context)
    raise AdapterError(f"Unknown TTS engine: {engine}")


def _resolve_device(device: str) -> str:
    if device in {"cpu", "cuda"}:
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _missing_dependency(extra_name: str) -> str:
    return (
        f"Required dependencies for '{extra_name}' are not installed. "
        f"Install them with: uv sync --extra {extra_name}"
    )


def _chatterbox_language(language: str, locale: str) -> str:
    normalized = locale.strip().lower()
    if normalized == "fr-ca":
        return "fr"
    return language.strip().lower() or "fr"


def _xtts_language(language: str, locale: str) -> str:
    normalized = locale.strip().lower()
    if normalized == "fr-ca":
        return "fr"
    return language.strip().lower() or "fr"


def _qwen3_language(language: str, locale: str) -> str:
    normalized = locale.strip().lower()
    if normalized == "fr-ca":
        return "French"
    mapping = {
        "fr": "French",
        "en": "English",
        "de": "German",
        "es": "Spanish",
        "it": "Italian",
        "ja": "Japanese",
        "ko": "Korean",
        "pt": "Portuguese",
        "ru": "Russian",
        "zh": "Chinese",
    }
    return mapping.get(language.strip().lower(), "French")


def _resolve_vibevoice_python() -> Path:
    root = Path.cwd()
    candidates = (
        root / ".venvs" / "vibevoice" / "Scripts" / "python.exe",
        root / ".venvs" / "vibevoice" / "bin" / "python",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _tail_text(value: str, limit: int = 600) -> str:
    cleaned = " ".join(value.strip().split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[-limit:]

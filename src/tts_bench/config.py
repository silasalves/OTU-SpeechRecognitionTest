from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    engine: str
    model_id: str
    package_extra: str
    enabled_by_default: bool = True
    supports_locales: tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""

    @property
    def run_id(self) -> str:
        return f"{self.engine}:{self.model_id}"


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        engine="kokoro",
        model_id="hexgrad/kokoro-82m",
        package_extra="kokoro",
        supports_locales=("fr", "fr-ca"),
        notes="Uses Kokoro's built-in French pipeline with the ff_siwis voice by default.",
    ),
    ModelSpec(
        engine="kitten-tts",
        model_id="KittenML/kitten-tts-mini-0.8",
        package_extra="kittentts",
        enabled_by_default=False,
        notes="English-only KittenTTS mini model. Added as a lightweight ONNX out-of-language baseline on the French dataset.",
    ),
    ModelSpec(
        engine="kitten-tts",
        model_id="KittenML/kitten-tts-micro-0.8",
        package_extra="kittentts",
        enabled_by_default=False,
        notes="English-only KittenTTS micro model. Added as a lightweight ONNX out-of-language baseline on the French dataset.",
    ),
    ModelSpec(
        engine="kitten-tts",
        model_id="KittenML/kitten-tts-nano-0.8-int8",
        package_extra="kittentts",
        enabled_by_default=False,
        notes="English-only KittenTTS nano int8 model. Added as a smallest-footprint ONNX out-of-language baseline on the French dataset.",
    ),
    ModelSpec(
        engine="chatterbox",
        model_id="ResembleAI/chatterbox-multilingual",
        package_extra="chatterbox",
        supports_locales=("fr", "fr-ca"),
        notes="Uses Chatterbox Multilingual with a cloned voice prompt from the selected reference sample.",
    ),
    ModelSpec(
        engine="f5-tts",
        model_id="F5TTS_v1_Base",
        package_extra="f5tts",
        enabled_by_default=False,
        supports_locales=("fr", "fr-ca"),
        notes="Uses the selected reference sample as the prompt voice and transcript for zero-shot generation.",
    ),
    ModelSpec(
        engine="voxcpm",
        model_id="openbmb/VoxCPM-0.5B",
        package_extra="",
        enabled_by_default=False,
        notes="Docker-backed VoxCPM 0.5B voice-cloning model. Current upstream is primarily optimized for Chinese and English, so this is treated as an experimental out-of-language baseline on the French dataset.",
    ),
    ModelSpec(
        engine="qwen3-tts",
        model_id="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        package_extra="",
        enabled_by_default=False,
        supports_locales=("fr", "fr-ca"),
        notes="Docker-backed Qwen3-TTS 0.6B voice-cloning model. Uses the official qwen-tts package through a CUDA container built on the official PyTorch runtime image.",
    ),
    ModelSpec(
        engine="cosyvoice2",
        model_id="FunAudioLLM/CosyVoice2-0.5B",
        package_extra="",
        enabled_by_default=False,
        supports_locales=("fr", "fr-ca"),
        notes="Docker-backed CosyVoice2 0.5B run. The upstream checkpoint does not ship built-in SFT speakers, so this repo's no-clone path injects the upstream fixed-speaker cache and runs inference_sft() against the English-translated interview-2/interview-3 smoke pair used for the repo's out-of-language baselines.",
    ),
    ModelSpec(
        engine="vibevoice",
        model_id="microsoft/VibeVoice-Realtime-0.5B",
        package_extra="",
        enabled_by_default=False,
        supports_locales=("fr", "fr-ca"),
        notes="Dedicated-venv VibeVoice Realtime 0.5B run around the official Microsoft repo. This repo currently validates the official single-speaker realtime path with the bundled experimental French voice presets rather than the disabled long-form VibeVoice-TTS path.",
    ),
    ModelSpec(
        engine="dia2",
        model_id="nari-labs/Dia2-1B",
        package_extra="",
        enabled_by_default=False,
        notes="Docker-backed Dia2 1B dialogue TTS model. Upstream currently documents English generation only, so this is treated as an experimental out-of-language baseline on the French dataset.",
    ),
    ModelSpec(
        engine="glm-tts",
        model_id="zai-org/GLM-TTS",
        package_extra="",
        enabled_by_default=False,
        notes="Docker-backed GLM-TTS model. Upstream currently documents Chinese plus English-mixed text, so the repo smoke test uses an English prompt and English translations of interview-2/interview-3 rather than the default French dataset.",
    ),
    ModelSpec(
        engine="orpheus-tts",
        model_id="canopylabs/3b-fr-ft-research_release",
        package_extra="orpheus",
        enabled_by_default=False,
        supports_locales=("fr", "fr-ca"),
        notes="French Orpheus research-release finetune, loaded through the official Orpheus package in Docker. Requires a Hugging Face token with gated-model access.",
    ),
    ModelSpec(
        engine="orpheus-tts",
        model_id="canopylabs/orpheus-tts-0.1-finetune-prod",
        package_extra="orpheus",
        enabled_by_default=False,
        notes="English Orpheus production finetune, loaded through the official Orpheus package in Docker. This is an out-of-language baseline on the French dataset.",
    ),
    ModelSpec(
        engine="orpheus-tts",
        model_id="canopylabs/orpheus-3b-0.1-pretrained",
        package_extra="orpheus",
        enabled_by_default=False,
        notes="English Orpheus pretrained model, loaded through the official Orpheus package in Docker. This is an out-of-language baseline on the French dataset.",
    ),
    ModelSpec(
        engine="chatterbox",
        model_id="ResembleAI/chatterbox",
        package_extra="chatterbox",
        notes="English-only Chatterbox baseline. Retained mainly for timing and human listening on the French set.",
    ),
    ModelSpec(
        engine="chatterbox",
        model_id="ResembleAI/chatterbox-turbo",
        package_extra="chatterbox",
        notes="English-only Chatterbox Turbo baseline. Retained mainly for timing and human listening on the French set.",
    ),
)


def resolve_models(
    include: list[str] | None = None,
    engines: list[str] | None = None,
) -> list[ModelSpec]:
    include_set = {value.strip() for value in include or [] if value.strip()}
    engine_set = {value.strip() for value in engines or [] if value.strip()}

    if include_set or engine_set:
        specs = list(MODEL_SPECS)
    else:
        specs = [spec for spec in MODEL_SPECS if spec.enabled_by_default]
    if engine_set:
        specs = [spec for spec in specs if spec.engine in engine_set]
    if include_set:
        specs = [
            spec
            for spec in specs
            if spec.run_id in include_set or spec.model_id in include_set
        ]
    return specs

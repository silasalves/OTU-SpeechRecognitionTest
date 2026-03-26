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

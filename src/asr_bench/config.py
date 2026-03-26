from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    engine: str
    model_id: str
    package_extra: str
    supports_locales: tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""

    @property
    def run_id(self) -> str:
        return f"{self.engine}:{self.model_id}"


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        engine="whisper",
        model_id="tiny",
        package_extra="whisper",
        supports_locales=("fr", "fr-ca"),
    ),
    ModelSpec(
        engine="whisper",
        model_id="base",
        package_extra="whisper",
        supports_locales=("fr", "fr-ca"),
    ),
    ModelSpec(
        engine="whisper",
        model_id="small",
        package_extra="whisper",
        supports_locales=("fr", "fr-ca"),
    ),
    ModelSpec(
        engine="whisper",
        model_id="medium",
        package_extra="whisper",
        supports_locales=("fr", "fr-ca"),
    ),
    ModelSpec(
        engine="whisper",
        model_id="large-v3",
        package_extra="whisper",
        supports_locales=("fr", "fr-ca"),
    ),
    ModelSpec(
        engine="moonshine",
        model_id="moonshine/tiny",
        package_extra="moonshine",
        notes="Official Moonshine models do not currently include French; this run is retained for timing and out-of-language comparison.",
    ),
    ModelSpec(
        engine="moonshine",
        model_id="moonshine/base",
        package_extra="moonshine",
        notes="Official Moonshine models do not currently include French; this run is retained for timing and out-of-language comparison.",
    ),
    ModelSpec(
        engine="parakeet",
        model_id="nvidia/parakeet-ctc-0.6b",
        package_extra="nemo",
        notes="Parakeet checkpoints are primarily English models in NeMo; keep this only if you want an out-of-language baseline on French audio.",
    ),
    ModelSpec(
        engine="parakeet",
        model_id="nvidia/parakeet-ctc-1.1b",
        package_extra="nemo",
        notes="Parakeet checkpoints are primarily English models in NeMo; keep this only if you want an out-of-language baseline on French audio.",
    ),
    ModelSpec(
        engine="parakeet",
        model_id="nvidia/parakeet-tdt-0.6b-v2",
        package_extra="nemo",
        notes="Parakeet checkpoints are primarily English models in NeMo; keep this only if you want an out-of-language baseline on French audio.",
    ),
    ModelSpec(
        engine="parakeet",
        model_id="nvidia/parakeet-tdt-1.1b",
        package_extra="nemo",
        notes="Parakeet checkpoints are primarily English models in NeMo; keep this only if you want an out-of-language baseline on French audio.",
    ),
    ModelSpec(
        engine="canary",
        model_id="nvidia/canary-180m-flash",
        package_extra="nemo",
        supports_locales=("fr", "fr-ca"),
    ),
    ModelSpec(
        engine="canary",
        model_id="nvidia/canary-1b",
        package_extra="nemo",
        supports_locales=("fr", "fr-ca"),
    ),
    ModelSpec(
        engine="canary",
        model_id="nvidia/canary-1b-flash",
        package_extra="nemo",
        supports_locales=("fr", "fr-ca"),
    ),
    ModelSpec(
        engine="canary",
        model_id="nvidia/canary-1b-v2",
        package_extra="nemo",
        supports_locales=("fr", "fr-ca"),
    ),
    ModelSpec(
        engine="owsm-ctc",
        model_id="espnet/owsm_ctc_v3.1_1B",
        package_extra="espnet",
        supports_locales=("fr", "fr-ca"),
    ),
    ModelSpec(
        engine="funasr",
        model_id="iic/SenseVoiceSmall",
        package_extra="funasr",
        notes="SenseVoiceSmall is multi-lingual, but verify French quality on your target audio before treating scores as production guidance.",
    ),
)


def resolve_models(
    include: list[str] | None = None,
    engines: list[str] | None = None,
) -> list[ModelSpec]:
    include_set = {value.strip() for value in include or [] if value.strip()}
    engine_set = {value.strip() for value in engines or [] if value.strip()}

    specs = list(MODEL_SPECS)
    if engine_set:
        specs = [spec for spec in specs if spec.engine in engine_set]
    if include_set:
        specs = [
            spec
            for spec in specs
            if spec.run_id in include_set or spec.model_id in include_set
        ]
    return specs

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TextSample:
    locale: str
    source_audio_path: Path
    text: str

    @property
    def sample_id(self) -> str:
        return f"{self.locale}/{self.source_audio_path.name}"


def discover_dataset(
    data_root: Path,
    locales: list[str] | None = None,
) -> list[TextSample]:
    locale_filter = {value.strip() for value in locales or [] if value.strip()}
    samples: list[TextSample] = []
    for locale_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        if locale_filter and locale_dir.name not in locale_filter:
            continue
        transcript_path = locale_dir / "transcriptions.txt"
        if not transcript_path.exists():
            continue
        transcripts = _read_transcriptions(transcript_path)
        for audio_name, transcript in transcripts.items():
            audio_path = locale_dir / audio_name
            if not audio_path.exists():
                raise FileNotFoundError(
                    f"Transcript references missing audio file: {audio_path}"
                )
            samples.append(
                TextSample(
                    locale=locale_dir.name,
                    source_audio_path=audio_path,
                    text=transcript,
                )
            )
    if not samples:
        raise FileNotFoundError(f"No TTS samples found under {data_root}")
    return samples


def resolve_reference_samples(
    samples: list[TextSample],
    reference_sample_name: str | None = None,
) -> dict[str, TextSample]:
    grouped: dict[str, list[TextSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.locale, []).append(sample)

    reference_by_locale: dict[str, TextSample] = {}
    for locale, locale_samples in grouped.items():
        if reference_sample_name:
            matched = next(
                (
                    sample
                    for sample in locale_samples
                    if sample.source_audio_path.name == reference_sample_name
                ),
                None,
            )
            if matched is None:
                raise FileNotFoundError(
                    f"Reference sample '{reference_sample_name}' was not found for locale '{locale}'."
                )
            reference_by_locale[locale] = matched
            continue
        reference_by_locale[locale] = locale_samples[0]
    return reference_by_locale


def _read_transcriptions(transcript_path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for raw_line in transcript_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(
                f"Invalid transcript line in {transcript_path}: {raw_line!r}"
            )
        audio_name, transcript = line.split(":", 1)
        mapping[audio_name.strip()] = transcript.strip()
    return mapping

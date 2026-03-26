from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import wave


@dataclass(frozen=True)
class AudioSample:
    locale: str
    audio_path: Path
    transcript: str
    duration_seconds: float

    @property
    def sample_id(self) -> str:
        return f"{self.locale}/{self.audio_path.name}"


def discover_dataset(data_root: Path) -> list[AudioSample]:
    samples: list[AudioSample] = []
    for locale_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
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
                AudioSample(
                    locale=locale_dir.name,
                    audio_path=audio_path,
                    transcript=transcript,
                    duration_seconds=_wav_duration(audio_path),
                )
            )
    if not samples:
        raise FileNotFoundError(f"No benchmark samples found under {data_root}")
    return samples


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


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        frame_rate = handle.getframerate()
        if frame_rate <= 0:
            raise ValueError(f"Invalid sample rate in {path}")
        return handle.getnframes() / frame_rate

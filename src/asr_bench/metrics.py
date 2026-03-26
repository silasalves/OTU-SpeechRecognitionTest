from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

from jiwer import wer


@dataclass(frozen=True)
class ScoredTranscript:
    raw_reference: str
    raw_hypothesis: str
    normalized_reference: str
    normalized_hypothesis: str
    wer: float


def score_transcript(reference: str, hypothesis: str) -> ScoredTranscript:
    normalized_reference = normalize_text(reference)
    normalized_hypothesis = normalize_text(hypothesis)
    return ScoredTranscript(
        raw_reference=reference,
        raw_hypothesis=hypothesis,
        normalized_reference=normalized_reference,
        normalized_hypothesis=normalized_hypothesis,
        wer=wer(normalized_reference, normalized_hypothesis),
    )


def normalize_text(text: str) -> str:
    lowered = text.casefold()
    cleaned = "".join(
        " " if _is_punctuation(char) else char for char in unicodedata.normalize("NFKC", lowered)
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _is_punctuation(char: str) -> bool:
    category = unicodedata.category(char)
    return category.startswith("P") or category.startswith("S")

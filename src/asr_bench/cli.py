from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Iterable

from asr_bench.adapters import BenchmarkContext, create_adapter
from asr_bench.config import ModelSpec, resolve_models
from asr_bench.dataset import AudioSample, discover_dataset
from asr_bench.metrics import score_transcript


@dataclass
class SampleResult:
    engine: str
    model_id: str
    sample_id: str
    locale: str
    audio_path: str
    duration_seconds: float
    elapsed_seconds: float
    rtf: float
    wer: float | None
    status: str
    error: str
    reference: str
    hypothesis: str
    normalized_reference: str
    normalized_hypothesis: str
    notes: str


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    samples = discover_dataset(data_root)
    specs = resolve_models(include=args.include, engines=args.engine)
    if not specs:
        raise SystemExit("No models selected.")

    context = BenchmarkContext(device=args.device, language=args.language)
    sample_results: list[SampleResult] = []

    for spec in specs:
        adapter = None
        adapter_error = ""
        try:
            adapter = create_adapter(spec.engine, spec.model_id, context)
        except Exception as exc:
            adapter_error = f"{type(exc).__name__}: {exc}"
            for sample in samples:
                sample_results.append(
                    SampleResult(
                        engine=spec.engine,
                        model_id=spec.model_id,
                        sample_id=sample.sample_id,
                        locale=sample.locale,
                        audio_path=str(sample.audio_path),
                        duration_seconds=sample.duration_seconds,
                        elapsed_seconds=0.0,
                        rtf=0.0,
                        wer=None,
                        status="adapter_error",
                        error=adapter_error,
                        reference=sample.transcript,
                        hypothesis="",
                        normalized_reference="",
                        normalized_hypothesis="",
                        notes=spec.notes,
                    )
                )
            continue

        for sample in samples:
            sample_results.append(run_sample(adapter, spec, sample))

    written_files = write_outputs(output_root, sample_results)
    print("Benchmark complete.")
    for path in written_files:
        print(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark open-source ASR models.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory containing locale folders with transcriptions.txt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where benchmark reports should be written.",
    )
    parser.add_argument(
        "--language",
        default="fr",
        help="Language code passed to models when supported. Default: fr",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Target device for backends that expose device selection.",
    )
    parser.add_argument(
        "--engine",
        action="append",
        default=[],
        help="Limit the run to one or more engines. Example: --engine whisper --engine canary",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Limit the run to explicit model IDs or engine:model IDs.",
    )
    return parser.parse_args()


def run_sample(adapter: object, spec: ModelSpec, sample: AudioSample) -> SampleResult:
    start = time.perf_counter()
    try:
        hypothesis = adapter.transcribe(sample.audio_path)  # type: ignore[attr-defined]
        elapsed = time.perf_counter() - start
        scored = score_transcript(sample.transcript, hypothesis)
        return SampleResult(
            engine=spec.engine,
            model_id=spec.model_id,
            sample_id=sample.sample_id,
            locale=sample.locale,
            audio_path=str(sample.audio_path),
            duration_seconds=sample.duration_seconds,
            elapsed_seconds=elapsed,
            rtf=elapsed / sample.duration_seconds,
            wer=scored.wer,
            status="ok",
            error="",
            reference=scored.raw_reference,
            hypothesis=scored.raw_hypothesis,
            normalized_reference=scored.normalized_reference,
            normalized_hypothesis=scored.normalized_hypothesis,
            notes=spec.notes,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return SampleResult(
            engine=spec.engine,
            model_id=spec.model_id,
            sample_id=sample.sample_id,
            locale=sample.locale,
            audio_path=str(sample.audio_path),
            duration_seconds=sample.duration_seconds,
            elapsed_seconds=elapsed,
            rtf=elapsed / sample.duration_seconds if sample.duration_seconds else 0.0,
            wer=None,
            status="runtime_error",
            error=f"{type(exc).__name__}: {exc}",
            reference=sample.transcript,
            hypothesis="",
            normalized_reference="",
            normalized_hypothesis="",
            notes=spec.notes,
        )


def write_outputs(output_root: Path, sample_results: list[SampleResult]) -> list[Path]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = output_root / f"benchmark-{timestamp}.json"
    summary_path = output_root / f"benchmark-summary-{timestamp}.csv"
    markdown_path = output_root / f"benchmark-summary-{timestamp}.md"

    raw_payload = {
        "generated_at_utc": timestamp,
        "python": sys.version,
        "results": [asdict(result) for result in sample_results],
    }
    raw_path.write_text(
        json.dumps(raw_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary_rows = summarize(sample_results)
    summary_path.write_text(render_csv(summary_rows), encoding="utf-8")
    markdown_path.write_text(render_markdown(summary_rows), encoding="utf-8")
    return [raw_path, summary_path, markdown_path]


def summarize(sample_results: list[SampleResult]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[SampleResult]] = {}
    for result in sample_results:
        grouped.setdefault((result.engine, result.model_id), []).append(result)

    rows: list[dict[str, str]] = []
    for (engine, model_id), results in sorted(grouped.items()):
        ok_results = [item for item in results if item.status == "ok" and item.wer is not None]
        row = {
            "engine": engine,
            "model_id": model_id,
            "samples": str(len(results)),
            "ok_samples": str(len(ok_results)),
            "avg_elapsed_seconds": _format_float(_mean(item.elapsed_seconds for item in ok_results)),
            "avg_rtf": _format_float(_mean(item.rtf for item in ok_results)),
            "avg_wer": _format_float(_mean(item.wer for item in ok_results)),
            "notes": next((item.notes for item in results if item.notes), ""),
            "errors": " | ".join(sorted({item.error for item in results if item.error})),
        }
        rows.append(row)
    return rows


def render_csv(rows: list[dict[str, str]]) -> str:
    headers = [
        "engine",
        "model_id",
        "samples",
        "ok_samples",
        "avg_elapsed_seconds",
        "avg_rtf",
        "avg_wer",
        "notes",
        "errors",
    ]
    lines = [",".join(headers)]
    for row in rows:
        values = [_csv_escape(row[header]) for header in headers]
        lines.append(",".join(values))
    return "\n".join(lines) + "\n"


def render_markdown(rows: list[dict[str, str]]) -> str:
    headers = [
        "engine",
        "model_id",
        "samples",
        "ok_samples",
        "avg_elapsed_seconds",
        "avg_rtf",
        "avg_wer",
    ]
    lines = [
        "# ASR Benchmark Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[header] for header in headers) + " |")

    notes = [row for row in rows if row["notes"] or row["errors"]]
    if notes:
        lines.extend(["", "## Notes", ""])
        for row in notes:
            label = f"{row['engine']} / {row['model_id']}"
            if row["notes"]:
                lines.append(f"- {label}: {row['notes']}")
            if row["errors"]:
                lines.append(f"- {label} errors: {row['errors']}")
    lines.append("")
    return "\n".join(lines)


def _mean(values: Iterable[float]) -> float | None:
    collected = list(values)
    if not collected:
        return None
    return sum(collected) / len(collected)


def _format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def _csv_escape(value: str) -> str:
    escaped = value.replace('"', '""')
    return f'"{escaped}"'


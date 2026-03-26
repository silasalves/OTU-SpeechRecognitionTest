from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Iterable

import soundfile as sf

from tts_bench.adapters import (
    BenchmarkContext,
    ReferencePrompt,
    SynthesisRequest,
    create_adapter,
)
from tts_bench.config import ModelSpec, resolve_models
from tts_bench.dataset import TextSample, discover_dataset, resolve_reference_samples


@dataclass
class SampleResult:
    engine: str
    model_id: str
    sample_id: str
    locale: str
    source_audio_path: str
    reference_audio_path: str
    output_path: str
    text: str
    input_characters: int
    elapsed_seconds: float
    generated_duration_seconds: float | None
    rtf: float | None
    status: str
    error: str
    notes: str


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    all_samples = discover_dataset(data_root, locales=args.locale)
    reference_samples = resolve_reference_samples(
        all_samples,
        reference_sample_name=args.reference_sample,
    )
    benchmark_samples = [
        sample
        for sample in all_samples
        if reference_samples[sample.locale].source_audio_path != sample.source_audio_path
    ]
    if not benchmark_samples:
        benchmark_samples = all_samples
    samples = benchmark_samples[: args.limit] if args.limit else benchmark_samples
    specs = resolve_models(include=args.include, engines=args.engine)
    if not specs:
        raise SystemExit("No models selected.")

    reference_prompts = {
        locale: ReferencePrompt(
            locale=sample.locale,
            audio_path=sample.source_audio_path,
            transcript=sample.text,
        )
        for locale, sample in reference_samples.items()
    }
    context = BenchmarkContext(
        device=args.device,
        language=args.language,
        reference_prompts=reference_prompts,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = output_root / f"tts-run-{timestamp}"
    audio_root = run_root / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)

    sample_results: list[SampleResult] = []
    for spec in specs:
        adapter = None
        try:
            adapter = create_adapter(spec.engine, spec.model_id, context)
        except Exception as exc:
            adapter_error = f"{type(exc).__name__}: {exc}"
            for sample in samples:
                reference = context.reference_prompts[sample.locale]
                sample_results.append(
                    SampleResult(
                        engine=spec.engine,
                        model_id=spec.model_id,
                        sample_id=sample.sample_id,
                        locale=sample.locale,
                        source_audio_path=str(sample.source_audio_path),
                        reference_audio_path=str(reference.audio_path),
                        output_path=str(_output_path_for(audio_root, spec, sample)),
                        text=sample.text,
                        input_characters=len(sample.text),
                        elapsed_seconds=0.0,
                        generated_duration_seconds=None,
                        rtf=None,
                        status="adapter_error",
                        error=adapter_error,
                        notes=spec.notes,
                    )
                )
            continue

        for sample in samples:
            reference = context.reference_prompts[sample.locale]
            output_path = _output_path_for(audio_root, spec, sample)
            request = SynthesisRequest(
                locale=sample.locale,
                text=sample.text,
                output_path=output_path,
                reference_audio_path=reference.audio_path,
                reference_text=reference.transcript,
            )
            sample_results.append(run_sample(adapter, spec, sample, request))

    written_files = write_outputs(
        run_root=run_root,
        timestamp=timestamp,
        sample_results=sample_results,
        reference_prompts=reference_prompts,
    )
    print("TTS benchmark complete.")
    for path in written_files:
        print(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark open-source TTS models.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory containing locale folders with transcriptions.txt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "tts",
        help="Directory where TTS benchmark runs should be written.",
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
        "--locale",
        action="append",
        default=[],
        help="Limit the run to one or more locale folders. Example: --locale fr-ca",
    )
    parser.add_argument(
        "--engine",
        action="append",
        default=[],
        help="Limit the run to one or more engines. Example: --engine kokoro --engine xtts-v2",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Limit the run to explicit model IDs or engine:model IDs.",
    )
    parser.add_argument(
        "--reference-sample",
        default="",
        help="Audio filename to use as the cloning prompt per locale. Defaults to the first transcript entry in each locale.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit the number of text samples generated per run.",
    )
    return parser.parse_args()


def run_sample(
    adapter: object,
    spec: ModelSpec,
    sample: TextSample,
    request: SynthesisRequest,
) -> SampleResult:
    start = time.perf_counter()
    try:
        adapter.synthesize(request)  # type: ignore[attr-defined]
        elapsed = time.perf_counter() - start
        generated_duration = _audio_duration_seconds(request.output_path)
        return SampleResult(
            engine=spec.engine,
            model_id=spec.model_id,
            sample_id=sample.sample_id,
            locale=sample.locale,
            source_audio_path=str(sample.source_audio_path),
            reference_audio_path=str(request.reference_audio_path or ""),
            output_path=str(request.output_path),
            text=sample.text,
            input_characters=len(sample.text),
            elapsed_seconds=elapsed,
            generated_duration_seconds=generated_duration,
            rtf=elapsed / generated_duration if generated_duration else None,
            status="ok",
            error="",
            notes=spec.notes,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start
        if request.output_path.exists():
            request.output_path.unlink()
        return SampleResult(
            engine=spec.engine,
            model_id=spec.model_id,
            sample_id=sample.sample_id,
            locale=sample.locale,
            source_audio_path=str(sample.source_audio_path),
            reference_audio_path=str(request.reference_audio_path or ""),
            output_path=str(request.output_path),
            text=sample.text,
            input_characters=len(sample.text),
            elapsed_seconds=elapsed,
            generated_duration_seconds=None,
            rtf=None,
            status="runtime_error",
            error=f"{type(exc).__name__}: {exc}",
            notes=spec.notes,
        )


def write_outputs(
    run_root: Path,
    timestamp: str,
    sample_results: list[SampleResult],
    reference_prompts: dict[str, ReferencePrompt],
) -> list[Path]:
    raw_path = run_root / f"tts-benchmark-{timestamp}.json"
    summary_path = run_root / f"tts-summary-{timestamp}.csv"
    markdown_path = run_root / f"tts-summary-{timestamp}.md"

    raw_payload = {
        "generated_at_utc": timestamp,
        "python": sys.version,
        "reference_prompts": {
            locale: {
                "audio_path": str(prompt.audio_path),
                "transcript": prompt.transcript,
            }
            for locale, prompt in reference_prompts.items()
        },
        "results": [asdict(result) for result in sample_results],
    }
    raw_path.write_text(
        json.dumps(raw_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary_rows = summarize(sample_results)
    summary_path.write_text(render_csv(summary_rows), encoding="utf-8")
    markdown_path.write_text(
        render_markdown(summary_rows, reference_prompts),
        encoding="utf-8",
    )
    return [raw_path, summary_path, markdown_path]


def summarize(sample_results: list[SampleResult]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[SampleResult]] = {}
    for result in sample_results:
        grouped.setdefault((result.engine, result.model_id), []).append(result)

    rows: list[dict[str, str]] = []
    for (engine, model_id), results in sorted(grouped.items()):
        ok_results = [item for item in results if item.status == "ok"]
        row = {
            "engine": engine,
            "model_id": model_id,
            "samples": str(len(results)),
            "ok_samples": str(len(ok_results)),
            "avg_elapsed_seconds": _format_float(
                _mean(item.elapsed_seconds for item in ok_results)
            ),
            "avg_generated_duration_seconds": _format_float(
                _mean(
                    item.generated_duration_seconds
                    for item in ok_results
                    if item.generated_duration_seconds is not None
                )
            ),
            "avg_rtf": _format_float(
                _mean(item.rtf for item in ok_results if item.rtf is not None)
            ),
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
        "avg_generated_duration_seconds",
        "avg_rtf",
        "notes",
        "errors",
    ]
    lines = [",".join(headers)]
    for row in rows:
        values = [_csv_escape(row[header]) for header in headers]
        lines.append(",".join(values))
    return "\n".join(lines) + "\n"


def render_markdown(
    rows: list[dict[str, str]],
    reference_prompts: dict[str, ReferencePrompt],
) -> str:
    headers = [
        "engine",
        "model_id",
        "samples",
        "ok_samples",
        "avg_elapsed_seconds",
        "avg_generated_duration_seconds",
        "avg_rtf",
    ]
    lines = [
        "# TTS Benchmark Summary",
        "",
        "Reference prompts by locale:",
        "",
    ]
    for locale, prompt in sorted(reference_prompts.items()):
        lines.append(f"- {locale}: {prompt.audio_path.name}")
    lines.extend(
        [
            "",
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
    )
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


def _output_path_for(audio_root: Path, spec: ModelSpec, sample: TextSample) -> Path:
    return (
        audio_root
        / spec.engine
        / _slug(spec.model_id)
        / sample.locale
        / sample.source_audio_path.name
    )


def _slug(value: str) -> str:
    return (
        value.replace("\\", "-")
        .replace("/", "-")
        .replace(":", "-")
        .replace(" ", "-")
    )


def _audio_duration_seconds(path: Path) -> float:
    info = sf.info(str(path))
    if info.samplerate <= 0:
        raise ValueError(f"Invalid sample rate in {path}")
    return info.frames / info.samplerate


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


if __name__ == "__main__":
    main()

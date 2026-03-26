# OTU-SpeechRecognitionTest

Benchmark harness for comparing several open-source ASR systems against the speech data under [`data/`](./data).

## What it measures

- Wall-clock transcription time per audio file
- Real-time factor (RTF): `elapsed_seconds / audio_duration_seconds`
- Word error rate (WER) using normalized text

The harness discovers locale folders under `data/` that contain a `transcriptions.txt` file in the format:

```text
filename.wav: Reference transcript
```

## Included engines and models

- Whisper: `tiny`, `base`, `small`, `medium`, `large-v3`
- Moonshine: `moonshine/tiny`, `moonshine/base`
- NVIDIA Parakeet: `nvidia/parakeet-ctc-0.6b`, `nvidia/parakeet-ctc-1.1b`, `nvidia/parakeet-tdt-0.6b-v2`, `nvidia/parakeet-tdt-1.1b`
- NVIDIA Canary: `nvidia/canary-180m-flash`, `nvidia/canary-1b`, `nvidia/canary-1b-flash`, `nvidia/canary-1b-v2`
- OWSM-CTC (ESPnet): `espnet/owsm_ctc_v3.1_1B`
- FunASR: `iic/SenseVoiceSmall`

Notes:

- Moonshine's official published models do not currently include French, so Moonshine runs on `fr-ca` are out-of-language comparisons.
- NVIDIA's published Parakeet checkpoints are primarily English, so Parakeet runs on this French dataset are also out-of-language comparisons unless you swap in a different checkpoint.
- Canary is the NVIDIA family in this list with documented French ASR support.

## Environment

This project is set up for Python `3.10` or `3.11`.

Recommended flow with `uv`:

```powershell
.\scripts\install-backends.ps1 -Extras whisper,moonshine,nemo,espnet,funasr
.\scripts\run-benchmark.ps1 -Language fr -Device cuda
```

If you only want part of the matrix:

```powershell
.\scripts\install-backends.ps1 -Extras whisper,nemo
.\scripts\run-benchmark.ps1 -Language fr -Engine whisper,canary
```

## Aggregated GPU Results

The consolidated GPU comparison for the `data/fr-ca` set is also stored in [`outputs/gpu-comparison-20260325.md`](./outputs/gpu-comparison-20260325.md) and [`outputs/gpu-comparison-20260325.csv`](./outputs/gpu-comparison-20260325.csv).

These numbers were obtained on March 25, 2026 by running the benchmark on this Windows machine with:

- CUDA-enabled PyTorch for Whisper and NeMo-based models
- local Hugging Face/model caches under the repo
- repo-local audio preprocessing for Windows compatibility
- the benchmark harness in `src/asr_bench`

Test machine:

- Laptop: `GIGABYTE G6 KF`
- OS: `Microsoft Windows 11 Home` 64-bit, version `10.0.26200`
- CPU: `13th Gen Intel Core i7-13620H` with `10` cores and `16` logical processors
- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`, driver `591.74`, `8188 MiB` VRAM
- Integrated GPU: `Intel UHD Graphics`
- System memory: about `31.7 GiB`

Successful runs:

| rank_by_wer | engine | model_id | avg_elapsed_seconds | avg_rtf | avg_wer | notes |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | whisper | large-v3 | 26.6114 | 0.9654 | 0.1140 | Best WER in this matrix. |
| 2 | whisper | medium | 6.8186 | 0.2474 | 0.1311 | Strong accuracy-speed tradeoff. |
| 3 | whisper | small | 4.2864 | 0.1543 | 0.1483 | Faster than `medium` with moderate WER loss. |
| 4 | canary | nvidia/canary-1b | 20.8662 | 0.7969 | 0.1692 | Best NVIDIA multilingual result in this set. |
| 5 | canary | nvidia/canary-1b-flash | 36.8523 | 1.5940 | 0.1835 | Better WER than `180m`, slower than `1b` on this host. |
| 6 | whisper | base | 2.1005 | 0.0762 | 0.2605 | Fast, but a clear WER drop from `small` and up. |
| 7 | whisper | tiny | 2.2619 | 0.0832 | 0.3663 | Fast, but accuracy degrades substantially. |
| 8 | owsm-ctc | espnet/owsm_ctc_v3.1_1B | 2.4075 | 0.0934 | 0.3869 | Good speed, middling French accuracy on this sample set. |
| 9 | canary | nvidia/canary-180m-flash | 15.5430 | 0.6127 | 0.4833 | Much weaker than the `1b` variants here. |
| 10 | parakeet | nvidia/parakeet-tdt-1.1b | 0.5261 | 0.0196 | 0.9471 | English-focused model; treated as out-of-language baseline. |
| 11 | parakeet | nvidia/parakeet-ctc-0.6b | 0.3363 | 0.0157 | 0.9552 | English-focused model; treated as out-of-language baseline. |
| 12 | parakeet | nvidia/parakeet-ctc-1.1b | 0.2008 | 0.0075 | 0.9718 | English-focused model; treated as out-of-language baseline. |
| 13 | parakeet | nvidia/parakeet-tdt-0.6b-v2 | 0.2223 | 0.0091 | 0.9769 | English-focused model; treated as out-of-language baseline. |
| 14 | moonshine | moonshine/base | 2.4428 | 0.0888 | 0.9887 | French is out-of-language. This run fell back to CPU because ONNX Runtime CUDA DLLs were missing. |
| 15 | funasr | iic/SenseVoiceSmall | 0.6378 | 0.0250 | 0.9935 | Ran after adapter fixes, but poor French output on this set. |
| 16 | moonshine | moonshine/tiny | 2.3212 | 0.0933 | 1.2029 | French is out-of-language. This run fell back to CPU because ONNX Runtime CUDA DLLs were missing. |

Failed run:

| engine | model_id | status | reason |
| --- | --- | --- | --- |
| canary | nvidia/canary-1b-v2 | failed | NeMo prompt-format assertion in worker process. |

Summary:

- Best accuracy: `whisper large-v3`
- Best accuracy-speed balance: `whisper medium`
- Best NVIDIA multilingual result: `nvidia/canary-1b`
- Fastest useful non-Whisper multilingual result: `espnet/owsm_ctc_v3.1_1B`
- Moonshine did not receive real GPU acceleration on this machine because the ONNX Runtime CUDA provider could not load `cublasLt64_12.dll`

## How To Reproduce The GPU Table

Install the dependencies:

```powershell
.\scripts\install-backends.ps1 -Extras whisper,moonshine,nemo,espnet,funasr
```

Run the same model groups used for the consolidated table:

```powershell
.\scripts\run-benchmark.ps1 -Engine whisper -Language fr -Device cuda
.\scripts\run-benchmark.ps1 -Include canary:nvidia/canary-180m-flash -Language fr -Device cuda
.\scripts\run-benchmark.ps1 -Include canary:nvidia/canary-1b -Language fr -Device cuda
.\scripts\run-benchmark.ps1 -Include canary:nvidia/canary-1b-flash -Language fr -Device cuda
.\scripts\run-benchmark.ps1 -Engine parakeet -Language fr -Device cuda
.\scripts\run-benchmark.ps1 -Engine moonshine -Language fr -Device cuda
.\scripts\run-benchmark.ps1 -Include owsm-ctc:espnet/owsm_ctc_v3.1_1B -Language fr -Device cuda
.\scripts\run-benchmark.ps1 -Include funasr:iic/SenseVoiceSmall -Language fr -Device cuda
```

Then compare the resulting `benchmark-summary-<timestamp>.md` files, or use the consolidated files under `outputs/` as the reference format.

## Outputs

Each run writes three files under `outputs/`:

- `benchmark-<timestamp>.json`: full per-sample results
- `benchmark-summary-<timestamp>.csv`: aggregated metrics by model
- `benchmark-summary-<timestamp>.md`: readable summary table

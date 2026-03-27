# OTU-SpeechRecognitionTest

Benchmark harness for comparing open-source ASR systems and testing open-source TTS systems against the speech data under [`data/`](./data).

## What it measures

- Wall-clock transcription time per audio file
- Real-time factor (RTF): `elapsed_seconds / audio_duration_seconds`
- Word error rate (WER) using normalized text

The harness discovers locale folders under `data/` that contain a `transcriptions.txt` file in the format:

```text
filename.wav: Reference transcript
```

## TTS Workbench

The repo includes a `tts_bench` workbench for generating audio from the texts in `data/fr-ca/transcriptions.txt` so humans can listen to the outputs directly.

Currently wired engines:

- Kokoro
- Kitten TTS
- Chatterbox Multilingual
- Chatterbox
- Chatterbox Turbo
- F5-TTS (opt-in, experimental on this dataset)
- Orpheus TTS (opt-in, gated Hugging Face access required)

The TTS run uses one reference clip per locale for the voice-cloning engines. By default, it picks the first transcript entry in that locale, which for `fr-ca` is `interview-1.wav`. The benchmark excludes that prompt clip from the generated target set so cloned-voice models are only timed on held-out texts. You can override the prompt clip with `-ReferenceSample`.

Install one or more TTS backends:

```powershell
.\scripts\install-tts-backends.ps1
```

Or install a subset:

```powershell
.\scripts\install-tts-backends.ps1 -Engine kokoro,kitten-tts,chatterbox,f5-tts,orpheus-tts
```

For CUDA-backed Docker runs, make sure Docker Desktop is running with NVIDIA GPU support enabled. The default GPU matrix uses local Kokoro plus Docker-backed Chatterbox variants.

Run the TTS workbench on the `fr-ca` texts:

```powershell
.\scripts\run-tts-benchmark.ps1 -Device cuda
```

Or limit the matrix explicitly:

```powershell
.\scripts\run-tts-benchmark.ps1 -Include kokoro:hexgrad/kokoro-82m -Include chatterbox:ResembleAI/chatterbox-multilingual -ReferenceSample interview-1.wav
```

Each TTS run writes a timestamped benchmark bundle containing:

- `tts-benchmark-<timestamp>.json`: full per-sample timing and output paths
- `tts-summary-<timestamp>.csv`: aggregated timing summary
- `tts-summary-<timestamp>.md`: readable summary table
- `audio/`: generated WAV files grouped by engine, model, and locale

Curated listening samples are also checked into the repo under `samples/tts/`, with two MP3 clips per tested TTS model (`interview-2` and `interview-3`).

If you add or refresh WAV samples later, you can regenerate MP3s for the full set with:

```powershell
.\scripts\convert-tts-samples-to-mp3.ps1
```

Notes:

- The TTS summary focuses on wall-clock generation time and generated audio duration.
- `kokoro` uses the built-in French pipeline and defaults to the `ff_siwis` voice.
- `kitten-tts` uses the official KittenTTS wheel and was benchmarked here with the `mini`, `micro`, and `nano-int8` English-only models as out-of-language baselines on the French dataset.
- `chatterbox` variants use the selected reference clip for zero-shot voice cloning.
- `f5-tts` currently uses the official upstream Docker image and CLI when selected explicitly.
- `orpheus-tts` currently uses the official Orpheus package in Docker and requires a valid `HF_TOKEN` plus approved access to the requested model repo on Hugging Face.
- `xtts-v2` is not part of the default matrix on this Windows host because the available local and Docker paths were not stable enough for repeatable runs.
- The tracked `samples/tts/` set keeps the curated listening examples in MP3 format to keep the repo smaller.

## ASR Engines and Models

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

This project is set up for Python `3.11`.

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

## How To Reproduce The ASR GPU Table

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

Then compare the resulting `benchmark-summary-<timestamp>.md` files against the summary tables below.

## Outputs

Each run writes three primary files:

- `benchmark-<timestamp>.json`: full per-sample results
- `benchmark-summary-<timestamp>.csv`: aggregated metrics by model
- `benchmark-summary-<timestamp>.md`: readable summary table

## Benchmark Results

### Testing Environment

The consolidated benchmark sections below were run on the same Windows machine unless noted otherwise.

Runtime setup:

- CUDA-enabled PyTorch for Whisper and NeMo-based ASR models
- Docker Desktop with NVIDIA GPU support for the Docker-backed TTS runs
- local Hugging Face and model caches under the repo
- repo-local audio preprocessing for Windows compatibility
- benchmark harnesses in `src/asr_bench` and `src/tts_bench`

Test machine:

- Laptop: `GIGABYTE G6 KF`
- OS: `Microsoft Windows 11 Home` 64-bit, version `10.0.26200`
- CPU: `13th Gen Intel Core i7-13620H` with `10` cores and `16` logical processors
- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`, driver `591.74`, `8188 MiB` VRAM
- Integrated GPU: `Intel UHD Graphics`
- System memory: about `31.7 GiB`

### Aggregated ASR GPU Results

These ASR numbers were obtained on March 25, 2026.

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

### Aggregated TTS GPU Results

These TTS numbers were obtained on March 26-27, 2026 over the 4 held-out `fr-ca` texts, using `interview-1.wav` as the cloning prompt where applicable.

Successful runs:

| engine | model_id | avg_elapsed_seconds | avg_generated_duration_seconds | avg_rtf | notes |
| --- | --- | --- | --- | --- | --- |
| kokoro | hexgrad/kokoro-82m | 7.2080 | 25.3750 | 0.2848 | Fastest TTS model in this matrix by a wide margin. |
| kitten-tts | KittenML/kitten-tts-nano-0.8-int8 | 15.8269 | 39.7646 | 0.3968 | Smallest KittenTTS variant tested, and the fastest Kitten model on this machine. English-only, so treated as an out-of-language baseline on French. |
| kitten-tts | KittenML/kitten-tts-micro-0.8 | 16.2807 | 39.9271 | 0.4077 | Slightly slower than `nano-int8`, but still much faster than the heavier neural voice-cloning stacks. English-only baseline on French. |
| kitten-tts | KittenML/kitten-tts-mini-0.8 | 27.0658 | 34.5458 | 0.7829 | Largest KittenTTS model tested here. Slower than `micro` and `nano-int8`, but still well below real time on CPU. English-only baseline on French. |
| chatterbox | ResembleAI/chatterbox | 162.9319 | 31.9500 | 5.1943 | English-only baseline; slower, but generated the longest audio on this set. |
| chatterbox | ResembleAI/chatterbox-multilingual | 137.7789 | 22.0200 | 6.3773 | Best language-matched Chatterbox variant for this French dataset. |
| chatterbox | ResembleAI/chatterbox-turbo | 140.3128 | 22.3600 | 7.1170 | Slightly faster wall-clock than base Chatterbox, but worse RTF than the multilingual checkpoint. |
| orpheus-tts | canopylabs/3b-fr-ft-research_release | 366.4933 | 7.2533 | 50.7938 | French gated Orpheus run completed successfully after tuning vLLM down to fit this 8 GB GPU, but generation was very slow and outputs were much shorter than the other engines. |

Excluded or unstable runs:

| engine | model_id | status | reason |
| --- | --- | --- | --- |
| f5-tts | F5TTS_v1_Base | unstable | Failed on the full Quebec French prompts with tensor-size mismatches even through the official upstream Docker image and CLI. |
| xtts-v2 | tts_models/multilingual/multi-dataset/xtts_v2 | failed | Local and Docker-backed paths were not stable enough on this Windows host for repeatable runs. |

Summary:

- Fastest overall: `kokoro`
- Fastest KittenTTS variant: `KittenML/kitten-tts-nano-0.8-int8`
- Fastest Chatterbox variant by wall-clock time: `ResembleAI/chatterbox-multilingual`
- Lowest Chatterbox RTF in this run: `ResembleAI/chatterbox`
- Orpheus can be made to run on this 8 GB GPU, but only with reduced vLLM limits and with a very poor speed-to-audio ratio
- The default TTS matrix in this repo currently stays on the models that produced repeatable audio on this machine

### TTS Sample Players

The embedded players below use the tracked MP3 samples from the models in the aggregated TTS results above.

| engine | model_id | interview-2 | interview-3 |
| --- | --- | --- | --- |
| kokoro | `hexgrad/kokoro-82m` | <audio controls src="./samples/tts/kokoro/hexgrad-kokoro-82m/interview-2.mp3"></audio> | <audio controls src="./samples/tts/kokoro/hexgrad-kokoro-82m/interview-3.mp3"></audio> |
| chatterbox | `ResembleAI/chatterbox-multilingual` | <audio controls src="./samples/tts/chatterbox/ResembleAI-chatterbox-multilingual/interview-2.mp3"></audio> | <audio controls src="./samples/tts/chatterbox/ResembleAI-chatterbox-multilingual/interview-3.mp3"></audio> |
| chatterbox | `ResembleAI/chatterbox` | <audio controls src="./samples/tts/chatterbox/ResembleAI-chatterbox/interview-2.mp3"></audio> | <audio controls src="./samples/tts/chatterbox/ResembleAI-chatterbox/interview-3.mp3"></audio> |
| chatterbox | `ResembleAI/chatterbox-turbo` | <audio controls src="./samples/tts/chatterbox/ResembleAI-chatterbox-turbo/interview-2.mp3"></audio> | <audio controls src="./samples/tts/chatterbox/ResembleAI-chatterbox-turbo/interview-3.mp3"></audio> |
| kitten-tts | `KittenML/kitten-tts-nano-0.8-int8` | <audio controls src="./samples/tts/kitten-tts/KittenML-kitten-tts-nano-0.8-int8/interview-2.mp3"></audio> | <audio controls src="./samples/tts/kitten-tts/KittenML-kitten-tts-nano-0.8-int8/interview-3.mp3"></audio> |
| kitten-tts | `KittenML/kitten-tts-micro-0.8` | <audio controls src="./samples/tts/kitten-tts/KittenML-kitten-tts-micro-0.8/interview-2.mp3"></audio> | <audio controls src="./samples/tts/kitten-tts/KittenML-kitten-tts-micro-0.8/interview-3.mp3"></audio> |
| kitten-tts | `KittenML/kitten-tts-mini-0.8` | <audio controls src="./samples/tts/kitten-tts/KittenML-kitten-tts-mini-0.8/interview-2.mp3"></audio> | <audio controls src="./samples/tts/kitten-tts/KittenML-kitten-tts-mini-0.8/interview-3.mp3"></audio> |
| orpheus-tts | `canopylabs/3b-fr-ft-research_release` | <audio controls src="./samples/tts/orpheus-tts/canopylabs-3b-fr-ft-research_release/interview-2.mp3"></audio> | <audio controls src="./samples/tts/orpheus-tts/canopylabs-3b-fr-ft-research_release/interview-3.mp3"></audio> |

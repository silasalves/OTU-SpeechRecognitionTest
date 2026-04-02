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
- VoxCPM 0.5B (opt-in, experimental on this dataset)
- Qwen3-TTS 0.6B Base (opt-in)
- CosyVoice2 0.5B (opt-in, fixed-speaker no-clone smoke path)
- VibeVoice Realtime 0.5B (opt-in, dedicated-venv official runtime)
- Dia2 1B (opt-in, experimental on this dataset)
- GLM-TTS (opt-in, English/Chinese mixed smoke path on this dataset)
- Orpheus TTS (opt-in, gated Hugging Face access required)

The TTS run uses one reference clip per locale for the voice-cloning engines. By default, it picks the first transcript entry in that locale, which for `fr-ca` is `interview-1.wav`. The benchmark excludes that prompt clip from the generated target set so cloned-voice models are only timed on held-out texts. You can override the prompt clip with `-ReferenceSample`.

Install one or more TTS backends:

```powershell
.\scripts\install-tts-backends.ps1
```

Or install a subset:

```powershell
.\scripts\install-tts-backends.ps1 -Engine kokoro,kitten-tts,chatterbox,f5-tts,voxcpm,qwen3-tts,cosyvoice2,vibevoice,dia2,glm-tts,orpheus-tts
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

Curated listening samples are also checked into the repo under `samples/tts/`, with one or two MP3 clips per tested TTS model.

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
- `voxcpm` currently uses a custom Docker image around `openbmb/VoxCPM-0.5B`, with `optimize=False` in the upstream package and a `soundfile` prompt-audio loader workaround for the current Docker `torchaudio` and `torchcodec` stack on this machine.
- `voxcpm` was first smoke-tested successfully on March 31, 2026 through `.\scripts\run-tts-benchmark.ps1 -Include voxcpm:openbmb/VoxCPM-0.5B -Limit 1 -Device cuda`. Because upstream is primarily oriented toward Chinese and English, it was later rerun on April 1, 2026 against the same English-translated `interview-2` and `interview-3` set used for the other English-only models, using the upstream English prompt. That English pair averaged about `73.47` seconds for `28.12` seconds of audio (`RTF 2.62`). VoxCPM's upstream API also supports generation without a prompt pair, and that prompt-free English path was smoke-tested successfully in `outputs/tts/voxcpm-english-noclone-smoke.wav`.
- `voxcpm` was also sanity-checked separately with an English sentence on March 31, 2026, using the same Docker path and the same `interview-1.wav` reference clip. That English-only check completed successfully and wrote `outputs/tts/voxcpm-english-smoke.wav`.
- `qwen3-tts` currently uses a custom Docker image built on the official `pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime` image because the upstream Qwen3-TTS repo did not expose an official Docker image or Dockerfile in this workspace check.
- `qwen3-tts` was first smoke-tested successfully on March 31, 2026 through `.\scripts\run-tts-benchmark.ps1 -Include qwen3-tts:Qwen/Qwen3-TTS-12Hz-0.6B-Base -Limit 1 -Device cuda`, and was then rerun on April 1, 2026 on held-out `fr-ca` `interview-2` and `interview-3` through a corrected 2-sample smoke set that uses the real `interview-1.wav` transcript as the voice-cloning prompt text. That corrected pair averaged about `301.07` seconds for `20.00` seconds of audio (`RTF 15.01`). An earlier March 31 rerun had mistakenly paired the prompt audio with the wrong prompt transcript, which produced much worse and sometimes runaway results.
- `cosyvoice2` currently uses a custom Docker image built on the official `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel` image because the upstream CosyVoice repo exposed source and Dockerfile workflows, but no published inference image in this workspace check. A native Windows source install was attempted on April 1, 2026 in an isolated venv and failed early while building `openai-whisper`, so this repo validates the Docker path instead.
- `FunAudioLLM/CosyVoice2-0.5B` does not ship a built-in `spk2info.pt`, so the repo's no-clone CosyVoice2 path downloads the upstream fixed-speaker cache referenced by the official Triton runtime docs and runs `inference_sft()` rather than any voice-cloning API. That upstream fixed-speaker cache currently exposes three Chinese-labeled speakers (`系统默认`, `龙小夏`, and `湾湾小何`). Because this validated no-clone stock-speaker path does not expose a native French preset voice in this repo, CosyVoice2 is documented here against the same English-translated `interview-2` and `interview-3` smoke set used for the other out-of-language TTS baselines. By default it prefers the cached `longxiaoxia` speaker, and you can override that with `TTS_BENCH_COSYVOICE2_SPEAKER_ID`.
- `cosyvoice2` was smoke-tested successfully on April 2, 2026 through `uv run --no-sync python -m tts_bench.cli --data-root .tmp/glmtts-smoke-data --output-dir outputs/tts --language en --device cuda --locale en --include cosyvoice2:FunAudioLLM/CosyVoice2-0.5B --reference-sample interview-1.wav --limit 2`, which exercised the English-translated `interview-2` and `interview-3` pair without voice cloning and averaged about `146.00` seconds for `20.48` seconds of audio (`RTF 7.13`). This row is kept here as a fixed-speaker out-of-language baseline rather than a French preset-voice comparison.
- `vibevoice` in this repo currently means the official `microsoft/VibeVoice-Realtime-0.5B` path, not the older long-form `VibeVoice-TTS-1.5B` path, because Microsoft removed the long-form TTS code from the upstream repo on September 5, 2025. The validated integration here uses a dedicated local venv at `.venvs/vibevoice`, installs the official repo in editable mode, and then runs the official realtime model locally rather than through Docker.
- `VibeVoice-Realtime-0.5B` is officially described upstream as primarily English, but the upstream repo also ships experimental multilingual speaker presets, including two French presets (`fr-Spk0_man` and `fr-Spk1_woman`). This repo uses the bundled `fr-Spk1_woman` preset by default for `fr-ca`, and you can override that with `TTS_BENCH_VIBEVOICE_SPEAKER_NAME`. On this Windows setup the dedicated venv uses the official CUDA 12.8 PyTorch wheel and falls back to SDPA when FlashAttention2 is unavailable.
- `vibevoice` was smoke-tested successfully on April 2, 2026 through `.\scripts\run-tts-benchmark.ps1 -Include vibevoice:microsoft/VibeVoice-Realtime-0.5B -Limit 2 -Device cuda`, which exercised the held-out French `interview-2` and `interview-3` pair with the upstream experimental French voice preset and averaged about `118.02` seconds for `19.00` seconds of audio (`RTF 5.97`).
- `dia2` currently uses a custom Docker image built on the official `pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime` image because the upstream Dia2 repo did not expose an official Docker image or Dockerfile in this workspace check. A native Windows source install may be possible in principle, but the upstream GitHub package path was not stable enough here, so the benchmark integration uses Docker only.
- `dia2` was first smoke-tested successfully on March 31, 2026 through `.\scripts\run-tts-benchmark.ps1 -Include dia2:nari-labs/Dia2-1B -Limit 1 -Device cuda`. Because upstream currently documents English generation only, it was rerun later that day on English translations of `interview-2` and `interview-3` using the upstream English prompt, averaging about `642.21` seconds for `21.88` seconds of audio (`RTF 29.36`).
- `glm-tts` currently uses a custom Docker image built on the official `pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime` image because the upstream GLM-TTS repo did not expose an official Docker image or Dockerfile in this workspace check. A native Windows source install looks possible in principle, but this repo only validated the Docker path.
- `glm-tts` was smoke-tested successfully on March 31, 2026 through `uv run --no-sync python -m tts_bench.cli --data-root .tmp/glmtts-smoke-data --output-dir outputs/tts --language en --device cuda --include glm-tts:zai-org/GLM-TTS --reference-sample interview-1.wav`, producing English-translated `interview-2` and `interview-3` samples in about `1308.91` seconds on average for `25.06` seconds of audio (`RTF 53.46`).
- `orpheus-tts` currently uses the official Orpheus package in Docker and requires a valid `HF_TOKEN` plus approved access to the requested model repo on Hugging Face.
- `xtts-v2` is not part of the default matrix on this Windows host because the available local and Docker paths were not stable enough for repeatable runs.
- Several smaller Voxtral TTS repos were evaluated on March 31, 2026. Native Windows install is blocked because `mlx` does not ship `win_amd64` wheels. Linux CUDA Docker prototypes could load `mlx-community/Voxtral-4B-TTS-2603-mlx-4bit` (~2.54 GB) and `mlx-community/Voxtral-4B-TTS-2603-mlx-6bit` (~3.50 GB), but both failed during synthesis because MLX CUDA currently reports no `quantized_matmul` implementation for the required 4-bit and 6-bit affine kernels on this setup. `AITRADER/Voxtral-4B-TTS-2603-mxfp4` (~3.03 GB) was also tried and failed earlier during model load because the current `mlx-audio` loader did not accept its parameter layout. Larger BF16 variants were not pursued further because their weights are about 8.0 GB and Mistral's official model card recommends at least 16 GB of GPU memory, while the discovered GGUF repo exposed 4.70 GB Q8_0 and 8.81 GB F16 files without a documented TTS runtime path in this repo.
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

| Rank by WER | Engine | Model | Avg elapsed (s) | Avg RTF | Avg WER | Notes |
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

| Engine | Model | Status | Reason |
| --- | --- | --- | --- |
| canary | nvidia/canary-1b-v2 | failed | NeMo prompt-format assertion in worker process. |

Summary:

- Best accuracy: `whisper large-v3`
- Best accuracy-speed balance: `whisper medium`
- Best NVIDIA multilingual result: `nvidia/canary-1b`
- Fastest useful non-Whisper multilingual result: `espnet/owsm_ctc_v3.1_1B`
- Moonshine did not receive real GPU acceleration on this machine because the ONNX Runtime CUDA provider could not load `cublasLt64_12.dll`

### Aggregated TTS GPU Results

These TTS numbers were obtained on the same Windows machine using `interview-1.wav` as the cloning prompt where applicable. Most rows come from the March 26-27, 2026 4-sample `fr-ca` run; `qwen3-tts` was rerun again on April 1, 2026 as a corrected 2-sample `fr-ca` smoke test on held-out `interview-2` and `interview-3`, `cosyvoice2` was rerun on April 2, 2026 as its own 2-sample English-translated no-clone smoke test on `interview-2` and `interview-3` because the validated stock-speaker path here does not expose a French preset voice, `vibevoice` was added later that same day as its own 2-sample held-out `fr-ca` smoke test using the upstream experimental French speaker preset through a dedicated local venv, `voxcpm` was rerun on April 1, 2026 on the English-translated `interview-2` and `interview-3` pair because its validated path here tracks upstream English support more closely, `dia2` was rerun later on March 31, 2026 as a 2-sample English-translated smoke test, and `glm-tts` was added later that same day as its own 2-sample English-translated smoke test.

Successful runs:

| Engine | Model | Avg elapsed (s) | Generated audio (s) | Avg RTF | Notes |
| --- | --- | --- | --- | --- | --- |
| kokoro | hexgrad/kokoro-82m | 7.2080 | 25.3750 | 0.2848 | Fastest TTS model in this matrix by a wide margin. |
| kitten-tts | KittenML/kitten-tts-nano-0.8-int8 | 15.8269 | 39.7646 | 0.3968 | Smallest KittenTTS variant tested, and the fastest Kitten model on this machine. English-only, so treated as an out-of-language baseline on French. |
| kitten-tts | KittenML/kitten-tts-micro-0.8 | 16.2807 | 39.9271 | 0.4077 | Slightly slower than `nano-int8`, but still much faster than the heavier neural voice-cloning stacks. English-only baseline on French. |
| kitten-tts | KittenML/kitten-tts-mini-0.8 | 27.0658 | 34.5458 | 0.7829 | Largest KittenTTS model tested here. Slower than `micro` and `nano-int8`, but still well below real time on CPU. English-only baseline on French. |
| chatterbox | ResembleAI/chatterbox | 162.9319 | 31.9500 | 5.1943 | English-only baseline; slower, but generated the longest audio on this set. |
| chatterbox | ResembleAI/chatterbox-multilingual | 137.7789 | 22.0200 | 6.3773 | Best language-matched Chatterbox variant for this French dataset. |
| chatterbox | ResembleAI/chatterbox-turbo | 140.3128 | 22.3600 | 7.1170 | Slightly faster wall-clock than base Chatterbox, but worse RTF than the multilingual checkpoint. |
| voxcpm | openbmb/VoxCPM-0.5B | 73.4698 | 28.1200 | 2.6175 | Added later as a 2-sample English-translated smoke test on `interview-2` and `interview-3`, using the upstream English prompt because the validated VoxCPM path here tracks English support more closely. Docker-backed VoxCPM 0.5B voice-cloning run. |
| qwen3-tts | Qwen/Qwen3-TTS-12Hz-0.6B-Base | 301.0716 | 20.0000 | 15.0053 | Added later as a corrected 2-sample `fr-ca` smoke test on `interview-2` and `interview-3`. Docker-backed Qwen3-TTS 0.6B voice-cloning run. |
| cosyvoice2 | FunAudioLLM/CosyVoice2-0.5B | 146.0015 | 20.4800 | 7.1326 | Added later as a 2-sample English-translated smoke test on `interview-2` and `interview-3`, using the upstream fixed-speaker cache and `inference_sft()` so this path avoids voice cloning. The validated stock speakers are not French presets, so this is kept as an out-of-language fixed-speaker baseline. |
| vibevoice | microsoft/VibeVoice-Realtime-0.5B | 118.0217 | 19.0000 | 5.9713 | Added later as a 2-sample held-out `fr-ca` smoke test on `interview-2` and `interview-3`, using the official dedicated-venv realtime runtime with the upstream experimental French `fr-Spk1_woman` preset. This repo validates the realtime single-speaker path because the older long-form VibeVoice-TTS code is disabled upstream. |
| orpheus-tts | canopylabs/3b-fr-ft-research_release | 366.4933 | 7.2533 | 50.7938 | French gated Orpheus run completed successfully after tuning vLLM down to fit this 8 GB GPU, but generation was very slow and outputs were much shorter than the other engines. |
| glm-tts | zai-org/GLM-TTS | 1308.9063 | 25.0600 | 53.4616 | Added later as a 2-sample English-translated smoke test on `interview-2` and `interview-3`, using the upstream English prompt because French is not documented upstream. |
| dia2 | nari-labs/Dia2-1B | 642.2125 | 21.8800 | 29.3624 | Added later as a 2-sample English-translated smoke test on `interview-2` and `interview-3`, using the upstream English prompt because Dia2 currently documents English generation only. |

Excluded or unstable runs:

| Engine | Model | Status | Reason |
| --- | --- | --- | --- |
| f5-tts | F5TTS_v1_Base | unstable | Failed on the full Quebec French prompts with tensor-size mismatches even through the official upstream Docker image and CLI. |
| voxtral-mlx | mlx-community/Voxtral-4B-TTS-2603-mlx-4bit | failed | Native Windows install was not possible because `mlx` has no `win_amd64` wheels, and a custom Linux CUDA Docker prototype loaded the model but failed at generation time with MLX CUDA reporting no implementation for the needed 4-bit affine `quantized_matmul`. |
| voxtral-mlx | mlx-community/Voxtral-4B-TTS-2603-mlx-6bit | failed | Same Docker-backed MLX path as the 4-bit variant, but generation still failed on this host because MLX CUDA reported no implementation for the required 6-bit affine `quantized_matmul`. |
| voxtral-mlx | AITRADER/Voxtral-4B-TTS-2603-mxfp4 | failed | This smaller MXFP4 variant was worth testing by size, but the current `mlx-audio` stack rejected the checkpoint during load with hundreds of unexpected parameters, so it did not reach synthesis on this setup. |
| xtts-v2 | tts_models/multilingual/multi-dataset/xtts_v2 | failed | Local and Docker-backed paths were not stable enough on this Windows host for repeatable runs. |

Summary:

- Fastest overall: `kokoro`
- Fastest KittenTTS variant: `KittenML/kitten-tts-nano-0.8-int8`
- Fastest Chatterbox variant by wall-clock time: `ResembleAI/chatterbox-multilingual`
- Lowest Chatterbox RTF in this run: `ResembleAI/chatterbox`
- Orpheus can be made to run on this 8 GB GPU, but only with reduced vLLM limits and with a very poor speed-to-audio ratio
- GLM-TTS can also run here through Docker, but the validated path in this repo currently uses the upstream English prompt plus English translations of `interview-2` and `interview-3` because French is not documented upstream
- VibeVoice currently works here through its own dedicated `.venvs/vibevoice` runtime using the official realtime 0.5B model and the upstream experimental French speaker preset; this repo does not validate the older long-form TTS path because its upstream code was removed
- An earlier Qwen3-TTS and VoxCPM French smoke rerun paired the prompt audio with the wrong prompt transcript, which turned out to be a real benchmarking bug and produced much worse outputs than the corrected rerun
- VoxCPM also supports prompt-free generation upstream, and that no-clone English path worked here in a separate smoke test
- The smallest alternate Voxtral repos were still not usable here: the MLX-community 4-bit and 6-bit builds hit MLX CUDA backend limits during generation, and the smaller AITRADER MXFP4 build failed to load in the current `mlx-audio` stack
- The default TTS matrix in this repo currently stays on the models that produced repeatable audio on this machine

### TTS Sample Players

The embedded players below use the tracked MP3 samples from the TTS results above. Most rows have the older curated `fr-ca` `interview-2` and `interview-3` pair; `qwen3-tts` now uses a later corrected `fr-ca` smoke-test pair, `vibevoice` uses a later held-out `fr-ca` pair through the official realtime runtime and experimental French voice preset, while `cosyvoice2`, `voxcpm`, `dia2`, and `glm-tts` use English-translated `interview-2` and `interview-3` smoke-test pairs because their validated paths in this repo currently follow upstream non-French usage more closely.

| Engine | Model | Interview 2 | Interview 3 |
| --- | --- | --- | --- |
| kokoro | `hexgrad/kokoro-82m` | <audio controls src="./samples/tts/kokoro/hexgrad-kokoro-82m/interview-2.mp3"><a href="./samples/tts/kokoro/hexgrad-kokoro-82m/interview-2.mp3">interview-2.mp3</a></audio> | <audio controls src="./samples/tts/kokoro/hexgrad-kokoro-82m/interview-3.mp3"><a href="./samples/tts/kokoro/hexgrad-kokoro-82m/interview-3.mp3">interview-3.mp3</a></audio> |
| chatterbox | `ResembleAI/chatterbox-multilingual` | <audio controls src="./samples/tts/chatterbox/ResembleAI-chatterbox-multilingual/interview-2.mp3"><a href="./samples/tts/chatterbox/ResembleAI-chatterbox-multilingual/interview-2.mp3">interview-2.mp3</a></audio> | <audio controls src="./samples/tts/chatterbox/ResembleAI-chatterbox-multilingual/interview-3.mp3"><a href="./samples/tts/chatterbox/ResembleAI-chatterbox-multilingual/interview-3.mp3">interview-3.mp3</a></audio> |
| chatterbox | `ResembleAI/chatterbox` | <audio controls src="./samples/tts/chatterbox/ResembleAI-chatterbox/interview-2.mp3"><a href="./samples/tts/chatterbox/ResembleAI-chatterbox/interview-2.mp3">interview-2.mp3</a></audio> | <audio controls src="./samples/tts/chatterbox/ResembleAI-chatterbox/interview-3.mp3"><a href="./samples/tts/chatterbox/ResembleAI-chatterbox/interview-3.mp3">interview-3.mp3</a></audio> |
| chatterbox | `ResembleAI/chatterbox-turbo` | <audio controls src="./samples/tts/chatterbox/ResembleAI-chatterbox-turbo/interview-2.mp3"><a href="./samples/tts/chatterbox/ResembleAI-chatterbox-turbo/interview-2.mp3">interview-2.mp3</a></audio> | <audio controls src="./samples/tts/chatterbox/ResembleAI-chatterbox-turbo/interview-3.mp3"><a href="./samples/tts/chatterbox/ResembleAI-chatterbox-turbo/interview-3.mp3">interview-3.mp3</a></audio> |
| kitten-tts | `KittenML/kitten-tts-nano-0.8-int8` | <audio controls src="./samples/tts/kitten-tts/KittenML-kitten-tts-nano-0.8-int8/interview-2.mp3"><a href="./samples/tts/kitten-tts/KittenML-kitten-tts-nano-0.8-int8/interview-2.mp3">interview-2.mp3</a></audio> | <audio controls src="./samples/tts/kitten-tts/KittenML-kitten-tts-nano-0.8-int8/interview-3.mp3"><a href="./samples/tts/kitten-tts/KittenML-kitten-tts-nano-0.8-int8/interview-3.mp3">interview-3.mp3</a></audio> |
| kitten-tts | `KittenML/kitten-tts-micro-0.8` | <audio controls src="./samples/tts/kitten-tts/KittenML-kitten-tts-micro-0.8/interview-2.mp3"><a href="./samples/tts/kitten-tts/KittenML-kitten-tts-micro-0.8/interview-2.mp3">interview-2.mp3</a></audio> | <audio controls src="./samples/tts/kitten-tts/KittenML-kitten-tts-micro-0.8/interview-3.mp3"><a href="./samples/tts/kitten-tts/KittenML-kitten-tts-micro-0.8/interview-3.mp3">interview-3.mp3</a></audio> |
| kitten-tts | `KittenML/kitten-tts-mini-0.8` | <audio controls src="./samples/tts/kitten-tts/KittenML-kitten-tts-mini-0.8/interview-2.mp3"><a href="./samples/tts/kitten-tts/KittenML-kitten-tts-mini-0.8/interview-2.mp3">interview-2.mp3</a></audio> | <audio controls src="./samples/tts/kitten-tts/KittenML-kitten-tts-mini-0.8/interview-3.mp3"><a href="./samples/tts/kitten-tts/KittenML-kitten-tts-mini-0.8/interview-3.mp3">interview-3.mp3</a></audio> |
| voxcpm | `openbmb/VoxCPM-0.5B` | <audio controls src="./samples/tts/voxcpm/openbmb-VoxCPM-0.5B/interview-2.mp3"><a href="./samples/tts/voxcpm/openbmb-VoxCPM-0.5B/interview-2.mp3">interview-2.mp3</a></audio> | <audio controls src="./samples/tts/voxcpm/openbmb-VoxCPM-0.5B/interview-3.mp3"><a href="./samples/tts/voxcpm/openbmb-VoxCPM-0.5B/interview-3.mp3">interview-3.mp3</a></audio> |
| qwen3-tts | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | <audio controls src="./samples/tts/qwen3-tts/Qwen-Qwen3-TTS-12Hz-0.6B-Base/interview-2.mp3"><a href="./samples/tts/qwen3-tts/Qwen-Qwen3-TTS-12Hz-0.6B-Base/interview-2.mp3">interview-2.mp3</a></audio> | <audio controls src="./samples/tts/qwen3-tts/Qwen-Qwen3-TTS-12Hz-0.6B-Base/interview-3.mp3"><a href="./samples/tts/qwen3-tts/Qwen-Qwen3-TTS-12Hz-0.6B-Base/interview-3.mp3">interview-3.mp3</a></audio> |
| cosyvoice2 | `FunAudioLLM/CosyVoice2-0.5B` | <audio controls src="./samples/tts/cosyvoice2/FunAudioLLM-CosyVoice2-0.5B/interview-2.mp3"><a href="./samples/tts/cosyvoice2/FunAudioLLM-CosyVoice2-0.5B/interview-2.mp3">interview-2.mp3</a></audio> | <audio controls src="./samples/tts/cosyvoice2/FunAudioLLM-CosyVoice2-0.5B/interview-3.mp3"><a href="./samples/tts/cosyvoice2/FunAudioLLM-CosyVoice2-0.5B/interview-3.mp3">interview-3.mp3</a></audio> |
| vibevoice | `microsoft/VibeVoice-Realtime-0.5B` | <audio controls src="./samples/tts/vibevoice/microsoft-VibeVoice-Realtime-0.5B/interview-2.mp3"><a href="./samples/tts/vibevoice/microsoft-VibeVoice-Realtime-0.5B/interview-2.mp3">interview-2.mp3</a></audio> | <audio controls src="./samples/tts/vibevoice/microsoft-VibeVoice-Realtime-0.5B/interview-3.mp3"><a href="./samples/tts/vibevoice/microsoft-VibeVoice-Realtime-0.5B/interview-3.mp3">interview-3.mp3</a></audio> |
| glm-tts | `zai-org/GLM-TTS` | <audio controls src="./samples/tts/glm-tts/zai-org-GLM-TTS/interview-2.mp3"><a href="./samples/tts/glm-tts/zai-org-GLM-TTS/interview-2.mp3">interview-2.mp3</a></audio> | <audio controls src="./samples/tts/glm-tts/zai-org-GLM-TTS/interview-3.mp3"><a href="./samples/tts/glm-tts/zai-org-GLM-TTS/interview-3.mp3">interview-3.mp3</a></audio> |
| dia2 | `nari-labs/Dia2-1B` | <audio controls src="./samples/tts/dia2/nari-labs-Dia2-1B/interview-2.mp3"><a href="./samples/tts/dia2/nari-labs-Dia2-1B/interview-2.mp3">interview-2.mp3</a></audio> | <audio controls src="./samples/tts/dia2/nari-labs-Dia2-1B/interview-3.mp3"><a href="./samples/tts/dia2/nari-labs-Dia2-1B/interview-3.mp3">interview-3.mp3</a></audio> |
| orpheus-tts | `canopylabs/3b-fr-ft-research_release` | <audio controls src="./samples/tts/orpheus-tts/canopylabs-3b-fr-ft-research_release/interview-2.mp3"><a href="./samples/tts/orpheus-tts/canopylabs-3b-fr-ft-research_release/interview-2.mp3">interview-2.mp3</a></audio> | <audio controls src="./samples/tts/orpheus-tts/canopylabs-3b-fr-ft-research_release/interview-3.mp3"><a href="./samples/tts/orpheus-tts/canopylabs-3b-fr-ft-research_release/interview-3.mp3">interview-3.mp3</a></audio> |

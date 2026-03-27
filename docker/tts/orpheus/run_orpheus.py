from __future__ import annotations

import os
import wave

from vllm import AsyncEngineArgs, AsyncLLMEngine
from orpheus_tts import OrpheusModel


def _setup_engine_for_small_gpu(self: OrpheusModel) -> AsyncLLMEngine:
    engine_args = AsyncEngineArgs(
        model=self.model_name,
        dtype=self.dtype,
        gpu_memory_utilization=0.995,
        max_model_len=768,
        disable_custom_all_reduce=True,
        enable_chunked_prefill=False,
        enforce_eager=True,
    )
    return AsyncLLMEngine.from_engine_args(engine_args)


def main() -> None:
    model_id = os.environ["MODEL_ID"]
    text = os.environ["TEXT"]
    output_path = os.environ["OUTPUT_PATH"]

    OrpheusModel._setup_engine = _setup_engine_for_small_gpu
    model = OrpheusModel(model_name=model_id)
    audio_chunks = model.generate_speech(prompt=text, voice=None)

    with wave.open(output_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        for chunk in audio_chunks:
            wav_file.writeframes(chunk)


if __name__ == "__main__":
    main()

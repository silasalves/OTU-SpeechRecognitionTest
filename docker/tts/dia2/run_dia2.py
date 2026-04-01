from __future__ import annotations

import os
from pathlib import Path
import types

import torch

# Dia2 currently expects a cudnn.conv backend shim that is not present in the
# official PyTorch 2.8 runtime image we use here.
if not hasattr(torch.backends.cudnn, "conv"):
    torch.backends.cudnn.conv = types.SimpleNamespace()

from dia2 import Dia2, GenerationConfig, PrefixConfig, SamplingConfig


def main() -> None:
    model_id = os.environ["MODEL_ID"]
    text = os.environ["TEXT"]
    output_path = Path(os.environ["OUTPUT_PATH"])
    reference_audio_path = os.environ["REFERENCE_AUDIO_PATH"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dia = Dia2.from_repo(
        model_id,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="bfloat16" if torch.cuda.is_available() else "float32",
    )
    config = GenerationConfig(
        cfg_scale=2.0,
        text=SamplingConfig(temperature=0.6, top_k=50),
        audio=SamplingConfig(temperature=0.8, top_k=50),
        prefix=PrefixConfig(
            speaker_1=reference_audio_path,
            include_audio=False,
        ),
        use_cuda_graph=False,
        use_torch_compile=False,
    )
    dia.generate(
        f"[S1] {text}",
        config=config,
        output_wav=str(output_path),
        verbose=False,
    )


if __name__ == "__main__":
    main()

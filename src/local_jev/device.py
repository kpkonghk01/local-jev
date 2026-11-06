"""Where a model runs: CUDA, then MPS, then CPU unless overridden; bf16 there, fp32 on CPU."""

from __future__ import annotations

DEVICES = ("auto", "cuda", "mps", "cpu")
DTYPES = ("auto", "bfloat16", "float16", "float32")


def pick_device(override: str = "auto") -> str:
    if override != "auto":
        return override
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str, override: str = "auto") -> str:
    if override != "auto":
        return override
    return "float32" if device == "cpu" else "bfloat16"

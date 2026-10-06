from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def encode_in_batches(encoder: Any, texts: Iterable[str], batch_size: int = 64) -> list[Any]:
    """Encode text in bounded batches for CPU or GPU memory stability."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    values = list(texts)
    return [
        embedding
        for start in range(0, len(values), batch_size)
        for embedding in encoder(values[start : start + batch_size])
    ]

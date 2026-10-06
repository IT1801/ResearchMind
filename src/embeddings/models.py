from __future__ import annotations

from dataclasses import dataclass
import os

from .encoder import DEFAULT_EMBEDDING_MODEL


@dataclass(frozen=True)
class EmbeddingModelConfig:
    name: str = os.getenv("BGE_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    normalize_embeddings: bool = True
    batch_size: int = 64

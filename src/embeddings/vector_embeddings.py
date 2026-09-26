from __future__ import annotations

import os
from typing import Any

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"


def bge_embedding_function(
    model_name: str | None = None,
) -> Any:
    """Create the normalized BGE embedding function used by Chroma."""
    return SentenceTransformerEmbeddingFunction(
        model_name=model_name or os.getenv("BGE_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        normalize_embeddings=True,
    )

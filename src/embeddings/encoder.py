from __future__ import annotations

import os
from typing import Any

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv

from ..core.exceptions import EmbeddingError
from ..core.logging import get_logger, log_exception

load_dotenv()

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
logger = get_logger(__name__)


def bge_embedding_function(model_name: str | None = None) -> Any:
    """Create the normalized BGE embedding function used by Chroma."""
    selected_model = model_name or os.getenv("BGE_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    try:
        logger.info("Initializing embedding model", extra={"model": selected_model})
        return SentenceTransformerEmbeddingFunction(
            model_name=selected_model,
            normalize_embeddings=True,
        )
    except Exception as error:
        log_exception(logger, "Unable to initialize embedding model", error, {
            "model": selected_model,
        })
        raise EmbeddingError(
            f"Unable to initialize embedding model: {selected_model}"
        ) from error

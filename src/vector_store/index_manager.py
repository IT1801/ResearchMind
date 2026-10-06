from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

from ..core.constants import HNSW_EF, HNSW_M

HNSW_METADATA = {
    "hnsw:M": HNSW_M,
    "hnsw:construction_ef": HNSW_EF,
    "hnsw:search_ef": HNSW_EF,
}


class VectorIndexManager:
    """Create, inspect, and rebuild Chroma indexes with explicit HNSW settings."""

    def __init__(self, path: str | Path = ".chroma") -> None:
        self.client = chromadb.PersistentClient(path=str(path))

    def create(self, collection_name: str, embedding_function: Any) -> Any:
        return self.client.get_or_create_collection(
            collection_name,
            metadata=HNSW_METADATA,
            embedding_function=embedding_function,
        )

    def count(self, collection_name: str) -> int:
        return self.client.get_collection(collection_name).count()

    def delete(self, collection_name: str) -> None:
        self.client.delete_collection(collection_name)

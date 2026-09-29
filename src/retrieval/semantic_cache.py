from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from langchain_core.documents import Document

from ..embeddings.vector_embeddings import bge_embedding_function
from ..infra.exceptions import RAGError
from ..vector_store.client import HNSW_METADATA

DEFAULT_CACHE_COLLECTION = "semantic_query_cache"
DEFAULT_SIMILARITY_THRESHOLD = 0.12


@dataclass(frozen=True)
class CachedResult:
    document: Document
    score: float
    vector_rank: int | None
    bm25_rank: int | None


class SemanticQueryCache:
    """Persist semantically similar query results in a separate Chroma collection."""

    def __init__(
        self,
        path: str | Path = ".chroma",
        collection_name: str = DEFAULT_CACHE_COLLECTION,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        collection: Any | None = None,
    ) -> None:
        if similarity_threshold < 0:
            raise ValueError("similarity_threshold must not be negative")
        self.similarity_threshold = similarity_threshold
        if collection is not None:
            self.collection = collection
            return
        try:
            client = chromadb.PersistentClient(path=str(path))
            self.collection = client.get_or_create_collection(
                collection_name,
                metadata=HNSW_METADATA,
                embedding_function=bge_embedding_function(),
            )
        except Exception as error:
            raise RAGError("Unable to initialize semantic query cache") from error

    def get(self, query: str) -> list[CachedResult] | None:
        """Return cached results when the nearest query is within the threshold."""
        if not query.strip() or self.collection.count() == 0:
            return None
        try:
            result = self.collection.query(
                query_texts=[query],
                n_results=1,
				include=["metadatas", "distances"],
            )
            distance = (result.get("distances") or [[]])[0]
            metadatas = (result.get("metadatas") or [[]])[0]
            if not distance or not metadatas or distance[0] > self.similarity_threshold:
                return None
            payload = json.loads(metadatas[0]["payload"])
            return [
                CachedResult(
                    document=Document(
                        page_content=item["page_content"],
                        metadata=item.get("metadata", {}),
                    ),
                    score=float(item["score"]),
                    vector_rank=item.get("vector_rank"),
                    bm25_rank=item.get("bm25_rank"),
                )
                for item in payload
            ]
        except Exception as error:
            raise RAGError("Unable to read semantic query cache") from error

    def put(self, query: str, results: list[CachedResult]) -> None:
        """Cache final retrieval results under a deterministic query ID."""
        if not query.strip() or not results:
            return
        payload = json.dumps(
            [
                {
                    "page_content": result.document.page_content,
                    "metadata": result.document.metadata,
                    "score": result.score,
                    "vector_rank": result.vector_rank,
                    "bm25_rank": result.bm25_rank,
                }
                for result in results
            ],
            default=str,
        )
        query_key = hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()
        try:
            self.collection.upsert(
                ids=[query_key],
                documents=[query.strip()],
                metadatas=[{"query": query.strip(), "payload": payload}],
            )
        except Exception as error:
            raise RAGError("Unable to write semantic query cache") from error

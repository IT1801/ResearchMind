from __future__ import annotations

import json
import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import chromadb
from langchain_core.documents import Document

from ..embeddings.vector_embeddings import bge_embedding_function


class ChromaVectorStore:
    """Persistent Chroma storage for LangChain documents."""

    def __init__(
        self,
        path: str | Path = ".chroma",
        collection_name: str = "arxiv_abstracts_bge_base",
        collection: Any | None = None,
        embedding_function: Any | None = None,
    ) -> None:
        if collection is not None:
            self.collection = collection
            return

        client = chromadb.PersistentClient(path=str(path))
        self.collection = client.get_or_create_collection(
            collection_name,
            embedding_function=embedding_function or bge_embedding_function(),
        )

    def add_documents(self, documents: Iterable[Document]) -> int:
        documents = list(documents)
        if not documents:
            return 0

        ids = [self._document_id(document) for document in documents]
        texts = [document.page_content for document in documents]
        metadatas = [self._metadata(document.metadata) for document in documents]
        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
        )
        return len(documents)

    @staticmethod
    def _document_id(document: Document) -> str:
        paper_id = str(document.metadata.get("id", "unknown"))
        start_index = str(document.metadata.get("start_index", "0"))
        value = f"{paper_id}:{start_index}:{document.page_content}"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                normalized[key] = value
            else:
                normalized[key] = json.dumps(value, default=str)
        return normalized

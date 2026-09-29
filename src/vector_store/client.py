from __future__ import annotations

import json
import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import chromadb
from langchain_core.documents import Document

from ..embeddings.vector_embeddings import bge_embedding_function
from ..infra.exceptions import VectorStoreError
from ..logging.loggers import get_logger, log_exception

logger = get_logger(__name__)


class ChromaVectorStore:
    """Persistent Chroma storage for LangChain documents."""

    def __init__(
        self,
        path: str | Path = ".chroma",
        collection_name: str = "arxiv_abstracts_parent_child",
        collection: Any | None = None,
        embedding_function: Any | None = None,
    ) -> None:
        if collection is not None:
            self.collection = collection
            return

        try:
            client = chromadb.PersistentClient(path=str(path))
            self.collection = client.get_or_create_collection(
                collection_name,
                embedding_function=embedding_function or bge_embedding_function(),
            )
        except Exception as error:
            log_exception(logger, "Unable to initialize vector store", error, {
                "path": str(path),
                "collection": collection_name,
            })
            raise VectorStoreError(
                f"Unable to initialize vector store collection: {collection_name}"
            ) from error

    def add_documents(self, documents: Iterable[Document]) -> int:
        documents = list(documents)
        if not documents:
            return 0

        ids = [self._document_id(document) for document in documents]
        texts = [document.page_content for document in documents]
        metadatas = [self._metadata(document.metadata) for document in documents]
        try:
            self.collection.upsert(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
            )
        except Exception as error:
            log_exception(logger, "Unable to persist document batch", error, {
                "collection": self.collection.name,
                "batch_size": len(documents),
            })
            raise VectorStoreError("Unable to persist document batch") from error
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

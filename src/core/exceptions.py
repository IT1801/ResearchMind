from __future__ import annotations


class RAGError(Exception):
    """Base class for expected application failures."""


class ConfigurationError(RAGError):
    """Raised when required application configuration is invalid or missing."""


class DataSourceError(RAGError):
    """Raised when a remote or local data source cannot be read."""


class DocumentParseError(RAGError):
    """Raised when a source record cannot become a document."""


class ChunkingError(RAGError):
    """Raised when chunking configuration or execution fails."""


class EmbeddingError(RAGError):
    """Raised when an embedding model cannot be initialized or used."""


class VectorStoreError(RAGError):
    """Raised when vector storage cannot be initialized or updated."""


class IngestionError(RAGError):
    """Raised when the ingestion workflow fails."""

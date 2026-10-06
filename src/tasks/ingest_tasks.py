from __future__ import annotations

import os

from ..ingestion.pipeline import IngestionPipeline, IngestionStats
from .worker import worker


async def ingest_documents(limit: int, start_offset: int = 0) -> IngestionStats:
    """Async task wrapper for a bounded ingestion run."""
    os.environ["HF_START_OFFSET"] = str(start_offset)
    return IngestionPipeline().run(limit=limit)


worker.register("ingest_documents", ingest_documents)

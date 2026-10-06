from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel, Field

from ...tasks.ingest_tasks import ingest_documents

router = APIRouter(prefix="/ingest", tags=["ingestion"])


class IngestRequest(BaseModel):
    limit: int = Field(gt=0, le=500_000)
    start_offset: int = Field(default=0, ge=0)


class IngestResponse(BaseModel):
    status: str
    message: str


@router.post("", response_model=IngestResponse, status_code=202)
def start_ingestion(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
) -> IngestResponse:
    background_tasks.add_task(ingest_documents, request.limit, request.start_offset)
    return IngestResponse(
        status="accepted",
        message=f"Ingestion queued for {request.limit} documents from offset {request.start_offset}.",
    )

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from ..core.observability import configure_observability

configure_observability()

from .middleware import request_context_middleware
from .routes.health import router as health_router
from .routes.ingest import router as ingest_router
from .routes.query import router as query_router

app = FastAPI(
    title="ResearchMind Retrieval API",
    description="Hybrid vector and BM25 retrieval over parent-child contexts.",
    version="0.1.0",
)

app.include_router(health_router)
app.include_router(query_router)
app.include_router(ingest_router)
app.middleware("http")(request_context_middleware)


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    return FileResponse(Path(__file__).parents[1] / "static" / "index.html")

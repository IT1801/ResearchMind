from __future__ import annotations

from pydantic import BaseModel, Field


class DocumentSchema(BaseModel):
    page_content: str
    metadata: dict[str, object] = Field(default_factory=dict)


class ChunkSchema(DocumentSchema):
    chunk_level: str = "child"
    parent_id: str | None = None


class QuerySchema(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    top_k: int = Field(default=10, ge=1, le=10)

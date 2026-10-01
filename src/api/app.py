from __future__ import annotations

from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..infra.exceptions import RAGError
from ..retrieval.hybrid import HybridRetriever, RetrievalResult

app = FastAPI(
	title="ResearchMind Retrieval API",
	description="Hybrid vector and BM25 retrieval over parent-child contexts.",
	version="0.1.0",
)

_retriever: HybridRetriever | None = None


class QueryRequest(BaseModel):
	query: str = Field(min_length=1, max_length=2_000)
	top_k: int = Field(default=10, ge=1, le=10)


class RetrievedContext(BaseModel):
	content: str
	score: float
	vector_rank: int | None = None
	bm25_rank: int | None = None
	metadata: dict[str, object]


class QueryResponse(BaseModel):
	query: str
	results: list[RetrievedContext]


def get_retriever() -> HybridRetriever:
	global _retriever
	if _retriever is None:
		_retriever = HybridRetriever()
	return _retriever


def _serialize_result(result: RetrievalResult) -> RetrievedContext:
	return RetrievedContext(
		content=result.document.page_content,
		score=result.score,
		vector_rank=result.vector_rank,
		bm25_rank=result.bm25_rank,
		metadata=result.document.metadata,
	)


@app.get("/health")
def health() -> dict[str, str]:
	return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(
	payload: QueryRequest,
	retriever: Annotated[HybridRetriever, Depends(get_retriever)],
) -> QueryResponse:
	try:
		results = retriever.retrieve(payload.query, top_k=payload.top_k)
	except ValueError as error:
		raise HTTPException(status_code=400, detail=str(error)) from error
	except RAGError as error:
		raise HTTPException(status_code=503, detail=str(error)) from error

	return QueryResponse(
		query=payload.query,
		results=[_serialize_result(result) for result in results],
	)
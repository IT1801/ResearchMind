from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ...core.exceptions import RAGError
from ...retrieval.hybrid import HybridRetriever, RetrievalResult
from ..dependencies import answer_query, get_guardrails, get_retriever
from ...guardrails import BLOCKED_MESSAGE, ResearchMindGuardrails

router = APIRouter(tags=["query"])


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
    answer: str
    results: list[RetrievedContext]
    blocked: bool = False
    guardrail_message: str | None = None


def serialize_result(result: RetrievalResult) -> RetrievedContext:
    return RetrievedContext(
        content=result.document.page_content,
        score=result.score,
        vector_rank=result.vector_rank,
        bm25_rank=result.bm25_rank,
        metadata=result.document.metadata,
    )


@router.post("/query", response_model=QueryResponse)
def query(
    payload: QueryRequest,
    retriever: Annotated[HybridRetriever, Depends(get_retriever)],
    guardrails: Annotated[ResearchMindGuardrails, Depends(get_guardrails)],
) -> QueryResponse:
    try:
        input_decision = guardrails.validate_input(payload.query)
        if not input_decision.allowed:
            return QueryResponse(
                query=payload.query,
                answer=input_decision.message or BLOCKED_MESSAGE,
                results=[],
                blocked=True,
                guardrail_message=input_decision.category,
            )
        results = retriever.retrieve(payload.query, top_k=payload.top_k)
        answer = answer_query(payload.query, results)
        output_decision = guardrails.validate_output(answer)
        if not output_decision.allowed:
            return QueryResponse(
                query=payload.query,
                answer=output_decision.message or BLOCKED_MESSAGE,
                results=[],
                blocked=True,
                guardrail_message=output_decision.category,
            )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RAGError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail="Unable to generate an answer") from error

    return QueryResponse(
        query=payload.query,
        answer=answer,
        results=[serialize_result(result) for result in results],
    )

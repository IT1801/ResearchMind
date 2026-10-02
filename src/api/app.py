from __future__ import annotations

import os
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

from ..infra.exceptions import RAGError
from ..retrieval.hybrid import HybridRetriever, RetrievalResult

app = FastAPI(
	title="ResearchMind Retrieval API",
	description="Hybrid vector and BM25 retrieval over parent-child contexts.",
	version="0.1.0",
)

_retriever: HybridRetriever | None = None
_llm: ChatGoogleGenerativeAI | None = None


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


def get_retriever() -> HybridRetriever:
	global _retriever
	if _retriever is None:
		_retriever = HybridRetriever()
	return _retriever


def get_llm() -> ChatGoogleGenerativeAI:
	global _llm
	if _llm is None:
		_llm = ChatGoogleGenerativeAI(
			model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
			google_api_key=os.getenv("GEMINI_API_KEY"),
			temperature=0,
			max_output_tokens=800,
		)
	return _llm


def _answer_query(query_text: str, results: list[RetrievalResult]) -> str:
	context = "\n\n".join(
		f"Source {index + 1}:\n{result.document.page_content}"
		for index, result in enumerate(results)
	)
	prompt = f"""Answer the question using only the supplied research context.
If the context is insufficient, say so clearly. Keep the answer concise and do
not invent citations or facts.

Question: {query_text}

Research context:
{context}"""
	response = get_llm().invoke(prompt)
	content = response.content
	if isinstance(content, list):
		return "".join(
			part.get("text", "") if isinstance(part, dict) else str(part)
			for part in content
		).strip()
	return str(content).strip()


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
	return FileResponse("src/api/static/index.html")


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
		answer = _answer_query(payload.query, results)
	except ValueError as error:
		raise HTTPException(status_code=400, detail=str(error)) from error
	except RAGError as error:
		raise HTTPException(status_code=503, detail=str(error)) from error
	except Exception as error:
		raise HTTPException(status_code=502, detail="Unable to generate an answer") from error

	return QueryResponse(
		query=payload.query,
		answer=answer,
		results=[_serialize_result(result) for result in results],
	)
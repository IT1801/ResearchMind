from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from ..core.exceptions import RAGError

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass(frozen=True)
class RerankedDocument:
	"""A parent document scored by a cross-encoder."""

	document: Document
	score: float


class CrossEncoderReranker:
	"""Score query-parent pairs with a sentence-transformers cross-encoder."""

	def __init__(
		self,
		model_name: str = DEFAULT_RERANKER_MODEL,
		model: Any | None = None,
		max_length: int = 512,
	) -> None:
		if max_length <= 0:
			raise ValueError("max_length must be greater than zero")
		self.model_name = model_name
		self._model = model
		self.max_length = max_length

	@property
	def model(self) -> Any:
		if self._model is None:
			try:
				self._model = CrossEncoder(
					self.model_name,
					max_length=self.max_length,
				)
			except Exception as error:
				raise RAGError(
					f"Unable to initialize reranker model: {self.model_name}"
				) from error
		return self._model

	def rerank(
		self,
		query: str,
		documents: Iterable[Document],
		top_k: int = 10,
	) -> list[RerankedDocument]:
		"""Return parent documents ordered by cross-encoder relevance."""
		if not query.strip():
			return []
		if top_k <= 0:
			raise ValueError("top_k must be greater than zero")
		candidates = list(documents)
		if not candidates:
			return []
		scores = self.model.predict(
			[(query, document.page_content) for document in candidates]
		)
		results = [
			RerankedDocument(document=document, score=float(score))
			for document, score in zip(candidates, scores)
		]
		return sorted(results, key=lambda result: result.score, reverse=True)[:top_k]
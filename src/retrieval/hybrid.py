from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from ..infra.exceptions import RAGError, VectorStoreError
from ..logging.loggers import get_logger, log_exception
from ..vector_store.client import ChromaVectorStore

logger = get_logger(__name__)


def _tokenize(text: str) -> list[str]:
	return text.lower().split()


@dataclass(frozen=True)
class RetrievalResult:
	"""A document returned by the hybrid ranker."""

	document: Document
	score: float
	vector_rank: int | None = None
	bm25_rank: int | None = None


class HybridRetriever:
	"""Combine Chroma vector search and BM25 with reciprocal rank fusion."""

	def __init__(
		self,
		store: ChromaVectorStore | None = None,
		candidate_k: int = 50,
		vector_weight: float = 0.5,
		bm25_weight: float = 0.5,
		get_batch_size: int = 5000,
		tokenizer: Callable[[str], list[str]] = _tokenize,
	) -> None:
		if candidate_k <= 0:
			raise ValueError("candidate_k must be greater than zero")
		if vector_weight < 0 or bm25_weight < 0:
			raise ValueError("retrieval weights cannot be negative")
		if vector_weight == 0 and bm25_weight == 0:
			raise ValueError("at least one retrieval weight must be positive")
		if get_batch_size <= 0:
			raise ValueError("get_batch_size must be greater than zero")

		self.store = store or ChromaVectorStore()
		self.candidate_k = candidate_k
		self.vector_weight = vector_weight
		self.bm25_weight = bm25_weight
		self.get_batch_size = get_batch_size
		self.tokenizer = tokenizer
		self._documents: list[Document] = []
		self._document_by_id: dict[str, Document] = {}
		self._document_ids: list[str] = []
		self._bm25: BM25Okapi | None = None

	def _ensure_bm25(self) -> None:
		if self._bm25 is not None:
			return
		try:
			collection = self.store.collection
			total = collection.count()
			for offset in range(0, total, self.get_batch_size):
				data = collection.get(
					offset=offset,
					limit=self.get_batch_size,
					include=["documents", "metadatas"],
				)
				for document_id, text, metadata in zip(
					data["ids"],
					data.get("documents") or [],
					data.get("metadatas") or [],
				):
					document = Document(page_content=text, metadata=metadata or {})
					self._documents.append(document)
					self._document_ids.append(document_id)
					self._document_by_id[document_id] = document
			self._bm25 = BM25Okapi(
				[self.tokenizer(document.page_content) for document in self._documents]
			)
		except Exception as error:
			log_exception(logger, "Unable to build BM25 index", error)
			raise VectorStoreError("Unable to build BM25 index") from error

	def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
		"""Return the top documents ranked by vector and lexical relevance."""
		if not query.strip():
			return []
		if top_k <= 0:
			raise ValueError("top_k must be greater than zero")

		self._ensure_bm25()
		assert self._bm25 is not None
		limit = self.candidate_k
		vector_results = self.store.collection.query(
			query_texts=[query],
			n_results=limit,
			include=["documents", "metadatas"],
		)
		vector_ids = vector_results.get("ids", [[]])[0]
		vector_documents = vector_results.get("documents", [[]])[0]
		vector_metadata = vector_results.get("metadatas", [[]])[0]

		bm25_scores = self._bm25.get_scores(self.tokenizer(query))
		bm25_indexes = sorted(
			range(len(bm25_scores)),
			key=lambda index: bm25_scores[index],
			reverse=True,
		)[:limit]

		fused: dict[str, dict[str, Any]] = {}
		for rank, document_id in enumerate(vector_ids, start=1):
			document = self._document_by_id.get(document_id)
			if document is None:
				document = Document(
					page_content=vector_documents[rank - 1],
					metadata=vector_metadata[rank - 1] or {},
				)
			fused[document_id] = {
				"document": document,
				"score": self.vector_weight / (60 + rank),
				"vector_rank": rank,
				"bm25_rank": None,
			}
		for rank, index in enumerate(bm25_indexes, start=1):
			document_id = self._document_ids[index]
			entry = fused.setdefault(
				document_id,
				{
					"document": self._documents[index],
					"score": 0.0,
					"vector_rank": None,
					"bm25_rank": None,
				},
			)
			entry["score"] += self.bm25_weight / (60 + rank)
			entry["bm25_rank"] = rank

		return [
			RetrievalResult(**entry)
			for entry in sorted(
				fused.values(), key=lambda entry: entry["score"], reverse=True
			)[:top_k]
		]

def search(query: str, top_k: int = 10) -> list[RetrievalResult]:
	"""Convenience wrapper using the default persistent collection."""
	return HybridRetriever().retrieve(query, top_k=top_k)


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Run hybrid vector and BM25 retrieval.")
	parser.add_argument("query")
	parser.add_argument("--top-k", type=int, default=10)
	parser.add_argument("--candidate-k", type=int, default=50)
	parser.add_argument("--persist-directory", default=".chroma")
	parser.add_argument("--collection", default="arxiv_abstracts_bge_base")
	return parser


def main() -> None:
	args = _build_parser().parse_args()
	try:
		store = ChromaVectorStore(args.persist_directory, args.collection)
		results = HybridRetriever(store, candidate_k=args.candidate_k).retrieve(
			args.query, top_k=args.top_k
		)
		print(json.dumps([
			{"score": result.score, "metadata": result.document.metadata,
			 "page_content": result.document.page_content}
			for result in results
		], indent=2, default=str))
	except RAGError as error:
		log_exception(logger, "Hybrid retrieval failed", error)
		raise SystemExit(str(error)) from error


if __name__ == "__main__":
	main()

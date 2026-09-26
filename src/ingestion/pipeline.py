from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass, asdict
from typing import Protocol

from langchain_core.documents import Document

from .chunkers import (
	DEFAULT_CHUNK_OVERLAP,
	DEFAULT_CHUNK_SIZE,
	recursive_chunk_documents,
)
from .parsers import huggingface_loader
from ..vector_store.client import ChromaVectorStore


class DocumentStore(Protocol):
	"""Store that embeds documents before persisting them."""

	def add_documents(self, documents: list[Document]) -> int:
		...


DocumentLoader = Callable[[int | None], Iterator[Document]]


@dataclass
class IngestionStats:
	source_documents: int = 0
	chunks: int = 0
	batches: int = 0


class IngestionPipeline:
	"""Parse, chunk, embed, and persist Hugging Face documents."""

	def __init__(
		self,
		document_loader: DocumentLoader = huggingface_loader,
		store: DocumentStore | None = None,
		chunk_size: int = DEFAULT_CHUNK_SIZE,
		chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
		batch_size: int = 100,
		progress_every: int = 10,
	) -> None:
		if batch_size <= 0:
			raise ValueError("batch_size must be greater than zero")
		if progress_every <= 0:
			raise ValueError("progress_every must be greater than zero")
		self.document_loader = document_loader
		self.store = store or ChromaVectorStore()
		self.chunk_size = chunk_size
		self.chunk_overlap = chunk_overlap
		self.batch_size = batch_size
		self.progress_every = progress_every

	def parse_documents(self, limit: int | None = None) -> Iterator[Document]:
		"""Stream parsed source documents from the configured loader."""
		return self.document_loader(limit)

	def chunk_documents(self, documents: Iterator[Document]) -> Iterator[Document]:
		"""Apply the configured recursive token chunking strategy."""
		return recursive_chunk_documents(
			documents,
			chunk_size=self.chunk_size,
			chunk_overlap=self.chunk_overlap,
		)

	def run(self, limit: int | None = None) -> IngestionStats:
		stats = IngestionStats()
		batch: list[Document] = []
		for document in self.parse_documents(limit):
			stats.source_documents += 1
			for chunk in self.chunk_documents(iter([document])):
				batch.append(chunk)
				if len(batch) >= self.batch_size:
					stats.chunks += self.store.add_documents(batch)
					stats.batches += 1
					batch = []
					if stats.batches % self.progress_every == 0:
						print(
							f"ingested {stats.source_documents} documents, "
							f"{stats.chunks} chunks",
							file=sys.stderr,
							flush=True,
						)

		if batch:
			stats.chunks += self.store.add_documents(batch)
			stats.batches += 1
		return stats


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Ingest arXiv abstracts into Chroma.")
	parser.add_argument(
		"--limit",
		type=int,
		default=10,
		metavar="N",
		help="Stream at most N source documents (default: 10).",
	)
	parser.add_argument("--batch-size", type=int, default=500, metavar="N")
	parser.add_argument("--progress-every", type=int, default=10, metavar="N")
	parser.add_argument("--persist-directory", default=".chroma")
	parser.add_argument("--collection", default="arxiv_abstracts_bge_base")
	return parser


def main() -> None:
	args = _build_parser().parse_args()
	store = ChromaVectorStore(args.persist_directory, args.collection)
	stats = IngestionPipeline(
		store=store,
		batch_size=args.batch_size,
		progress_every=args.progress_every,
	).run(args.limit)
	print(json.dumps(asdict(stats), indent=2))


if __name__ == "__main__":
	main()

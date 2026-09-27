from __future__ import annotations

from collections.abc import Iterable, Iterator

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..infra.exceptions import ChunkingError
from ..logging.loggers import get_logger, log_exception


DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 60
logger = get_logger(__name__)


def recursive_chunk_documents(
	documents: Iterable[Document],
	chunk_size: int = DEFAULT_CHUNK_SIZE,
	chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Iterator[Document]:
	"""Yield recursively split documents while retaining their metadata."""
	try:
		if chunk_size <= 0:
			raise ValueError("chunk_size must be greater than zero")
		if chunk_overlap < 0 or chunk_overlap >= chunk_size:
			raise ValueError("chunk_overlap must be between zero and chunk_size")
		splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
			encoding_name="cl100k_base",
			chunk_size=chunk_size,
			chunk_overlap=chunk_overlap,
			add_start_index=True,
		)
	except Exception as error:
		log_exception(logger, "Unable to configure document chunker", error, {
			"chunk_size": chunk_size,
			"chunk_overlap": chunk_overlap,
		})
		raise ChunkingError("Unable to configure document chunker") from error
	for document in documents:
		yield from splitter.split_documents([document])

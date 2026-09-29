from __future__ import annotations

from collections.abc import Iterable, Iterator
import hashlib

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..infra.exceptions import ChunkingError
from ..logging.loggers import get_logger, log_exception


DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 60
DEFAULT_CHILD_CHUNK_SIZE = 200
DEFAULT_CHILD_CHUNK_OVERLAP = 20
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


def parent_child_chunk_documents(
	documents: Iterable[Document],
	parent_chunk_size: int = DEFAULT_CHUNK_SIZE,
	parent_chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
	child_chunk_size: int = DEFAULT_CHILD_CHUNK_SIZE,
	child_chunk_overlap: int = DEFAULT_CHILD_CHUNK_OVERLAP,
) -> Iterator[Document]:
	"""Yield searchable child chunks linked to their larger parent chunks."""
	if child_chunk_size >= parent_chunk_size:
		raise ChunkingError("child_chunk_size must be smaller than parent_chunk_size")
	for parent in recursive_chunk_documents(
		documents,
		chunk_size=parent_chunk_size,
		chunk_overlap=parent_chunk_overlap,
	):
		parent_id = hashlib.sha256(
			f"{parent.metadata.get('id', 'unknown')}:"
			f"{parent.metadata.get('start_index', '0')}:"
			f"{parent.page_content}".encode("utf-8")
		).hexdigest()
		child_metadata = {
			**parent.metadata,
			"parent_id": parent_id,
			"parent_content": parent.page_content,
			"parent_start_index": parent.metadata.get("start_index", 0),
			"chunk_level": "child",
		}
		child_document = Document(
			page_content=parent.page_content,
			metadata=child_metadata,
		)
		for child in recursive_chunk_documents(
			[child_document],
			chunk_size=child_chunk_size,
			chunk_overlap=child_chunk_overlap,
		):
			yield child

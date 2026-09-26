from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Iterator

from datasets import load_dataset
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()


@dataclass(frozen=True)
class HuggingFaceDatasetConfig:
	"""Settings for a streamed Hugging Face dataset."""

	dataset_id: str = "common-pile/arxiv_abstracts"
	split: str = "train"
	text_field: str = "abstract"

	@classmethod
	def from_environment(cls) -> "HuggingFaceDatasetConfig":
		return cls(
			dataset_id=os.getenv("HF_DATASET_ID", cls.dataset_id),
			split=os.getenv("HF_DATASET_SPLIT", cls.split),
			text_field=os.getenv("HF_DATASET_TEXT_FIELD", cls.text_field),
		)


class HuggingFaceDatasetConnector:
	"""Stream records from a public or authenticated Hugging Face dataset."""

	def __init__(self, config: HuggingFaceDatasetConfig | None = None):
		self.config = config or HuggingFaceDatasetConfig.from_environment()

	def records(self) -> Iterator[dict[str, Any]]:
		"""Yield dataset records without downloading the complete split."""
		dataset = load_dataset(
			self.config.dataset_id,
			split=self.config.split,
			streaming=True,
		)
		for record in dataset:
			yield dict(record)

	def documents(self, limit: int | None = None) -> Iterator[Document]:
		"""Yield abstract records as LangChain documents."""
		from .parsers import record_to_document

		for index, record in enumerate(self.records()):
			document = record_to_document(
				record,
				text_field=self.config.text_field,
				dataset=self.config.dataset_id,
			)
			if document is None:
				continue
			yield document
			if limit is not None and index + 1 >= limit:
				return


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Inspect a Hugging Face dataset.")
	parser.add_argument(
		"--limit",
		type=int,
		default=1,
		metavar="N",
		help="Stream at most N documents (default: 1).",
	)
	return parser


def main() -> None:
	args = _build_parser().parse_args()
	connector = HuggingFaceDatasetConnector()
	for document in connector.documents(limit=args.limit):
		print(document.model_dump_json())


if __name__ == "__main__":
	main()
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Iterator
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()


@dataclass(frozen=True)
class HuggingFaceDatasetConfig:
	"""Settings for a streamed Hugging Face dataset."""

	dataset_id: str = "common-pile/arxiv_abstracts"
	split: str = "train"
	text_field: str = "text"
	rows_api_url: str = "https://datasets-server.huggingface.co/rows"

	@classmethod
	def from_environment(cls) -> "HuggingFaceDatasetConfig":
		return cls(
			dataset_id=os.getenv("HF_DATASET_ID", cls.dataset_id),
			split=os.getenv("HF_DATASET_SPLIT", cls.split),
			text_field=os.getenv("HF_DATASET_TEXT_FIELD", cls.text_field),
			rows_api_url=os.getenv("HF_ROWS_API_URL", cls.rows_api_url),
		)


class HuggingFaceDatasetConnector:
	"""Stream records from a public or authenticated Hugging Face dataset."""

	def __init__(self, config: HuggingFaceDatasetConfig | None = None):
		self.config = config or HuggingFaceDatasetConfig.from_environment()

	def records(self) -> Iterator[dict[str, Any]]:
		"""Yield dataset records through the Hugging Face rows API."""
		offset = 0
		page_size = 100
		while True:
			query = urlencode(
				{
					"dataset": self.config.dataset_id,
					"config": "default",
					"split": self.config.split,
					"offset": offset,
					"length": page_size,
				}
			)
			request = Request(
				f"{self.config.rows_api_url}?{query}",
				headers={"Authorization": f"Bearer {os.getenv('HF_TOKEN', '')}"},
			)
			with urlopen(request, timeout=30) as response:
				payload = json.load(response)
			rows = payload.get("rows", [])
			if not rows:
				return
			for item in rows:
				yield dict(item["row"])
			offset += len(rows)
			if len(rows) < page_size:
				return

	def documents(self, limit: int | None = None) -> Iterator[Document]:
		"""Yield abstract records as LangChain documents."""
		from .parsers import record_to_document

		if limit is not None and limit <= 0:
			return
		returned = 0
		for record in self.records():
			document = record_to_document(
				record,
				text_field=self.config.text_field,
				dataset=self.config.dataset_id,
			)
			if document is None:
				continue
			yield document
			returned += 1
			if limit is not None and returned >= limit:
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
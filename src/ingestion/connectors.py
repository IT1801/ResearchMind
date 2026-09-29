from __future__ import annotations

import argparse
import certifi
import json
import os
import ssl
import time
from dataclasses import dataclass
from typing import Any, Iterator
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from langchain_core.documents import Document

from ..infra.exceptions import DataSourceError
from ..logging.loggers import get_logger, log_exception

load_dotenv()
logger = get_logger(__name__)


@dataclass(frozen=True)
class HuggingFaceDatasetConfig:
	"""Settings for a streamed Hugging Face dataset."""

	dataset_id: str = "common-pile/arxiv_abstracts"
	split: str = "train"
	text_field: str = "text"
	rows_api_url: str = "https://datasets-server.huggingface.co/rows"
	page_size: int = 100
	request_timeout: float = 30.0
	max_retries: int = 5
	retry_backoff_seconds: float = 2.0
	start_offset: int = 0
	ca_bundle: str = certifi.where()

	@classmethod
	def from_environment(cls) -> "HuggingFaceDatasetConfig":
		return cls(
			dataset_id=os.getenv("HF_DATASET_ID", cls.dataset_id),
			split=os.getenv("HF_DATASET_SPLIT", cls.split),
			text_field=os.getenv("HF_DATASET_TEXT_FIELD", cls.text_field),
			rows_api_url=os.getenv("HF_ROWS_API_URL", cls.rows_api_url),
			page_size=int(os.getenv("HF_PAGE_SIZE", str(cls.page_size))),
			request_timeout=float(
				os.getenv("HF_REQUEST_TIMEOUT", str(cls.request_timeout))
			),
			max_retries=int(os.getenv("HF_MAX_RETRIES", str(cls.max_retries))),
			retry_backoff_seconds=float(
				os.getenv(
					"HF_RETRY_BACKOFF_SECONDS",
					str(cls.retry_backoff_seconds),
				)
			),
			start_offset=int(os.getenv("HF_START_OFFSET", str(cls.start_offset))),
			ca_bundle=os.getenv("HF_CA_BUNDLE", cls.ca_bundle),
		)


class HuggingFaceDatasetConnector:
	"""Stream records from a public or authenticated Hugging Face dataset."""

	def __init__(self, config: HuggingFaceDatasetConfig | None = None):
		self.config = config or HuggingFaceDatasetConfig.from_environment()

	def records(self) -> Iterator[dict[str, Any]]:
		"""Yield dataset records through the Hugging Face rows API."""
		offset = self.config.start_offset
		ssl_context = ssl.create_default_context(cafile=self.config.ca_bundle)
		while True:
			query = urlencode(
				{
					"dataset": self.config.dataset_id,
					"config": "default",
					"split": self.config.split,
					"offset": offset,
					"length": self.config.page_size,
				}
			)
			request = Request(
				f"{self.config.rows_api_url}?{query}",
				headers={"Authorization": f"Bearer {os.getenv('HF_TOKEN', '')}"},
			)
			for attempt in range(self.config.max_retries + 1):
				try:
					with urlopen(
						request,
						timeout=self.config.request_timeout,
						context=ssl_context,
					) as response:
						payload = json.load(response)
					break
				except Exception as error:
					if attempt >= self.config.max_retries:
						log_exception(
							logger,
							"Unable to fetch Hugging Face dataset page",
							error,
							{
								"dataset": self.config.dataset_id,
								"split": self.config.split,
								"offset": offset,
								"attempts": attempt + 1,
							},
						)
						raise DataSourceError(
							f"Unable to fetch dataset page at offset {offset}"
						) from error
					delay = self.config.retry_backoff_seconds * (2**attempt)
					logger.warning(
						"Retrying Hugging Face dataset page",
						extra={"offset": offset, "attempt": attempt + 1, "delay": delay},
					)
					time.sleep(delay)
			rows = payload.get("rows", [])
			if not rows:
				return
			for item in rows:
				yield dict(item["row"])
			offset += len(rows)
			if len(rows) < self.config.page_size:
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
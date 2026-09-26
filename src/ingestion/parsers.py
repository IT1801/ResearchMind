from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from langchain_core.documents import Document


def record_to_document(
    record: Mapping[str, Any],
    text_field: str = "text",
    dataset: str | None = None,
) -> Document | None:
    """Convert one metadata record into a searchable LangChain document."""
    text = str(record.get(text_field, "")).strip()
    if not text:
        return None

    title = str(record.get("title", "")).strip()
    page_content = f"{title}\n\n{text}" if title else text
    metadata = {
        key: value
        for key, value in record.items()
        if key != text_field
    }
    if dataset:
        metadata["dataset"] = dataset
    return Document(page_content=page_content, metadata=metadata)


def huggingface_loader(
    limit: int | None = None,
    text_field: str | None = None,
    dataset_id: str | None = None,
    split: str | None = None,
) -> Iterator[Document]:
    """Stream Hugging Face records as documents without materializing the split."""
    from .connectors import HuggingFaceDatasetConfig, HuggingFaceDatasetConnector

    defaults = HuggingFaceDatasetConfig.from_environment()
    config = HuggingFaceDatasetConfig(
        dataset_id=dataset_id or defaults.dataset_id,
        split=split or defaults.split,
        text_field=text_field or defaults.text_field,
    )
    connector = HuggingFaceDatasetConnector(config)
    return connector.documents(limit=limit)

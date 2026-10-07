from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Iterable

from dotenv import load_dotenv
from langchain_core.documents import Document

from ..core.exceptions import RAGError
from ..core.observability import traced

load_dotenv()

DEFAULT_COMPRESSION_MODEL = "gemini-3.1-flash-lite"


@dataclass(frozen=True)
class CompressedContext:
    document: Document
    source_index: int
    retained: bool


class LLMContextCompressor:
    """Use one LLM call to keep only query-relevant parent evidence."""

    def __init__(
        self,
        model_name: str | None = None,
        client: Any | None = None,
        max_output_tokens: int = 2000,
    ) -> None:
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be greater than zero")
        self.model_name = model_name or os.getenv(
            "GEMINI_MODEL", DEFAULT_COMPRESSION_MODEL
        )
        self._client = client
        self.max_output_tokens = max_output_tokens

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI

                self._client = ChatGoogleGenerativeAI(
                    model=self.model_name,
                    google_api_key=os.getenv("GEMINI_API_KEY"),
                    temperature=0,
                    max_output_tokens=self.max_output_tokens,
                )
            except Exception as error:
                raise RAGError(
                    f"Unable to initialize context compression model: {self.model_name}"
                ) from error
        return self._client

    @traced("researchmind.context_compression", run_type="chain")
    def compress(
        self,
        query: str,
        documents: Iterable[Document],
    ) -> list[CompressedContext]:
        """Return only parent evidence required to answer the query."""
        candidates = list(documents)
        if not query.strip() or not candidates:
            return []

        prompt = self._build_prompt(query, candidates)
        try:
            response = self.client.invoke(prompt)
            payload = self._parse_response(response)
        except Exception as error:
            raise RAGError("Unable to compress retrieved context") from error

        compressed: list[CompressedContext] = []
        for item in payload:
            index = item.get("chunk_index")
            if not isinstance(index, int) or not 0 <= index < len(candidates):
                continue
            if not item.get("keep", False):
                continue
            text = str(item.get("context", "")).strip()
            if not text:
                continue
            metadata = dict(candidates[index].metadata)
            metadata["compression_applied"] = True
            metadata["compression_source_index"] = index
            compressed.append(
                CompressedContext(
                    document=Document(page_content=text, metadata=metadata),
                    source_index=index,
                    retained=True,
                )
            )
        return compressed

    @staticmethod
    def _build_prompt(query: str, documents: list[Document]) -> str:
        contexts = "\n\n".join(
            f"CHUNK {index}:\n{document.page_content}"
            for index, document in enumerate(documents)
        )
        return f"""You compress retrieval context for a question-answering system.

Question:
{query}

For each parent chunk, decide whether it is required to answer the question.
If required, extract only the minimum self-contained facts needed. Preserve
technical names, numbers, equations, and qualifications. Do not invent facts.
Return JSON only, as an array with this exact shape:
[{{"chunk_index": 0, "keep": true, "context": "essential evidence"}}]
Include one object for every input chunk. Use keep=false and context="" when it
is not required.

Parent chunks:
{contexts}"""

    @staticmethod
    def _parse_response(response: Any) -> list[dict[str, Any]]:
        content = response.content if hasattr(response, "content") else str(response)
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        match = re.search(r"\[.*\]", str(content), re.DOTALL)
        if match is None:
            raise ValueError("compression model did not return a JSON array")
        payload = json.loads(match.group(0))
        if not isinstance(payload, list):
            raise ValueError("compression response must be a JSON array")
        return [item for item in payload if isinstance(item, dict)]

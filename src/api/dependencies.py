from __future__ import annotations

import os
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from ..core.config import get_settings
from ..retrieval.hybrid import HybridRetriever
from ..guardrails import ResearchMindGuardrails

_retriever: HybridRetriever | None = None
_llm: ChatGoogleGenerativeAI | None = None
_guardrails: ResearchMindGuardrails | None = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def get_guardrails() -> ResearchMindGuardrails:
    global _guardrails
    if _guardrails is None:
        _guardrails = ResearchMindGuardrails()
    return _guardrails


def get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        settings = get_settings()
        _llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key or os.getenv("GEMINI_API_KEY"),
            temperature=0,
            max_output_tokens=800,
        )
    return _llm


def answer_query(query_text: str, results: list[Any]) -> str:
    context = "\n\n".join(
        f"Source {index + 1}:\n{result.document.page_content}"
        for index, result in enumerate(results)
    )
    prompt = f"""Answer the question using only the supplied research context.
If the context is insufficient, say so clearly. Keep the answer concise and do
not invent citations or facts.

Question: {query_text}

Research context:
{context}"""
    response = get_llm().invoke(prompt)
    content = response.content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        ).strip()
    return str(content).strip()

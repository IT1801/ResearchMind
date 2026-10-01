from __future__ import annotations

import argparse
import ast
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import tiktoken
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.context_compressor import LLMContextCompressor
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.semantic_cache import SemanticQueryCache
from src.vector_store.client import ChromaVectorStore


class NoOpCache:
    def get(self, query: str) -> None:
        return None

    def put(self, query: str, results: list[Any]) -> None:
        return None


def parse_json_response(response: Any) -> dict[str, Any]:
    content = response.content if hasattr(response, "content") else str(response)
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    start = str(content).find("{")
    end = str(content).rfind("}")
    if start < 0 or end < start:
        raise ValueError("LLM did not return a JSON object")
    payload = str(content)[start : end + 1]
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        value = ast.literal_eval(payload)
    if not isinstance(value, dict):
        raise ValueError("LLM response was not a JSON object")
    return value


def response_text(response: Any) -> str:
    content = response.content if hasattr(response, "content") else response
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def context_text(documents: list[Document]) -> str:
    return "\n\n".join(
        f"CONTEXT {index + 1}:\n{document.page_content}"
        for index, document in enumerate(documents)
    )


def answer_prompt(query: str, documents: list[Document]) -> str:
    return f"""Answer the question using only the supplied context. If the context is insufficient, say so. Be concise and preserve important qualifications.

Question: {query}

{context_text(documents)}"""


def judge_prompt(query: str, reference: str, answer: str, documents: list[Document]) -> str:
    return f"""Evaluate a retrieval-augmented answer. Return JSON only with integer scores from 0 to 2:
{{"accuracy": 0, "faithfulness": 0}}
Accuracy: 2 means the answer agrees with the reference answer, 1 means partially correct, 0 means incorrect.
Faithfulness: 2 means every claim is supported by the supplied context, 1 means partly supported, 0 means unsupported or contradicted.

Question: {query}
Reference answer: {reference}
Answer: {answer}

Supplied context:
{context_text(documents)}"""


def token_count(encoding: Any, documents: list[Document]) -> int:
    return len(encoding.encode(context_text(documents)))


def precision(documents: list[Document], relevant_ids: set[str]) -> float:
    if not documents:
        return 0.0
    relevant = sum(
        1
        for document in documents
        if str(document.metadata.get("id")) in relevant_ids
    )
    return relevant / len(documents)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the end-to-end RAG evaluation.")
    parser.add_argument("--dataset", default="data/eval_queries.jsonl")
    parser.add_argument("--collection", default="arxiv_abstracts_parent_child")
    parser.add_argument("--output", default="data/eval_end_to_end.json")
    parser.add_argument("--cache-collection", default="semantic_query_cache_eval")
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.dataset).read_text().splitlines()
        if line.strip()
    ]
    encoding = tiktoken.get_encoding("cl100k_base")
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0,
        max_output_tokens=800,
    )
    store = ChromaVectorStore(collection_name=args.collection)
    reranker = CrossEncoderReranker()
    compressor = LLMContextCompressor(client=llm)
    cache = SemanticQueryCache(collection_name=args.cache_collection)
    hybrid = HybridRetriever(
        store=store,
        reranker=reranker,
        compressor=compressor,
        cache=cache,
    )
    baseline = HybridRetriever(
        store=store,
        reranker=reranker,
        compressor=type("NoOpCompressor", (), {"compress": lambda _, query, docs: []})(),
        cache=NoOpCache(),
    )

    records: list[dict[str, Any]] = []
    for row in rows:
        query = row["query"]
        relevant_ids = {str(value) for value in row["relevant_document_ids"]}
        vector_data = store.collection.query(
            query_texts=[query],
            n_results=10,
            include=["documents", "metadatas"],
        )
        vector_documents = [
            Document(page_content=text, metadata=metadata or {})
            for text, metadata in zip(
                (vector_data.get("documents") or [[]])[0],
                (vector_data.get("metadatas") or [[]])[0],
            )
        ]
        hybrid_results = hybrid.retrieve(query, top_k=10)
        hybrid_documents = [result.document for result in hybrid_results]
        baseline_results = baseline.retrieve(query, top_k=10)
        baseline_documents = [result.document for result in baseline_results]

        baseline_answer = response_text(
            llm.invoke(answer_prompt(query, vector_documents))
        )
        hybrid_answer = response_text(
            llm.invoke(answer_prompt(query, hybrid_documents))
        )
        baseline_judge = parse_json_response(
            llm.invoke(judge_prompt(query, row["reference_answer"], baseline_answer, vector_documents))
        )
        hybrid_judge = parse_json_response(
            llm.invoke(judge_prompt(query, row["reference_answer"], hybrid_answer, hybrid_documents))
        )
        records.append({
            "query": query,
            "vector_context_precision": precision(vector_documents, relevant_ids),
            "hybrid_context_precision": precision(hybrid_documents, relevant_ids),
            "baseline_context_tokens": token_count(encoding, vector_documents),
            "compressed_context_tokens": token_count(encoding, hybrid_documents),
            "baseline_accuracy": int(baseline_judge.get("accuracy", 0)),
            "hybrid_accuracy": int(hybrid_judge.get("accuracy", 0)),
            "baseline_faithfulness": int(baseline_judge.get("faithfulness", 0)),
            "hybrid_faithfulness": int(hybrid_judge.get("faithfulness", 0)),
        })
        print(f"completed={len(records)}/{len(rows)} query={query}")

    cache_probe = HybridRetriever(
        store=store,
        reranker=reranker,
        compressor=compressor,
        cache=cache,
    )
    cache_times: list[float] = []
    for row in rows:
        started = time.perf_counter()
        cache_probe.retrieve(row["query"], top_k=10)
        cache_times.append(time.perf_counter() - started)

    total_baseline_tokens = sum(row["baseline_context_tokens"] for row in records)
    total_compressed_tokens = sum(row["compressed_context_tokens"] for row in records)
    summary = {
        "queries": len(records),
        "context_precision": {
            "vector_only": statistics.mean(row["vector_context_precision"] for row in records),
            "hybrid_compressed": statistics.mean(row["hybrid_context_precision"] for row in records),
        },
        "answer_accuracy_0_to_2": {
            "vector_only": statistics.mean(row["baseline_accuracy"] for row in records),
            "hybrid_compressed": statistics.mean(row["hybrid_accuracy"] for row in records),
        },
        "faithfulness_0_to_2": {
            "vector_only": statistics.mean(row["baseline_faithfulness"] for row in records),
            "hybrid_compressed": statistics.mean(row["hybrid_faithfulness"] for row in records),
        },
        "context_tokens": {
            "baseline_total": total_baseline_tokens,
            "compressed_total": total_compressed_tokens,
            "reduction_percent": (1 - total_compressed_tokens / total_baseline_tokens) * 100,
        },
        "semantic_cache": {
            "repeat_queries": len(cache_times),
            "mean_repeat_latency_seconds": statistics.mean(cache_times),
        },
        "records": records,
    }
    Path(args.output).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.vector_store.client import ChromaVectorStore


class NoOpCache:
    def get(self, query: str) -> None:
        return None

    def put(self, query: str, results: list[Any]) -> None:
        return None


class NoOpCompressor:
    def compress(self, query: str, documents: list[Any]) -> list[Any]:
        return []


def percentile(values: list[float], value: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * value
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def recall_at_10(result_ids: list[str], relevant_ids: set[str]) -> float:
    if not relevant_ids:
        return 0.0
    return len(set(result_ids) & relevant_ids) / len(relevant_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate vector and hybrid retrieval.")
    parser.add_argument("--dataset", default="evals/datasets/eval_queries.jsonl")
    parser.add_argument("--persist-directory", default=".chroma")
    parser.add_argument("--collection", default="arxiv_abstracts_parent_child")
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--output", default="evals/eval_results.json")
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.dataset).read_text().splitlines()
        if line.strip()
    ]
    store = ChromaVectorStore(args.persist_directory, args.collection)
    collection = store.collection
    reranker = CrossEncoderReranker()
    hybrid = HybridRetriever(
        store=store,
        candidate_k=args.candidate_k,
        reranker=reranker,
        compressor=NoOpCompressor(),
        cache=NoOpCache(),
    )
    results: list[dict[str, Any]] = []
    vector_latencies: list[float] = []
    hybrid_latencies: list[float] = []
    vector_recalls: list[float] = []
    hybrid_recalls: list[float] = []

    for row in rows:
        query = row["query"]
        relevant_ids = {str(value) for value in row["relevant_document_ids"]}

        started = time.perf_counter()
        vector_result = collection.query(
            query_texts=[query],
            n_results=10,
            include=["metadatas"],
        )
        vector_elapsed = time.perf_counter() - started
        vector_ids = [
            str(metadata.get("id"))
            for metadata in (vector_result.get("metadatas") or [[]])[0]
            if metadata and metadata.get("id") is not None
        ]

        started = time.perf_counter()
        hybrid_result = hybrid.retrieve(query, top_k=10)
        hybrid_elapsed = time.perf_counter() - started
        hybrid_ids = [
            str(result.document.metadata.get("id"))
            for result in hybrid_result
            if result.document.metadata.get("id") is not None
        ]

        vector_latencies.append(vector_elapsed)
        hybrid_latencies.append(hybrid_elapsed)
        vector_recall = recall_at_10(vector_ids, relevant_ids)
        hybrid_recall = recall_at_10(hybrid_ids, relevant_ids)
        vector_recalls.append(vector_recall)
        hybrid_recalls.append(hybrid_recall)
        results.append({
            "query": query,
            "relevant_document_ids": sorted(relevant_ids),
            "vector_document_ids": vector_ids,
            "hybrid_document_ids": hybrid_ids,
            "vector_recall_at_10": vector_recall,
            "hybrid_recall_at_10": hybrid_recall,
            "vector_latency_seconds": vector_elapsed,
            "hybrid_latency_seconds": hybrid_elapsed,
        })

    summary = {
        "queries": len(rows),
        "candidate_k": args.candidate_k,
        "vector_only": {
            "mean_recall_at_10": statistics.mean(vector_recalls),
            "mean_latency_seconds": statistics.mean(vector_latencies),
            "p95_latency_seconds": percentile(vector_latencies, 0.95),
        },
        "hybrid_reranked": {
            "mean_recall_at_10": statistics.mean(hybrid_recalls),
            "mean_latency_seconds": statistics.mean(hybrid_latencies),
            "p95_latency_seconds": percentile(hybrid_latencies, 0.95),
        },
        "results": results,
        "notes": [
            "Gemini compression was disabled to isolate retrieval quality.",
            "Answer accuracy, faithfulness, token reduction, and cache hit rate require separate measured runs.",
        ],
    }
    Path(args.output).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()

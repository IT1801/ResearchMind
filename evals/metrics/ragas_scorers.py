from __future__ import annotations

from collections.abc import Iterable


def token_reduction_percent(original_tokens: int, compressed_tokens: int) -> float:
    """Return prompt-token reduction as a percentage."""
    if original_tokens <= 0:
        return 0.0
    return (1 - compressed_tokens / original_tokens) * 100


def context_precision(retrieved_ids: Iterable[str], relevant_ids: set[str]) -> float:
    retrieved = list(retrieved_ids)
    if not retrieved:
        return 0.0
    return sum(document_id in relevant_ids for document_id in retrieved) / len(retrieved)


def recall_at_k(retrieved_ids: Iterable[str], relevant_ids: set[str], k: int) -> float:
    relevant = set(relevant_ids)
    if not relevant or k <= 0:
        return 0.0
    return len(set(list(retrieved_ids)[:k]) & relevant) / len(relevant)


def faithfulness_score(claims_supported: int, total_claims: int) -> float:
    """Simple deterministic faithfulness score for custom evaluation reports."""
    if total_claims <= 0:
        return 0.0
    return claims_supported / total_claims

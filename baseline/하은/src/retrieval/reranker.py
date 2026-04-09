from __future__ import annotations

from collections import defaultdict

from .base_retriever import RetrieverResult


def rerank(results: list[RetrieverResult], top_k: int = 5) -> list[RetrieverResult]:
    per_file_counts: dict[str, int] = defaultdict(int)
    reranked: list[RetrieverResult] = []
    for result in sorted(results, key=lambda item: (-item.score, item.metadata.get("file_name", ""))):
        file_name = result.metadata.get("file_name", "")
        penalty = per_file_counts[file_name] * 0.75
        adjusted = RetrieverResult(
            chunk_id=result.chunk_id,
            score=round(result.score - penalty, 4),
            text=result.text,
            metadata=result.metadata,
        )
        reranked.append(adjusted)
        per_file_counts[file_name] += 1
    reranked.sort(key=lambda item: (-item.score, item.metadata.get("file_name", "")))
    return reranked[:top_k]

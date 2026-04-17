from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from typing import Any

from rank_bm25 import BM25Okapi

from src.chunking.chunker import chunk_document, tokenize
from src.db.parsed_store import load_parsed_documents

from .base_retriever import RetrieverResult


class HybridRetriever:
    """BM25 + metadata overlap retriever for parsed document chunks."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200, strategy: str = "section_aware") -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strategy = strategy

    @staticmethod
    @lru_cache(maxsize=8)
    def _build_index(chunk_size: int, overlap: int, strategy: str) -> tuple[list[dict[str, Any]], BM25Okapi]:
        chunks: list[dict[str, Any]] = []
        tokenized_corpus: list[list[str]] = []
        for document in load_parsed_documents():
            doc_chunks = chunk_document(
                document["text"],
                metadata=document["metadata"],
                chunk_size=chunk_size,
                overlap=overlap,
                strategy=strategy,
            )
            for chunk in doc_chunks:
                chunks.append(chunk)
                tokenized_corpus.append(tokenize(chunk["text"]))
        bm25 = BM25Okapi(tokenized_corpus or [[""]])
        return chunks, bm25

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrieverResult]:
        chunks, bm25 = self._build_index(self.chunk_size, self.overlap, self.strategy)
        if not chunks:
            return []

        query_tokens = tokenize(query)
        raw_scores = bm25.get_scores(query_tokens or [""])
        results: list[RetrieverResult] = []

        for chunk, bm25_score in zip(chunks, raw_scores):
            metadata = chunk["metadata"]
            if filters:
                file_name = (filters.get("file_name") or "").strip()
                agency = (filters.get("agency") or "").strip()
                keyword = (filters.get("keyword") or "").strip().lower()
                if file_name and metadata.get("file_name") != file_name:
                    continue
                if agency and metadata.get("agency") != agency:
                    continue
                if keyword:
                    haystack = " ".join(
                        [
                            metadata.get("project_name", ""),
                            metadata.get("file_name", ""),
                            metadata.get("agency", ""),
                        ]
                    ).lower()
                    if keyword not in haystack and keyword not in chunk["text"].lower():
                        continue

            metadata_bonus = 0.0
            metadata_bonus += sum(token in metadata.get("project_name", "").lower() for token in query_tokens) * 1.5
            metadata_bonus += sum(token in metadata.get("agency", "").lower() for token in query_tokens)
            score = float(bm25_score + metadata_bonus)
            if score <= 0 and query_tokens:
                continue
            results.append(
                RetrieverResult(
                    chunk_id=chunk["chunk_id"],
                    score=round(score, 4),
                    text=chunk["text"],
                    metadata=metadata,
                )
            )

        return sorted(results, key=lambda item: (-item.score, item.metadata.get("file_name", "")))[: max(top_k * 3, top_k)]

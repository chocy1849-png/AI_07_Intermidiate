"""RAG 파이프라인 — 검색 + 응답 생성.

retrieve()          : 검색 모드(기본/MMR/하이브리드)에 따라 관련 청크 반환
generate_answer()   : LLM 기반 논스트리밍 응답
stream_answer()     : LLM 스트리밍 제너레이터
answer_query()      : retrieve + generate 한 번에 (논스트리밍)
answer_query_stream(): retrieve + stream (스트리밍용, retrieved_docs도 반환)

ChromaDB가 비어있을 때는 BM25 폴백으로 자동 전환한다.
"""
from __future__ import annotations

from typing import Any, Generator

from src.db.vector_store import count as chroma_count
from src.db.vector_store import query as vector_query
from src.db.vector_store import query_mmr
from src.embedding.embedder import get_embedder
from src.generation.chain import (
    generate_answer as chain_generate,
    stream_answer as chain_stream,
)
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import rerank

_BM25_RETRIEVER = None


def _get_bm25() -> HybridRetriever:
    global _BM25_RETRIEVER
    if _BM25_RETRIEVER is None:
        _BM25_RETRIEVER = HybridRetriever(
            chunk_size=1000, overlap=200, strategy="langchain_recursive"
        )
    return _BM25_RETRIEVER


def _bm25_retrieve(query: str, top_k: int, filters: dict | None) -> list[dict]:
    results = _get_bm25().retrieve(query=query, top_k=top_k, filters=filters)
    reranked = rerank(results, top_k=top_k)
    max_score = max((r.score for r in reranked), default=1.0) or 1.0
    return [
        {
            "chunk_id": r.chunk_id,
            "text": r.text,
            "metadata": r.metadata,
            "distance": max(0.0, 1 - r.score / max_score),
        }
        for r in reranked
    ]


def _build_chroma_filter(filters: dict | None) -> dict | None:
    if not filters:
        return None
    agency = (filters.get("agency") or "").strip()
    if agency:
        return {"agency": {"$eq": agency}}
    return None


def _merge_hybrid(
    vector_results: list[dict],
    bm25_results: list[dict],
    top_k: int,
    alpha: float = 0.5,
) -> list[dict]:
    """벡터 유사도와 BM25 점수를 정규화 후 합산한다."""
    max_bm25 = max((1 - r["distance"] for r in bm25_results), default=1.0)
    merged: dict[str, dict] = {}

    for r in vector_results:
        sim = 1 - r["distance"]
        merged[r["chunk_id"]] = {**r, "_hybrid": alpha * sim}

    for r in bm25_results:
        norm = (1 - r["distance"]) / (max_bm25 + 1e-10)
        if r["chunk_id"] in merged:
            merged[r["chunk_id"]]["_hybrid"] += (1 - alpha) * norm
        else:
            merged[r["chunk_id"]] = {**r, "_hybrid": (1 - alpha) * norm}

    sorted_docs = sorted(merged.values(), key=lambda x: x["_hybrid"], reverse=True)
    for doc in sorted_docs:
        doc.pop("_hybrid", None)
    return sorted_docs[:top_k]


def retrieve(
    query: str,
    search_mode: str = "기본 검색",
    filters: dict[str, Any] | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """검색 모드에 따라 관련 청크를 반환한다.

    search_mode:
        "기본 검색" — 벡터 유사도 검색
        "MMR"       — 다양성 보장 검색 (Maximum Marginal Relevance)
        "하이브리드" — BM25 + 벡터 앙상블

    ChromaDB가 비어있으면 BM25로 폴백한다.
    """
    chroma_filter = _build_chroma_filter(filters)
    use_chroma = chroma_count() > 0

    if not use_chroma:
        return _bm25_retrieve(query, top_k=top_k, filters=filters)

    embedder = get_embedder()
    query_emb = embedder.embed_one(query)

    if search_mode == "MMR":
        return query_mmr(
            query_emb,
            top_k=top_k,
            lambda_mult=0.5,
            fetch_k=top_k * 4,
            filters=chroma_filter,
        )

    if search_mode == "하이브리드":
        vec_results = vector_query(query_emb, top_k=top_k * 2, filters=chroma_filter)
        bm25_results = _bm25_retrieve(query, top_k=top_k * 2, filters=filters)
        return _merge_hybrid(vec_results, bm25_results, top_k)

    # 기본 검색
    return vector_query(query_emb, top_k=top_k, filters=chroma_filter)


def generate_answer(
    query: str,
    retrieved_docs: list[dict[str, Any]],
    chat_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return chain_generate(query, retrieved_docs, chat_history)


def stream_answer(
    query: str,
    retrieved_docs: list[dict[str, Any]],
    chat_history: list[dict[str, Any]] | None = None,
) -> Generator[str, None, None]:
    yield from chain_stream(query, retrieved_docs, chat_history)


def answer_query(
    query: str,
    search_mode: str = "기본 검색",
    filters: dict[str, Any] | None = None,
    top_k: int = 5,
    chat_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    retrieved_docs = retrieve(query, search_mode=search_mode, filters=filters, top_k=top_k)
    result = generate_answer(query, retrieved_docs, chat_history)
    result["retrieved_docs"] = retrieved_docs
    result["debug"]["search_mode"] = search_mode
    result["debug"]["filters"] = filters or {}
    return result


def answer_query_stream(
    query: str,
    search_mode: str = "기본 검색",
    filters: dict[str, Any] | None = None,
    top_k: int = 5,
    chat_history: list[dict[str, Any]] | None = None,
) -> tuple[Generator[str, None, None], list[dict[str, Any]]]:
    """스트리밍 응답과 retrieved_docs를 함께 반환한다.

    Returns:
        (stream_generator, retrieved_docs)
    """
    retrieved_docs = retrieve(query, search_mode=search_mode, filters=filters, top_k=top_k)
    return stream_answer(query, retrieved_docs, chat_history), retrieved_docs

"""ChromaDB 기반 벡터 저장소.

add_documents() : 청크 + 임베딩을 ChromaDB에 upsert
query()         : 벡터 유사도 검색 (기본 검색)
query_mmr()     : MMR(Maximum Marginal Relevance) 검색 — 다양성 보장
count() / reset(): 유틸리티
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import chromadb


CHROMA_PATH = Path(__file__).parent.parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "rfp_documents"

_client: chromadb.PersistentClient | None = None
_collection: chromadb.Collection | None = None


def _get_collection() -> chromadb.Collection:
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _cosine_sim(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / (denom + 1e-10))


def add_documents(
    chunks: list[dict],
    embeddings: list[list[float]] | None = None,
) -> None:
    """청크 목록을 ChromaDB에 upsert한다.

    Args:
        chunks: chunk_document() 반환 포맷 리스트.
                각 항목은 chunk_id, text, metadata 키를 가진다.
        embeddings: 각 청크에 대응하는 임베딩 벡터 리스트.
                    None이면 더미 벡터([0.0] * 1536)를 사용한다.
    """
    if not chunks:
        return

    collection = _get_collection()

    ids = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    if embeddings is None:
        embeddings = [[0.0] * 1536 for _ in chunks]

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )


def query(
    query_embedding: list[float],
    top_k: int = 5,
    filters: dict | None = None,
) -> list[dict]:
    """벡터 유사도 검색 + 메타데이터 필터.

    Returns:
        [{"chunk_id", "text", "metadata", "distance"}]
    """
    collection = _get_collection()
    if collection.count() == 0:
        return []

    kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": min(top_k, collection.count()),
        "include": ["documents", "metadatas", "distances"],
    }
    if filters:
        kwargs["where"] = filters

    result = collection.query(**kwargs)

    return [
        {"chunk_id": cid, "text": text, "metadata": meta, "distance": dist}
        for cid, text, meta, dist in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        )
    ]


def query_mmr(
    query_embedding: list[float],
    top_k: int = 5,
    lambda_mult: float = 0.5,
    fetch_k: int = 20,
    filters: dict | None = None,
) -> list[dict]:
    """MMR(Maximum Marginal Relevance) 검색.

    관련성 높은 문서 중 서로 유사한 문서를 제거해 다양성을 보장한다.

    Args:
        lambda_mult: 관련성(1.0) vs 다양성(0.0) 가중치. 기본 0.5.
        fetch_k: MMR 후보 풀 크기. top_k 보다 크게 설정.
    """
    collection = _get_collection()
    if collection.count() == 0:
        return []

    n = min(fetch_k, collection.count())
    kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": n,
        "include": ["documents", "metadatas", "distances", "embeddings"],
    }
    if filters:
        kwargs["where"] = filters

    result = collection.query(**kwargs)

    candidates = [
        {
            "chunk_id": cid,
            "text": text,
            "metadata": meta,
            "distance": dist,
            "embedding": emb,
        }
        for cid, text, meta, dist, emb in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
            result["embeddings"][0],
        )
    ]

    if not candidates:
        return []

    selected: list[int] = []
    remaining = list(range(len(candidates)))

    while remaining and len(selected) < top_k:
        if not selected:
            best = min(remaining, key=lambda i: candidates[i]["distance"])
        else:
            best = max(
                remaining,
                key=lambda i: (
                    lambda_mult * (1 - candidates[i]["distance"])
                    - (1 - lambda_mult)
                    * max(_cosine_sim(candidates[i]["embedding"], candidates[s]["embedding"]) for s in selected)
                ),
            )
        selected.append(best)
        remaining.remove(best)

    return [
        {
            "chunk_id": candidates[i]["chunk_id"],
            "text": candidates[i]["text"],
            "metadata": candidates[i]["metadata"],
            "distance": candidates[i]["distance"],
        }
        for i in selected
    ]


def get_indexed_ids() -> set[str]:
    """ChromaDB에 적재된 chunk_id 전체를 반환한다."""
    collection = _get_collection()
    if collection.count() == 0:
        return set()
    return set(collection.get(include=[])["ids"])


def count() -> int:
    return _get_collection().count()


def reset() -> None:
    """Collection을 초기화한다 (테스트용)."""
    global _collection
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    client.delete_collection(COLLECTION_NAME)
    _collection = None

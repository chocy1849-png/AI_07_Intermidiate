from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any


def BM25_모듈_가져오기():
    try:
        from rank_bm25 import BM25Okapi
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "rank-bm25 패키지가 설치되어 있지 않습니다. requirements.txt를 다시 설치하세요."
        ) from exc
    return BM25Okapi


def BM25_토큰화(text: str) -> list[str]:
    value = str(text or "").lower().strip()
    return re.findall(r"[0-9a-zA-Z가-힣]+", value)


def BM25_인덱스_구성(chunk_rows: list[dict[str, Any]]) -> dict[str, Any]:
    BM25Okapi = BM25_모듈_가져오기()

    tokenized_corpus = [BM25_토큰화(row.get("contextual_chunk_text", "")) for row in chunk_rows]
    model = BM25Okapi(tokenized_corpus)
    return {
        "tokenized_corpus": tokenized_corpus,
        "chunk_rows": chunk_rows,
        "model": model,
    }


def BM25_인덱스_저장(path: Path, index_payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(index_payload, file)


def BM25_인덱스_불러오기(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        return pickle.load(file)


def BM25_검색(index_payload: dict[str, Any], query: str, top_k: int) -> list[dict[str, Any]]:
    tokenized_query = BM25_토큰화(query)
    model = index_payload["model"]
    chunk_rows = index_payload["chunk_rows"]
    scores = model.get_scores(tokenized_query)

    ranked_pairs = sorted(
        enumerate(scores),
        key=lambda item: item[1],
        reverse=True,
    )

    results: list[dict[str, Any]] = []
    for rank, (row_index, score) in enumerate(ranked_pairs[:top_k], start=1):
        row = chunk_rows[row_index]
        results.append(
            {
                "rank": rank,
                "score": float(score),
                "chunk_id": row.get("chunk_id", ""),
                "document": row.get("contextual_chunk_text", ""),
                "metadata": {
                    "chunk_id": row.get("chunk_id", ""),
                    "document_id": row.get("document_id", ""),
                    "source_file_name": row.get("source_file_name", ""),
                    "source_path": row.get("source_path", ""),
                    "source_extension": row.get("source_extension", ""),
                    "사업명": row.get("사업명", ""),
                    "발주 기관": row.get("발주 기관", ""),
                    "공고 번호": row.get("공고 번호", ""),
                    "공개 일자": row.get("공개 일자", ""),
                    "파일형식": row.get("파일형식", ""),
                    "raw_chunk_chars": row.get("raw_chunk_chars", ""),
                    "contextual_chunk_chars": row.get("contextual_chunk_chars", ""),
                },
            }
        )
    return results

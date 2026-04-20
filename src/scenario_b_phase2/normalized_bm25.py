from __future__ import annotations

import re


NORMALIZE_RULES: list[tuple[str, str]] = [
    ("예산", "금액"),
    ("사업비", "금액"),
    ("추정금액", "금액"),
    ("사업기간", "기간"),
    ("수행기간", "기간"),
    ("계약방식", "계약 방식"),
    ("입찰방식", "입찰 방식"),
    ("평가기준", "평가 기준"),
    ("참가자격", "참가 자격"),
]


def normalize_query_for_bm25(query: str) -> str:
    normalized = str(query or "")
    for source, target in NORMALIZE_RULES:
        normalized = normalized.replace(source, target)
    return re.sub(r"\s+", " ", normalized).strip()


def build_normalized_bm25_queries(query: str) -> list[str]:
    normalized = normalize_query_for_bm25(query)
    if normalized and normalized != str(query or "").strip():
        return [normalized]
    return []

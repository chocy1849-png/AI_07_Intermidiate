from __future__ import annotations

import re
from typing import Any


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[0-9A-Za-z가-힣]+", str(text or "").lower()))


def _metadata_text(metadata: dict[str, Any]) -> str:
    fields = [
        "source_file_name",
        "사업명",
        "발주 기관",
        "agency",
        "issuer",
        "project_name",
        "section_title",
        "section_label",
        "parent_section_label",
        "item_title",
        "chunk_role",
        "chunk_role_tags",
        "period_raw",
        "deadline_text",
        "budget_text",
        "evaluation_text",
        "contract_method",
        "bid_method",
    ]
    return " ".join(str(metadata.get(field, "")) for field in fields)


def _contains(pattern: str, text: str) -> bool:
    return bool(re.search(pattern, text, re.IGNORECASE))


def compute_metadata_soft_boost(
    *,
    question: str,
    profile: dict[str, Any],
    metadata: dict[str, Any],
) -> float:
    text = _metadata_text(metadata)
    q_tokens = _tokenize(question)
    m_tokens = _tokenize(text)

    boost = 0.0
    token_overlap = len(q_tokens & m_tokens)
    boost += min(0.0045, token_overlap * 0.00055)

    if profile.get("budget") and _contains(r"(예산|금액|사업비|기초금액|추정금액|원)", text):
        boost += 0.0038
    if profile.get("schedule") and _contains(r"(일정|기간|마감|착수|완료|기한|납기|deadline)", text):
        boost += 0.0038
    if profile.get("comparison") and _contains(r"(비교|차이|공통|대비)", text):
        boost += 0.0032
    if profile.get("contract") and _contains(r"(계약|입찰|방식|협상)", text):
        boost += 0.0032

    if _contains(r"(발주|기관|주관|issuer|agency)", question) and _contains(r"(발주|기관|주관|issuer|agency)", text):
        boost += 0.0035
    if _contains(r"(평가|배점|심사|정량|정성|evaluation)", question) and _contains(
        r"(평가|배점|심사|정량|정성|evaluation)", text
    ):
        boost += 0.0035
    if _contains(r"(마감|기한|납기|완료일|deadline)", question) and _contains(
        r"(마감|기한|납기|완료일|deadline)", text
    ):
        boost += 0.0035

    return round(min(0.02, boost), 6)

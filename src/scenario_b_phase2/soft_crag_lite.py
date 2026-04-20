from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from scenario_a.common_pipeline import CandidateRow


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[0-9A-Za-z가-힣]{2,}", str(text or ""))}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _chunk_type(row: CandidateRow) -> str:
    return str(row.metadata.get("chunk_type", "") or "").strip().lower()


def _has_body(row: CandidateRow) -> bool:
    has_table = int(row.metadata.get("has_table", 0) or 0)
    if has_table == 0:
        return True
    return _chunk_type(row) in {"paired_body_chunk", "table_body_pack", "row_body_pack"}


def _has_table(row: CandidateRow) -> bool:
    has_table = int(row.metadata.get("has_table", 0) or 0)
    if has_table == 1:
        return True
    return _chunk_type(row) in {
        "table_true_ocr_v2",
        "raw_table_ocr",
        "header_value_pair",
        "cell_row_block",
        "row_summary_chunk",
    }


@dataclass(slots=True)
class SoftCragLiteConfig:
    top_n: int = 6
    score_weight: float = 0.045
    keep_k: int = 5
    scope_mode: str = "targeted"  # targeted | all
    factual_mode: str = "off"  # off | weak | on
    downrank_threshold: float = 0.36
    low_conf_threshold: float = 0.22
    duplicate_penalty_weight: float = 0.20
    weak_context_penalty: float = 0.20
    table_pair_bonus: float = 0.18
    downrank_penalty: float = 0.008
    low_conf_penalty: float = 0.015


@dataclass(slots=True)
class SoftCragDecision:
    chunk_id: str
    decision: str  # keep | down-rank | flag_low_confidence
    quality_score: float
    query_overlap: float
    metadata_overlap: float
    section_consistency: float
    duplicated_evidence_ratio: float
    weak_context: float
    table_body_pair_consistency: float


def should_apply_soft_crag(profile: dict[str, Any], config: SoftCragLiteConfig) -> bool:
    answer_type = str(profile.get("answer_type", "")).strip().lower()
    follow_up = bool(profile.get("follow_up"))
    comparison = bool(profile.get("comparison")) or answer_type == "comparison"
    table_plus_text = answer_type == "table_plus_text"

    if str(config.scope_mode).strip().lower() == "all":
        if answer_type in {"factual", "table_factual"}:
            return str(config.factual_mode).strip().lower() != "off"
        return True

    if comparison or table_plus_text or follow_up:
        return True

    if answer_type in {"factual", "table_factual"}:
        return str(config.factual_mode).strip().lower() != "off"
    return False


def _top_n_section_counts(candidates: list[CandidateRow], top_n: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in candidates[: max(1, top_n)]:
        section = str(row.metadata.get("section_label", "") or row.metadata.get("section_title", "") or "").strip().lower()
        if not section:
            continue
        counts[section] = counts.get(section, 0) + 1
    return counts


def _duplicate_ratio(candidates: list[CandidateRow], top_n: int) -> float:
    signatures: dict[str, int] = {}
    selected = candidates[: max(1, top_n)]
    for row in selected:
        source = str(row.metadata.get("source_file_name", "")).strip().lower()
        chunk_type = _chunk_type(row)
        text = re.sub(r"\s+", " ", str(row.text or "").strip().lower())[:220]
        signature = f"{source}|{chunk_type}|{text}"
        signatures[signature] = signatures.get(signature, 0) + 1
    duplicate_count = sum(count - 1 for count in signatures.values() if count > 1)
    return round(duplicate_count / max(1, len(selected)), 4)


def _query_overlap(question_tokens: set[str], row: CandidateRow) -> float:
    row_tokens = _tokenize(row.text)
    if not question_tokens:
        return 0.0
    return round(len(question_tokens & row_tokens) / max(1, len(question_tokens)), 4)


def _metadata_overlap(question_tokens: set[str], row: CandidateRow) -> float:
    fields = [
        "section_label",
        "parent_section_label",
        "section_title",
        "item_title",
        "chunk_role",
        "source_file_name",
        "project_name",
        "agency",
        "header_path",
        "value_text",
    ]
    text = " ".join(str(row.metadata.get(key, "") or "") for key in fields)
    meta_tokens = _tokenize(text)
    if not question_tokens:
        return 0.0
    return round(len(question_tokens & meta_tokens) / max(1, len(question_tokens)), 4)


def _section_consistency(section_counts: dict[str, int], row: CandidateRow, top_n: int) -> float:
    if not section_counts:
        return 0.0
    section = str(row.metadata.get("section_label", "") or row.metadata.get("section_title", "") or "").strip().lower()
    if not section:
        return 0.0
    return round(section_counts.get(section, 0) / max(1, top_n), 4)


def _table_body_pair_consistency(answer_type: str, row: CandidateRow, candidates: list[CandidateRow], top_n: int) -> float:
    if answer_type != "table_plus_text":
        return 0.5
    selected = candidates[: max(1, top_n)]
    has_table = any(_has_table(item) for item in selected)
    has_body = any(_has_body(item) for item in selected)
    pair_hit = any(
        _chunk_type(item) in {"paired_body_chunk", "table_body_pack", "row_body_pack"}
        or _safe_float(item.metadata.get("pairing_score"), 0.0) >= 0.25
        for item in selected
    )
    if not has_table or not has_body:
        return 0.0
    if pair_hit:
        return 1.0
    return 0.4


def _weak_context(answer_type: str, row: CandidateRow, pair_consistency: float, query_overlap: float, metadata_overlap: float) -> float:
    if answer_type == "table_plus_text":
        return 1.0 if pair_consistency < 0.5 else 0.0
    if answer_type == "follow_up":
        return 1.0 if (query_overlap < 0.10 and metadata_overlap < 0.10) else 0.0
    if answer_type in {"comparison", "factual", "table_factual"}:
        return 1.0 if query_overlap < 0.06 else 0.0
    return 0.0


def _quality_score(
    *,
    query_overlap: float,
    metadata_overlap: float,
    section_consistency: float,
    duplicate_ratio: float,
    weak_context: float,
    pair_consistency: float,
    config: SoftCragLiteConfig,
    factual_strength: float,
) -> float:
    base = (
        (0.36 * query_overlap)
        + (0.22 * metadata_overlap)
        + (0.17 * section_consistency)
        + (0.15 * pair_consistency)
    )
    penalty = (duplicate_ratio * config.duplicate_penalty_weight) + (weak_context * config.weak_context_penalty)
    score = (base - penalty) * factual_strength
    return round(max(0.0, min(1.0, score)), 4)


def apply_soft_crag_lite(
    *,
    question: str,
    profile: dict[str, Any],
    candidates: list[CandidateRow],
    config: SoftCragLiteConfig,
    top_k: int,
) -> tuple[list[CandidateRow], list[SoftCragDecision], dict[str, Any]]:
    if not candidates:
        return [], [], {
            "apply_scope": str(config.scope_mode),
            "factual_mode": str(config.factual_mode),
            "decision_keep_count": 0,
            "decision_downrank_count": 0,
            "decision_low_conf_count": 0,
            "low_confidence_flag": 0.0,
            "duplicate_ratio": 0.0,
        }

    answer_type = str(profile.get("answer_type", "")).strip().lower()
    question_tokens = _tokenize(question)
    top_n = max(1, min(len(candidates), int(config.top_n)))
    target_rows = candidates[:top_n]
    section_counts = _top_n_section_counts(target_rows, top_n)
    duplicate_ratio = _duplicate_ratio(target_rows, top_n)

    factual_mode = str(config.factual_mode).strip().lower()
    factual_strength = 1.0
    if answer_type in {"factual", "table_factual"} and factual_mode == "weak":
        factual_strength = 0.55

    decisions: list[SoftCragDecision] = []
    reranked: list[CandidateRow] = []
    for index, row in enumerate(candidates):
        query_overlap = _query_overlap(question_tokens, row)
        metadata_overlap = _metadata_overlap(question_tokens, row)
        section_consistency = _section_consistency(section_counts, row, top_n)
        pair_consistency = _table_body_pair_consistency(answer_type, row, target_rows, top_n)
        weak_context = _weak_context(answer_type, row, pair_consistency, query_overlap, metadata_overlap)
        quality = _quality_score(
            query_overlap=query_overlap,
            metadata_overlap=metadata_overlap,
            section_consistency=section_consistency,
            duplicate_ratio=duplicate_ratio,
            weak_context=weak_context,
            pair_consistency=pair_consistency,
            config=config,
            factual_strength=factual_strength,
        )

        decision = "keep"
        if quality < config.low_conf_threshold:
            decision = "flag_low_confidence"
        elif quality < config.downrank_threshold:
            decision = "down-rank"

        base_score = float(row.adjusted_score if row.adjusted_score is not None else row.fusion_score)
        delta = quality * float(config.score_weight) * factual_strength
        if decision == "down-rank":
            delta -= float(config.downrank_penalty)
        elif decision == "flag_low_confidence":
            delta -= float(config.low_conf_penalty)
        adjusted_score = base_score + delta

        enriched_metadata = dict(row.metadata)
        enriched_metadata.update(
            {
                "soft_crag_query_overlap": query_overlap,
                "soft_crag_metadata_overlap": metadata_overlap,
                "soft_crag_section_consistency": section_consistency,
                "soft_crag_duplicate_ratio": duplicate_ratio,
                "soft_crag_weak_context": weak_context,
                "soft_crag_table_body_pair_consistency": pair_consistency,
                "soft_crag_quality_score": quality,
                "soft_crag_decision": decision,
                "soft_crag_candidate_rank": float(index + 1),
            }
        )

        reranked.append(
            CandidateRow(
                chunk_id=row.chunk_id,
                text=row.text,
                metadata=enriched_metadata,
                fusion_score=row.fusion_score,
                adjusted_score=adjusted_score,
                crag_label="SOFT",
                crag_reason=f"{decision}|q={quality:.3f}",
            )
        )
        decisions.append(
            SoftCragDecision(
                chunk_id=row.chunk_id,
                decision=decision,
                quality_score=quality,
                query_overlap=query_overlap,
                metadata_overlap=metadata_overlap,
                section_consistency=section_consistency,
                duplicated_evidence_ratio=duplicate_ratio,
                weak_context=weak_context,
                table_body_pair_consistency=pair_consistency,
            )
        )

    reranked = sorted(reranked, key=lambda item: item.adjusted_score or item.fusion_score, reverse=True)
    keep_k = max(int(top_k), int(config.keep_k))
    selected = reranked[:keep_k]

    keep_count = sum(1 for row in decisions[:top_n] if row.decision == "keep")
    downrank_count = sum(1 for row in decisions[:top_n] if row.decision == "down-rank")
    low_conf_count = sum(1 for row in decisions[:top_n] if row.decision == "flag_low_confidence")
    summary = {
        "apply_scope": str(config.scope_mode),
        "factual_mode": factual_mode,
        "decision_keep_count": keep_count,
        "decision_downrank_count": downrank_count,
        "decision_low_conf_count": low_conf_count,
        "low_confidence_flag": 1.0 if low_conf_count > 0 else 0.0,
        "duplicate_ratio": duplicate_ratio,
    }
    return selected, decisions, summary

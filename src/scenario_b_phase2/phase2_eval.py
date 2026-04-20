from __future__ import annotations

import csv
import re
import time
from pathlib import Path
from typing import Any

from eval_utils import (
    build_auto_summary,
    build_dependency_components,
    build_manual_summary,
    pack_components_greedily,
    parse_question_rows,
    sort_result_rows,
    write_csv,
    write_json,
)
from scenario_a.judge import judge_row
from scenario_b_phase2.retrieval_metrics import build_phase2_coverage_summary


REJECTION_PATTERNS = [
    r"문맥에\s*없",
    r"문서에\s*없",
    r"문서에서\s*확인되지\s*않",
    r"제공된\s*문서.*없",
    r"포함되어\s*있지\s*않",
    r"명시(?:되어)?\s*있지\s*않",
    r"기재(?:되어)?\s*있지\s*않",
    r"확인할\s*수\s*없",
    r"정보가\s*없",
    r"자료가\s*없",
]


def _normalize_doc_name(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").lower())


def _safe_float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_question_rows_from_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    parsed: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        question_id = str(row.get("question_id", "")).strip() or f"CSVQ{index:03d}"
        depends_on = str(row.get("depends_on", "")).strip()
        parsed.append(
            {
                "question_id": question_id,
                "question_index": int(row.get("question_index") or index),
                "type_group": row.get("type_group", ""),
                "type_label": row.get("type_label", ""),
                "scenario_label": row.get("scenario_label", ""),
                "turn_index": _safe_float(row.get("turn_index")),
                "question": row.get("question", ""),
                "answer_type": row.get("answer_type", ""),
                "ground_truth_doc": row.get("ground_truth_doc", ""),
                "ground_truth_docs": row.get("ground_truth_docs", ""),
                "ground_truth_hint": row.get("ground_truth_hint", ""),
                "eval_focus": row.get("eval_focus", ""),
                "expected": row.get("expected", ""),
                "depends_on": depends_on,
                "depends_on_list": [part.strip() for part in depends_on.split(",") if part.strip() and part.strip() != "-"],
            }
        )
    return parsed


def _apply_question_filter(rows: list[dict[str, Any]], question_id_file: Path | None = None) -> list[dict[str, Any]]:
    if question_id_file is None:
        return rows
    selected = {
        line.strip()
        for line in question_id_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    return [row for row in rows if row["question_id"] in selected]


def _apply_sharding(rows: list[dict[str, Any]], shard_count: int, shard_index: int) -> list[dict[str, Any]]:
    if shard_count <= 1:
        return rows
    components = build_dependency_components(rows)
    buckets = pack_components_greedily(components, shard_count)
    if shard_index < 0 or shard_index >= len(buckets):
        return []
    return buckets[shard_index]


def read_eval_rows(
    question_set_path: Path,
    *,
    question_id_file: Path | None = None,
    shard_count: int = 1,
    shard_index: int = 0,
) -> list[dict[str, Any]]:
    if question_set_path.suffix.lower() == ".csv":
        rows = _read_question_rows_from_csv(question_set_path)
    else:
        rows = parse_question_rows(question_set_path)
    rows = _apply_question_filter(rows, question_id_file)
    rows = _apply_sharding(rows, max(1, shard_count), shard_index)
    return rows


def _compute_doc_hits(question_row: dict[str, Any], source_docs: list[str]) -> tuple[float | None, float | None, float | None]:
    gt_doc = str(question_row.get("ground_truth_doc", "")).strip()
    gt_docs = [x.strip() for x in str(question_row.get("ground_truth_docs", "")).split("|") if x.strip()]
    targets = [*([gt_doc] if gt_doc else []), *gt_docs]
    if not targets:
        return None, None, None

    source_norm = [_normalize_doc_name(x) for x in source_docs if x]
    target_norm = [_normalize_doc_name(x) for x in targets if x]
    top1_hit = 1.0 if source_norm and any(t in source_norm[0] or source_norm[0] in t for t in target_norm) else 0.0
    topk_hit = 1.0 if any(any(t in src or src in t for t in target_norm) for src in source_norm) else 0.0
    hit_count = sum(1 for target in target_norm if any(target in src or src in target for src in source_norm))
    hit_rate = hit_count / len(target_norm) if target_norm else None
    return top1_hit, topk_hit, hit_rate


def _detect_rejection(answer_text: str) -> int:
    text = str(answer_text or "")
    return int(any(re.search(pattern, text) for pattern in REJECTION_PATTERNS))


def _build_row(question_row: dict[str, Any], answered: Any, elapsed_sec: float, model_key: str, embedding_backend_key: str) -> dict[str, Any]:
    source_docs = [str(candidate.metadata.get("source_file_name", "")).strip() for candidate in answered.candidates]
    top1_hit, topk_hit, hit_rate = _compute_doc_hits(question_row, source_docs)
    profile = answered.profile or {}
    rejection_expected = int(str(question_row.get("answer_type", "")).strip().lower() == "rejection")
    rejection_detected = _detect_rejection(answered.answer_text)
    rejection_success = rejection_detected if rejection_expected else None
    return {
        "question_id": question_row.get("question_id", ""),
        "question_index": question_row.get("question_index", ""),
        "type_group": question_row.get("type_group", ""),
        "group_label": question_row.get("group_label", ""),
        "answer_type": question_row.get("answer_type", ""),
        "question": question_row.get("question", ""),
        "ground_truth_doc": question_row.get("ground_truth_doc", ""),
        "ground_truth_docs": question_row.get("ground_truth_docs", ""),
        "ground_truth_hint": question_row.get("ground_truth_hint", ""),
        "expected": question_row.get("expected", ""),
        "eval_focus": question_row.get("eval_focus", ""),
        "selected_pipeline": answered.route,
        "embedding_backend": embedding_backend_key,
        "model_key": model_key,
        "answer_text": answered.answer_text,
        "answer_chars": len(str(answered.answer_text or "")),
        "elapsed_sec": round(elapsed_sec, 2),
        "retrieval_context": answered.context_text,
        "source_docs": " | ".join(source_docs),
        "top1_doc_hit": top1_hit,
        "topk_doc_hit": topk_hit,
        "ground_truth_doc_hit_rate": hit_rate,
        "rejection_expected": rejection_expected,
        "rejection_detected": rejection_detected,
        "rejection_success": rejection_success,
        "query_variant_count": int(_safe_float(profile.get("query_variant_count")) or 0),
        "controlled_query_expansion_used": _safe_float(profile.get("controlled_query_expansion_used")),
        "soft_crag_lite_used": _safe_float(profile.get("soft_crag_lite_used")),
        "soft_crag_scope_mode": profile.get("soft_crag_scope_mode", ""),
        "soft_crag_factual_mode": profile.get("soft_crag_factual_mode", ""),
        "soft_crag_decision_keep_count": _safe_float(profile.get("soft_crag_decision_keep_count")),
        "soft_crag_decision_downrank_count": _safe_float(profile.get("soft_crag_decision_downrank_count")),
        "soft_crag_decision_low_conf_count": _safe_float(profile.get("soft_crag_decision_low_conf_count")),
        "soft_crag_low_confidence_flag": _safe_float(profile.get("soft_crag_low_confidence_flag")),
        "soft_crag_duplicate_ratio": _safe_float(profile.get("soft_crag_duplicate_ratio")),
        "fallback_triggered": _safe_float(profile.get("fallback_triggered")),
        "normalized_bm25_used": _safe_float(profile.get("normalized_bm25_used")),
        "metadata_aware_used": _safe_float(profile.get("metadata_aware_used")),
        "dual_doc_coverage": _safe_float(profile.get("dual_doc_coverage")),
        "comparison_evidence_coverage": _safe_float(profile.get("comparison_evidence_coverage")),
        "table_plus_body_coverage": _safe_float(profile.get("table_plus_body_coverage")),
        "source_diversity": _safe_float(profile.get("source_diversity")),
        "table_hit": _safe_float(profile.get("table_hit")),
        "body_hit": _safe_float(profile.get("body_hit")),
        "pair_hit": _safe_float(profile.get("pair_hit")),
        "answer_hit": _safe_float(profile.get("answer_hit")),
        "nearby_body_hit": _safe_float(profile.get("nearby_body_hit")),
        "parent_section_hit": _safe_float(profile.get("parent_section_hit")),
        "pairing_score_max": _safe_float(profile.get("pairing_score_max")),
        "structured_evidence_hit": _safe_float(profile.get("structured_evidence_hit")),
        "exact_header_match_hit": _safe_float(profile.get("exact_header_match_hit")),
        "exact_row_match_hit": _safe_float(profile.get("exact_row_match_hit")),
        "generic_row_pollution_count": _safe_float(profile.get("generic_row_pollution_count")),
        "answer_type_router_used": _safe_float(profile.get("answer_type_router_used")),
        "answer_type_router_confidence": _safe_float(profile.get("answer_type_router_confidence")),
        "answer_type_router_route": profile.get("answer_type_router_route", ""),
        "answer_type_router_signals": profile.get("answer_type_router_signals", "[]"),
        "query_variants": profile.get("query_variants", "[]"),
    }


def evaluate_phase2(
    pipeline: Any,
    adapter: Any,
    question_rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    judge_model: str | None = None,
    run_label: str = "phase2",
    extra_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict[str, Any]] = []

    for row in question_rows:
        started = time.time()
        answered = pipeline.answer(row, adapter)
        elapsed = time.time() - started
        result_rows.append(
            _build_row(
                row,
                answered,
                elapsed,
                model_key=adapter.config.model_key,
                embedding_backend_key=pipeline.embedding_backend.config.backend_key,
            )
        )

    result_rows = sort_result_rows(result_rows)
    write_csv(output_dir / "phase2_eval_results.csv", result_rows)
    write_csv(output_dir / "phase2_eval_auto_summary.csv", build_auto_summary(result_rows))
    write_csv(output_dir / "phase2_eval_coverage_summary.csv", build_phase2_coverage_summary(result_rows))

    manual_rows: list[dict[str, Any]] = []
    if judge_model:
        for row in result_rows:
            judged = judge_row(pipeline.openai_client, judge_model, row)
            merged = dict(row)
            merged.update(judged)
            manual_rows.append(merged)
        write_csv(output_dir / "phase2_eval_manual_completed.csv", manual_rows)
        write_csv(output_dir / "phase2_eval_manual_summary.csv", build_manual_summary(manual_rows))

    manifest = {
        "run_label": run_label,
        "question_count": len(result_rows),
        "model_key": adapter.config.model_key,
        "embedding_backend": pipeline.embedding_backend.config.backend_key,
        "collection_name": pipeline.embedding_backend.config.collection_name,
        "bm25_index_path": str(pipeline.resolve_bm25_index_path()),
        "judge_model": judge_model or "",
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    write_json(output_dir / "phase2_eval_manifest.json", manifest)

    return {"result_rows": result_rows, "manual_rows": manual_rows, "manifest": manifest}

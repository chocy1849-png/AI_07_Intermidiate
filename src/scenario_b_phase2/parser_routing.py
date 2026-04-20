from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _nested_get(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


@dataclass(slots=True)
class ParserRouteDecision:
    document_id: str
    source_file_name: str
    parser_route: str
    need_table_parser: bool
    need_ocr: bool
    is_high_priority: bool
    score: int
    reason: str


def decide_parser_route(document_row: dict[str, Any]) -> ParserRouteDecision:
    metadata = document_row.get("metadata", {}) or {}
    parser_info = document_row.get("parser_info", {}) or {}
    artifacts = document_row.get("artifacts", {}) or {}

    table_markers = _safe_int(metadata.get("table_markers"))
    figure_markers = _safe_int(metadata.get("figure_markers"))
    ocr_candidates = _safe_int(metadata.get("ocr_candidate_count"))
    enriched_tables = _safe_int(metadata.get("enriched_table_count"))
    text_length = _safe_int(parser_info.get("text_length"))
    table_blocks = len(artifacts.get("table_blocks", []) or [])
    visual_candidates = len(artifacts.get("visual_candidates", []) or [])
    source_extension = str(document_row.get("source_extension", "")).lower()

    score = 0
    reasons: list[str] = []

    if table_markers > 0:
        score += 2
        reasons.append(f"table_markers={table_markers}")
    if figure_markers > 0:
        score += 1
        reasons.append(f"figure_markers={figure_markers}")
    if ocr_candidates > 0 or visual_candidates > 0:
        score += 2
        reasons.append(f"ocr_candidates={max(ocr_candidates, visual_candidates)}")
    if text_length < 1500:
        score += 1
        reasons.append(f"short_text={text_length}")
    if source_extension in {".hwp", ".hwpx"}:
        score += 1
        reasons.append(f"format={source_extension}")
    if table_blocks > 0 and enriched_tables == 0:
        score += 2
        reasons.append("table_blocks_without_enrichment")
    if _safe_int(_nested_get(parser_info, "html_probe", "html_table_count")) > 0:
        score += 1
        reasons.append("html_probe_table_detected")

    need_table_parser = table_markers > 0 or table_blocks > 0
    need_ocr = (ocr_candidates > 0 or visual_candidates > 0) and score >= 3
    is_high_priority = score >= 4

    if need_ocr and need_table_parser:
        route = "ocr_plus_table"
    elif need_ocr:
        route = "ocr_first"
    elif need_table_parser:
        route = "table_first"
    else:
        route = "text_only"

    return ParserRouteDecision(
        document_id=str(document_row.get("document_id", "")).strip(),
        source_file_name=str(document_row.get("source_file_name", "")).strip(),
        parser_route=route,
        need_table_parser=need_table_parser,
        need_ocr=need_ocr,
        is_high_priority=is_high_priority,
        score=score,
        reason=", ".join(reasons) if reasons else "default_text_only",
    )


def build_parser_routing_rows(document_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in document_rows:
        decision = decide_parser_route(row)
        rows.append(
            {
                "document_id": decision.document_id,
                "source_file_name": decision.source_file_name,
                "parser_route": decision.parser_route,
                "need_table_parser": int(decision.need_table_parser),
                "need_ocr": int(decision.need_ocr),
                "is_high_priority": int(decision.is_high_priority),
                "route_score": decision.score,
                "route_reason": decision.reason,
            }
        )
    return rows

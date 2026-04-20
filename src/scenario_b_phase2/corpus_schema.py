from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


def _as_str(value: Any) -> str:
    return "" if value is None else str(value)


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value in (None, "", "None"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class EnrichedChunkRecord:
    chunk_id: str
    doc_id: str
    source_file_name: str
    page: str
    source_position: str
    section_label: str
    parent_section_label: str
    item_title: str
    nearby_body_text: str
    table_markdown: str
    figure_text: str
    extracted_from: str
    ocr_confidence: float | None
    parser_source: str
    chunk_type: str
    is_high_priority: int
    contextual_chunk_text: str

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


def build_enriched_chunk_record(chunk_row: dict[str, Any]) -> EnrichedChunkRecord:
    metadata = chunk_row.get("metadata", {}) or {}
    parser_info = chunk_row.get("parser_info", {}) or {}

    extracted_from = _as_str(
        chunk_row.get("extracted_from")
        or metadata.get("extracted_from")
        or ("mixed" if int(chunk_row.get("has_table", 0) or 0) == 1 else "body")
    )
    chunk_type = _as_str(chunk_row.get("chunk_type") or metadata.get("chunk_type") or extracted_from or "body")

    return EnrichedChunkRecord(
        chunk_id=_as_str(chunk_row.get("chunk_id")),
        doc_id=_as_str(chunk_row.get("document_id") or chunk_row.get("doc_id")),
        source_file_name=_as_str(chunk_row.get("source_file_name")),
        page=_as_str(chunk_row.get("page") or metadata.get("page") or ""),
        source_position=_as_str(chunk_row.get("source_position") or metadata.get("source_position") or ""),
        section_label=_as_str(chunk_row.get("section_label")),
        parent_section_label=_as_str(chunk_row.get("parent_section_label")),
        item_title=_as_str(chunk_row.get("item_title")),
        nearby_body_text=_as_str(chunk_row.get("nearby_body_text")),
        table_markdown=_as_str(chunk_row.get("table_markdown")),
        figure_text=_as_str(chunk_row.get("figure_text")),
        extracted_from=extracted_from,
        ocr_confidence=_as_float(chunk_row.get("ocr_confidence"), _as_float(parser_info.get("ocr_confidence"))),
        parser_source=_as_str(chunk_row.get("parser_source") or parser_info.get("parser_used") or ""),
        chunk_type=chunk_type,
        is_high_priority=int(chunk_row.get("is_high_priority", 0) or 0),
        contextual_chunk_text=_as_str(chunk_row.get("contextual_chunk_text")),
    )


def schema_field_spec() -> list[dict[str, str]]:
    return [
        {"field": "doc_id", "description": "문서 단위 식별자"},
        {"field": "page", "description": "문서 내 페이지 정보(가능한 경우)"},
        {"field": "source_position", "description": "원문 내 상대 위치 또는 섹션 포지션"},
        {"field": "section_label", "description": "현재 청크의 섹션 라벨"},
        {"field": "parent_section_label", "description": "상위 섹션 라벨"},
        {"field": "item_title", "description": "표/항목 단위 제목"},
        {"field": "nearby_body_text", "description": "표/그림 인접 본문 텍스트"},
        {"field": "table_markdown", "description": "표 복원 결과(마크다운)"},
        {"field": "figure_text", "description": "그림 OCR/설명 텍스트"},
        {"field": "extracted_from", "description": "table|figure|body|mixed"},
        {"field": "ocr_confidence", "description": "OCR 신뢰도(0~1 또는 None)"},
        {"field": "parser_source", "description": "적용된 파서 식별자"},
        {"field": "chunk_type", "description": "청크 유형(body/table/figure/mixed)"},
        {"field": "is_high_priority", "description": "구조 보존 우선 처리 여부(0/1)"},
    ]

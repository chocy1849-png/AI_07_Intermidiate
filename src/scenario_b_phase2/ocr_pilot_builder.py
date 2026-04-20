from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval_utils import read_csv
from scenario_b_phase2.parser_routing import build_parser_routing_rows


MARKER_PATTERN = re.compile(r"<(표|그림|차트)>")


def _norm_name(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").lower())


def _contains_budget(text: str) -> int:
    return int(bool(re.search(r"(예산|금액|사업비|추정금액|기초금액|원)", text)))


def _contains_schedule(text: str) -> int:
    return int(bool(re.search(r"(기간|일정|마감|착수|완료|기한|납기|개월|일\s*이내)", text)))


def _contains_contract(text: str) -> int:
    return int(bool(re.search(r"(계약|입찰|협상|방식|제한경쟁|수의)", text)))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            raw = line.strip()
            if not raw:
                continue
            rows.append(json.loads(raw))
    return rows


def read_target_doc_names(eval_set_paths: list[Path]) -> set[str]:
    targets: set[str] = set()
    for path in eval_set_paths:
        if not path.exists():
            continue
        rows = read_csv(path)
        for row in rows:
            one = str(row.get("ground_truth_doc", "")).strip()
            if one:
                targets.add(_norm_name(one))
            many = [x.strip() for x in str(row.get("ground_truth_docs", "")).split("|") if x.strip()]
            for item in many:
                targets.add(_norm_name(item))
    return targets


def _priority_sort_key(doc: dict[str, Any], route: dict[str, Any]) -> tuple[int, int, int, int]:
    metadata = doc.get("metadata", {}) or {}
    return (
        int(route.get("need_ocr", 0) or 0),
        int(route.get("is_high_priority", 0) or 0),
        int(route.get("route_score", 0) or 0),
        int(metadata.get("table_markers", 0) or 0),
    )


def select_pilot_documents(
    document_rows: list[dict[str, Any]],
    *,
    required_doc_names: set[str],
    pilot_doc_count: int = 12,
    min_doc_count: int = 10,
    max_doc_count: int = 15,
) -> list[dict[str, Any]]:
    route_rows = build_parser_routing_rows(document_rows)
    route_by_doc = {str(row.get("document_id", "")): row for row in route_rows}
    doc_by_norm_name: dict[str, list[dict[str, Any]]] = {}
    for doc in document_rows:
        key = _norm_name(str(doc.get("source_file_name", "")))
        doc_by_norm_name.setdefault(key, []).append(doc)

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    # 1) table/group-bc 평가셋 관련 문서 우선 포함
    for target_name in sorted(required_doc_names):
        for key, docs in doc_by_norm_name.items():
            if target_name and (target_name in key or key in target_name):
                for doc in docs:
                    doc_id = str(doc.get("document_id", ""))
                    if doc_id in seen:
                        continue
                    selected.append(doc)
                    seen.add(doc_id)

    # 2) OCR 우선 점수 순으로 채우기
    candidates = sorted(
        document_rows,
        key=lambda doc: _priority_sort_key(doc, route_by_doc.get(str(doc.get("document_id", "")), {})),
        reverse=True,
    )
    target_count = max(min_doc_count, min(max_doc_count, pilot_doc_count))
    for doc in candidates:
        if len(selected) >= target_count:
            break
        doc_id = str(doc.get("document_id", ""))
        if doc_id in seen:
            continue
        selected.append(doc)
        seen.add(doc_id)

    return selected[:max_doc_count]


def _scan_sections(lines: list[str]) -> tuple[list[str], list[str]]:
    parent_labels: list[str] = []
    section_labels: list[str] = []
    current_parent = ""
    current_section = ""
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("□") or re.match(r"^\d+\.", stripped) or re.match(r"^[가-힣]\.", stripped):
            current_parent = stripped
            current_section = stripped
        elif stripped.startswith("◦") or stripped.startswith("-") or stripped.startswith("※"):
            current_section = stripped
        parent_labels.append(current_parent)
        section_labels.append(current_section or current_parent)
    return parent_labels, section_labels


def _window_body(lines: list[str], center: int, radius: int = 6) -> str:
    start = max(0, center - radius)
    end = min(len(lines), center + radius + 1)
    selected: list[str] = []
    for line in lines[start:end]:
        stripped = line.strip()
        if not stripped:
            continue
        if MARKER_PATTERN.search(stripped):
            continue
        selected.append(stripped)
    return "\n".join(selected)


def _to_context_text(row: dict[str, Any]) -> str:
    prefix = [
        f"[문서] {row.get('source_file_name', '')}",
        f"[섹션] {row.get('section_label', '')}",
        f"[상위섹션] {row.get('parent_section_label', '')}",
        f"[출처유형] {row.get('extracted_from', '')}",
    ]
    evidence_parts = []
    if row.get("table_markdown"):
        evidence_parts.append(f"[TABLE]\n{row['table_markdown']}")
    if row.get("figure_text"):
        evidence_parts.append(f"[FIGURE]\n{row['figure_text']}")
    if row.get("nearby_body_text"):
        evidence_parts.append(f"[BODY]\n{row['nearby_body_text']}")
    return "\n".join(prefix + evidence_parts)


@dataclass(slots=True)
class PilotBuildResult:
    selected_documents: list[dict[str, Any]]
    chunk_rows: list[dict[str, Any]]


def build_ocr_pilot_chunks(
    selected_documents: list[dict[str, Any]],
    *,
    body_chunk_chars: int = 850,
    body_chunk_overlap: int = 120,
) -> PilotBuildResult:
    chunk_rows: list[dict[str, Any]] = []

    for doc in selected_documents:
        doc_id = str(doc.get("document_id", "")).strip()
        file_name = str(doc.get("source_file_name", "")).strip()
        source_path = str(doc.get("source_path", "")).strip()
        source_extension = str(doc.get("source_extension", "")).strip()
        metadata = doc.get("metadata", {}) or {}
        parser_info = doc.get("parser_info", {}) or {}
        text = str(doc.get("text", "") or "")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            continue
        parent_labels, section_labels = _scan_sections(lines)

        parser_source = str(parser_info.get("parser_used", "")).strip()
        visual_priority = float(metadata.get("visual_priority_max_score", 0.0) or 0.0)
        ocr_confidence = round(min(1.0, max(0.0, visual_priority / 5.0)), 4)

        # 1) table/figure + nearby body 결합 chunk
        marker_indexes = [idx for idx, line in enumerate(lines) if MARKER_PATTERN.search(line)]
        for marker_order, idx in enumerate(marker_indexes, start=1):
            marker_text = lines[idx]
            extracted_from = "figure" if "그림" in marker_text else "table"
            nearby_body = _window_body(lines, idx, radius=6)
            section_label = section_labels[idx] if idx < len(section_labels) else ""
            parent_section_label = parent_labels[idx] if idx < len(parent_labels) else ""
            item_title = section_label or parent_section_label
            row = {
                "chunk_id": f"{doc_id}__pilot_mixed_{marker_order:04d}",
                "document_id": doc_id,
                "source_file_name": file_name,
                "source_path": source_path,
                "source_extension": source_extension,
                "사업명": str(metadata.get("사업명", "")),
                "발주 기관": str(metadata.get("발주 기관", "")),
                "공고 번호": str(metadata.get("공고 번호", "")),
                "공개 일자": str(metadata.get("공개 일자", "")),
                "section_label": section_label,
                "parent_section_label": parent_section_label,
                "item_title": item_title,
                "nearby_body_text": nearby_body,
                "table_markdown": "[TABLE_MARKER]" if extracted_from == "table" else "",
                "figure_text": "[FIGURE_MARKER]" if extracted_from == "figure" else "",
                "extracted_from": "mixed",
                "chunk_type": "table_plus_body",
                "parser_source": parser_source,
                "ocr_confidence": ocr_confidence,
                "is_high_priority": 1,
                "has_table": 1 if extracted_from == "table" else 0,
            }
            row["contextual_chunk_text"] = _to_context_text(row)
            row["has_budget_signal"] = _contains_budget(row["contextual_chunk_text"])
            row["has_schedule_signal"] = _contains_schedule(row["contextual_chunk_text"])
            row["has_contract_signal"] = _contains_contract(row["contextual_chunk_text"])
            chunk_rows.append(row)

        # 2) body chunk (baseline 호환 유지)
        body_lines = [line for line in lines if not MARKER_PATTERN.search(line)]
        body_text = "\n".join(body_lines)
        if not body_text.strip():
            continue
        start = 0
        body_order = 1
        while start < len(body_text):
            end = min(len(body_text), start + body_chunk_chars)
            snippet = body_text[start:end].strip()
            if not snippet:
                break
            ratio = start / max(1, len(body_text))
            idx = min(len(section_labels) - 1, int(ratio * (len(section_labels) - 1)))
            section_label = section_labels[idx] if section_labels else ""
            parent_section_label = parent_labels[idx] if parent_labels else ""
            row = {
                "chunk_id": f"{doc_id}__pilot_body_{body_order:04d}",
                "document_id": doc_id,
                "source_file_name": file_name,
                "source_path": source_path,
                "source_extension": source_extension,
                "사업명": str(metadata.get("사업명", "")),
                "발주 기관": str(metadata.get("발주 기관", "")),
                "공고 번호": str(metadata.get("공고 번호", "")),
                "공개 일자": str(metadata.get("공개 일자", "")),
                "section_label": section_label,
                "parent_section_label": parent_section_label,
                "item_title": section_label or parent_section_label,
                "nearby_body_text": snippet,
                "table_markdown": "",
                "figure_text": "",
                "extracted_from": "body",
                "chunk_type": "body",
                "parser_source": parser_source,
                "ocr_confidence": ocr_confidence,
                "is_high_priority": 1 if int(metadata.get("table_markers", 0) or 0) > 0 else 0,
                "has_table": 0,
            }
            row["contextual_chunk_text"] = _to_context_text(row)
            row["has_budget_signal"] = _contains_budget(row["contextual_chunk_text"])
            row["has_schedule_signal"] = _contains_schedule(row["contextual_chunk_text"])
            row["has_contract_signal"] = _contains_contract(row["contextual_chunk_text"])
            chunk_rows.append(row)
            body_order += 1
            if end >= len(body_text):
                break
            start = max(0, end - body_chunk_overlap)

    return PilotBuildResult(selected_documents=selected_documents, chunk_rows=chunk_rows)

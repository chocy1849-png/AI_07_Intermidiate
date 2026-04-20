from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rag_bm25 import BM25_인덱스_구성, BM25_인덱스_저장
from scenario_a.common_pipeline import PipelinePaths, ScenarioACommonPipeline


try:
    from ocr.extract_hwp_artifacts import extract_hwp_artifacts
    from ocr.run_hwp_ocr_pipeline import run_image_ocr
except Exception as exc:  # noqa: BLE001
    raise RuntimeError("ocr 모듈 import 실패: ocr.extract_hwp_artifacts / ocr.run_hwp_ocr_pipeline") from exc


@dataclass(slots=True)
class StageSpec:
    stage_key: str
    embedding_backend_key: str
    collection_name: str
    bm25_index_name: str
    include_image_ocr: bool
    include_discarded_hints: bool


STAGE_SPECS: dict[str, StageSpec] = {
    "A": StageSpec(
        stage_key="A",
        embedding_backend_key="openai_text_embedding_3_small_ocr_aug_a",
        collection_name="rfp_contextual_chunks_v2_ocr_aug_a",
        bm25_index_name="bm25_index_phase2_ocr_aug_a.pkl",
        include_image_ocr=False,
        include_discarded_hints=False,
    ),
    "B": StageSpec(
        stage_key="B",
        embedding_backend_key="openai_text_embedding_3_small_ocr_aug_b",
        collection_name="rfp_contextual_chunks_v2_ocr_aug_b",
        bm25_index_name="bm25_index_phase2_ocr_aug_b.pkl",
        include_image_ocr=True,
        include_discarded_hints=False,
    ),
    "C": StageSpec(
        stage_key="C",
        embedding_backend_key="openai_text_embedding_3_small_ocr_aug_c",
        collection_name="rfp_contextual_chunks_v2_ocr_aug_c",
        bm25_index_name="bm25_index_phase2_ocr_aug_c.pkl",
        include_image_ocr=True,
        include_discarded_hints=True,
    ),
}

DEFAULT_DISCARDED_HINT_FILE_PATTERNS = [
    "벤처기업협회_2024년 벤처확인종합관리시스템 기능 고도화 용역사업",
]
DEFAULT_DISCARDED_HINT_TABLE_INDEXES = {18, 19, 21, 40, 42, 204}


def _require_openai() -> Any:
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError("openai 패키지가 필요합니다.") from exc
    return OpenAI


def _require_chromadb() -> Any:
    try:
        import chromadb
    except ModuleNotFoundError as exc:
        raise RuntimeError("chromadb 패키지가 필요합니다.") from exc
    return chromadb


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            raw = line.strip()
            if not raw:
                continue
            rows.append(json.loads(raw))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_slug(value: str, max_len: int = 48) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", str(value or "")).strip("_").lower()
    if not slug:
        slug = "x"
    return slug[:max_len]


def _doc_hash(document_id: str) -> str:
    return hashlib.sha1(document_id.encode("utf-8")).hexdigest()[:10]


def _contains_budget(text: str) -> int:
    return int(bool(re.search(r"(예산|금액|사업비|추정금액|기초금액|원)", str(text or ""))))


def _contains_schedule(text: str) -> int:
    return int(bool(re.search(r"(기간|일정|마감|착수|완료|기한|납기|개월|일\s*이내)", str(text or ""))))


def _contains_contract(text: str) -> int:
    return int(bool(re.search(r"(계약|입찰|협상|방식|제한경쟁|수의)", str(text or ""))))


def _first_line(text: str, fallback: str = "정보 없음", limit: int = 160) -> str:
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:limit]
    return fallback


def _split_text(text: str, max_chars: int = 1400, overlap: int = 120) -> list[str]:
    value = str(text or "").strip()
    if not value:
        return []
    if len(value) <= max_chars:
        return [value]
    chunks: list[str] = []
    start = 0
    while start < len(value):
        end = min(len(value), start + max_chars)
        snippet = value[start:end].strip()
        if snippet:
            chunks.append(snippet)
        if end >= len(value):
            break
        start = max(0, end - overlap)
    return chunks


def _table_context_text(row: dict[str, Any]) -> str:
    prefix = [
        "[문서 구조 요약]",
        f"- 사업명: {row.get('사업명', '정보 없음')}",
        f"- 발주기관: {row.get('발주 기관', '정보 없음')}",
        f"- 파일명: {row.get('source_file_name', '정보 없음')}",
        f"- 섹션제목: {row.get('section_label', '정보 없음')}",
        f"- 청크유형: {row.get('chunk_type', '정보 없음')}",
        f"- linked_parent_text: {row.get('linked_parent_text', '')}",
        f"- 표포함: {'예' if int(row.get('has_table', 0) or 0) == 1 else '아니오'}",
        f"- OCR신뢰도: {row.get('ocr_confidence', '')}",
        "[/문서 구조 요약]",
    ]
    body = [
        "[본문 청크]",
        str(row.get("nearby_body_text", "")).strip(),
        "[/본문 청크]",
    ]
    return "\n".join(prefix + [""] + body).strip()


def _image_context_text(row: dict[str, Any]) -> str:
    prefix = [
        "[문서 구조 요약]",
        f"- 사업명: {row.get('사업명', '정보 없음')}",
        f"- 발주기관: {row.get('발주 기관', '정보 없음')}",
        f"- 파일명: {row.get('source_file_name', '정보 없음')}",
        f"- 청크유형: {row.get('chunk_type', '정보 없음')}",
        f"- 이미지명: {row.get('item_title', '')}",
        f"- OCR신뢰도: {row.get('ocr_confidence', '')}",
        "[/문서 구조 요약]",
    ]
    body = [
        "[본문 청크]",
        str(row.get("figure_text", "")).strip(),
        "[/본문 청크]",
    ]
    return "\n".join(prefix + [""] + body).strip()


def _simple_metadata(row: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in row.items():
        if key == "contextual_chunk_text":
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            output[key] = value if value is not None else ""
        else:
            output[key] = json.dumps(value, ensure_ascii=False)
    return output


def _parse_discarded_indexes(shortlist_path: Path) -> set[int]:
    if not shortlist_path.exists():
        return set(DEFAULT_DISCARDED_HINT_TABLE_INDEXES)
    text = shortlist_path.read_text(encoding="utf-8", errors="replace")
    # Prefer explicit "구조표로 다시 볼 우선 후보" section to avoid
    # accidentally capturing unrelated numeric tokens in the markdown.
    shortlist_block = re.search(
        r"구조표로 다시 볼 우선 후보:\s*(.+?)(?:\n\s*-\s*설명문/제목 블록|\n## |\Z)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if shortlist_block:
        indexes = {int(value) for value in re.findall(r"`(\d+)`", shortlist_block.group(1))}
        if indexes:
            return indexes

    indexes = {int(value) for value in re.findall(r"table_index\s*`?\s*(\d+)", text)}
    if indexes:
        return indexes
    return set(DEFAULT_DISCARDED_HINT_TABLE_INDEXES)


def _is_hint_target_file(file_name: str) -> bool:
    value = str(file_name or "")
    return any(pattern in value for pattern in DEFAULT_DISCARDED_HINT_FILE_PATTERNS)


def _iter_hwp_documents(processed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in processed_rows:
        source_extension = str(row.get("source_extension", "")).strip().lower()
        if source_extension != ".hwp":
            continue
        source_path = Path(str(row.get("source_path", "")).strip())
        if not source_path.exists():
            continue
        selected.append(row)
    return selected


def _table_row_from_artifact(
    *,
    stage_key: str,
    doc_row: dict[str, Any],
    table: dict[str, Any],
    row_kind: str,
    is_high_priority: int,
    chunk_text: str,
    part_index: int,
    part_count: int,
) -> dict[str, Any]:
    metadata = doc_row.get("metadata", {}) or {}
    parser_info = doc_row.get("parser_info", {}) or {}
    document_id = str(doc_row.get("document_id", "")).strip()
    file_name = str(doc_row.get("source_file_name", "")).strip()
    doc_code = _doc_hash(document_id)
    table_index = int(table.get("table_index", 0) or 0)
    part_suffix = f"_p{part_index:02d}" if part_count > 1 else ""
    chunk_id = f"ocraug_{stage_key.lower()}_{doc_code}_{row_kind}_{table_index:04d}{part_suffix}"

    linked_parent_text = str(table.get("linked_parent_text", "") or "").strip()
    raw_text = str(table.get("text", "") or "").strip()
    section_label = linked_parent_text or _first_line(raw_text)
    parent_section_label = linked_parent_text or section_label
    item_title = _first_line(raw_text)
    nearby_body_text = str(chunk_text or "").strip()

    row = {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "source_file_name": file_name,
        "source_path": str(doc_row.get("source_path", "")),
        "source_extension": str(doc_row.get("source_extension", "")),
        "사업명": str(metadata.get("사업명", "")),
        "발주 기관": str(metadata.get("발주 기관", "")),
        "공고 번호": str(metadata.get("공고 번호", "")),
        "공개 일자": str(metadata.get("공개 일자", "")),
        "파일형식": str(metadata.get("파일형식", "")),
        "section_title": section_label,
        "section_path": str(table.get("section", "")),
        "section_label": section_label,
        "parent_section_label": parent_section_label,
        "item_title": item_title,
        "source_position": str(table.get("record_start_index", "")),
        "linked_parent_text": linked_parent_text,
        "linked_parent_table_index": str(table.get("linked_parent_table_index", "") or ""),
        "table_markdown": raw_text,
        "figure_text": "",
        "nearby_body_text": nearby_body_text,
        "extracted_from": "table",
        "chunk_type": row_kind,
        "chunk_role": "표/구조",
        "chunk_role_tags": row_kind,
        "parser_source": "extract_hwp_artifacts",
        "ocr_confidence": 0.0,
        "ocr_status": "deferred_table_ocr",
        "has_table": 1,
        "is_high_priority": int(is_high_priority),
        "raw_chunk_chars": len(nearby_body_text),
        "table_part_index": part_index,
        "table_part_count": part_count,
    }
    row["contextual_chunk_text"] = _table_context_text(row)
    row["contextual_chunk_chars"] = len(row["contextual_chunk_text"])
    row["has_budget_signal"] = _contains_budget(row["contextual_chunk_text"])
    row["has_schedule_signal"] = _contains_schedule(row["contextual_chunk_text"])
    row["has_contract_signal"] = _contains_contract(row["contextual_chunk_text"])
    row["budget_text"] = str(metadata.get("사업 금액", "") or "")
    row["period_raw"] = ""
    row["contract_method"] = ""
    row["bid_method"] = ""
    row["purpose_summary"] = ""
    row["parser_used"] = str(parser_info.get("parser_used", ""))
    row["source_marker"] = "ocr_augment"
    return row


def _image_confidence(priority: str) -> float:
    if priority == "high":
        return 0.85
    if priority == "medium":
        return 0.65
    return 0.45


def _image_row_from_ocr(
    *,
    stage_key: str,
    doc_row: dict[str, Any],
    image: dict[str, Any],
    ocr_text: str,
) -> dict[str, Any]:
    metadata = doc_row.get("metadata", {}) or {}
    document_id = str(doc_row.get("document_id", "")).strip()
    doc_code = _doc_hash(document_id)
    image_name = str(image.get("name", "")).strip() or "image"
    image_slug = _safe_slug(image_name, max_len=24)
    chunk_id = f"ocraug_{stage_key.lower()}_{doc_code}_img_{image_slug}"
    priority = str(image.get("ocr_candidate_priority", "")).strip().lower()
    row = {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "source_file_name": str(doc_row.get("source_file_name", "")),
        "source_path": str(doc_row.get("source_path", "")),
        "source_extension": str(doc_row.get("source_extension", "")),
        "사업명": str(metadata.get("사업명", "")),
        "발주 기관": str(metadata.get("발주 기관", "")),
        "공고 번호": str(metadata.get("공고 번호", "")),
        "공개 일자": str(metadata.get("공개 일자", "")),
        "파일형식": str(metadata.get("파일형식", "")),
        "section_title": "이미지 OCR",
        "section_path": "",
        "section_label": "이미지 OCR",
        "parent_section_label": "",
        "item_title": image_name,
        "source_position": "",
        "linked_parent_text": "",
        "linked_parent_table_index": "",
        "table_markdown": "",
        "figure_text": ocr_text.strip(),
        "nearby_body_text": "",
        "extracted_from": "figure",
        "chunk_type": "image_ocr",
        "chunk_role": "이미지/OCR",
        "chunk_role_tags": "image_ocr",
        "parser_source": "run_hwp_ocr_pipeline",
        "ocr_confidence": _image_confidence(priority),
        "ocr_status": "ok",
        "has_table": 0,
        "is_high_priority": 1 if priority == "high" else 0,
        "raw_chunk_chars": len(ocr_text.strip()),
        "saved_path": str(image.get("saved_path", "")),
        "source_marker": "ocr_augment",
    }
    row["contextual_chunk_text"] = _image_context_text(row)
    row["contextual_chunk_chars"] = len(row["contextual_chunk_text"])
    row["has_budget_signal"] = _contains_budget(row["contextual_chunk_text"])
    row["has_schedule_signal"] = _contains_schedule(row["contextual_chunk_text"])
    row["has_contract_signal"] = _contains_contract(row["contextual_chunk_text"])
    row["budget_text"] = str(metadata.get("사업 금액", "") or "")
    row["period_raw"] = ""
    row["contract_method"] = ""
    row["bid_method"] = ""
    row["purpose_summary"] = ""
    return row


def _select_structural_tables(payload: dict[str, Any]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_indexes: set[int] = set()
    for table in payload.get("tables", []):
        storage_bucket = str(table.get("storage_bucket", "")).strip()
        final_class = str(table.get("final_classification", "")).strip()
        table_index = int(table.get("table_index", 0) or 0)
        if table_index in seen_indexes:
            continue

        should_add = False
        if storage_bucket == "section_header_block":
            should_add = True
        elif storage_bucket == "structural_table":
            should_add = final_class == "final_review_table" or int(table.get("ocr_candidate_score", 0) or 0) >= 4

        if should_add:
            selected.append(table)
            seen_indexes.add(table_index)
    return selected


def _extract_image_rows(
    *,
    stage_key: str,
    doc_row: dict[str, Any],
    payload: dict[str, Any],
    max_per_doc: int,
) -> list[dict[str, Any]]:
    image_candidates = sorted(
        payload.get("image_ocr_candidates", []),
        key=lambda item: int(item.get("ocr_candidate_score", 0) or 0),
        reverse=True,
    )[: max(0, max_per_doc)]
    rows: list[dict[str, Any]] = []
    for image in image_candidates:
        saved_path = str(image.get("saved_path", "")).strip()
        if not saved_path:
            continue
        path = Path(saved_path)
        if not path.exists():
            continue
        ocr_result = run_image_ocr(path)
        if str(ocr_result.get("status", "")).strip() != "ok":
            continue
        ocr_text = str(ocr_result.get("ocr_text", "")).strip()
        if not ocr_text:
            continue
        rows.append(_image_row_from_ocr(stage_key=stage_key, doc_row=doc_row, image=image, ocr_text=ocr_text))
    return rows


def _load_baseline_chunks(project_root: Path) -> list[dict[str, Any]]:
    baseline_path = project_root / "rag_outputs" / "b02_prefix_v2_chunks.jsonl"
    if not baseline_path.exists():
        raise FileNotFoundError(f"baseline chunk 파일이 없습니다: {baseline_path}")
    return _read_jsonl(baseline_path)


def _load_processed_documents(project_root: Path) -> list[dict[str, Any]]:
    path = project_root / "processed_data" / "processed_documents.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"processed_documents.jsonl 파일이 없습니다: {path}")
    return _read_jsonl(path)


def _build_stage_rows(
    *,
    stage: StageSpec,
    hwp_rows: list[dict[str, Any]],
    ocr_work_root: Path,
    selected_discarded_indexes: set[int],
    max_image_candidates_per_doc: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stage_rows: list[dict[str, Any]] = []
    counts = {
        "documents": 0,
        "structural_rows": 0,
        "section_header_rows": 0,
        "discarded_hint_rows": 0,
        "image_ocr_rows": 0,
    }

    for doc_row in hwp_rows:
        source_path = Path(str(doc_row.get("source_path", "")).strip())
        document_id = str(doc_row.get("document_id", "")).strip()
        doc_dir = ocr_work_root / _doc_hash(document_id)
        payload = extract_hwp_artifacts(source_path, save_images=stage.include_image_ocr, output_dir=doc_dir)
        counts["documents"] += 1

        for table in _select_structural_tables(payload):
            storage_bucket = str(table.get("storage_bucket", "")).strip()
            if storage_bucket == "section_header_block":
                row_kind = "section_header_block"
            else:
                row_kind = "structural_table"
            table_chunks = _split_text(str(table.get("text", "") or ""))
            if not table_chunks:
                continue
            for part_index, chunk_text in enumerate(table_chunks, start=1):
                row = _table_row_from_artifact(
                    stage_key=stage.stage_key,
                    doc_row=doc_row,
                    table=table,
                    row_kind=row_kind,
                    is_high_priority=1,
                    chunk_text=chunk_text,
                    part_index=part_index,
                    part_count=len(table_chunks),
                )
                stage_rows.append(row)
                if storage_bucket == "section_header_block":
                    counts["section_header_rows"] += 1
                else:
                    counts["structural_rows"] += 1

        if stage.include_discarded_hints and _is_hint_target_file(str(doc_row.get("source_file_name", ""))):
            for table in payload.get("discarded_tables", []):
                table_index = int(table.get("table_index", 0) or 0)
                if table_index not in selected_discarded_indexes:
                    continue
                table_chunks = _split_text(str(table.get("text", "") or ""))
                if not table_chunks:
                    continue
                for part_index, chunk_text in enumerate(table_chunks, start=1):
                    row = _table_row_from_artifact(
                        stage_key=stage.stage_key,
                        doc_row=doc_row,
                        table=table,
                        row_kind="selected_discarded_hint",
                        is_high_priority=1,
                        chunk_text=chunk_text,
                        part_index=part_index,
                        part_count=len(table_chunks),
                    )
                    stage_rows.append(row)
                    counts["discarded_hint_rows"] += 1

        if stage.include_image_ocr:
            image_rows = _extract_image_rows(
                stage_key=stage.stage_key,
                doc_row=doc_row,
                payload=payload,
                max_per_doc=max_image_candidates_per_doc,
            )
            stage_rows.extend(image_rows)
            counts["image_ocr_rows"] += len(image_rows)

    deduped: dict[str, dict[str, Any]] = {}
    for row in stage_rows:
        deduped[row["chunk_id"]] = row
    stage_rows = list(deduped.values())
    counts["stage_row_count"] = len(stage_rows)
    return stage_rows, counts


def _find_collection_dir(project_root: Path, collection_name: str) -> Path:
    chromadb = _require_chromadb()
    candidates = [
        project_root / "rag_outputs" / "chroma_db",
        project_root.parent / "rfp_rag_chroma_db",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            client = chromadb.PersistentClient(path=str(path))
            client.get_collection(collection_name)
            return path
        except Exception:  # noqa: BLE001
            continue
    raise FileNotFoundError(
        f"컬렉션 {collection_name} 를 찾을 수 없습니다. 검색 경로: {', '.join(str(path) for path in candidates)}"
    )


def _copy_baseline_collection(
    *,
    baseline_client: Any,
    baseline_collection_name: str,
    target_client: Any,
    target_collection_name: str,
    batch_size: int,
) -> int:
    baseline_collection = baseline_client.get_collection(baseline_collection_name)
    target_collection = target_client.get_or_create_collection(name=target_collection_name)
    total = int(baseline_collection.count() or 0)
    copied = 0
    for offset in range(0, total, max(1, batch_size)):
        batch = baseline_collection.get(
            include=["documents", "metadatas", "embeddings"],
            limit=max(1, batch_size),
            offset=offset,
        )
        ids = batch.get("ids")
        if ids is None:
            ids = []
        if not ids:
            continue
        documents = batch.get("documents")
        if documents is None:
            documents = []
        metadatas = batch.get("metadatas")
        if metadatas is None:
            metadatas = []
        embeddings = batch.get("embeddings")
        if embeddings is None:
            embeddings = []
        target_collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        copied += len(ids)
    return copied


def _embed_and_add_rows(
    *,
    target_collection: Any,
    rows: list[dict[str, Any]],
    openai_client: Any,
    embedding_model: str,
    batch_size: int,
) -> int:
    added = 0
    for start in range(0, len(rows), max(1, batch_size)):
        batch = rows[start : start + max(1, batch_size)]
        embed_texts = [str(row.get("contextual_chunk_text", ""))[:6000] for row in batch]
        response = openai_client.embeddings.create(model=embedding_model, input=embed_texts)
        vectors = [item.embedding for item in response.data]
        target_collection.add(
            ids=[str(row.get("chunk_id", "")) for row in batch],
            documents=embed_texts,
            metadatas=[_simple_metadata(row) for row in batch],
            embeddings=vectors,
        )
        added += len(batch)
    return added


def _build_stage_chroma(
    *,
    project_root: Path,
    stage: StageSpec,
    augment_rows: list[dict[str, Any]],
    chroma_output_dir: Path,
    baseline_collection_name: str,
    embedding_model: str,
    embedding_batch_size: int,
    copy_batch_size: int,
    reset_collection: bool,
) -> dict[str, Any]:
    chromadb = _require_chromadb()
    OpenAI = _require_openai()

    load_dotenv(project_root / ".env", override=False)
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 가 필요합니다.")
    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip() or "https://api.openai.com/v1"
    openai_client = OpenAI(api_key=api_key, base_url=base_url)

    baseline_dir = _find_collection_dir(project_root, baseline_collection_name)
    baseline_client = chromadb.PersistentClient(path=str(baseline_dir))

    chroma_output_dir.mkdir(parents=True, exist_ok=True)
    target_client = chromadb.PersistentClient(path=str(chroma_output_dir))
    if reset_collection:
        try:
            target_client.delete_collection(stage.collection_name)
        except Exception:  # noqa: BLE001
            pass
    copied = _copy_baseline_collection(
        baseline_client=baseline_client,
        baseline_collection_name=baseline_collection_name,
        target_client=target_client,
        target_collection_name=stage.collection_name,
        batch_size=copy_batch_size,
    )
    target_collection = target_client.get_collection(stage.collection_name)
    added = _embed_and_add_rows(
        target_collection=target_collection,
        rows=augment_rows,
        openai_client=openai_client,
        embedding_model=embedding_model,
        batch_size=embedding_batch_size,
    )
    return {
        "baseline_chroma_dir": str(baseline_dir),
        "target_chroma_dir": str(chroma_output_dir),
        "collection_name": stage.collection_name,
        "baseline_rows_copied": copied,
        "augment_rows_added": added,
        "collection_count": int(target_collection.count() or 0),
    }


def build_stage_assets(
    *,
    project_root: Path,
    stage: StageSpec,
    baseline_rows: list[dict[str, Any]],
    hwp_rows: list[dict[str, Any]],
    output_root: Path,
    selected_discarded_indexes: set[int],
    max_image_candidates_per_doc: int,
    baseline_collection_name: str,
    chroma_output_dir: Path,
    embedding_model: str,
    embedding_batch_size: int,
    copy_batch_size: int,
    reset_collection: bool,
) -> dict[str, Any]:
    stage_dir = output_root / f"stage_{stage.stage_key.lower()}"
    ocr_work_root = stage_dir / "ocr_work"
    stage_rows, counts = _build_stage_rows(
        stage=stage,
        hwp_rows=hwp_rows,
        ocr_work_root=ocr_work_root,
        selected_discarded_indexes=selected_discarded_indexes,
        max_image_candidates_per_doc=max_image_candidates_per_doc,
    )
    combined_rows = [*baseline_rows, *stage_rows]

    augment_jsonl = stage_dir / f"phase2_ocr_aug_stage_{stage.stage_key.lower()}_augment.jsonl"
    combined_jsonl = stage_dir / f"phase2_ocr_aug_stage_{stage.stage_key.lower()}_combined.jsonl"
    _write_jsonl(augment_jsonl, stage_rows)
    _write_jsonl(combined_jsonl, combined_rows)

    bm25_path = project_root / "rag_outputs" / stage.bm25_index_name
    bm25_payload = BM25_인덱스_구성(combined_rows)
    BM25_인덱스_저장(bm25_path, bm25_payload)

    chroma = _build_stage_chroma(
        project_root=project_root,
        stage=stage,
        augment_rows=stage_rows,
        chroma_output_dir=chroma_output_dir,
        baseline_collection_name=baseline_collection_name,
        embedding_model=embedding_model,
        embedding_batch_size=embedding_batch_size,
        copy_batch_size=copy_batch_size,
        reset_collection=reset_collection,
    )

    manifest = {
        "stage": stage.stage_key,
        "embedding_backend_key": stage.embedding_backend_key,
        "collection_name": stage.collection_name,
        "bm25_index_name": stage.bm25_index_name,
        "include_image_ocr": stage.include_image_ocr,
        "include_discarded_hints": stage.include_discarded_hints,
        "counts": counts,
        "baseline_row_count": len(baseline_rows),
        "combined_row_count": len(combined_rows),
        "augment_jsonl": str(augment_jsonl),
        "combined_jsonl": str(combined_jsonl),
        "bm25_index_path": str(bm25_path),
        "chroma": chroma,
    }
    _write_json(stage_dir / f"phase2_ocr_aug_stage_{stage.stage_key.lower()}_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Build OCR augment enrichment assets for Phase2 A/B/C.")
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument("--output-root", default=str(root / "rag_outputs" / "phase2_ocr_aug_assets"))
    parser.add_argument("--chroma-output-dir", default=str(root.parent / "rfp_rag_chroma_db_phase2_ocr_aug"))
    parser.add_argument("--baseline-embedding-backend", default="openai_text_embedding_3_small")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--copy-batch-size", type=int, default=256)
    parser.add_argument("--max-image-candidates-per-doc", type=int, default=2)
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["A", "B", "C"],
        help="Stages to build. Choices: A B C",
    )
    parser.add_argument(
        "--discarded-shortlist-path",
        default=str(root / "ocr" / "discarded_shortlist.md"),
    )
    parser.add_argument("--reset-collection", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_root).resolve()
    chroma_output_dir = Path(args.chroma_output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    baseline_rows = _load_baseline_chunks(project_root)
    processed_rows = _load_processed_documents(project_root)
    hwp_rows = _iter_hwp_documents(processed_rows)

    baseline_pipeline = ScenarioACommonPipeline(PipelinePaths(project_root=project_root))
    baseline_embedding_config = baseline_pipeline.load_embedding_config(args.baseline_embedding_backend)
    baseline_collection_name = baseline_embedding_config.collection_name

    selected_discarded_indexes = _parse_discarded_indexes(Path(args.discarded_shortlist_path).resolve())

    selected_stage_keys = []
    for stage_key in args.stages:
        stage_upper = str(stage_key).strip().upper()
        if stage_upper not in STAGE_SPECS:
            raise ValueError(f"Unsupported stage key: {stage_key}")
        selected_stage_keys.append(stage_upper)

    manifests: list[dict[str, Any]] = []
    for stage_key in selected_stage_keys:
        stage = STAGE_SPECS[stage_key]
        manifest = build_stage_assets(
            project_root=project_root,
            stage=stage,
            baseline_rows=baseline_rows,
            hwp_rows=hwp_rows,
            output_root=output_root,
            selected_discarded_indexes=selected_discarded_indexes,
            max_image_candidates_per_doc=max(0, args.max_image_candidates_per_doc),
            baseline_collection_name=baseline_collection_name,
            chroma_output_dir=chroma_output_dir,
            embedding_model=args.embedding_model,
            embedding_batch_size=max(1, args.embedding_batch_size),
            copy_batch_size=max(1, args.copy_batch_size),
            reset_collection=bool(args.reset_collection),
        )
        manifests.append(manifest)
        print(
            f"[done] stage={stage.stage_key} augment={manifest['counts']['stage_row_count']} "
            f"combined={manifest['combined_row_count']} collection={stage.collection_name}"
        )

    summary = {
        "project_root": str(project_root),
        "output_root": str(output_root),
        "chroma_output_dir": str(chroma_output_dir),
        "hwp_document_count": len(hwp_rows),
        "baseline_row_count": len(baseline_rows),
        "baseline_collection_name": baseline_collection_name,
        "selected_discarded_indexes": sorted(selected_discarded_indexes),
        "stage_manifests": manifests,
    }
    _write_json(output_root / "phase2_ocr_aug_build_summary.json", summary)
    print(f"[done] summary={output_root / 'phase2_ocr_aug_build_summary.json'}")


if __name__ == "__main__":
    main()

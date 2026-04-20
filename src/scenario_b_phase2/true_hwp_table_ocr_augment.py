from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import os
import re
import subprocess
import sys
import time
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
    from bs4 import BeautifulSoup  # type: ignore
except Exception as exc:  # noqa: BLE001
    raise RuntimeError("bs4 패키지가 필요합니다. pip install beautifulsoup4") from exc

try:
    from PIL import Image, ImageDraw, ImageFont  # type: ignore
except Exception as exc:  # noqa: BLE001
    raise RuntimeError("Pillow 패키지가 필요합니다. pip install pillow") from exc

try:
    from ocr.extract_hwp_artifacts import extract_hwp_artifacts
    from ocr.run_hwp_ocr_pipeline import run_image_ocr
except Exception as exc:  # noqa: BLE001
    raise RuntimeError("ocr 모듈 import 실패: ocr.extract_hwp_artifacts / ocr.run_hwp_ocr_pipeline") from exc


EMBEDDING_BACKEND_KEY = "openai_text_embedding_3_small_true_table_ocr"
COLLECTION_NAME = "rfp_contextual_chunks_v2_true_table_ocr"
BM25_INDEX_NAME = "bm25_index_phase2_true_table_ocr.pkl"


@dataclass(slots=True)
class HtmlTable:
    index: int
    text: str
    rows: list[list[str]]
    normalized: str


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


def _doc_hash(document_id: str) -> str:
    return hashlib.sha1(document_id.encode("utf-8")).hexdigest()[:10]


def _norm_name(value: str) -> str:
    lowered = re.sub(r"\s+", "", str(value or "").lower())
    return re.sub(r"[^0-9a-z가-힣]", "", lowered)


def _normalize_text(value: str) -> str:
    text = str(value or "").lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^0-9a-z가-힣]", "", text)
    return text


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


def _contains_budget(text: str) -> int:
    return int(bool(re.search(r"(예산|금액|사업비|추정금액|기초금액|원)", str(text or ""))))


def _contains_schedule(text: str) -> int:
    return int(bool(re.search(r"(기간|일정|마감|착수|완료|기한|납기|개월|일\s*이내)", str(text or ""))))


def _contains_contract(text: str) -> int:
    return int(bool(re.search(r"(계약|입찰|협상|방식|제한경쟁|수의)", str(text or ""))))


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
        target_collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
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


def _build_chroma(
    *,
    project_root: Path,
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
            target_client.delete_collection(COLLECTION_NAME)
        except Exception:  # noqa: BLE001
            pass

    copied = _copy_baseline_collection(
        baseline_client=baseline_client,
        baseline_collection_name=baseline_collection_name,
        target_client=target_client,
        target_collection_name=COLLECTION_NAME,
        batch_size=copy_batch_size,
    )
    target_collection = target_client.get_collection(COLLECTION_NAME)
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
        "collection_name": COLLECTION_NAME,
        "baseline_rows_copied": copied,
        "augment_rows_added": added,
        "collection_count": int(target_collection.count() or 0),
    }


def _read_target_doc_names(eval_set_paths: list[Path]) -> set[str]:
    names: set[str] = set()
    for path in eval_set_paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                one = str(row.get("ground_truth_doc", "")).strip()
                if one.lower().endswith(".hwp"):
                    names.add(_norm_name(one))
                many = [x.strip() for x in str(row.get("ground_truth_docs", "")).split("|") if x.strip()]
                for item in many:
                    if item.lower().endswith(".hwp"):
                        names.add(_norm_name(item))
    return names


def _select_target_hwp_docs(processed_rows: list[dict[str, Any]], target_names: set[str]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in processed_rows:
        ext = str(row.get("source_extension", "")).strip().lower()
        if ext != ".hwp":
            continue
        file_name = str(row.get("source_file_name", "")).strip()
        if not file_name:
            continue
        norm_file = _norm_name(file_name)
        matched = False
        for target in target_names:
            if not target:
                continue
            if min(len(target), len(norm_file)) >= 8 and (target in norm_file or norm_file in target):
                matched = True
                break
            ratio = difflib.SequenceMatcher(None, target, norm_file).ratio()
            if ratio >= 0.78:
                matched = True
                break
        if not matched:
            continue
        source_path = Path(str(row.get("source_path", "")).strip())
        if not source_path.exists():
            continue
        doc_id = str(row.get("document_id", "")).strip()
        if not doc_id or doc_id in seen:
            continue
        selected.append(row)
        seen.add(doc_id)
    return selected


def _run_hwp5html(*, source_path: Path, html_path: Path, refresh: bool) -> tuple[bool, str]:
    if html_path.exists() and not refresh:
        return True, ""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "hwp5html",
        "--loglevel",
        "error",
        "--html",
        "--output",
        str(html_path),
        str(source_path),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        tail = "\n".join((completed.stderr or completed.stdout or "").splitlines()[-20:])
        return False, tail.strip()
    return True, ""


def _parse_html_tables(html_path: Path) -> list[HtmlTable]:
    text = html_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(text, "html.parser")
    output: list[HtmlTable] = []
    for index, table in enumerate(soup.find_all("table"), start=1):
        rows: list[list[str]] = []
        for tr in table.find_all("tr"):
            cells = [re.sub(r"\s+", " ", cell.get_text(" ", strip=True)).strip() for cell in tr.find_all(["th", "td"])]
            cells = [cell for cell in cells if cell]
            if cells:
                rows.append(cells)
        if rows:
            lines = [" | ".join(cells) for cells in rows]
            table_text = "\n".join(lines).strip()
        else:
            table_text = re.sub(r"\s+", " ", table.get_text(" ", strip=True)).strip()
        if not table_text:
            continue
        output.append(HtmlTable(index=index, text=table_text, rows=rows, normalized=_normalize_text(table_text)))
    return output


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    value = _normalize_text(text)
    if len(value) < n:
        return {value} if value else set()
    return {value[i : i + n] for i in range(len(value) - n + 1)}


def _match_html_table(
    *,
    structural_text: str,
    linked_parent_text: str,
    html_tables: list[HtmlTable],
) -> tuple[HtmlTable | None, float]:
    norm_struct = _normalize_text(structural_text)
    if not norm_struct or not html_tables:
        return None, 0.0
    struct_grams = _char_ngrams(norm_struct, n=3)

    best_table: HtmlTable | None = None
    best_score = 0.0
    parent_norm = _normalize_text(linked_parent_text)
    for table in html_tables:
        if not table.normalized:
            continue
        table_grams = _char_ngrams(table.normalized, n=3)
        if not table_grams:
            continue
        inter = len(struct_grams & table_grams)
        union = len(struct_grams | table_grams)
        jaccard = inter / union if union else 0.0

        prefix_score = 0.0
        if norm_struct[:80] and norm_struct[:80] in table.normalized:
            prefix_score = 0.20
        elif table.normalized[:80] and table.normalized[:80] in norm_struct:
            prefix_score = 0.15

        parent_boost = 0.0
        if parent_norm and parent_norm in table.normalized:
            parent_boost = 0.15

        score = jaccard + prefix_score + parent_boost
        if score > best_score:
            best_score = score
            best_table = table
    return best_table, round(best_score, 4)


def _render_table_crop(*, table: HtmlTable, out_path: Path) -> tuple[bool, str]:
    lines: list[str] = []
    if table.rows:
        for row in table.rows:
            lines.append(" | ".join(row))
    else:
        lines = [line.strip() for line in table.text.splitlines() if line.strip()]
    if not lines:
        return False, "table text is empty"

    lines = [line[:220] for line in lines[:80]]
    font = ImageFont.load_default()
    dummy = Image.new("RGB", (1, 1), color="white")
    draw = ImageDraw.Draw(dummy)
    max_width = 0
    line_height = 16
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        width = max(1, int(bbox[2] - bbox[0]))
        height = max(1, int(bbox[3] - bbox[1]))
        max_width = max(max_width, width)
        line_height = max(line_height, height + 4)

    width = min(3600, max(900, max_width + 40))
    height = min(5200, max(300, len(lines) * line_height + 40))
    image = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(image)
    y = 20
    for line in lines:
        draw.text((20, y), line, fill="black", font=font)
        y += line_height
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return True, ""


def _build_doc_body_index(baseline_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_file: dict[str, list[dict[str, Any]]] = {}
    for row in baseline_rows:
        file_name = str(row.get("source_file_name", "")).strip()
        if not file_name:
            continue
        if int(row.get("has_table", 0) or 0) == 1:
            continue
        by_file.setdefault(file_name, []).append(row)
    return by_file


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[0-9A-Za-z가-힣]{2,}", str(text or "").lower())


def _find_nearby_body_text(
    *,
    body_rows: list[dict[str, Any]],
    structural_text: str,
    linked_parent_text: str,
) -> str:
    query_tokens = set(_tokenize(linked_parent_text))
    if len(query_tokens) < 3:
        query_tokens.update(_tokenize(structural_text)[:18])
    if not query_tokens:
        return ""

    best_row: dict[str, Any] | None = None
    best_score = 0.0
    for row in body_rows:
        body_text = str(row.get("raw_chunk_text", "") or row.get("contextual_chunk_text", "")).strip()
        if not body_text:
            continue
        tokens = set(_tokenize(body_text))
        if not tokens:
            continue
        inter = len(query_tokens & tokens)
        if inter <= 0:
            continue
        score = inter / max(1, len(query_tokens))
        if linked_parent_text and linked_parent_text in body_text:
            score += 0.2
        if score > best_score:
            best_score = score
            best_row = row
    if best_row is None:
        return ""
    return str(best_row.get("raw_chunk_text", "") or best_row.get("contextual_chunk_text", ""))[:1500]


def _table_context_text(row: dict[str, Any]) -> str:
    parts = [
        "[문서 구조 요약]",
        f"- 사업명: {row.get('사업명', '정보 없음')}",
        f"- 발주기관: {row.get('발주 기관', '정보 없음')}",
        f"- 파일명: {row.get('source_file_name', '정보 없음')}",
        f"- 섹션제목: {row.get('section_label', '정보 없음')}",
        f"- 청크유형: {row.get('chunk_type', '정보 없음')}",
        f"- linked_parent_text: {row.get('linked_parent_text', '')}",
        f"- table_match_score: {row.get('table_match_score', 0.0)}",
        f"- table_ocr_status: {row.get('ocr_status', '')}",
        "[/문서 구조 요약]",
        "",
        "[STRUCTURAL_TABLE]",
        str(row.get("table_markdown", "")).strip(),
        "[/STRUCTURAL_TABLE]",
        "",
        "[TABLE_OCR_TEXT]",
        str(row.get("table_ocr_text", "")).strip(),
        "[/TABLE_OCR_TEXT]",
        "",
        "[NEARBY_BODY]",
        str(row.get("nearby_body_text", "")).strip(),
        "[/NEARBY_BODY]",
    ]
    return "\n".join(parts).strip()


def _table_row(
    *,
    doc_row: dict[str, Any],
    table: dict[str, Any],
    table_ocr_text: str,
    nearby_body_text: str,
    table_match_score: float,
    ocr_status: str,
    ocr_confidence: float,
    crop_path: str,
    row_kind: str,
    part_index: int,
    part_count: int,
) -> dict[str, Any]:
    metadata = doc_row.get("metadata", {}) or {}
    parser_info = doc_row.get("parser_info", {}) or {}
    document_id = str(doc_row.get("document_id", "")).strip()
    file_name = str(doc_row.get("source_file_name", "")).strip()
    table_index = int(table.get("table_index", 0) or 0)
    doc_code = _doc_hash(document_id)
    part_suffix = f"_p{part_index:02d}" if part_count > 1 else ""
    chunk_id = f"trueocr_{doc_code}_{row_kind}_{table_index:04d}{part_suffix}"

    linked_parent_text = str(table.get("linked_parent_text", "") or "").strip()
    raw_text = str(table.get("text", "") or "").strip()
    section_label = linked_parent_text or (raw_text.splitlines()[0][:140] if raw_text else "정보 없음")
    source_position = str(table.get("record_start_index", "") or "")

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
        "parent_section_label": linked_parent_text or section_label,
        "item_title": f"table_{table_index}",
        "source_position": source_position,
        "linked_parent_text": linked_parent_text,
        "linked_parent_table_index": str(table.get("linked_parent_table_index", "") or ""),
        "table_markdown": raw_text,
        "table_ocr_text": table_ocr_text,
        "figure_text": "",
        "nearby_body_text": nearby_body_text,
        "extracted_from": "table",
        "chunk_type": row_kind,
        "chunk_role": "표/OCR",
        "chunk_role_tags": row_kind,
        "parser_source": "hwp5html+table_crop_ocr",
        "ocr_confidence": round(float(ocr_confidence), 4),
        "ocr_status": ocr_status,
        "ocr_crop_path": crop_path,
        "table_match_score": round(float(table_match_score), 4),
        "has_table": 1,
        "is_high_priority": 1,
        "raw_chunk_chars": len(raw_text),
        "table_part_index": part_index,
        "table_part_count": part_count,
        "parser_used": str(parser_info.get("parser_used", "")),
        "source_marker": "true_table_ocr_augment",
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
    return row


def _section_header_support_row(
    *,
    doc_row: dict[str, Any],
    table: dict[str, Any],
    part_text: str,
    part_index: int,
    part_count: int,
) -> dict[str, Any]:
    metadata = doc_row.get("metadata", {}) or {}
    document_id = str(doc_row.get("document_id", "")).strip()
    doc_code = _doc_hash(document_id)
    table_index = int(table.get("table_index", 0) or 0)
    part_suffix = f"_p{part_index:02d}" if part_count > 1 else ""
    chunk_id = f"trueocr_{doc_code}_section_header_{table_index:04d}{part_suffix}"
    linked_parent_text = str(table.get("linked_parent_text", "") or "").strip()
    text = str(part_text or "").strip()
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
        "section_title": linked_parent_text or text[:120],
        "section_path": str(table.get("section", "")),
        "section_label": linked_parent_text or text[:120],
        "parent_section_label": linked_parent_text or text[:120],
        "item_title": f"section_header_{table_index}",
        "source_position": str(table.get("record_start_index", "") or ""),
        "linked_parent_text": linked_parent_text,
        "linked_parent_table_index": "",
        "table_markdown": text,
        "table_ocr_text": "",
        "figure_text": "",
        "nearby_body_text": "",
        "extracted_from": "table",
        "chunk_type": "section_header_support",
        "chunk_role": "표/구조",
        "chunk_role_tags": "section_header_support",
        "parser_source": "extract_hwp_artifacts",
        "ocr_confidence": 0.0,
        "ocr_status": "not_applicable",
        "ocr_crop_path": "",
        "table_match_score": 0.0,
        "has_table": 1,
        "is_high_priority": 0,
        "raw_chunk_chars": len(text),
        "table_part_index": part_index,
        "table_part_count": part_count,
        "source_marker": "true_table_ocr_augment",
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
    return row


def build_true_table_ocr_rows(
    *,
    hwp_docs: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    output_root: Path,
    max_table_candidates_per_doc: int,
    min_match_score: float,
    refresh_html: bool,
    include_section_header_support: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    augment_rows: list[dict[str, Any]] = []
    summary = {
        "target_hwp_doc_count": len(hwp_docs),
        "processed_doc_count": 0,
        "hwp5html_success_count": 0,
        "hwp5html_fail_count": 0,
        "table_candidate_count": 0,
        "table_matched_count": 0,
        "table_unmatched_count": 0,
        "table_crop_success_count": 0,
        "table_crop_fail_count": 0,
        "table_ocr_ok_count": 0,
        "table_ocr_fallback_count": 0,
        "section_header_support_rows": 0,
        "errors": [],
        "docs": [],
    }
    body_by_file = _build_doc_body_index(baseline_rows)

    for doc_row in hwp_docs:
        document_id = str(doc_row.get("document_id", "")).strip()
        source_file_name = str(doc_row.get("source_file_name", "")).strip()
        source_path = Path(str(doc_row.get("source_path", "")).strip())
        doc_hash = _doc_hash(document_id)
        doc_dir = output_root / "ocr_work" / doc_hash
        html_path = doc_dir / "rendered.html"
        crop_dir = doc_dir / "table_crops"
        t0 = time.time()

        ok, err = _run_hwp5html(source_path=source_path, html_path=html_path, refresh=refresh_html)
        if not ok:
            summary["hwp5html_fail_count"] += 1
            summary["errors"].append(
                {
                    "document_id": document_id,
                    "source_file_name": source_file_name,
                    "error_type": "hwp5html_failed",
                    "message": err,
                }
            )
            continue

        summary["hwp5html_success_count"] += 1
        payload = extract_hwp_artifacts(source_path, save_images=False, output_dir=doc_dir)
        html_tables = _parse_html_tables(html_path)
        candidates = sorted(
            payload.get("table_ocr_candidates", []),
            key=lambda row: int(row.get("ocr_candidate_score", 0) or 0),
            reverse=True,
        )
        if max_table_candidates_per_doc > 0:
            candidates = candidates[: max_table_candidates_per_doc]
        summary["table_candidate_count"] += len(candidates)

        doc_body_rows = body_by_file.get(source_file_name, [])
        matched_indexes: set[int] = set()
        added_rows_before = len(augment_rows)
        doc_table_matches = 0
        doc_crop_ok = 0
        doc_ocr_ok = 0

        for table in candidates:
            table_index = int(table.get("table_index", 0) or 0)
            structural_text = str(table.get("text", "") or "").strip()
            linked_parent_text = str(table.get("linked_parent_text", "") or "").strip()
            matched, match_score = _match_html_table(
                structural_text=structural_text,
                linked_parent_text=linked_parent_text,
                html_tables=html_tables,
            )

            ocr_text = ""
            ocr_status = "fallback_structural_text"
            ocr_confidence = 0.35
            crop_path = ""

            if matched is not None and match_score >= min_match_score:
                summary["table_matched_count"] += 1
                doc_table_matches += 1
                matched_indexes.add(table_index)
                matched_text = matched.text.strip() or structural_text
                crop_file = crop_dir / f"table_{table_index:04d}.png"
                crop_ok, crop_err = _render_table_crop(table=matched, out_path=crop_file)
                if crop_ok:
                    summary["table_crop_success_count"] += 1
                    doc_crop_ok += 1
                    crop_path = str(crop_file)
                    ocr_result = run_image_ocr(crop_file)
                    if str(ocr_result.get("status", "")).strip() == "ok":
                        ocr_text = str(ocr_result.get("ocr_text", "")).strip()
                    if ocr_text:
                        summary["table_ocr_ok_count"] += 1
                        doc_ocr_ok += 1
                        ocr_status = "ok"
                        ocr_confidence = 0.85
                    else:
                        summary["table_ocr_fallback_count"] += 1
                        ocr_text = matched_text
                        ocr_status = "ocr_failed_fallback_html_table_text"
                        ocr_confidence = 0.55
                else:
                    summary["table_crop_fail_count"] += 1
                    summary["table_ocr_fallback_count"] += 1
                    summary["errors"].append(
                        {
                            "document_id": document_id,
                            "source_file_name": source_file_name,
                            "error_type": "crop_failed",
                            "table_index": table_index,
                            "message": crop_err,
                        }
                    )
                    ocr_text = matched_text
                    ocr_status = "crop_failed_fallback_html_table_text"
                    ocr_confidence = 0.50
            else:
                summary["table_unmatched_count"] += 1
                summary["table_ocr_fallback_count"] += 1
                ocr_text = structural_text
                ocr_status = "html_table_not_matched_fallback_structural_text"
                ocr_confidence = 0.30

            nearby_body = _find_nearby_body_text(
                body_rows=doc_body_rows,
                structural_text=structural_text,
                linked_parent_text=linked_parent_text,
            )
            merged_text = "\n".join(part for part in [structural_text, ocr_text, nearby_body] if part.strip()).strip()
            parts = _split_text(merged_text, max_chars=1400, overlap=120)
            if not parts:
                continue
            for part_index, _part in enumerate(parts, start=1):
                row = _table_row(
                    doc_row=doc_row,
                    table=table,
                    table_ocr_text=ocr_text,
                    nearby_body_text=nearby_body,
                    table_match_score=match_score,
                    ocr_status=ocr_status,
                    ocr_confidence=ocr_confidence,
                    crop_path=crop_path,
                    row_kind="table_true_ocr",
                    part_index=part_index,
                    part_count=len(parts),
                )
                augment_rows.append(row)

        if include_section_header_support:
            for table in payload.get("section_header_blocks", []):
                child_indices = {int(value) for value in table.get("linked_child_table_indices", []) if int(value or 0) > 0}
                if not child_indices.intersection(matched_indexes):
                    continue
                parts = _split_text(str(table.get("text", "") or ""), max_chars=900, overlap=80)
                for part_index, part_text in enumerate(parts, start=1):
                    augment_rows.append(
                        _section_header_support_row(
                            doc_row=doc_row,
                            table=table,
                            part_text=part_text,
                            part_index=part_index,
                            part_count=len(parts),
                        )
                    )
                    summary["section_header_support_rows"] += 1

        elapsed = round(time.time() - t0, 2)
        summary["processed_doc_count"] += 1
        summary["docs"].append(
            {
                "document_id": document_id,
                "source_file_name": source_file_name,
                "html_table_count": len(html_tables),
                "table_candidate_count": len(candidates),
                "doc_table_matches": doc_table_matches,
                "doc_crop_success": doc_crop_ok,
                "doc_ocr_ok": doc_ocr_ok,
                "augment_rows_added": len(augment_rows) - added_rows_before,
                "elapsed_sec": elapsed,
            }
        )

    deduped: dict[str, dict[str, Any]] = {}
    for row in augment_rows:
        deduped[row["chunk_id"]] = row
    augment_rows = list(deduped.values())
    summary["augment_row_count"] = len(augment_rows)
    return augment_rows, summary


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Build true HWP table OCR augment corpus (table crop + OCR + augment-only merge on baseline)."
    )
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument("--output-root", default=str(root / "rag_outputs" / "phase2_true_table_ocr_assets"))
    parser.add_argument("--chroma-output-dir", default=str(root.parent / "rfp_rag_chroma_db_phase2_true_table_ocr"))
    parser.add_argument(
        "--table-eval-set-path",
        default=str(root / "rag_outputs" / "eval_sets" / "b05_table_eval_questions_v2.csv"),
    )
    parser.add_argument(
        "--groupbc-eval-set-path",
        default=str(root / "rag_outputs" / "eval_sets" / "b05_group_bc_questions_v1.csv"),
    )
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--copy-batch-size", type=int, default=256)
    parser.add_argument("--max-table-candidates-per-doc", type=int, default=10)
    parser.add_argument("--min-match-score", type=float, default=0.22)
    parser.add_argument("--refresh-html", action="store_true")
    parser.add_argument("--include-section-header-support", action="store_true")
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
    table_eval_path = Path(args.table_eval_set_path).resolve()
    group_eval_path = Path(args.groupbc_eval_set_path).resolve()
    target_names = _read_target_doc_names([table_eval_path, group_eval_path])
    target_hwp_docs = _select_target_hwp_docs(processed_rows, target_names)

    baseline_pipeline = ScenarioACommonPipeline(PipelinePaths(project_root=project_root))
    embedding_config = baseline_pipeline.load_embedding_config("openai_text_embedding_3_small")
    baseline_collection_name = embedding_config.collection_name

    augment_rows, build_summary = build_true_table_ocr_rows(
        hwp_docs=target_hwp_docs,
        baseline_rows=baseline_rows,
        output_root=output_root,
        max_table_candidates_per_doc=max(1, args.max_table_candidates_per_doc),
        min_match_score=max(0.0, args.min_match_score),
        refresh_html=bool(args.refresh_html),
        include_section_header_support=bool(args.include_section_header_support),
    )

    combined_rows = [*baseline_rows, *augment_rows]
    augment_jsonl = output_root / "true_table_ocr_augment.jsonl"
    combined_jsonl = output_root / "true_table_ocr_combined.jsonl"
    _write_jsonl(augment_jsonl, augment_rows)
    _write_jsonl(combined_jsonl, combined_rows)

    bm25_path = project_root / "rag_outputs" / BM25_INDEX_NAME
    bm25_payload = BM25_인덱스_구성(combined_rows)
    BM25_인덱스_저장(bm25_path, bm25_payload)

    chroma = _build_chroma(
        project_root=project_root,
        augment_rows=augment_rows,
        chroma_output_dir=chroma_output_dir,
        baseline_collection_name=baseline_collection_name,
        embedding_model=args.embedding_model,
        embedding_batch_size=max(1, args.embedding_batch_size),
        copy_batch_size=max(1, args.copy_batch_size),
        reset_collection=bool(args.reset_collection),
    )

    summary = {
        "project_root": str(project_root),
        "embedding_backend_key": EMBEDDING_BACKEND_KEY,
        "collection_name": COLLECTION_NAME,
        "bm25_index_name": BM25_INDEX_NAME,
        "baseline_row_count": len(baseline_rows),
        "augment_row_count": len(augment_rows),
        "combined_row_count": len(combined_rows),
        "target_hwp_docs": [str(row.get("source_file_name", "")) for row in target_hwp_docs],
        "table_eval_set_path": str(table_eval_path),
        "groupbc_eval_set_path": str(group_eval_path),
        "augment_jsonl": str(augment_jsonl),
        "combined_jsonl": str(combined_jsonl),
        "bm25_index_path": str(bm25_path),
        "chroma": chroma,
        "build_summary": build_summary,
        "true_table_ocr_feasibility": {
            "render_path": "hwp5html --html output",
            "table_crop_generation": "enabled (PIL render from matched HTML table rows)",
            "table_ocr": "enabled (run_image_ocr on generated table crop)",
            "fallback_policy": "OCR 실패/매칭 실패 시 structural/html text fallback",
        },
    }
    summary_path = output_root / "phase2_true_table_ocr_summary.json"
    _write_json(summary_path, summary)

    print(f"[done] target_hwp_docs={len(target_hwp_docs)}")
    print(f"[done] augment_rows={len(augment_rows)} combined_rows={len(combined_rows)}")
    print(f"[done] bm25={bm25_path}")
    print(f"[done] collection={COLLECTION_NAME} chroma_dir={chroma_output_dir}")
    print(f"[done] summary={summary_path}")


if __name__ == "__main__":
    main()

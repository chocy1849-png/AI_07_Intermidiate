from __future__ import annotations

import argparse
import base64
import csv
import difflib
import importlib.util
import json
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

from rag_공통 import INPUT_JSONL_PATH, OUTPUT_DIR, OpenAI_클라이언트_가져오기, csv_저장, jsonl_불러오기, jsonl_저장


B02_CHUNK_PATH = OUTPUT_DIR / "b02_prefix_v2_chunks.jsonl"
TABLE_EVAL_CSV_PATH = OUTPUT_DIR / "eval_sets" / "b05_table_eval_questions_v2.csv"
B05_CHUNK_PATH = OUTPUT_DIR / "b05_table_enriched_chunks.jsonl"
B05_SUMMARY_PATH = OUTPUT_DIR / "b05_table_enriched_chunk_summary.csv"
B05_DOCMAP_PATH = OUTPUT_DIR / "b05_table_doc_mapping.csv"
B05_MANIFEST_PATH = OUTPUT_DIR / "b05_table_enriched_manifest.json"
B05_HWP_CACHE_DIR = OUTPUT_DIR / "b05_hwp_xml_cache"
B05_FALLBACK_DIR = OUTPUT_DIR / "b05_vision_fallback"

TABLE_GENERIC_KEYWORDS = [
    "기능",
    "요구사항",
    "목록",
    "모듈",
    "구성",
    "장비",
    "지역",
    "앱",
    "현황",
    "소프트웨어",
    "AI",
    "ERP",
    "그룹웨어",
    "상황실",
    "삭제지원",
    "모바일오피스",
    "정의",
    "역할",
    "구분",
]


def load_b02_module():
    module_path = Path(__file__).resolve().parent / "10_B02_prefix_v2_청킹.py"
    spec = importlib.util.spec_from_file_location("b02_prefix_v2_chunking", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"B-02 모듈을 불러오지 못했습니다: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_text(value: str) -> str:
    text = str(value or "").lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^0-9a-z가-힣]+", "", text)
    return text


def clean_cell_text(text: str) -> str:
    value = str(text or "")
    value = value.replace("\u00a0", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def rows_to_markdown(rows: list[list[str]], max_rows: int = 16, max_cols: int = 8) -> str:
    normalized_rows: list[list[str]] = []
    for row in rows[:max_rows]:
        cells = [clean_cell_text(cell).replace("|", "/") for cell in row[:max_cols]]
        if any(cells):
            normalized_rows.append(cells)
    if not normalized_rows:
        return ""
    width = max(len(row) for row in normalized_rows)
    normalized_rows = [row + [""] * (width - len(row)) for row in normalized_rows]
    header = normalized_rows[0]
    body = normalized_rows[1:] if len(normalized_rows) > 1 else []
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def tokenize_keywords(text: str) -> list[str]:
    tokens = re.findall(r"[0-9A-Za-z가-힣]{2,}", str(text or ""))
    return [
        token
        for token in tokens
        if token not in {"정리해줘", "알려줘", "있어", "같이", "주요", "기존", "새로", "대상", "시스템"}
    ]


def parse_table_eval_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def load_doc_keywords(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    by_doc: dict[str, list[str]] = {}
    for row in rows:
        doc = row.get("ground_truth_doc", "").strip()
        if not doc:
            continue
        tokens = tokenize_keywords(row.get("question", "")) + tokenize_keywords(row.get("eval_focus", ""))
        by_doc.setdefault(doc, []).extend(tokens)
    return {doc: [token for token, _ in Counter(tokens).most_common(20)] for doc, tokens in by_doc.items()}


def match_target_documents(target_names: list[str], processed_documents: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    by_name = {str(row.get("source_file_name", "")): row for row in processed_documents}
    processed_names = list(by_name.keys())
    normalized_processed = {name: normalize_text(name) for name in processed_names}

    matched: dict[str, dict[str, Any]] = {}
    mapping_rows: list[dict[str, Any]] = []
    for target in target_names:
        target = str(target or "").strip().strip('"').strip("'")
        hit = by_name.get(target)
        reason = "exact"
        if hit is None:
            target_norm = normalize_text(target)
            best_name = None
            best_score = -1.0
            for name in processed_names:
                name_norm = normalized_processed[name]
                score = 0.0
                if target_norm in name_norm or name_norm in target_norm:
                    score = 1000.0 - abs(len(target_norm) - len(name_norm))
                else:
                    raw_tokens = re.findall(r"[0-9a-z가-힣]{2,}", target.lower())
                    token_overlap = sum(1 for token in raw_tokens if token and token in name.lower())
                    similarity = difflib.SequenceMatcher(None, target_norm, name_norm).ratio()
                    score = token_overlap + similarity * 10.0
                if score > best_score:
                    best_name = name
                    best_score = score
            if best_name is not None and best_score >= 4:
                hit = by_name[best_name]
                reason = f"fuzzy:{round(best_score, 4)}"

        mapping_rows.append(
            {
                "target_doc_name": target,
                "matched_source_file_name": hit.get("source_file_name", "") if hit else "",
                "matched_document_id": hit.get("document_id", "") if hit else "",
                "source_extension": hit.get("source_extension", "") if hit else "",
                "match_reason": reason if hit else "missing",
            }
        )
        if hit is not None:
            matched[target] = hit
    return matched, mapping_rows


def extract_hwp_tables_via_xml(source_path: Path, cache_dir: Path, timeout_sec: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    xml_path = cache_dir / "document.xml"
    if not xml_path.exists():
        result = subprocess.run(
            ["hwp5proc", "xml", "--output", str(xml_path), str(source_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=timeout_sec,
        )
        if result.returncode != 0 or not xml_path.exists():
            return [], {"mode": "hwp5proc_xml", "success": False, "error": (result.stderr or "")[:1000]}

    root = ET.parse(xml_path).getroot()
    tables: list[dict[str, Any]] = []
    for table_index, table in enumerate(root.iter("TableControl"), start=1):
        rows: list[list[str]] = []
        for tr in table.iter("TableRow"):
            row_cells: list[str] = []
            for tc in tr.findall("TableCell"):
                texts = [clean_cell_text(node.text or "") for node in tc.iter("Text") if clean_cell_text(node.text or "")]
                row_cells.append(" ".join(texts).strip())
            if any(row_cells):
                rows.append(row_cells)
        markdown = rows_to_markdown(rows)
        if markdown:
            tables.append(
                {
                    "table_index": table_index,
                    "source": "hwp5proc_xml",
                    "row_count": len(rows),
                    "markdown": markdown,
                    "table_text": " ".join(" ".join(row) for row in rows)[:4000],
                }
            )
    return tables, {"mode": "hwp5proc_xml", "success": True, "table_count": len(tables), "xml_path": str(xml_path)}


def extract_pdf_tables_from_processed(document: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts = document.get("artifacts", {}) or {}
    tables = []
    for block in artifacts.get("table_blocks", []) or []:
        markdown = str(block.get("markdown", "")).strip()
        if not markdown:
            continue
        tables.append(
            {
                "table_index": block.get("table_index"),
                "page_number": block.get("page_number"),
                "source": block.get("source", "pdf_preextract"),
                "row_count": block.get("row_count"),
                "markdown": markdown,
                "table_text": re.sub(r"[|\\-]+", " ", markdown),
            }
        )
    return tables


def score_table_block(table_block: dict[str, Any], doc_keywords: list[str]) -> float:
    text = table_block.get("table_text", "")
    row_count = int(table_block.get("row_count") or 0)
    keyword_hits = sum(1 for keyword in doc_keywords if keyword and keyword in text)
    generic_hits = sum(1 for keyword in TABLE_GENERIC_KEYWORDS if keyword in text)
    nonempty_tokens = len(re.findall(r"[가-힣A-Za-z0-9]{2,}", text))
    penalty = 0.0
    if row_count <= 1:
        penalty += 4.0
    if nonempty_tokens < 8:
        penalty += 2.0
    if len(text) < 25:
        penalty += 3.0
    return keyword_hits * 3.0 + generic_hits * 1.2 + min(row_count, 20) * 0.2 + min(nonempty_tokens, 40) * 0.05 - penalty


def select_relevant_tables(table_blocks: list[dict[str, Any]], doc_keywords: list[str], max_tables: int) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    seen = set()
    for block in table_blocks:
        key = normalize_text(block.get("markdown", "")[:1000])
        if not key or key in seen:
            continue
        seen.add(key)
        score = score_table_block(block, doc_keywords)
        if score < 1.0:
            continue
        new_block = dict(block)
        new_block["selection_score"] = round(score, 4)
        scored.append(new_block)
    scored.sort(key=lambda row: (row["selection_score"], len(row.get("markdown", ""))), reverse=True)
    return scored[:max_tables]


def image_to_data_url(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    mime = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "bmp": "image/bmp",
        "gif": "image/gif",
        "webp": "image/webp",
    }.get(ext, "image/png")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def vision_transcribe_structured_image(client, image_path: Path) -> str:
    response = client.responses.create(
        model="gpt-5",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "이미지가 표나 구조화된 목록이면 문서에 보이는 텍스트만 기반으로 markdown 표 또는 bullet list로 전사하라. "
                            "추론하지 말고, 구조화 정보가 아니면 정확히 NO_TABLE 만 출력하라."
                        ),
                    },
                    {"type": "input_image", "image_url": image_to_data_url(image_path)},
                ],
            }
        ],
    )
    return response.output_text.strip()


def extract_hwp_tables_via_vision_fallback(source_path: Path, cache_dir: Path, client, timeout_sec: int, max_images: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    html_dir = cache_dir / "html_fallback"
    html_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["hwp5html", "--output", str(html_dir), str(source_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
        timeout=timeout_sec,
    )
    if result.returncode != 0:
        return [], {"mode": "hwp5html_vision", "success": False, "error": (result.stderr or "")[:1000]}

    image_paths = sorted(
        [path for path in html_dir.rglob("*") if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}],
        key=lambda path: path.stat().st_size,
        reverse=True,
    )[:max_images]

    tables: list[dict[str, Any]] = []
    for image_index, image_path in enumerate(image_paths, start=1):
        transcription = vision_transcribe_structured_image(client, image_path)
        if transcription == "NO_TABLE":
            continue
        tables.append(
            {
                "table_index": image_index,
                "source": "gpt5_vision_fallback",
                "row_count": transcription.count("\n"),
                "markdown": transcription,
                "table_text": transcription,
            }
        )
    return tables, {"mode": "hwp5html_vision", "success": True, "image_count": len(image_paths), "table_count": len(tables)}


def build_table_chunk_rows(selected_tables: list[dict[str, Any]], processed_document: dict[str, Any], base_doc_row: dict[str, Any], b02_module) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    document_fields = {
        "사업명": base_doc_row.get("사업명", "정보 없음"),
        "발주 기관": base_doc_row.get("발주 기관", "정보 없음"),
        "공고 번호": base_doc_row.get("공고 번호", "정보 없음"),
        "공개 일자": base_doc_row.get("공개 일자", "정보 없음"),
        "budget_text": base_doc_row.get("budget_text", "정보 없음"),
        "budget_amount_krw": base_doc_row.get("budget_amount_krw", None),
        "bid_deadline": base_doc_row.get("bid_deadline", "정보 없음"),
        "period_raw": base_doc_row.get("period_raw", "정보 없음"),
        "period_days": base_doc_row.get("period_days", None),
        "period_months": base_doc_row.get("period_months", None),
        "contract_method": base_doc_row.get("contract_method", "정보 없음"),
        "bid_method": base_doc_row.get("bid_method", "정보 없음"),
        "purpose_summary": base_doc_row.get("purpose_summary", "정보 없음"),
        "document_type": base_doc_row.get("document_type", "정보 없음"),
        "domain_tags": base_doc_row.get("domain_tags", ""),
    }

    for index, table_block in enumerate(selected_tables, start=1):
        table_text = table_block["markdown"]
        chunk_role, chunk_role_tags, role_scores = b02_module.detect_chunk_roles(table_text)
        section_title = f"표 보강 {index}"
        chunk_role_value = "표" if chunk_role == "일반" else f"표/{chunk_role}"
        prefix_lines = [
            "[표 보강 요약]",
            f"- 사업명: {document_fields['사업명']}",
            f"- 발주기관: {document_fields['발주 기관']}",
            f"- 문서유형: {document_fields['document_type']}",
            f"- 도메인태그: {document_fields['domain_tags'] or '정보 없음'}",
            f"- 예산: {document_fields['budget_text']}",
            f"- 사업기간: {document_fields['period_raw']}",
            f"- 계약방식: {document_fields['contract_method']}",
            f"- 표추출출처: {table_block.get('source', 'unknown')}",
            f"- 표인덱스: {table_block.get('table_index', index)}",
            f"- 표행수: {table_block.get('row_count', '')}",
            f"- 선택점수: {table_block.get('selection_score', '')}",
            f"- 청크역할: {chunk_role_value}",
            f"- 청크태그: {('표|' + chunk_role_tags).strip('|') or '표'}",
            "[/표 보강 요약]",
        ]
        contextual = "\n".join(prefix_lines) + "\n\n[보강 표]\n" + table_text + "\n[/보강 표]"
        rows.append(
            {
                "chunk_id": f"{processed_document['document_id']}__table_{index:04d}",
                "document_id": processed_document["document_id"],
                "chunk_index": 10000 + index,
                "source_file_name": processed_document["source_file_name"],
                "source_path": processed_document["source_path"],
                "source_extension": processed_document["source_extension"],
                "사업명": document_fields["사업명"],
                "발주 기관": document_fields["발주 기관"],
                "공고 번호": document_fields["공고 번호"],
                "공개 일자": document_fields["공개 일자"],
                "파일형식": base_doc_row.get("파일형식", processed_document["source_extension"]),
                "metadata_prefix": "\n".join(prefix_lines),
                "raw_chunk_text": table_text,
                "contextual_chunk_text": contextual,
                "raw_chunk_chars": len(table_text),
                "contextual_chunk_chars": len(contextual),
                "budget_text": document_fields["budget_text"],
                "budget_amount_krw": document_fields["budget_amount_krw"],
                "bid_deadline": document_fields["bid_deadline"],
                "period_raw": document_fields["period_raw"],
                "period_days": document_fields["period_days"],
                "period_months": document_fields["period_months"],
                "contract_method": document_fields["contract_method"],
                "bid_method": document_fields["bid_method"],
                "purpose_summary": document_fields["purpose_summary"],
                "document_type": document_fields["document_type"],
                "domain_tags": document_fields["domain_tags"],
                "section_title": section_title,
                "section_path": section_title,
                "chunk_role": chunk_role_value,
                "chunk_role_tags": ("표|" + chunk_role_tags).strip("|") or "표",
                "has_table": 1,
                "has_budget_signal": int(role_scores.get("예산", 0) > 0),
                "has_schedule_signal": int(role_scores.get("기간", 0) > 0 or role_scores.get("일정", 0) > 0),
                "has_contract_signal": int(role_scores.get("계약", 0) > 0),
                "mentioned_systems": b02_module.detect_system_mentions(table_text),
                "table_source": table_block.get("source", ""),
                "table_index": table_block.get("table_index", index),
                "table_row_count": table_block.get("row_count", ""),
                "selection_score": table_block.get("selection_score", ""),
            }
        )
    return rows


def summarize_chunks(chunk_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for row in chunk_rows:
        by_doc.setdefault(str(row["document_id"]), []).append(row)
    summary = []
    for document_id, rows in by_doc.items():
        first = rows[0]
        summary.append(
            {
                "document_id": document_id,
                "source_file_name": first["source_file_name"],
                "사업명": first["사업명"],
                "발주 기관": first["발주 기관"],
                "chunk_count": len(rows),
                "table_chunk_count": sum(int(row.get("has_table", 0)) for row in rows),
                "supplemental_table_chunk_count": sum(1 for row in rows if "__table_" in str(row.get("chunk_id", ""))),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="B-05 표 enrichment supplemental chunk를 생성합니다.")
    parser.add_argument("--입력문서경로", default=str(INPUT_JSONL_PATH))
    parser.add_argument("--기준청크경로", default=str(B02_CHUNK_PATH))
    parser.add_argument("--표질문CSV경로", default=str(TABLE_EVAL_CSV_PATH))
    parser.add_argument("--출력청크경로", default=str(B05_CHUNK_PATH))
    parser.add_argument("--출력요약경로", default=str(B05_SUMMARY_PATH))
    parser.add_argument("--출력문서매핑경로", default=str(B05_DOCMAP_PATH))
    parser.add_argument("--manifest경로", default=str(B05_MANIFEST_PATH))
    parser.add_argument("--문서당최대표수", type=int, default=12)
    parser.add_argument("--xml타임아웃초", type=int, default=120)
    parser.add_argument("--fallback타임아웃초", type=int, default=360)
    parser.add_argument("--vision최대이미지수", type=int, default=2)
    args = parser.parse_args()

    b02_module = load_b02_module()
    processed_documents = jsonl_불러오기(Path(args.입력문서경로))
    b02_chunk_rows = jsonl_불러오기(Path(args.기준청크경로))
    question_rows = parse_table_eval_rows(Path(args.표질문CSV경로))
    target_names = list(dict.fromkeys(row["ground_truth_doc"] for row in question_rows if row.get("ground_truth_doc")))
    doc_keywords = load_doc_keywords(question_rows)

    matched_docs, mapping_rows = match_target_documents(target_names, processed_documents)
    csv_저장(Path(args.출력문서매핑경로), mapping_rows)

    base_doc_row_by_name: dict[str, dict[str, Any]] = {}
    for row in b02_chunk_rows:
        base_doc_row_by_name.setdefault(str(row.get("source_file_name", "")), row)

    output_chunk_rows = list(b02_chunk_rows)
    extraction_rows: list[dict[str, Any]] = []
    client = None

    for target_name, processed_doc in matched_docs.items():
        source_name = str(processed_doc.get("source_file_name", ""))
        base_doc_row = base_doc_row_by_name.get(source_name)
        if base_doc_row is None:
            continue

        if str(processed_doc.get("source_extension", "")).lower() == ".pdf":
            candidate_tables = extract_pdf_tables_from_processed(processed_doc)
            selected_tables = select_relevant_tables(candidate_tables, doc_keywords.get(target_name, []), args.문서당최대표수)
            extraction_meta: dict[str, Any] = {"mode": "pdf_artifacts", "candidate_count": len(candidate_tables), "selected_count": len(selected_tables)}
        else:
            candidate_tables, extraction_meta = extract_hwp_tables_via_xml(
                source_path=Path(processed_doc["source_path"]),
                cache_dir=B05_HWP_CACHE_DIR / normalize_text(source_name),
                timeout_sec=args.xml타임아웃초,
            )
            selected_tables = select_relevant_tables(candidate_tables, doc_keywords.get(target_name, []), args.문서당최대표수)
            if not selected_tables:
                if client is None:
                    client = OpenAI_클라이언트_가져오기()
                fallback_tables, fallback_meta = extract_hwp_tables_via_vision_fallback(
                    source_path=Path(processed_doc["source_path"]),
                    cache_dir=B05_FALLBACK_DIR / normalize_text(source_name),
                    client=client,
                    timeout_sec=args.fallback타임아웃초,
                    max_images=args.vision최대이미지수,
                )
                selected_tables = select_relevant_tables(fallback_tables, doc_keywords.get(target_name, []), args.문서당최대표수)
                extraction_meta = {"primary": extraction_meta, "fallback": fallback_meta}

        supplemental_rows = build_table_chunk_rows(selected_tables, processed_doc, base_doc_row, b02_module)
        output_chunk_rows.extend(supplemental_rows)
        extraction_rows.append(
            {
                "target_doc_name": target_name,
                "matched_source_file_name": source_name,
                "source_extension": processed_doc.get("source_extension", ""),
                "selected_table_count": len(selected_tables),
                "supplemental_chunk_count": len(supplemental_rows),
                "extraction_meta": json.dumps(extraction_meta, ensure_ascii=False),
            }
        )

    jsonl_저장(Path(args.출력청크경로), output_chunk_rows)
    csv_저장(Path(args.출력요약경로), summarize_chunks(output_chunk_rows))
    csv_저장(OUTPUT_DIR / "b05_table_extraction_summary.csv", extraction_rows)

    manifest = {
        "input_document_path": str(args.입력문서경로),
        "base_chunk_path": str(args.기준청크경로),
        "output_chunk_path": str(args.출력청크경로),
        "target_doc_count": len(target_names),
        "matched_doc_count": len(matched_docs),
        "base_chunk_count": len(b02_chunk_rows),
        "final_chunk_count": len(output_chunk_rows),
        "supplemental_chunk_count": len(output_chunk_rows) - len(b02_chunk_rows),
        "max_tables_per_doc": args.문서당최대표수,
        "hwp_primary_extractor": "hwp5proc xml",
        "hwp_fallback_extractor": "hwp5html + gpt-5 vision",
    }
    Path(args.manifest경로).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest경로).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[완료] B-05 표 enrichment supplemental chunk 생성이 끝났습니다.")
    print(f"- 대상 문서 수: {len(target_names)}")
    print(f"- 매칭 문서 수: {len(matched_docs)}")
    print(f"- 기존 청크 수: {len(b02_chunk_rows)}")
    print(f"- 최종 청크 수: {len(output_chunk_rows)}")
    print(f"- 추가 table chunk 수: {len(output_chunk_rows) - len(b02_chunk_rows)}")


if __name__ == "__main__":
    main()

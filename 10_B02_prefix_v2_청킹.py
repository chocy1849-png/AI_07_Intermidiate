from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

from rag_공통 import INPUT_JSONL_PATH, OUTPUT_DIR


B02_CHUNK_JSONL_PATH = OUTPUT_DIR / "b02_prefix_v2_chunks.jsonl"
B02_CHUNK_SUMMARY_CSV_PATH = OUTPUT_DIR / "b02_prefix_v2_chunk_summary.csv"
B02_DOCUMENT_FIELDS_CSV_PATH = OUTPUT_DIR / "b02_prefix_v2_document_fields.csv"
B02_MANIFEST_PATH = OUTPUT_DIR / "b02_prefix_v2_manifest.json"


ROLE_RULES: dict[str, list[str]] = {
    "예산": ["사업예산", "소요예산", "사업 금액", "기초금액", "추정금액", "예정가격", "금액"],
    "기간": ["사업기간", "계약기간", "수행기간", "용역기간", "입찰 참여 마감일", "마감일", "완료일", "계약체결일로부터"],
    "계약": ["계약방식", "사업추진방식", "입찰방식", "협상에 의한 계약", "제한경쟁", "일반경쟁", "수의계약"],
    "목적": ["사업목적", "사업목표", "추진목적", "추진배경", "필요성"],
    "요구사항": ["요구사항", "기능 요구사항", "성능 요구사항", "인터페이스 요구사항", "데이터 요구사항", "보안 요구사항", "테스트 요구사항"],
    "평가": ["평가기준", "기술평가", "가격평가", "협상적격자", "배점"],
    "보안": ["보안", "개인정보", "정보보호", "접근통제", "비밀유지", "암호화"],
    "연계": ["연계", "인터페이스", "API", "외부 시스템", "내/외부", "SSO", "연동"],
    "일정": ["착수", "중간보고", "최종보고", "일정", "단계", "M+", "개월", "일 이내"],
}

DOMAIN_RULES: dict[str, list[str]] = {
    "학사": ["학사", "수강", "교육과정", "졸업", "장학", "포털"],
    "ERP": ["ERP", "경영관리", "그룹웨어", "GW"],
    "상담": ["상담", "채팅", "챗봇", "콜센터"],
    "지도": ["지도", "공간정보", "지오코딩", "플랫폼"],
    "ISP": ["ISP", "정보화전략계획", "전략계획"],
    "AI": ["AI", "인공지능", "생성형", "지능정보", "머신러닝", "자연어"],
    "연구": ["연구비", "과제", "연구", "실험"],
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def clean_text(text: str) -> str:
    value = str(text or "").replace("\x00", " ")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"\n{3,}", "\n\n", value)
    value = re.sub(r"[ \t]{2,}", " ", value)
    return value.strip()


def choose_document_text(document: dict[str, Any]) -> str:
    rag_ready = clean_text(document.get("rag_ready_text", ""))
    if rag_ready:
        return rag_ready
    return clean_text(document.get("text", ""))


def split_text(text: str, chunk_size: int, overlap_size: int) -> list[str]:
    cleaned = clean_text(text)
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    minimum_split = max(200, int(chunk_size * 0.6))

    while start < len(cleaned):
        tentative_end = min(start + chunk_size, len(cleaned))
        if tentative_end >= len(cleaned):
            chunk = cleaned[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        search_floor = min(len(cleaned), start + minimum_split)
        split_candidates = [
            cleaned.rfind("\n\n", search_floor, tentative_end),
            cleaned.rfind("\n", search_floor, tentative_end),
            cleaned.rfind(". ", search_floor, tentative_end),
            cleaned.rfind("  ", search_floor, tentative_end),
            cleaned.rfind(" ", search_floor, tentative_end),
        ]
        split_point = max(split_candidates)
        if split_point <= start:
            split_point = tentative_end

        chunk = cleaned[start:split_point].strip()
        if chunk:
            chunks.append(chunk)

        next_start = max(split_point - overlap_size, start + 1)
        if next_start <= start:
            next_start = split_point
        start = next_start

    return chunks


def safe_str(value: Any, default: str = "정보 없음") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() == "none" or text.lower() == "nan":
        return default
    return text


def chunk_id_from_document_id(document_id: str, chunk_index: int) -> str:
    safe_document_id = re.sub(r"[^0-9A-Za-z가-힣-]+", "_", str(document_id))
    return f"{safe_document_id}__chunk_{chunk_index:04d}"


def coalesce_inline_value(value: str) -> str:
    text = clean_text(value).replace("\n", " ")
    text = re.split(
        r"\s+(?=(?:□|○|ㅇ)\s)|"
        r"\s+(?=(?:사업예산|소요예산|기초금액|추정금액|계약방식|사업추진방식|입찰방식|사업목적|사업목표|추진목적|추진배경|사업내용|과업내용)\s*[:：])",
        text,
        maxsplit=1,
    )[0]
    text = re.sub(r"\s+", " ", text).strip(" -:|")
    return text


def search_field_value(text: str, labels: list[str], max_chars: int = 160) -> str:
    label_pattern = "|".join(re.escape(label) for label in labels)
    match = re.search(rf"(?:{label_pattern})\s*[:：]\s*(.+)", text)
    if not match:
        return ""
    raw_value = match.group(1)[:max_chars]
    return coalesce_inline_value(raw_value)


def extract_contract_method(text: str) -> str:
    value = search_field_value(text, ["계약방식", "사업추진방식", "계약 방법"], max_chars=120)
    parts: list[str] = []
    if value:
        parts.append(value)
    for phrase in ["협상에 의한 계약", "수의계약", "총액계약"]:
        if phrase in text and all(phrase not in part for part in parts):
            parts.append(phrase)
    return " / ".join(parts[:3])


def extract_bid_method(text: str) -> str:
    value = search_field_value(text, ["입찰방식", "입찰 방법"], max_chars=120)
    parts: list[str] = []
    if value:
        parts.append(value)
    for phrase in ["제한경쟁입찰", "일반경쟁입찰", "제한 경쟁입찰", "긴급공고"]:
        if phrase in text and all(phrase not in part for part in parts):
            parts.append(phrase)
    return " / ".join(parts[:3])


def extract_period_raw(text: str) -> str:
    value = search_field_value(text, ["사업기간", "계약기간", "수행기간", "용역기간", "기 간"], max_chars=120)
    if value:
        return value
    for pattern in [
        r"계약체결일로부터[^\n]{0,80}",
        r"착수일로부터[^\n]{0,80}",
    ]:
        match = re.search(pattern, text)
        if match:
            return coalesce_inline_value(match.group(0))
    return ""


def extract_period_numbers(period_raw: str) -> tuple[int | None, int | None]:
    if not period_raw:
        return None, None
    month_match = re.search(r"(\d+)\s*개월", period_raw)
    day_match = re.search(r"(\d+)\s*일", period_raw)
    months = int(month_match.group(1)) if month_match else None
    days = int(day_match.group(1)) if day_match else None
    return days, months


def extract_purpose_summary(text: str) -> str:
    labels = ["사업목적", "사업목표", "추진목적", "추진배경", "추진배경 및 필요성"]
    for label in labels:
        match = re.search(rf"{re.escape(label)}\s*[:：]?\s*(.+)", text)
        if not match:
            continue
        candidate = text[match.start(): match.start() + 500]
        candidate = re.sub(rf"^{re.escape(label)}\s*[:：]?\s*", "", candidate)
        candidate = candidate.replace("\n", " ")
        candidate = re.split(r"\s+(?=(?:□|○|ㅇ)\s)|\s+(?=(?:사업내용|주요 과업내용|주요요구사항|과업내용)\s*[:：])", candidate, maxsplit=1)[0]
        candidate = re.sub(r"\s+", " ", candidate).strip(" -:")
        return candidate[:180]
    return ""


def detect_document_type(source_file_name: str, text: str) -> str:
    filename = source_file_name.lower()
    corpus = f"{source_file_name} {text[:1200]}".lower()
    if "isp" in corpus or "정보화전략계획" in text:
        return "ISP"
    if "컨설팅" in text or "컨설팅" in source_file_name:
        return "컨설팅"
    if "운영" in text or "유지보수" in text or "운영 용역" in source_file_name:
        return "운영/유지보수"
    if "구축" in text or "개발" in text:
        return "구축/개발"
    if filename.endswith(".pdf"):
        return "제안요청서/PDF"
    return "제안요청서/HWP"


def detect_domain_tags(source_file_name: str, text: str) -> str:
    corpus = f"{source_file_name} {text[:3000]}"
    tags = [tag for tag, keywords in DOMAIN_RULES.items() if any(keyword in corpus for keyword in keywords)]
    return "|".join(tags)


def detect_heading(chunk_text: str, primary_role: str) -> str:
    lines = [line.strip() for line in chunk_text.splitlines() if line.strip()]
    for line in lines[:20]:
        candidate = re.sub(r"\s+", " ", line).strip()
        if len(candidate) > 80:
            continue
        if re.match(r"^(?:제\s*\d+\s*[장절]|[0-9]+[.)]|[가-힣A-Za-z]+[.)]|[□■○ㅇ▶-])", candidate):
            return candidate[:80]
        if any(keyword in candidate for keyword in ["사업목적", "추진배경", "사업기간", "사업예산", "계약방식", "입찰방식", "요구사항", "평가기준"]):
            return candidate[:80]
    return primary_role if primary_role != "일반" else ""


def detect_chunk_roles(chunk_text: str) -> tuple[str, str, dict[str, int]]:
    scores: dict[str, int] = {}
    for role, keywords in ROLE_RULES.items():
        score = sum(1 for keyword in keywords if keyword in chunk_text)
        scores[role] = score

    matched_roles = [role for role, score in scores.items() if score > 0]
    if not matched_roles:
        return "일반", "", scores

    matched_roles.sort(key=lambda role: (-scores[role], role))
    primary_role = matched_roles[0]
    role_tags = "|".join(matched_roles)
    return primary_role, role_tags, scores


def detect_system_mentions(chunk_text: str) -> str:
    matches = re.findall(r"([가-힣A-Za-z0-9()·/\- ]{2,40}(?:시스템|플랫폼|ERP|GW|포털|챗봇|LMS|RCMS))", chunk_text)
    seen: list[str] = []
    for match in matches:
        cleaned = re.sub(r"\s+", " ", match).strip()
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
    return " | ".join(seen[:5])


def normalize_budget(metadata: dict[str, Any]) -> tuple[str, int | None]:
    raw_amount = safe_str(metadata.get("사업 금액"), default="")
    if not raw_amount:
        return "정보 없음", None
    try:
        amount = int(float(raw_amount))
    except ValueError:
        return raw_amount, None
    return f"{amount:,}원", amount


def build_document_fields(document: dict[str, Any], text: str) -> dict[str, Any]:
    metadata = document.get("metadata", {}) or {}
    budget_text, budget_amount_krw = normalize_budget(metadata)
    period_raw = extract_period_raw(text)
    period_days, period_months = extract_period_numbers(period_raw)

    return {
        "document_id": document.get("document_id", ""),
        "source_file_name": document.get("source_file_name", ""),
        "사업명": safe_str(metadata.get("사업명")),
        "발주 기관": safe_str(metadata.get("발주 기관")),
        "공고 번호": safe_str(metadata.get("공고 번호")),
        "공개 일자": safe_str(metadata.get("공개 일자")),
        "budget_text": budget_text,
        "budget_amount_krw": budget_amount_krw,
        "bid_deadline": safe_str(metadata.get("입찰 참여 마감일")),
        "period_raw": period_raw or "정보 없음",
        "period_days": period_days,
        "period_months": period_months,
        "contract_method": extract_contract_method(text) or "정보 없음",
        "bid_method": extract_bid_method(text) or "정보 없음",
        "purpose_summary": extract_purpose_summary(text) or "정보 없음",
        "document_type": detect_document_type(document.get("source_file_name", ""), text),
        "domain_tags": detect_domain_tags(document.get("source_file_name", ""), text),
        "table_marker_count": metadata.get("table_markers", 0),
        "enriched_table_count": metadata.get("enriched_table_count", 0),
    }


def build_prefix_v2(document_fields: dict[str, Any], chunk_fields: dict[str, Any]) -> str:
    lines = [
        "[문서 구조 요약]",
        f"- 사업명: {document_fields['사업명']}",
        f"- 발주기관: {document_fields['발주 기관']}",
        f"- 문서유형: {document_fields['document_type']}",
        f"- 도메인태그: {document_fields['domain_tags'] or '정보 없음'}",
        f"- 예산: {document_fields['budget_text']}",
        f"- 입찰마감일: {document_fields['bid_deadline']}",
        f"- 사업기간: {document_fields['period_raw']}",
        f"- 계약방식: {document_fields['contract_method']}",
        f"- 입찰방식: {document_fields['bid_method']}",
        f"- 핵심목적: {document_fields['purpose_summary']}",
        f"- 섹션제목: {chunk_fields['section_title'] or '정보 없음'}",
        f"- 청크역할: {chunk_fields['chunk_role']}",
        f"- 청크태그: {chunk_fields['chunk_role_tags'] or '정보 없음'}",
        f"- 표포함: {'예' if chunk_fields['has_table'] else '아니오'}",
        f"- 예산신호: {'예' if chunk_fields['has_budget_signal'] else '아니오'}",
        f"- 일정신호: {'예' if chunk_fields['has_schedule_signal'] else '아니오'}",
        f"- 계약신호: {'예' if chunk_fields['has_contract_signal'] else '아니오'}",
        f"- 주요 시스템: {chunk_fields['mentioned_systems'] or '정보 없음'}",
        "[/문서 구조 요약]",
    ]
    return "\n".join(lines)


def build_chunk_rows(document: dict[str, Any], chunk_texts: list[str], document_fields: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, chunk_text in enumerate(chunk_texts, start=1):
        has_table = int("<표>" in chunk_text or "TABLE BLOCK" in chunk_text or "| ---" in chunk_text)
        chunk_role, chunk_role_tags, role_scores = detect_chunk_roles(chunk_text)
        section_title = detect_heading(chunk_text, chunk_role)
        section_path = section_title
        mentioned_systems = detect_system_mentions(chunk_text)

        chunk_fields = {
            "section_title": section_title,
            "section_path": section_path,
            "chunk_role": chunk_role,
            "chunk_role_tags": chunk_role_tags,
            "has_table": has_table,
            "has_budget_signal": int(role_scores.get("예산", 0) > 0),
            "has_schedule_signal": int(role_scores.get("기간", 0) > 0 or role_scores.get("일정", 0) > 0),
            "has_contract_signal": int(role_scores.get("계약", 0) > 0),
            "mentioned_systems": mentioned_systems,
        }
        prefix_v2 = build_prefix_v2(document_fields, chunk_fields)
        contextual_chunk_text = f"{prefix_v2}\n\n[본문 청크]\n{chunk_text}\n[/본문 청크]"

        row = {
            "chunk_id": chunk_id_from_document_id(document.get("document_id", "문서"), index),
            "document_id": document.get("document_id"),
            "chunk_index": index,
            "source_file_name": document.get("source_file_name"),
            "source_path": document.get("source_path"),
            "source_extension": document.get("source_extension"),
            "사업명": document_fields["사업명"],
            "발주 기관": document_fields["발주 기관"],
            "공고 번호": document_fields["공고 번호"],
            "공개 일자": document_fields["공개 일자"],
            "파일형식": safe_str((document.get("metadata", {}) or {}).get("파일형식"), document.get("source_extension", "정보 없음")),
            "metadata_prefix": prefix_v2,
            "raw_chunk_text": chunk_text,
            "contextual_chunk_text": contextual_chunk_text,
            "raw_chunk_chars": len(chunk_text),
            "contextual_chunk_chars": len(contextual_chunk_text),
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
            "section_path": section_path,
            "chunk_role": chunk_role,
            "chunk_role_tags": chunk_role_tags,
            "has_table": has_table,
            "has_budget_signal": chunk_fields["has_budget_signal"],
            "has_schedule_signal": chunk_fields["has_schedule_signal"],
            "has_contract_signal": chunk_fields["has_contract_signal"],
            "mentioned_systems": mentioned_systems,
        }
        rows.append(row)
    return rows


def build_summary_rows(chunk_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in chunk_rows:
        grouped.setdefault(str(row["document_id"]), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for document_id, rows in grouped.items():
        raw_lengths = [int(row["raw_chunk_chars"]) for row in rows]
        contextual_lengths = [int(row["contextual_chunk_chars"]) for row in rows]
        first = rows[0]
        summary_rows.append(
            {
                "document_id": document_id,
                "source_file_name": first["source_file_name"],
                "사업명": first["사업명"],
                "발주 기관": first["발주 기관"],
                "document_type": first["document_type"],
                "domain_tags": first["domain_tags"],
                "chunk_count": len(rows),
                "raw_chunk_chars_avg": round(mean(raw_lengths), 2),
                "raw_chunk_chars_max": max(raw_lengths),
                "contextual_chunk_chars_avg": round(mean(contextual_lengths), 2),
                "contextual_chunk_chars_max": max(contextual_lengths),
                "table_chunk_count": sum(int(row["has_table"]) for row in rows),
                "budget_chunk_count": sum(int(row["has_budget_signal"]) for row in rows),
                "schedule_chunk_count": sum(int(row["has_schedule_signal"]) for row in rows),
                "contract_chunk_count": sum(int(row["has_contract_signal"]) for row in rows),
            }
        )
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="B-02 prefix-v2 청킹과 구조화 필드 추출을 수행합니다.")
    parser.add_argument("--입력경로", default=str(INPUT_JSONL_PATH), help="processed_documents.jsonl 경로")
    parser.add_argument("--출력경로", default=str(B02_CHUNK_JSONL_PATH), help="prefix-v2 chunk JSONL 경로")
    parser.add_argument("--요약CSV경로", default=str(B02_CHUNK_SUMMARY_CSV_PATH), help="chunk summary CSV 경로")
    parser.add_argument("--문서필드CSV경로", default=str(B02_DOCUMENT_FIELDS_CSV_PATH), help="document fields CSV 경로")
    parser.add_argument("--manifest경로", default=str(B02_MANIFEST_PATH), help="manifest JSON 경로")
    parser.add_argument("--청크크기", type=int, default=500, help="본문 기준 chunk size")
    parser.add_argument("--겹침크기", type=int, default=80, help="본문 기준 overlap size")
    parser.add_argument("--최대문서수", type=int, default=None, help="테스트용 문서 수 제한")
    args = parser.parse_args()

    input_path = Path(args.입력경로)
    if not input_path.exists():
        raise FileNotFoundError(f"입력 JSONL이 없습니다: {input_path}")

    documents = read_jsonl(input_path)
    if args.최대문서수 is not None:
        documents = documents[: args.최대문서수]

    all_chunk_rows: list[dict[str, Any]] = []
    document_field_rows: list[dict[str, Any]] = []
    skipped_documents = 0

    for document in documents:
        document_text = choose_document_text(document)
        if not document_text:
            skipped_documents += 1
            continue

        document_fields = build_document_fields(document, document_text)
        document_field_rows.append(document_fields)

        chunk_texts = split_text(document_text, chunk_size=args.청크크기, overlap_size=args.겹침크기)
        all_chunk_rows.extend(build_chunk_rows(document, chunk_texts, document_fields))

    if not all_chunk_rows:
        raise RuntimeError("생성된 prefix-v2 청크가 없습니다.")

    output_path = Path(args.출력경로)
    summary_path = Path(args.요약CSV경로)
    document_fields_path = Path(args.문서필드CSV경로)
    manifest_path = Path(args.manifest경로)

    write_jsonl(output_path, all_chunk_rows)
    write_csv(summary_path, build_summary_rows(all_chunk_rows))
    write_csv(document_fields_path, document_field_rows)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "summary_csv_path": str(summary_path),
                "document_fields_csv_path": str(document_fields_path),
                "document_count": len(documents),
                "skipped_documents": skipped_documents,
                "chunk_count": len(all_chunk_rows),
                "chunk_size": args.청크크기,
                "overlap_size": args.겹침크기,
                "prefix_version": "v2_structured_fields",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("[완료] B-02 prefix-v2 청킹이 끝났습니다.")
    print(f"- 입력 문서 수: {len(documents)}")
    print(f"- 건너뛴 문서 수: {skipped_documents}")
    print(f"- 생성 청크 수: {len(all_chunk_rows)}")
    print(f"- 청크 경로: {output_path}")
    print(f"- 문서 필드 CSV: {document_fields_path}")
    print(f"- 요약 CSV: {summary_path}")


if __name__ == "__main__":
    main()

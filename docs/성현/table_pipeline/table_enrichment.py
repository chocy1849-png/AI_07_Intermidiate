import re
from typing import List, Optional

Pair = tuple[str, str]


TABLE_FIELD_HINTS = (
    "요구사항",
    "요구",
    "사항",
    "분류",
    "고유번호",
    "ID",
    "명칭",
    "정의",
    "설명",
    "세부",
    "내용",
    "산출정보",
    "관련",
    "사용자",
    "기능",
    "구분",
    "내역",
    "수량",
    "비고",
    "장비",
    "지역",
    "대상",
    "기간",
    "일정",
    "금액",
    "배점",
    "기준",
)


def normalize_table_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def looks_like_table_label(line: str) -> bool:
    stripped = normalize_table_line(line)
    if not stripped:
        return False
    compact = re.sub(r"\s+", "", stripped)
    if len(compact) <= 10:
        return True
    if any(hint in stripped for hint in TABLE_FIELD_HINTS):
        return True
    if re.fullmatch(r"[A-Z]{2,5}[-_ ]?\d{2,4}", stripped):
        return True
    return False


def join_table_value_lines(lines: List[str]) -> str:
    normalized = []
    for line in lines:
        cleaned = normalize_table_line(line)
        if cleaned:
            normalized.append(cleaned)
    return " / ".join(normalized)


def extract_key_value_pairs(table_text: str) -> List[Pair]:
    lines = extract_table_lines(table_text)
    pairs: List[Pair] = []
    index = 0

    while index < len(lines) - 1:
        label = lines[index]
        if not looks_like_table_label(label):
            index += 1
            continue

        value_lines: List[str] = []
        cursor = index + 1
        while cursor < len(lines):
            candidate = lines[cursor]
            if value_lines and looks_like_table_label(candidate):
                break
            if not value_lines and candidate == label:
                break
            value_lines.append(candidate)
            cursor += 1
            if len(value_lines) >= 6:
                break

        value = join_table_value_lines(value_lines)
        if value and value != label:
            pairs.append((label, value))
            index = cursor
            continue

        index += 1

    deduped: List[Pair] = []
    seen = set()
    for label, value in pairs:
        key = (label, value)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((label, value))
    return deduped


def build_table_key_value_lines(table_text: str, pairs: Optional[List[Pair]] = None) -> List[str]:
    pairs = pairs or extract_key_value_pairs(table_text)
    return [f"{label}: {value}" for label, value in pairs[:8]]


def build_table_row_summary_lines(table_text: str, pairs: Optional[List[Pair]] = None) -> List[str]:
    pairs = pairs or extract_key_value_pairs(table_text)
    summaries: List[str] = []

    current: List[str] = []
    for label, value in pairs:
        current.append(f"{label}={value}")
        if len(current) == 3:
            summaries.append(" / ".join(current))
            current = []

    if current:
        summaries.append(" / ".join(current))

    return summaries[:4]


def extract_table_lines(table_text: str) -> List[str]:
    return [normalize_table_line(line) for line in table_text.splitlines() if normalize_table_line(line)]


def find_pair_value(pairs: List[Pair], *keywords: str) -> Optional[str]:
    for label, value in pairs:
        if any(keyword in label for keyword in keywords):
            return value
    return None


def detect_table_type(table: dict, table_text: str, pairs: List[Pair]) -> str:
    parent_text = normalize_table_line(table.get("linked_parent_text") or "")
    body = f"{parent_text}\n{table_text}"

    if any(keyword in body for keyword in ["장비", "교환기", "CTI", "IP녹취", "대상지역", "대상 지역", "중앙상황실", "광역상황실"]):
        return "equipment_region"
    if any(keyword in body for keyword in ["모바일오피스", "업무지원 앱", "업무전용앱", "탑재 앱", "앱 목록", "서비스 구성", "그룹포털", "전자결재", "모바일메신저", "VPN", "MDM"]):
        return "service_status"
    if any(keyword in body for keyword in ["AI", "삭제지원", "피해촬영물", "검출", "일치율", "판별", "기존 시스템", "기존 업무시스템", "한계", "통합 연계"]):
        return "ai_requirement"
    if any(keyword in body for keyword in ["요구사항 고유번호", "요구사항 ID", "요구사항 명칭"]) or find_pair_value(pairs, "요구사항 고유번호", "요구사항 ID"):
        return "requirement_table"
    if any(keyword in body for keyword in ["배점", "평가기준", "평가항목"]):
        return "score_table"
    if any(keyword in body for keyword in ["일정", "기간", "제출", "마감"]):
        return "schedule_table"
    return "general_table"


def append_if_value(lines: List[str], label: str, value: Optional[str], max_length: Optional[int] = None) -> None:
    if not value:
        return
    if max_length is not None:
        value = value[:max_length]
    lines.append(f"{label}: {value}")


DOC_TITLE_STOPWORDS = {
    "사업",
    "구축",
    "개선",
    "고도화",
    "용역",
    "운영",
    "시스템",
    "정보시스템",
    "통합",
    "재구축",
    "기능개선",
    "개발",
    "및",
}


def extract_doc_focus_terms(doc_title: str, limit: int = 3) -> List[str]:
    cleaned = re.sub(r"[\[\]\(\)_]", " ", doc_title)
    tokens = [token.strip() for token in re.split(r"\s+", cleaned) if token.strip()]
    focus_terms: List[str] = []

    for token in tokens:
        normalized = re.sub(r"[^0-9A-Za-z가-힣]+", "", token)
        if not normalized or len(normalized) <= 1:
            continue
        if normalized in DOC_TITLE_STOPWORDS:
            continue
        if normalized not in focus_terms:
            focus_terms.append(normalized)
        if len(focus_terms) >= limit:
            break

    return focus_terms


def build_doc_focus_prefix(doc_title: str) -> str:
    return " ".join(extract_doc_focus_terms(doc_title, limit=2)).strip()


def infer_table_title(table: dict, doc_title: str, table_type: str) -> str:
    parent_text = normalize_table_line(table.get("linked_parent_text") or "")
    if parent_text:
        return parent_text

    focus_prefix = build_doc_focus_prefix(doc_title)
    title_by_type = {
        "equipment_region": "도입 장비 및 대상 지역",
        "service_status": "기존 기능 및 서비스 구성 현황",
        "ai_requirement": "AI 기반 기능 요구사항",
        "requirement_table": "요구사항 상세 표",
        "score_table": "평가 및 배점 표",
        "schedule_table": "일정 및 기간 표",
        "general_table": f"{doc_title} 구조형 표",
    }
    base_title = title_by_type.get(table_type, f"{doc_title} 구조형 표")
    if focus_prefix and focus_prefix not in base_title:
        return f"{focus_prefix} {base_title}".strip()
    return base_title


def collect_matching_lines(table_text: str, keywords: List[str], limit: int = 3) -> List[str]:
    matches: List[str] = []
    for line in extract_table_lines(table_text):
        if any(keyword in line for keyword in keywords):
            matches.append(line)
        if len(matches) >= limit:
            break
    return matches


def summarize_terms(table_text: str, terms: List[str], limit: int = 6) -> str:
    found: List[str] = []
    for term in terms:
        if term in table_text and term not in found:
            found.append(term)
        if len(found) >= limit:
            break
    return ", ".join(found)


def build_type_template_summary(table_type: str, table_text: str, pairs: List[Pair]) -> List[str]:
    lines: List[str] = []

    if table_type == "equipment_region":
        id_value = find_pair_value(pairs, "요구사항 고유번호", "요구사항 ID")
        name_value = find_pair_value(pairs, "요구사항 명칭", "명칭")
        equipment_terms = summarize_terms(table_text, ["CTI", "교환기", "IP녹취", "IP전화기", "게이트웨이", "서버", "SIP", "PSTN"])
        region_terms = summarize_terms(table_text, ["중앙", "서울", "대전", "대구", "광주"])
        append_if_value(lines, "요구사항 ID", id_value)
        append_if_value(lines, "요구사항 명칭", name_value)
        append_if_value(lines, "장비 목록", equipment_terms)
        append_if_value(lines, "대상 지역", region_terms)
        return lines

    if table_type == "service_status":
        service_terms = summarize_terms(table_text, ["모바일오피스", "업무전용앱", "업무지원 앱", "VPN", "MDM", "메신저", "그룹채팅", "전화걸기"])
        purpose_lines = collect_matching_lines(table_text, ["통합", "이관", "폐지", "사용", "기능"], limit=2)
        append_if_value(lines, "주요 서비스/앱", service_terms)
        lines.extend(f"현황 요약: {line}" for line in purpose_lines)
        return lines

    if table_type == "ai_requirement":
        feature_lines = collect_matching_lines(table_text, ["AI", "삭제지원", "검출", "검색", "판별", "일치율"], limit=4)
        lines.extend(f"AI 요구사항: {line}" for line in feature_lines)
        return lines

    if table_type == "requirement_table":
        id_value = find_pair_value(pairs, "요구사항 고유번호", "요구사항 ID")
        name_value = find_pair_value(pairs, "요구사항 명칭", "명칭")
        user_value = find_pair_value(pairs, "사용자")
        detail_value = find_pair_value(pairs, "세부", "내용", "정의")
        append_if_value(lines, "요구사항 ID", id_value)
        append_if_value(lines, "요구사항 명칭", name_value)
        append_if_value(lines, "대상 사용자", user_value)
        append_if_value(lines, "핵심 내용", detail_value, max_length=220)
        return lines

    return []


def build_comparison_summary(table_type: str, table_text: str) -> List[str]:
    lines: List[str] = []

    if table_type == "equipment_region":
        equipment_terms = summarize_terms(table_text, ["CTI", "교환기", "IP녹취", "IP전화기", "게이트웨이", "호처리 서버", "SIP", "PSTN"])
        region_terms = summarize_terms(table_text, ["중앙", "서울", "대전", "대구", "광주"])
        if equipment_terms:
            lines.append(f"도입 장비 축: {equipment_terms}")
        if region_terms:
            lines.append(f"지역 축: {region_terms}")

    if table_type == "service_status":
        current_lines = collect_matching_lines(table_text, ["기존", "현황", "이관", "통합", "폐지"], limit=3)
        lines.extend(f"서비스 구성 축: {line}" for line in current_lines)

    if table_type == "ai_requirement":
        problem_lines = collect_matching_lines(table_text, ["검색", "검출", "삭제지원", "판별", "일치율"], limit=3)
        lines.extend(f"해결 기능 축: {line}" for line in problem_lines)

    return lines[:4]


DOC_HINT_KEYWORDS = {
    "equipment_region": ["도입장비", "도입 장비", "대상지역", "대상 지역", "CTI", "교환기", "IP녹취", "중앙", "서울", "대전", "대구", "광주"],
    "service_status": ["모바일오피스", "업무전용앱", "업무지원 앱", "서비스 구성", "기존 기능", "기존", "이관", "통합", "폐지", "VPN", "MDM"],
    "ai_requirement": ["AI", "삭제지원", "기존 시스템", "한계", "문제", "피해촬영물", "검출", "검색", "판별", "통합 연계"],
}


def extract_doc_context_hints(payload: dict) -> dict[str, List[str]]:
    hints: dict[str, List[str]] = {key: [] for key in DOC_HINT_KEYWORDS}
    candidate_blocks: List[str] = []

    for block in payload.get("explanatory_blocks", []):
        text = (block.get("text") or "").strip()
        if text:
            candidate_blocks.append(text)

    for block in payload.get("section_header_blocks", []):
        text = (block.get("text") or "").strip()
        if text:
            candidate_blocks.append(text)

    for candidate in candidate_blocks:
        normalized_lines = extract_table_lines(candidate)
        joined = "\n".join(normalized_lines)
        for hint_type, keywords in DOC_HINT_KEYWORDS.items():
            if not any(keyword in joined for keyword in keywords):
                continue
            for line in normalized_lines:
                if any(keyword in line for keyword in keywords):
                    if line not in hints[hint_type]:
                        hints[hint_type].append(line)
                if len(hints[hint_type]) >= 4:
                    break

    return hints


def build_doc_context_blocks(doc_title: str, payload: dict) -> List[str]:
    hint_map = extract_doc_context_hints(payload)
    blocks: List[str] = []

    for hint_type, lines in hint_map.items():
        if not lines:
            continue
        block_lines = [
            "[DOC_CONTEXT_HINTS]",
            f"[DOC_TITLE] {doc_title}",
            f"[HINT_TYPE] {hint_type}",
            *lines[:4],
        ]
        blocks.append("\n".join(block_lines).strip())

    return blocks


def build_hint_summary_lines(table_type: str, hint_map: dict[str, List[str]]) -> List[str]:
    lines = hint_map.get(table_type, [])
    return lines[:3]


def build_doc_focus_hint_lines(doc_title: str, table_type: str) -> List[str]:
    focus_terms = extract_doc_focus_terms(doc_title, limit=3)
    if not focus_terms:
        return []

    label_by_type = {
        "equipment_region": "장비/구성 문맥",
        "service_status": "서비스 현황 문맥",
        "ai_requirement": "AI 요구사항 문맥",
        "requirement_table": "요구사항 문맥",
        "score_table": "평가/배점 문맥",
        "schedule_table": "일정/산출물 문맥",
        "general_table": "문서 문맥",
    }
    label = label_by_type.get(table_type, "문서 문맥")
    return [f"{label}: {' / '.join(focus_terms)}"]


def build_table_bridge_line(doc_title: str, table: dict) -> str | None:
    parent_text = normalize_table_line(table.get("linked_parent_text") or "")
    if parent_text:
        return f"이 표는 {doc_title} 문서의 '{parent_text}' 관련 내용을 구조적으로 정리한 것이다."
    return f"이 표는 {doc_title} 문서에서 추출한 구조형 표이다."


def build_table_block(table: dict, doc_title: str, hint_map: Optional[dict[str, List[str]]] = None) -> str | None:
    body = (table.get("text") or "").strip()
    if not body:
        return None
    pairs = extract_key_value_pairs(body)
    table_type = detect_table_type(table, body, pairs)
    table_title = infer_table_title(table, doc_title, table_type)

    prefixes: List[str] = ["[STRUCTURAL_TABLE]"]
    final_classification = table.get("final_classification")
    if final_classification:
        prefixes.append(f"[{final_classification}]")
    if table.get("ocr_candidate_priority") in {"high", "medium"}:
        prefixes.append(f"[OCR_CANDIDATE:{table['ocr_candidate_priority']}]")

    header = " ".join(prefixes)
    lines = [
        header,
        f"[DOC_TITLE] {doc_title}",
        f"[TABLE_INDEX] {table['table_index']}",
        f"[TABLE_TITLE] {table_title}",
        f"[TABLE_TYPE] {table_type}",
    ]

    if table.get("linked_parent_text"):
        lines.append(f"[SECTION_HEADER] {table['linked_parent_text']}")

    bridge_line = build_table_bridge_line(doc_title, table)
    if bridge_line:
        lines.append(f"[TABLE_CONTEXT] {bridge_line}")

    type_template_lines = build_type_template_summary(table_type, body, pairs)
    if type_template_lines:
        lines.append("[TYPE_TEMPLATE_SUMMARY]")
        lines.extend(type_template_lines)

    comparison_lines = build_comparison_summary(table_type, body)
    if comparison_lines:
        lines.append("[COMPARISON_SUMMARY]")
        lines.extend(comparison_lines)

    if hint_map:
        hint_lines = build_hint_summary_lines(table_type, hint_map)
        if hint_lines:
            lines.append("[DOC_HINT_SUMMARY]")
            lines.extend(hint_lines)

    focus_hint_lines = build_doc_focus_hint_lines(doc_title, table_type)
    if focus_hint_lines:
        lines.append("[DOC_FOCUS_SUMMARY]")
        lines.extend(focus_hint_lines)

    key_value_lines = build_table_key_value_lines(body, pairs)
    if key_value_lines:
        lines.append("[KEY_VALUE_SUMMARY]")
        lines.extend(key_value_lines)

    row_summary_lines = build_table_row_summary_lines(body, pairs)
    if row_summary_lines:
        lines.append("[ROW_SUMMARY]")
        lines.extend(row_summary_lines)

    lines.append("[RAW_TABLE_TEXT]")
    lines.append(body)
    return "\n".join(lines).strip()

import argparse
import json
import struct
import sys
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import olefile

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


TABLE_CONTROL_ID = "tbl "
GRAPHIC_CONTROL_ID = "gso "
TABLE_RELATED_TYPES = {66, 67, 68, 69, 72, 77}

TOC_KEYWORDS = (
    "목차",
    "차례",
    "추진개요",
    "추진방안",
    "추진내용",
    "입찰관련사항",
    "제안요청내용",
    "제안서작성요령",
    "별지서식",
    "붙임",
)

COVER_KEYWORDS = (
    "제안요청서",
    "제안서",
    "용역사업",
    "사업명",
    "사업개요",
    "재단",
    "협회",
    "공사",
    "대학교",
    "주식회사",
)

DATA_TABLE_KEYWORDS = (
    "구분",
    "항목",
    "내용",
    "비고",
    "금액",
    "예산",
    "평가",
    "배점",
    "일정",
    "기간",
    "제출",
    "자격",
    "요구사항",
    "기능",
    "수량",
    "단가",
    "합계",
)

HEADER_KEYWORDS = (
    "구분",
    "항목",
    "내용",
    "비고",
    "금액",
    "배점",
    "평가기준",
    "요구사항",
    "요구사항 ID",
    "요구사항 명칭",
    "수량",
    "단가",
    "합계",
)

BULLET_MARKERS = ("□", "○", "-", "◈", "※", "ㆍ")

HEADER_GROUPS = (
    ("구분", "항목", "내용"),
    ("구분", "내용", "비고"),
    ("요구사항", "ID", "명칭"),
    ("요구사항", "분류", "명칭"),
    ("평가", "배점", "기준"),
    ("평가항목", "배점", "평가기준"),
    ("일정", "기간", "비고"),
    ("예산", "금액", "합계"),
    ("수량", "단가", "금액"),
    ("구분", "역할", "비고"),
    ("번호", "보안약점", "설명"),
    ("구분", "항목", "적용계획"),
    ("적용계획", "미적용", "사유"),
)

SECOND_PASS_KEYWORD_GROUPS = {
    "basic_headers": ("구분", "항목", "내용", "비고"),
    "requirements_eval": ("요구사항", "평가", "배점", "기준", "평가기준"),
    "schedule_amount": ("일정", "기간", "예산", "금액", "합계", "수량", "단가"),
    "form_fields": ("업체명", "대표자", "주소", "전화번호", "접수번호", "처리일", "제출서류"),
    "roles_org": ("역할", "담당", "부서", "책임자", "PM"),
    "system_map": ("신청", "보고", "결과", "조회", "자료실", "통계", "출력", "완료", "보류"),
    "software_stack": ("상용SW", "버전", "서버", "DB", "검색엔진", "프레임워크", "WAS"),
    "security_plan": ("보안약점", "적용계획", "미적용", "사유", "인증", "암호화"),
}


@dataclass
class HwpTableArtifact:
    table_index: int
    section: str
    record_start_index: int
    row_hint_count: int
    paragraph_count: int
    text_length: int
    first_pass_classification: str
    toc_score: int
    cover_score: int
    data_score: int
    header_group_hits: int
    missing_signals: list[str]
    second_pass_bonus: int
    second_pass_reason: list[str]
    final_classification: str
    storage_bucket: str
    section_header_role: str | None
    linked_parent_table_index: int | None
    linked_parent_text: str | None
    linked_child_table_indices: list[int]
    ocr_candidate_score: int
    ocr_candidate_priority: str
    ocr_candidate_reasons: list[str]
    text: str


@dataclass
class HwpImageArtifact:
    name: str
    ext: str
    size_bytes: int
    saved_path: str | None
    ocr_candidate_score: int
    ocr_candidate_priority: str
    ocr_candidate_reasons: list[str]


def decode_para_text(payload: bytes) -> str:
    chars: list[str] = []
    j = 0
    while j < len(payload) - 1:
        code = struct.unpack_from("<H", payload, j)[0]

        if code in (1, 2, 3, 11, 12, 14, 15, 16, 17, 18, 21, 22, 23):
            j += 14
            continue
        if code in (4, 5, 6, 7, 8, 9, 19, 20):
            j += 10
            continue
        if code in (10, 13):
            chars.append("\n")
            j += 2
            continue
        if code == 24:
            chars.append("\t")
            j += 14
            continue
        if code == 0 or 0xD800 <= code <= 0xDFFF:
            j += 2
            continue

        chars.append(chr(code))
        j += 2

    text = "".join(chars)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def normalize_for_match(text: str) -> str:
    return "".join(text.replace("\x1f", "").split())


def count_keyword_hits(text: str, keywords: tuple[str, ...]) -> int:
    normalized_text = normalize_for_match(text)
    return sum(max(text.count(keyword), normalized_text.count(normalize_for_match(keyword))) for keyword in keywords)


def count_group_hits(text: str) -> dict[str, int]:
    normalized_text = normalize_for_match(text)
    hits: dict[str, int] = {}
    for group_name, keywords in SECOND_PASS_KEYWORD_GROUPS.items():
        hits[group_name] = sum(1 for keyword in keywords if normalize_for_match(keyword) in normalized_text)
    return hits


def count_outline_lines(lines: list[str]) -> int:
    count = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("Ⅰ.", "Ⅱ.", "Ⅲ.", "Ⅳ.", "Ⅴ.", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")):
            count += 1
    return count


def count_numeric_lines(lines: list[str]) -> int:
    count = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if any(char.isdigit() for char in stripped):
            count += 1
    return count


def classify_table_text(text: str, row_hint_count: int, paragraph_count: int) -> tuple[str, int, int, int]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    normalized_text = normalize_for_match(text)
    group_hits = count_group_hits(text)

    toc_score = count_keyword_hits(text, TOC_KEYWORDS)
    cover_score = count_keyword_hits(text, COVER_KEYWORDS)
    data_score = count_keyword_hits(text, DATA_TABLE_KEYWORDS)

    outline_lines = count_outline_lines(lines)
    numeric_lines = count_numeric_lines(lines)
    header_hit_count = sum(1 for keyword in HEADER_KEYWORDS if normalize_for_match(keyword) in normalized_text)
    bullet_line_count = sum(1 for line in lines if line.startswith(BULLET_MARKERS))
    long_line_count = sum(1 for line in lines if len(line) >= 40)
    header_group_hits = max(
        sum(1 for keyword in group if normalize_for_match(keyword) in normalized_text) for group in HEADER_GROUPS
    )
    active_group_count = sum(1 for hit in group_hits.values() if hit > 0)
    strongest_group_hit = max(group_hits.values(), default=0)

    if outline_lines >= 3:
        toc_score += 2
    if any(line == "목차" or line == "차례" for line in lines):
        toc_score += 3

    if paragraph_count <= 4 and row_hint_count <= 5 and normalize_for_match("제안요청서") in normalized_text:
        cover_score += 2
    if paragraph_count <= 4 and any("2024." in line or "2025." in line or "2026." in line for line in lines):
        cover_score += 1

    if row_hint_count >= 3:
        data_score += 1
    if numeric_lines >= max(3, len(lines) // 2):
        data_score += 1
    if any(normalize_for_match(keyword) in normalized_text for keyword in ("구분", "항목", "내용", "비고")):
        data_score += 2

    if toc_score >= 4:
        return "toc", toc_score, cover_score, data_score, header_group_hits
    if cover_score >= 4 and data_score <= 3:
        return "cover", toc_score, cover_score, data_score, header_group_hits

    # Gate-based table classification:
    # - sure_table: strong structural signals + a full 3-of-3 header group match.
    # - review_needed: enough structure + a 2-of-3 header group match.
    passes_structure_gate = row_hint_count >= 3 and paragraph_count >= 3
    passes_strict_structure_gate = row_hint_count >= 5 and paragraph_count >= 4
    passes_content_gate = data_score >= 4
    passes_field_gate = data_score >= 3 and (active_group_count >= 2 or strongest_group_hit >= 3)
    looks_like_explanatory_block = bullet_line_count >= 2 and long_line_count >= 2 and header_hit_count < 2

    if (
        passes_strict_structure_gate
        and header_group_hits >= 3
        and passes_content_gate
        and not looks_like_explanatory_block
    ):
        return "sure_table", toc_score, cover_score, data_score, header_group_hits

    if passes_structure_gate and header_group_hits >= 2 and passes_content_gate:
        return "review_needed", toc_score, cover_score, data_score, header_group_hits

    # Some useful business tables are field-heavy rather than header-heavy.
    # We keep them in review if structure exists and multiple field groups repeat.
    if (
        row_hint_count >= 5
        and paragraph_count >= 4
        and header_group_hits >= 1
        and passes_field_gate
        and not looks_like_explanatory_block
    ):
        return "review_needed", toc_score, cover_score, data_score, header_group_hits

    if (
        row_hint_count >= 8
        and paragraph_count >= 6
        and active_group_count >= 2
        and data_score >= 3
        and not looks_like_explanatory_block
    ):
        return "review_needed", toc_score, cover_score, data_score, header_group_hits

    return "uncertain", toc_score, cover_score, data_score, header_group_hits


def analyze_missing_signals(
    text: str,
    first_pass_classification: str,
    row_hint_count: int,
    paragraph_count: int,
    data_score: int,
    header_group_hits: int,
) -> tuple[list[str], int, list[str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    normalized_text = normalize_for_match(text)
    group_hits = count_group_hits(text)
    total_group_hits = sum(group_hits.values())
    bullet_line_count = sum(1 for line in lines if line.startswith(BULLET_MARKERS))
    long_line_count = sum(1 for line in lines if len(line) >= 40)
    short_label_lines = sum(1 for line in lines if 1 <= len(line) <= 12)
    arrow_lines = sum(1 for line in lines if "→" in line or "->" in line)
    label_keywords = (
        "대민사이트",
        "BackOffice",
        "추가사항",
        "추진목표",
        "추진과제",
        "추진일정",
        "추진체계",
        "역할",
        "구분",
    )
    label_keyword_hits = sum(1 for keyword in label_keywords if normalize_for_match(keyword) in normalized_text)
    first_lines_compact = "".join(normalize_for_match(line) for line in lines[:3])
    stacked_label_hit = any(
        keyword in first_lines_compact for keyword in ("추진목표", "추진과제", "추진일정", "추진체계및역할", "추진체계")
    )

    missing_signals: list[str] = []
    reasons: list[str] = []
    bonus = 0

    if row_hint_count >= 5 and data_score >= 4 and header_group_hits < 2:
        missing_signals.append("weak_header_match")
        reasons.append("row hints are strong but header match is weak")
        bonus += 2

    if (short_label_lines >= 3 and label_keyword_hits >= 1) or stacked_label_hit:
        missing_signals.append("left_label_column_pattern")
        reasons.append("short label-like lines repeat with category keywords")
        bonus += 2

    if row_hint_count >= 6 and (arrow_lines >= 1 or label_keyword_hits >= 2 or stacked_label_hit):
        missing_signals.append("box_layout_like")
        reasons.append("layout looks like a boxed or diagram-like table")
        bonus += 2

    if total_group_hits >= 6 and sum(1 for hit in group_hits.values() if hit > 0) >= 2:
        missing_signals.append("important_field_block")
        reasons.append("multiple business field groups appear strongly in the same block")
        bonus += 2

    if group_hits["system_map"] >= 4:
        missing_signals.append("system_map_like")
        reasons.append("system/menu map terms repeat across the block")
        bonus += 2

    if (
        "제도안내" in normalized_text
        or "신청방법" in normalized_text
        or ("결과" in normalized_text and "보고" in normalized_text)
    ) and group_hits["system_map"] >= 2:
        missing_signals.append("menu_flow_like")
        reasons.append("menu or workflow labels repeat like a system map")
        bonus += 2

    if group_hits["form_fields"] >= 3:
        missing_signals.append("form_field_like")
        reasons.append("structured business form fields repeat across the block")
        bonus += 2

    if (
        row_hint_count >= 5
        and (
            group_hits["form_fields"] >= 2
            or group_hits["schedule_amount"] >= 2
            or group_hits["requirements_eval"] >= 2
            or group_hits["roles_org"] >= 2
        )
    ):
        missing_signals.append("field_value_table_like")
        reasons.append("field-value pairs repeat with enough row structure to look tabular")
        bonus += 2

    if "제출서류" in normalized_text and ("목록" in normalized_text or "제출" in normalized_text) and short_label_lines >= 4:
        missing_signals.append("submission_list_like")
        reasons.append("submission document list looks like a structured checklist table")
        bonus += 2

    overview_keywords = ("사업개요", "사업명", "사업기간", "소요예산", "계약방식", "선정절차")
    overview_hits = sum(1 for keyword in overview_keywords if keyword in normalized_text)
    if overview_hits >= 3:
        missing_signals.append("project_overview_like")
        reasons.append("project overview fields repeat in a structured block")
        bonus += 2

    if group_hits["schedule_amount"] >= 3 and row_hint_count >= 3:
        missing_signals.append("schedule_amount_like")
        reasons.append("schedule or amount fields repeat with table-like structure")
        bonus += 2

    if ("대상업체" in normalized_text or "참가자격" in normalized_text) and ("사업금액" in normalized_text or "매출액" in normalized_text):
        missing_signals.append("eligibility_threshold_like")
        reasons.append("eligibility and threshold fields repeat in a structured qualification block")
        bonus += 2

    if group_hits["software_stack"] >= 3 and row_hint_count >= 3:
        missing_signals.append("software_stack_like")
        reasons.append("software or infrastructure inventory terms repeat across the block")
        bonus += 2

    if ("PM" in normalized_text or "사업관리자" in normalized_text) and "부문" in normalized_text:
        missing_signals.append("staffing_plan_like")
        reasons.append("staffing or responsibility fields repeat in a plan block")
        bonus += 2

    if (
        bullet_line_count >= 2
        and long_line_count >= 2
        and header_group_hits < 2
        and label_keyword_hits == 0
        and total_group_hits < 6
        and overview_hits < 3
    ):
        missing_signals.append("explanatory_block")
        reasons.append("bullet-heavy explanatory prose dominates the block")
        bonus -= 2

    if first_pass_classification == "review_needed" and header_group_hits >= 2 and row_hint_count >= 5:
        missing_signals.append("promotable_review_table")
        reasons.append("review table has enough structure to be promoted on second pass")
        bonus += 1

    return missing_signals, bonus, reasons


def second_pass_classify(
    first_pass_classification: str,
    missing_signals: list[str],
    second_pass_bonus: int,
) -> str:
    if first_pass_classification in {"cover", "toc"}:
        return first_pass_classification

    if first_pass_classification == "sure_table":
        return "final_sure_table"

    if first_pass_classification == "review_needed":
        if second_pass_bonus >= 1 and "explanatory_block" not in missing_signals:
            return "final_sure_table"
        if "explanatory_block" in missing_signals:
            return "explanatory_block"
        return "final_review_table"

    if first_pass_classification == "uncertain":
        has_structural_recovery_signal = any(
            signal in missing_signals
            for signal in (
                "left_label_column_pattern",
                "box_layout_like",
                "weak_header_match",
                "important_field_block",
                "system_map_like",
                "menu_flow_like",
                "form_field_like",
                "submission_list_like",
                "project_overview_like",
                "schedule_amount_like",
                "eligibility_threshold_like",
                "software_stack_like",
                "staffing_plan_like",
                "field_value_table_like",
            )
        )
        if second_pass_bonus >= 2 and has_structural_recovery_signal and "explanatory_block" not in missing_signals:
            return "final_review_table"
        if "explanatory_block" in missing_signals:
            return "explanatory_block"
        return "discarded_table"

    return "discarded_table"


def classify_storage_bucket(final_classification: str) -> str:
    if final_classification in {"final_sure_table", "final_review_table"}:
        return "structural_table"
    if final_classification == "explanatory_block":
        return "explanatory_block"
    if final_classification in {"cover", "toc"}:
        return "excluded"
    return "discarded"


def classify_candidate_priority(score: int) -> str:
    if score >= 8:
        return "high"
    if score >= 4:
        return "medium"
    return "low"


def score_table_ocr_candidate(table: HwpTableArtifact) -> tuple[int, str, list[str]]:
    reasons: list[str] = []
    score = 0

    if table.storage_bucket != "structural_table":
        return 0, "low", reasons

    # Tables already promoted to review usually need visual recovery more than
    # fully structured tables, so they are our default OCR-first targets.
    if table.final_classification == "final_review_table":
        score += 4
        reasons.append("review table needs visual recovery more than fully structured tables")
    elif table.final_classification == "final_sure_table":
        score += 1
        reasons.append("structured table is already usable, so OCR need starts low")

    if table.header_group_hits <= 1 and table.row_hint_count >= 5:
        score += 3
        reasons.append("row structure is strong but header recovery is weak")
    elif table.header_group_hits == 2 and table.row_hint_count >= 5:
        score += 1
        reasons.append("table is mostly structured but still has partial header ambiguity")

    structural_signals = {
        "weak_header_match": 2,
        "left_label_column_pattern": 2,
        "box_layout_like": 3,
        "menu_flow_like": 3,
        "field_value_table_like": 2,
        "schedule_amount_like": 2,
        "submission_list_like": 2,
        "project_overview_like": 2,
        "software_stack_like": 2,
        "staffing_plan_like": 2,
        "eligibility_threshold_like": 2,
    }
    for signal, bonus in structural_signals.items():
        if signal in table.missing_signals:
            score += bonus
            reasons.append(f"missing signal detected: {signal}")

    if table.linked_parent_text:
        score += 1
        reasons.append("section header is attached, which usually means this table belongs to a larger structured set")

    if table.text_length >= 300:
        score += 1
        reasons.append("table text is large enough that structure loss can meaningfully hurt retrieval")

    priority = classify_candidate_priority(score)
    return score, priority, reasons


def score_image_ocr_candidate(image: HwpImageArtifact) -> tuple[int, str, list[str]]:
    reasons: list[str] = []
    score = 0

    if image.size_bytes >= 300_000:
        score += 5
        reasons.append("large embedded image is more likely to be a meaningful diagram or screen capture")
    elif image.size_bytes >= 100_000:
        score += 3
        reasons.append("medium-size embedded image is worth checking for OCR")
    elif image.size_bytes >= 30_000:
        score += 1
        reasons.append("image is not tiny, so it may still contain readable content")

    if image.ext in {"png", "jpg", "jpeg", "bmp"}:
        score += 1
        reasons.append("common raster image format is suitable for OCR")

    if image.name.lower().startswith(("bin", "image", "img")):
        score += 1
        reasons.append("generic embedded image name suggests a document-inserted visual rather than metadata")

    priority = classify_candidate_priority(score)
    return score, priority, reasons


def is_section_header_candidate(table: HwpTableArtifact) -> bool:
    lines = [line.strip() for line in table.text.splitlines() if line.strip()]
    normalized_text = normalize_for_match(table.text)
    if table.storage_bucket != "discarded":
        return False
    if len(lines) > 3:
        return False
    if table.text_length > 80:
        return False

    header_terms = (
        "요구사항",
        "정의표",
        "추진내용",
        "추진개요",
        "추진방안",
        "제안요청",
        "기능요구사항",
        "성능요구사항",
        "데이터요구사항",
        "보안요구사항",
        "테스트요구사항",
        "인터페이스요구사항",
        "프로젝트관리요구사항",
        "프로젝트지원요구사항",
    )
    has_header_term = any(term in normalized_text for term in header_terms)
    looks_like_compact_header = table.paragraph_count <= 3 and table.row_hint_count <= 4
    return has_header_term and looks_like_compact_header


def attach_section_header_links(tables: list[HwpTableArtifact]) -> None:
    for idx, table in enumerate(tables):
        if not is_section_header_candidate(table):
            continue

        child_indices: list[int] = []
        for candidate in tables[idx + 1 : idx + 6]:
            if candidate.section != table.section:
                break
            if is_section_header_candidate(candidate):
                break
            if candidate.record_start_index - table.record_start_index > 1000:
                break
            if candidate.storage_bucket == "structural_table":
                child_indices.append(candidate.table_index)
                if candidate.linked_parent_table_index is None:
                    candidate.linked_parent_table_index = table.table_index
                    candidate.linked_parent_text = table.text.replace("\n", " / ")
            elif candidate.storage_bucket == "discarded" and candidate.paragraph_count <= 2 and candidate.text_length <= 80:
                # Allow one more compact title/subtitle block before the actual tables.
                continue
            else:
                break

        if child_indices:
            table.section_header_role = "section_header_block"
            table.linked_child_table_indices = child_indices
            table.storage_bucket = "section_header_block"


def build_table_artifact(
    tables: list[HwpTableArtifact],
    active_table: dict,
    text: str,
) -> HwpTableArtifact:
    classification, toc_score, cover_score, data_score, header_group_hits = classify_table_text(
        text=text,
        row_hint_count=active_table["row_hints"],
        paragraph_count=len(active_table["paragraphs"]),
    )
    missing_signals, second_pass_bonus, second_pass_reason = analyze_missing_signals(
        text=text,
        first_pass_classification=classification,
        row_hint_count=active_table["row_hints"],
        paragraph_count=len(active_table["paragraphs"]),
        data_score=data_score,
        header_group_hits=header_group_hits,
    )
    final_classification = second_pass_classify(
        first_pass_classification=classification,
        missing_signals=missing_signals,
        second_pass_bonus=second_pass_bonus,
    )
    placeholder = HwpTableArtifact(
        table_index=len(tables) + 1,
        section=active_table["section"],
        record_start_index=active_table["record_start_index"],
        row_hint_count=active_table["row_hints"],
        paragraph_count=len(active_table["paragraphs"]),
        text_length=len(text),
        first_pass_classification=classification,
        toc_score=toc_score,
        cover_score=cover_score,
        data_score=data_score,
        header_group_hits=header_group_hits,
        missing_signals=missing_signals,
        second_pass_bonus=second_pass_bonus,
        second_pass_reason=second_pass_reason,
        final_classification=final_classification,
        storage_bucket=classify_storage_bucket(final_classification),
        section_header_role=None,
        linked_parent_table_index=None,
        linked_parent_text=None,
        linked_child_table_indices=[],
        ocr_candidate_score=0,
        ocr_candidate_priority="low",
        ocr_candidate_reasons=[],
        text=text,
    )
    score, priority, reasons = score_table_ocr_candidate(placeholder)
    placeholder.ocr_candidate_score = score
    placeholder.ocr_candidate_priority = priority
    placeholder.ocr_candidate_reasons = reasons
    return placeholder


def iter_hwp_records(file_path: Path):
    ole = olefile.OleFileIO(str(file_path))
    header_data = ole.openstream("FileHeader").read()
    is_compressed = (header_data[36] & 1) == 1

    try:
        for entry in ole.listdir():
            if not entry or entry[0] != "BodyText":
                continue

            section_name = "/".join(entry)
            data = ole.openstream(section_name).read()
            if is_compressed:
                data = zlib.decompress(data, -15)

            i = 0
            record_index = 0
            while i + 4 <= len(data):
                header = struct.unpack_from("<I", data, i)[0]
                rec_type = header & 0x3FF
                rec_len = (header >> 20) & 0xFFF
                if rec_len == 0xFFF:
                    if i + 8 > len(data):
                        break
                    rec_len = struct.unpack_from("<I", data, i + 4)[0]
                    i += 4

                payload = data[i + 4 : i + 4 + rec_len]
                control_id = ""
                if rec_type in (71, 76):
                    control_id = payload[:4].decode("ascii", errors="ignore")[::-1]

                yield {
                    "section": section_name,
                    "record_index": record_index,
                    "type": rec_type,
                    "length": rec_len,
                    "control_id": control_id,
                    "payload": payload,
                }

                i += 4 + rec_len
                record_index += 1
    finally:
        ole.close()


def extract_hwp_tables(file_path: Path) -> list[HwpTableArtifact]:
    tables: list[HwpTableArtifact] = []
    active_table: dict | None = None

    for record in iter_hwp_records(file_path):
        rec_type = record["type"]
        control_id = record["control_id"]

        if rec_type == 71 and control_id == TABLE_CONTROL_ID:
            if active_table and active_table["paragraphs"]:
                text = "\n".join(active_table["paragraphs"]).strip()
                tables.append(build_table_artifact(tables, active_table, text))

            active_table = {
                "section": record["section"],
                "record_start_index": record["record_index"],
                "row_hints": 0,
                "paragraphs": [],
            }
            continue

        if active_table is None:
            continue

        # Table-related records continue the current table body. Any other
        # control header usually means the next object started, so we flush.
        if rec_type == 71 and control_id != TABLE_CONTROL_ID:
            if active_table["paragraphs"]:
                text = "\n".join(active_table["paragraphs"]).strip()
                tables.append(build_table_artifact(tables, active_table, text))
            active_table = None
            continue

        if rec_type not in TABLE_RELATED_TYPES:
            if active_table["paragraphs"]:
                text = "\n".join(active_table["paragraphs"]).strip()
                tables.append(build_table_artifact(tables, active_table, text))
            active_table = None
            continue

        if rec_type == 72:
            active_table["row_hints"] += 1
        elif rec_type == 67:
            text = decode_para_text(record["payload"])
            if text:
                active_table["paragraphs"].append(text)

    if active_table and active_table["paragraphs"]:
        text = "\n".join(active_table["paragraphs"]).strip()
        tables.append(build_table_artifact(tables, active_table, text))

    attach_section_header_links(tables)
    for table in tables:
        score, priority, reasons = score_table_ocr_candidate(table)
        table.ocr_candidate_score = score
        table.ocr_candidate_priority = priority
        table.ocr_candidate_reasons = reasons
    return tables


def extract_hwp_images(file_path: Path, save_dir: Path | None = None) -> list[HwpImageArtifact]:
    ole = olefile.OleFileIO(str(file_path))
    images: list[HwpImageArtifact] = []

    try:
        for entry in ole.listdir(streams=True, storages=False):
            if not entry or entry[0] != "BinData":
                continue

            stream_name = "/".join(entry)
            name = entry[-1]
            ext = Path(name).suffix.lower().lstrip(".") or "bin"
            data = ole.openstream(stream_name).read()

            saved_path: str | None = None
            if save_dir is not None:
                save_dir.mkdir(parents=True, exist_ok=True)
                image_path = save_dir / name
                image_path.write_bytes(data)
                saved_path = str(image_path.resolve())

            temp_image = HwpImageArtifact(
                name=name,
                ext=ext,
                size_bytes=len(data),
                saved_path=saved_path,
                ocr_candidate_score=0,
                ocr_candidate_priority="low",
                ocr_candidate_reasons=[],
            )
            image_score, image_priority, image_reasons = score_image_ocr_candidate(temp_image)

            images.append(
                HwpImageArtifact(
                    name=name,
                    ext=ext,
                    size_bytes=len(data),
                    saved_path=saved_path,
                    ocr_candidate_score=image_score,
                    ocr_candidate_priority=image_priority,
                    ocr_candidate_reasons=image_reasons,
                )
            )
    finally:
        ole.close()

    return images


def extract_hwp_artifacts(file_path: Path, save_images: bool = False, output_dir: Path | None = None) -> dict:
    image_dir = output_dir / "hwp_images" if save_images and output_dir is not None else None
    tables = extract_hwp_tables(file_path)
    images = extract_hwp_images(file_path, save_dir=image_dir)
    table_ocr_candidates = [
        asdict(table)
        for table in tables
        if table.storage_bucket == "structural_table" and table.ocr_candidate_score >= 4
    ]
    image_ocr_candidates = [asdict(image) for image in images if image.ocr_candidate_score >= 4]

    return {
        "source_path": str(file_path.resolve()),
        "source_name": file_path.name,
        "file_type": ".hwp",
        "status": "ok",
        "table_count": len(tables),
        "toc_table_count": sum(1 for table in tables if table.first_pass_classification == "toc"),
        "cover_table_count": sum(1 for table in tables if table.first_pass_classification == "cover"),
        "sure_table_count": sum(1 for table in tables if table.first_pass_classification == "sure_table"),
        "review_needed_count": sum(1 for table in tables if table.first_pass_classification == "review_needed"),
        "uncertain_table_count": sum(1 for table in tables if table.first_pass_classification == "uncertain"),
        "final_sure_table_count": sum(1 for table in tables if table.final_classification == "final_sure_table"),
        "final_review_table_count": sum(1 for table in tables if table.final_classification == "final_review_table"),
        "explanatory_block_count": sum(1 for table in tables if table.final_classification == "explanatory_block"),
        "discarded_table_count": sum(1 for table in tables if table.final_classification == "discarded_table"),
        "structural_table_count": sum(1 for table in tables if table.storage_bucket == "structural_table"),
        "section_header_block_count": sum(1 for table in tables if table.storage_bucket == "section_header_block"),
        "excluded_table_count": sum(1 for table in tables if table.storage_bucket == "excluded"),
        "image_count": len(images),
        "table_ocr_candidate_count": len(table_ocr_candidates),
        "image_ocr_candidate_count": len(image_ocr_candidates),
        "structural_tables": [asdict(table) for table in tables if table.storage_bucket == "structural_table"],
        "section_header_blocks": [asdict(table) for table in tables if table.storage_bucket == "section_header_block"],
        "explanatory_blocks": [asdict(table) for table in tables if table.storage_bucket == "explanatory_block"],
        "discarded_tables": [asdict(table) for table in tables if table.storage_bucket == "discarded"],
        "excluded_tables": [asdict(table) for table in tables if table.storage_bucket == "excluded"],
        "table_ocr_candidates": table_ocr_candidates,
        "image_ocr_candidates": image_ocr_candidates,
        "tables": [asdict(table) for table in tables],
        "images": [asdict(image) for image in images],
    }


def summarize(payload: dict) -> str:
    lines = [
        f"source: {payload['source_name']}",
        f"tables: {payload['table_count']}",
        f"  sure_table: {payload['sure_table_count']}",
        f"  review_needed: {payload['review_needed_count']}",
        f"  toc: {payload['toc_table_count']}",
        f"  cover: {payload['cover_table_count']}",
        f"  uncertain: {payload['uncertain_table_count']}",
        f"  final_sure_table: {payload['final_sure_table_count']}",
        f"  final_review_table: {payload['final_review_table_count']}",
        f"  section_header_block: {payload['section_header_block_count']}",
        f"  explanatory_block: {payload['explanatory_block_count']}",
        f"  discarded_table: {payload['discarded_table_count']}",
        f"  structural_table: {payload['structural_table_count']}",
        f"  table_ocr_candidate: {payload['table_ocr_candidate_count']}",
        f"  excluded_table: {payload['excluded_table_count']}",
        f"images: {payload['image_count']}",
        f"  image_ocr_candidate: {payload['image_ocr_candidate_count']}",
        "",
    ]

    for table in payload["tables"][:3]:
        lines.append(
            f"table {table['table_index']}: first={table['first_pass_classification']} final={table['final_classification']} "
            f"section={table['section']} row_hints={table['row_hint_count']} "
            f"paragraphs={table['paragraph_count']} text_length={table['text_length']} "
            f"(toc={table['toc_score']}, cover={table['cover_score']}, data={table['data_score']}, header_hits={table['header_group_hits']}, bonus={table['second_pass_bonus']}, bucket={table['storage_bucket']}, ocr_score={table['ocr_candidate_score']}, ocr_priority={table['ocr_candidate_priority']})"
        )
        preview = table["text"][:200].replace("\n", " / ")
        lines.append(f"   preview: {preview}")
        if table["missing_signals"]:
            lines.append(f"   missing_signals: {', '.join(table['missing_signals'])}")
        if table["ocr_candidate_reasons"]:
            lines.append(f"   ocr_candidate_reasons: {', '.join(table['ocr_candidate_reasons'][:3])}")

    for index, image in enumerate(payload["images"][:5], start=1):
        lines.append(
            f"image {index}: {image['name']} ext={image['ext']} size={image['size_bytes']} "
            f"ocr_score={image['ocr_candidate_score']} ocr_priority={image['ocr_candidate_priority']} saved={image['saved_path']}"
        )

    return "\n".join(lines)


def safe_console(text: str) -> str:
    return text.encode("cp949", errors="replace").decode("cp949")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract table and image artifacts directly from an HWP file.")
    parser.add_argument("--file", type=Path, required=True, help="HWP file path.")
    parser.add_argument("--json", type=Path, default=None, help="Optional output JSON path.")
    parser.add_argument("--save-images", action="store_true", help="Save BinData images to disk.")
    args = parser.parse_args()

    output_dir = args.json.parent if args.json is not None else Path("ocr/output")
    payload = extract_hwp_artifacts(args.file, save_images=args.save_images, output_dir=output_dir)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(safe_console(summarize(payload)))


if __name__ == "__main__":
    main()

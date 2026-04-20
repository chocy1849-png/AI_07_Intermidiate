import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rag_system.config import SETTINGS
from rag_system.rag_utils import clean_text, extract_text, iter_source_files


# These keywords represent table-heavy business information that is usually
# valuable for RAG question answering in RFP documents.
IMPORTANT_TABLE_KEYWORDS = {
    "예산": 3.0,
    "사업비": 3.0,
    "금액": 2.0,
    "평가": 2.5,
    "평가기준": 3.0,
    "평가 기준": 3.0,
    "배점": 3.0,
    "일정": 2.0,
    "기간": 1.5,
    "마감": 2.5,
    "제출": 2.0,
    "제출서류": 3.0,
    "제출 서류": 3.0,
    "제출방법": 3.0,
    "제출 방법": 3.0,
    "자격": 2.0,
    "참가자격": 3.0,
    "참가 자격": 3.0,
    "요건": 2.0,
    "요구사항": 2.5,
    "요구 사항": 2.5,
    "기능": 1.5,
    "범위": 1.5,
}

# These are common "form-like" sections that often contain many tables but are
# less useful as first-priority OCR targets.
FORM_KEYWORDS = {
    "별지": 2.0,
    "서식": 2.0,
    "첨부": 1.0,
    "일반현황": 2.0,
    "일반 현황": 2.0,
    "참여인력": 2.5,
    "참여 인력": 2.5,
    "서명": 2.0,
    "날인": 2.0,
    "확약서": 2.5,
    "이력서": 2.0,
    "실적증명": 2.5,
    "실적 증명": 2.5,
    "사용인감": 2.0,
}

MULTISPACE_PATTERN = re.compile(r"\S(?:\s{2,}|\t)\S")
NUMBER_HEAVY_PATTERN = re.compile(r"(\d[\d,./-]*\s+){2,}\d[\d,./-]*")
LISTING_PATTERN = re.compile(r"^(\S+\s{2,}){2,}\S+$")
SHORT_TOKEN_SPLIT_PATTERN = re.compile(r"[ \t]{2,}")


@dataclass
class DocumentPriority:
    source_path: str
    source_name: str
    line_count: int
    text_length: int
    table_structure_signal: float
    important_table_keyword_score: float
    form_penalty: float
    document_complexity_bonus: float
    priority_score: float
    reasons: list[str]


def weighted_keyword_score(text: str, keywords: dict[str, float], max_count_per_keyword: int = 8) -> tuple[float, list[str]]:
    score = 0.0
    matched: list[str] = []
    lowered = text.lower()
    for keyword, weight in keywords.items():
        count = lowered.count(keyword.lower())
        if count <= 0:
            continue
        capped_count = min(count, max_count_per_keyword)
        score += capped_count * weight
        matched.append(f"{keyword}x{capped_count}")
    return score, matched


def score_table_structure(lines: list[str]) -> tuple[float, list[str], list[int]]:
    score = 0.0
    reasons: list[str] = []
    structured_indices: list[int] = []

    multispace_lines = 0
    listing_lines = 0
    number_heavy_lines = 0
    short_column_lines = 0

    for line in lines:
        stripped = line.strip()
        if len(stripped) < 6:
            continue

        if MULTISPACE_PATTERN.search(stripped):
            multispace_lines += 1
            structured_indices.append(len(structured_indices))

        if NUMBER_HEAVY_PATTERN.search(stripped):
            number_heavy_lines += 1

        if LISTING_PATTERN.match(stripped):
            listing_lines += 1

        tokens = [token for token in SHORT_TOKEN_SPLIT_PATTERN.split(stripped) if token]
        short_tokens = [token for token in tokens if len(token) <= 12]
        if len(tokens) >= 3 and len(short_tokens) >= 3:
            short_column_lines += 1
    # Re-scan to keep real line indices for context extraction.
    structured_indices = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if len(stripped) < 6:
            continue
        looks_structured = (
            MULTISPACE_PATTERN.search(stripped)
            or NUMBER_HEAVY_PATTERN.search(stripped)
            or LISTING_PATTERN.match(stripped)
            or (
                len([token for token in SHORT_TOKEN_SPLIT_PATTERN.split(stripped) if token]) >= 3
                and len([token for token in SHORT_TOKEN_SPLIT_PATTERN.split(stripped) if token and len(token) <= 12]) >= 3
            )
        )
        if looks_structured:
            structured_indices.append(index)

    score += multispace_lines * 1.0
    score += listing_lines * 1.5
    score += number_heavy_lines * 1.7
    score += short_column_lines * 1.3

    if multispace_lines:
        reasons.append(f"column-like lines {multispace_lines}")
    if listing_lines:
        reasons.append(f"repeated row lines {listing_lines}")
    if number_heavy_lines:
        reasons.append(f"number-heavy lines {number_heavy_lines}")
    if short_column_lines:
        reasons.append(f"short-token rows {short_column_lines}")

    return score, reasons, structured_indices


def build_table_context_text(lines: list[str], structured_indices: list[int], window: int = 1) -> str:
    selected_indices: set[int] = set()
    for index in structured_indices:
        start = max(0, index - window)
        end = min(len(lines), index + window + 1)
        selected_indices.update(range(start, end))
    return "\n".join(lines[index] for index in sorted(selected_indices))


def score_document_complexity(lines: list[str], text_length: int) -> tuple[float, str | None]:
    non_empty_lines = [line for line in lines if line.strip()]
    line_bonus = min(len(non_empty_lines) / 40.0, 8.0)
    text_bonus = min(text_length / 5000.0, 4.0)
    score = round(line_bonus + text_bonus, 2)
    if score <= 0:
        return score, None
    return score, f"complexity bonus {score}"


def analyze_document(file_path: Path) -> DocumentPriority:
    raw_text = extract_text(file_path)
    cleaned_text = clean_text(raw_text)
    lines = cleaned_text.splitlines()

    table_signal_score, structure_reasons, structured_indices = score_table_structure(lines)
    table_context_text = build_table_context_text(lines, structured_indices)
    important_score, important_matches = weighted_keyword_score(table_context_text, IMPORTANT_TABLE_KEYWORDS)
    form_penalty, form_matches = weighted_keyword_score(cleaned_text, FORM_KEYWORDS)
    complexity_bonus, complexity_reason = score_document_complexity(lines, len(cleaned_text))

    priority_score = (
        table_signal_score * 0.5
        + important_score * 0.4
        + complexity_bonus * 0.1
        - form_penalty * 0.2
    )

    reasons: list[str] = []
    reasons.extend(structure_reasons[:2])
    if important_matches:
        reasons.append(f"important keywords {', '.join(important_matches[:4])}")
    if form_matches:
        reasons.append(f"form penalty {', '.join(form_matches[:3])}")
    if complexity_reason:
        reasons.append(complexity_reason)
    if not reasons:
        reasons.append("no strong table signals detected")

    return DocumentPriority(
        source_path=str(file_path.resolve()),
        source_name=file_path.name,
        line_count=len([line for line in lines if line.strip()]),
        text_length=len(cleaned_text),
        table_structure_signal=round(table_signal_score, 2),
        important_table_keyword_score=round(important_score, 2),
        form_penalty=round(form_penalty, 2),
        document_complexity_bonus=round(complexity_bonus, 2),
        priority_score=round(priority_score, 2),
        reasons=reasons,
    )


def format_report(priorities: list[DocumentPriority], top_k: int) -> str:
    lines = ["Top priority documents", ""]
    for index, item in enumerate(priorities[:top_k], start=1):
        lines.append(
            f"{index}. {item.source_name} | total={item.priority_score} | "
            f"structure={item.table_structure_signal} | important={item.important_table_keyword_score} | "
            f"penalty={item.form_penalty}"
        )
        lines.append(f"   why: {', '.join(item.reasons)}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank documents by table-first OCR priority.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top documents to print.")
    parser.add_argument("--json", type=Path, default=None, help="Optional output JSON path.")
    args = parser.parse_args()

    source_files = list(iter_source_files(SETTINGS.base_dir))
    if not source_files:
        raise FileNotFoundError(f"No PDF or HWP files found in: {SETTINGS.base_dir}")

    priorities = sorted(
        (analyze_document(file_path) for file_path in source_files),
        key=lambda item: item.priority_score,
        reverse=True,
    )

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps([asdict(item) for item in priorities], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(format_report(priorities, top_k=args.top_k))


if __name__ == "__main__":
    main()

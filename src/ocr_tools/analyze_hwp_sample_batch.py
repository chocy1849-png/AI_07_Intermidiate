import argparse
import json
import re
from collections import Counter
from pathlib import Path

from extract_hwp_artifacts import extract_hwp_artifacts


ROOT_DIR = Path(__file__).resolve().parent.parent
FILES_DIR = ROOT_DIR / "files"
DEFAULT_OUTPUT = ROOT_DIR / "ocr" / "hwp_sample_batch_summary.json"

# These substrings intentionally cover different document styles:
# requirements/evaluation, security, finance/form fields, ISP planning,
# workflow/system map, and survey/input forms.
SAMPLE_PATTERNS = (
    "벤처기업협회",
    "국방과학연구소_기록관리시스템",
    "한국사학진흥재단",
    "인천일자리플랫폼 정보시스템 구축 ISP",
    "보험개발원_실손보험 청구 전산화",
    "한국연구재단_2024년 대학산학협력활동 실태조사",
)

KEYWORD_GROUPS = {
    "basic_headers": ("구분", "항목", "내용", "비고"),
    "requirements_eval": ("요구사항", "평가", "배점", "기준", "평가기준"),
    "schedule_amount": ("일정", "기간", "예산", "금액", "합계", "수량", "단가"),
    "form_fields": ("업체명", "대표자", "주소", "전화번호", "접수번호", "처리일", "제출서류"),
    "roles_org": ("역할", "담당", "부서", "책임자", "PM"),
    "system_map": ("신청", "보고", "결과", "조회", "자료실", "통계", "출력", "완료", "보류"),
    "software_stack": ("상용SW", "버전", "서버", "DB", "검색엔진", "프레임워크", "WAS"),
    "security_plan": ("보안약점", "적용계획", "미적용", "사유", "인증", "암호화"),
}


def normalize(text: str) -> str:
    return "".join(text.replace("\x1f", "").split())


def select_sample_files(files_dir: Path) -> list[Path]:
    all_hwps = sorted(files_dir.glob("*.hwp"))
    selected: list[Path] = []
    for pattern in SAMPLE_PATTERNS:
        match = next((path for path in all_hwps if pattern in path.name), None)
        if match is not None:
            selected.append(match)
    return selected


def keyword_group_hits(text: str) -> dict[str, int]:
    normalized_text = normalize(text)
    hits: dict[str, int] = {}
    for group_name, keywords in KEYWORD_GROUPS.items():
        hits[group_name] = sum(1 for keyword in keywords if normalize(keyword) in normalized_text)
    return hits


def extract_top_tokens(text: str) -> list[str]:
    normalized_text = text.replace("\x1f", " ")
    tokens = re.findall(r"[가-힣A-Za-z][가-힣A-Za-z0-9·()/-]{1,}", normalized_text)
    filtered = [token for token in tokens if len(token) >= 2]
    counts = Counter(filtered)
    return [token for token, _ in counts.most_common(12)]


def summarize_file(file_path: Path) -> dict:
    payload = extract_hwp_artifacts(file_path, save_images=False, output_dir=ROOT_DIR / "ocr")
    interesting_candidates = []

    for table in payload["tables"]:
        group_hits = keyword_group_hits(table["text"])
        total_group_hits = sum(group_hits.values())
        if table["final_classification"] in {"discarded_table", "final_review_table"} and total_group_hits > 0:
            interesting_candidates.append(
                {
                    "table_index": table["table_index"],
                    "first_pass_classification": table["first_pass_classification"],
                    "final_classification": table["final_classification"],
                    "row_hint_count": table["row_hint_count"],
                    "paragraph_count": table["paragraph_count"],
                    "data_score": table["data_score"],
                    "header_group_hits": table["header_group_hits"],
                    "second_pass_bonus": table["second_pass_bonus"],
                    "missing_signals": table["missing_signals"],
                    "keyword_group_hits": group_hits,
                    "top_tokens": extract_top_tokens(table["text"]),
                    "preview": table["text"][:280].replace("\n", " / "),
                }
            )

    interesting_candidates.sort(
        key=lambda item: (
            -sum(item["keyword_group_hits"].values()),
            -item["data_score"],
            -item["row_hint_count"],
        )
    )

    return {
        "source_name": payload["source_name"],
        "source_path": payload["source_path"],
        "table_count": payload["table_count"],
        "image_count": payload["image_count"],
        "first_pass": {
            "sure_table": payload["sure_table_count"],
            "review_needed": payload["review_needed_count"],
            "toc": payload["toc_table_count"],
            "cover": payload["cover_table_count"],
            "uncertain": payload["uncertain_table_count"],
        },
        "second_pass": {
            "final_sure_table": payload["final_sure_table_count"],
            "final_review_table": payload["final_review_table_count"],
            "discarded_table": payload["discarded_table_count"],
        },
        "top_remaining_candidates": interesting_candidates[:15],
    }


def aggregate_group_stats(file_summaries: list[dict]) -> dict:
    final_review_counter: Counter[str] = Counter()
    discarded_counter: Counter[str] = Counter()

    for summary in file_summaries:
        for candidate in summary["top_remaining_candidates"]:
            target_counter = (
                final_review_counter
                if candidate["final_classification"] == "final_review_table"
                else discarded_counter
            )
            for group_name, hit_count in candidate["keyword_group_hits"].items():
                target_counter[group_name] += hit_count

    return {
        "final_review_group_hits": dict(final_review_counter.most_common()),
        "discarded_group_hits": dict(discarded_counter.most_common()),
    }


def format_console_report(report: dict) -> str:
    lines = [
        f"sample files: {len(report['files'])}",
        "",
    ]

    for file_summary in report["files"]:
        lines.append(f"source: {file_summary['source_name']}")
        lines.append(
            "  first_pass: "
            f"sure={file_summary['first_pass']['sure_table']} "
            f"review={file_summary['first_pass']['review_needed']} "
            f"uncertain={file_summary['first_pass']['uncertain']} "
            f"toc={file_summary['first_pass']['toc']} "
            f"cover={file_summary['first_pass']['cover']}"
        )
        lines.append(
            "  second_pass: "
            f"final_sure={file_summary['second_pass']['final_sure_table']} "
            f"final_review={file_summary['second_pass']['final_review_table']} "
            f"discarded={file_summary['second_pass']['discarded_table']}"
        )
        for candidate in file_summary["top_remaining_candidates"][:3]:
            lines.append(
                f"  candidate {candidate['table_index']}: "
                f"first={candidate['first_pass_classification']} "
                f"final={candidate['final_classification']} "
                f"groups={candidate['keyword_group_hits']} "
                f"signals={candidate['missing_signals']}"
            )
            lines.append(f"    preview: {candidate['preview']}")
        lines.append("")

    lines.append("aggregate discarded group hits:")
    lines.append(f"  {report['aggregate']['discarded_group_hits']}")
    lines.append("aggregate final_review group hits:")
    lines.append(f"  {report['aggregate']['final_review_group_hits']}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HWP table extraction on a fixed sample set and summarize common patterns.")
    parser.add_argument("--json", type=Path, default=DEFAULT_OUTPUT, help="Output JSON summary path.")
    args = parser.parse_args()

    sample_files = select_sample_files(FILES_DIR)
    file_summaries = [summarize_file(path) for path in sample_files]

    report = {
        "sample_patterns": list(SAMPLE_PATTERNS),
        "files": file_summaries,
        "aggregate": aggregate_group_stats(file_summaries),
    }

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(format_console_report(report).encode("cp949", errors="replace").decode("cp949"))


if __name__ == "__main__":
    main()

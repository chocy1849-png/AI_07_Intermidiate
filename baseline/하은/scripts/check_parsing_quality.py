from __future__ import annotations

import json
import random
import re
from datetime import datetime
from statistics import mean

from config import REPORTS_DIR, ensure_directories
from src.db.metadata_store import get_document_by_file_name
from src.db.parsed_store import load_parsed_documents


WEIRD_CHAR_PATTERN = re.compile(r"[^가-힣A-Za-z0-9\s\.,;:()\[\]{}\-/~\"'“”‘’·&<>%=+*#?!ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ○□※]")


def _weird_ratio(text: str) -> float:
    if not text:
        return 0.0
    weird = len(WEIRD_CHAR_PATTERN.findall(text))
    return round(weird / len(text), 4)


def _section_hits(text: str) -> int:
    return len(re.findall(r"(?m)^(?:\d+\.\s+|[가-하]\.\s+|Ⅰ\.|Ⅱ\.|Ⅲ\.|제\s*\d+\s*조)", text))


def build_parsing_quality_report(sample_size: int = 10, seed: int = 42) -> dict:
    ensure_directories()
    documents = load_parsed_documents()
    sampled = random.Random(seed).sample(documents, min(sample_size, len(documents)))

    samples = []
    for document in sampled:
        metadata = document["metadata"]
        text = document["text"]
        csv_row = get_document_by_file_name(metadata["file_name"]) or {}
        ratio = _weird_ratio(text)
        section_count = _section_hits(text)
        csv_length = len(csv_row.get("csv_text", "") or "")
        parsed_length = len(text)

        if ratio <= 0.01 and section_count >= 5:
            status = "양호"
        elif ratio <= 0.03:
            status = "검토 필요"
        else:
            status = "주의"

        samples.append(
            {
                "file_name": metadata["file_name"],
                "file_format": metadata["file_format"],
                "parser": metadata["parser"],
                "parsed_chars": parsed_length,
                "csv_chars": csv_length,
                "length_gain_x": round(parsed_length / max(csv_length, 1), 2),
                "weird_char_ratio": ratio,
                "section_hits": section_count,
                "status": status,
                "excerpt": text[:180].replace("\n", " "),
            }
        )

    issue_summary = [
        "HWP 문서는 전반적으로 본문 추출이 양호하지만, 표 셀 구조는 줄 단위 텍스트로 평탄화된다.",
        "일부 PDF는 MuPDF 경고가 발생했으나 폴백 없이도 결과가 생성되었고, 현재는 예외 시 pdfplumber 폴백이 준비되어 있다.",
        "CSV 텍스트 대비 원본 파싱 텍스트 길이는 대체로 수 배 이상 증가해, 검색/청킹에는 parsed JSON을 우선 사용해야 한다.",
    ]
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sample_size": len(samples),
        "average_weird_char_ratio": round(mean(item["weird_char_ratio"] for item in samples), 4) if samples else 0.0,
        "average_length_gain_x": round(mean(item["length_gain_x"] for item in samples), 2) if samples else 0.0,
        "issue_summary": issue_summary,
    }
    report = {"summary": summary, "samples": samples}
    (REPORTS_DIR / "parsing_quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    markdown = [
        "# Parsing Quality Report",
        "",
        f"- 생성 시각: {summary['generated_at']}",
        f"- 표본 수: {summary['sample_size']}",
        f"- 평균 이상문자 비율: {summary['average_weird_char_ratio']}",
        f"- 평균 길이 증가 배수(parsed/csv): {summary['average_length_gain_x']}",
        "",
        "## 주요 이슈",
        "",
    ]
    markdown.extend(f"- {item}" for item in issue_summary)
    markdown.extend(["", "## 샘플 점검 결과", ""])
    markdown.extend(
        f"- {item['file_name']}: 상태={item['status']}, 이상문자비율={item['weird_char_ratio']}, "
        f"섹션헤더={item['section_hits']}, 길이배수={item['length_gain_x']}"
        for item in samples
    )
    (REPORTS_DIR / "parsing_quality_report.md").write_text("\n".join(markdown), encoding="utf-8")
    return report


if __name__ == "__main__":
    build_parsing_quality_report()

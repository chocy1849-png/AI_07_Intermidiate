from __future__ import annotations

import json
import signal
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from statistics import mean
from time import perf_counter

from config import REPORTS_DIR, ensure_directories
from src.db.metadata_store import list_source_documents
from src.parser.hwp_parser import parse_hwp
from src.parser.pdf_parser import _parse_with_pdfplumber, _parse_with_pymupdf

HWP5TXT_PATH = Path("/Users/apple/workspace/codeit22/.venv/bin/hwp5txt")


def _attempt_import(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


class ParserTimeoutError(TimeoutError):
    """Raised when parser comparison exceeds the allotted timeout."""


def _run_with_timeout(fn, path: Path, timeout_sec: int = 10) -> tuple[str, int, float, str | None]:
    def handler(signum, frame):  # pragma: no cover - signal-based runtime path
        raise ParserTimeoutError(f"Timed out after {timeout_sec}s")

    started = perf_counter()
    previous_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout_sec)
    try:
        text, pages = fn(path)
        duration = round(perf_counter() - started, 3)
        return text, pages, duration, None
    except Exception as exc:  # pragma: no cover - runtime reporting path
        duration = round(perf_counter() - started, 3)
        return "", 0, duration, f"{type(exc).__name__}: {exc}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def _run_hwp5txt(path: Path, timeout_sec: int = 10) -> tuple[str, float, str | None]:
    started = perf_counter()
    try:
        completed = subprocess.run(
            [str(HWP5TXT_PATH), str(path)],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        duration = round(perf_counter() - started, 3)
        if completed.returncode != 0:
            return "", duration, completed.stderr.strip() or f"returncode={completed.returncode}"
        return completed.stdout, duration, None
    except Exception as exc:  # pragma: no cover - runtime reporting path
        duration = round(perf_counter() - started, 3)
        return "", duration, f"{type(exc).__name__}: {exc}"


def build_parser_comparison_report() -> dict:
    ensure_directories()
    documents = list_source_documents()
    pdf_docs = [doc for doc in documents if doc["file_format"] == "pdf"]
    hwp_docs = [doc for doc in documents if doc["file_format"] == "hwp"]

    pdf_rows = []
    for document in pdf_docs:
        path = Path(document["source_path"])
        print(f"comparing pdf parsers: {document['file_name']}")
        pymupdf_text, pymupdf_pages, pymupdf_time, pymupdf_error = _run_with_timeout(_parse_with_pymupdf, path)
        plumber_text, plumber_pages, plumber_time, plumber_error = _run_with_timeout(_parse_with_pdfplumber, path)

        pdf_rows.append(
            {
                "file_name": document["file_name"],
                "pymupdf_chars": len(pymupdf_text),
                "pymupdf_pages": pymupdf_pages,
                "pymupdf_time_sec": pymupdf_time,
                "pymupdf_error": pymupdf_error,
                "pdfplumber_chars": len(plumber_text),
                "pdfplumber_pages": plumber_pages,
                "pdfplumber_time_sec": plumber_time,
                "pdfplumber_error": plumber_error,
                "recommended": "pymupdf" if len(pymupdf_text) >= len(plumber_text) else "pdfplumber",
            }
        )

    hwp_rows = []
    for document in hwp_docs[:10]:
        path = Path(document["source_path"])
        started = perf_counter()
        parsed = parse_hwp(path)
        olefile_time = round(perf_counter() - started, 3)
        hwp5txt_text, hwp5txt_time, hwp5txt_error = _run_hwp5txt(path)
        hwp_rows.append(
            {
                "file_name": document["file_name"],
                "olefile_chars": parsed["metadata"]["char_count"],
                "olefile_time_sec": olefile_time,
                "hwp5txt_chars": len(hwp5txt_text),
                "hwp5txt_time_sec": hwp5txt_time,
                "hwp5txt_error": hwp5txt_error,
                "recommended": "olefile" if parsed["metadata"]["char_count"] >= len(hwp5txt_text) else "hwp5txt",
            }
        )

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pdf_document_count": len(pdf_docs),
        "hwp_document_count": len(hwp_docs),
        "pdf_average_pymupdf_time_sec": round(mean(row["pymupdf_time_sec"] for row in pdf_rows), 3) if pdf_rows else 0.0,
        "pdf_average_pdfplumber_time_sec": round(mean(row["pdfplumber_time_sec"] for row in pdf_rows), 3) if pdf_rows else 0.0,
        "hwp_candidates": {
            "olefile": True,
            "pyhwp": _attempt_import("hwp5"),
            "hwp5txt_cli": HWP5TXT_PATH.exists(),
        },
        "hwp_average_olefile_time_sec": round(mean(row["olefile_time_sec"] for row in hwp_rows), 3) if hwp_rows else 0.0,
        "hwp_average_hwp5txt_time_sec": round(mean(row["hwp5txt_time_sec"] for row in hwp_rows), 3) if hwp_rows else 0.0,
        "recommendation": {
            "hwp": "olefile 기반 커스텀 파서를 우선 사용하고, hwp5txt는 보조 비교 도구로 사용",
            "pdf": "PyMuPDF 우선, 오류 또는 경고 발생 시 pdfplumber 폴백",
        },
    }

    report = {"summary": summary, "pdf_rows": pdf_rows, "hwp_rows": hwp_rows}
    json_path = REPORTS_DIR / "parser_comparison_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    markdown = [
        "# Parser Comparison Report",
        "",
        f"- 생성 시각: {summary['generated_at']}",
        f"- HWP 문서 수: {summary['hwp_document_count']}",
        f"- PDF 문서 수: {summary['pdf_document_count']}",
        "",
        "## HWP 비교 결과",
        "",
        f"- `olefile`: {'사용 가능' if summary['hwp_candidates']['olefile'] else '불가'}",
        f"- `pyhwp`: {'사용 가능' if summary['hwp_candidates']['pyhwp'] else '미설치'}",
        f"- `hwp5txt`: {'사용 가능' if summary['hwp_candidates']['hwp5txt_cli'] else '미설치'}",
        f"- olefile 평균 처리 시간: {summary['hwp_average_olefile_time_sec']}초",
        f"- hwp5txt 평균 처리 시간: {summary['hwp_average_hwp5txt_time_sec']}초",
        "- 결론: 현재 환경에서는 `olefile` 기반 커스텀 파서가 본문 보존 측면에서 더 현실적이었다.",
        "",
        "## PDF 비교 결과",
        "",
        f"- PyMuPDF 평균 처리 시간: {summary['pdf_average_pymupdf_time_sec']}초",
        f"- pdfplumber 평균 처리 시간: {summary['pdf_average_pdfplumber_time_sec']}초",
        "- 결론: 기본값은 `PyMuPDF`, 예외 시 `pdfplumber` 폴백 전략이 적절하다.",
        "",
        "## 샘플 행",
        "",
    ]
    for row in pdf_rows:
        markdown.append(
            f"- {row['file_name']}: pymupdf={row['pymupdf_chars']}자/{row['pymupdf_time_sec']}초, "
            f"pdfplumber={row['pdfplumber_chars']}자/{row['pdfplumber_time_sec']}초, 추천={row['recommended']}"
        )
    markdown.extend(["", "## HWP 샘플 행", ""])
    for row in hwp_rows:
        markdown.append(
            f"- {row['file_name']}: olefile={row['olefile_chars']}자/{row['olefile_time_sec']}초, "
            f"hwp5txt={row['hwp5txt_chars']}자/{row['hwp5txt_time_sec']}초, 추천={row['recommended']}"
        )
    (REPORTS_DIR / "parser_comparison_report.md").write_text("\n".join(markdown), encoding="utf-8")
    return report


if __name__ == "__main__":
    build_parser_comparison_report()

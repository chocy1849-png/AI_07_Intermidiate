from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

from config import PARSED_DATA_DIR, REPORTS_DIR, ensure_directories
from src.db.metadata_store import list_source_documents
from src.parser import parse_document


def _slugify(file_name: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in file_name)
    return safe.strip("_") or "document"


def _build_output_path(file_name: str) -> Path:
    return PARSED_DATA_DIR / f"{_slugify(file_name)}.json"


def parse_all_documents(limit: int | None = None, force: bool = False) -> dict:
    ensure_directories()
    documents = list_source_documents()
    if limit is not None:
        documents = documents[:limit]

    started_at = datetime.now().isoformat(timespec="seconds")
    success_count = 0
    skipped_count = 0
    failures: list[dict] = []
    parsed_outputs: list[dict] = []

    for index, document in enumerate(documents, start=1):
        file_name = document["file_name"]
        source_path = document["source_path"]
        output_path = _build_output_path(file_name)

        if output_path.exists() and not force:
            skipped_count += 1
            continue

        started = perf_counter()
        try:
            parsed = parse_document(source_path)
            payload = {
                "text": parsed["text"],
                "metadata": {
                    **document,
                    **parsed["metadata"],
                    "parsed_at": datetime.now().isoformat(timespec="seconds"),
                },
            }
            output_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            duration = round(perf_counter() - started, 3)
            success_count += 1
            parsed_outputs.append(
                {
                    "index": index,
                    "file_name": file_name,
                    "status": "success",
                    "output_path": str(output_path),
                    "duration_sec": duration,
                }
            )
            print(f"[{index}/{len(documents)}] parsed {file_name} ({duration}s)")
        except Exception as exc:  # pragma: no cover - runtime reporting path
            duration = round(perf_counter() - started, 3)
            failure = {
                "index": index,
                "file_name": file_name,
                "source_path": source_path,
                "status": "failed",
                "duration_sec": duration,
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            print(f"[{index}/{len(documents)}] failed {file_name} -> {failure['error']}", file=sys.stderr)

    finished_at = datetime.now().isoformat(timespec="seconds")
    report = {
        "started_at": started_at,
        "finished_at": finished_at,
        "total_candidates": len(documents),
        "success_count": success_count,
        "skipped_count": skipped_count,
        "failure_count": len(failures),
        "failures": failures,
        "outputs": parsed_outputs,
    }
    report_path = REPORTS_DIR / f"parse_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"report": report, "report_path": report_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse source RFP documents into JSON files.")
    parser.add_argument("--limit", type=int, default=None, help="Only parse the first N source documents.")
    parser.add_argument("--force", action="store_true", help="Re-parse documents even if JSON already exists.")
    args = parser.parse_args()

    result = parse_all_documents(limit=args.limit, force=args.force)
    report = result["report"]
    print(
        json.dumps(
            {
                "success_count": report["success_count"],
                "skipped_count": report["skipped_count"],
                "failure_count": report["failure_count"],
                "report_path": str(result["report_path"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

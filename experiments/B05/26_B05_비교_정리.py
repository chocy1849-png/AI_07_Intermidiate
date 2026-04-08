from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from rag_공통 import OUTPUT_DIR


BASELINE_DIR = OUTPUT_DIR / "b05_table_eval_b02"
ENRICHED_DIR = OUTPUT_DIR / "b05_table_eval_enriched"
COMPARE_CSV_PATH = OUTPUT_DIR / "b05_table_compare.csv"
COMPARE_JSON_PATH = OUTPUT_DIR / "b05_table_compare.json"


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def numeric(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compare_rows(baseline_rows: list[dict[str, Any]], enriched_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline_map = {row["group_name"]: row for row in baseline_rows}
    enriched_map = {row["group_name"]: row for row in enriched_rows}
    group_names = sorted(set(baseline_map) | set(enriched_map))
    out = []
    for group_name in group_names:
        base = baseline_map.get(group_name, {})
        new = enriched_map.get(group_name, {})
        row = {"group_name": group_name}
        for key in [
            "question_count",
            "avg_faithfulness_score",
            "avg_completeness_score",
            "avg_groundedness_score",
            "avg_relevancy_score",
            "avg_manual_eval_score",
        ]:
            row[f"baseline_{key}"] = base.get(key)
            row[f"enriched_{key}"] = new.get(key)
            b = numeric(base.get(key))
            e = numeric(new.get(key))
            row[f"delta_{key}"] = round(e - b, 4) if b is not None and e is not None else ""
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="B-05 table eval 비교 요약을 생성합니다.")
    parser.add_argument("--기준디렉토리", default=str(BASELINE_DIR))
    parser.add_argument("--비교디렉토리", default=str(ENRICHED_DIR))
    parser.add_argument("--출력CSV경로", default=str(COMPARE_CSV_PATH))
    parser.add_argument("--출력JSON경로", default=str(COMPARE_JSON_PATH))
    args = parser.parse_args()

    baseline_summary = read_csv(Path(args.기준디렉토리) / "baseline_eval_manual_summary.csv")
    enriched_summary = read_csv(Path(args.비교디렉토리) / "baseline_eval_manual_summary.csv")
    rows = compare_rows(baseline_summary, enriched_summary)
    write_csv(Path(args.출력CSV경로), rows)
    Path(args.출력JSON경로).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[완료] B-05 비교 요약 생성이 끝났습니다.")
    print(f"- 비교 CSV: {args.출력CSV경로}")
    print(f"- 비교 JSON: {args.출력JSON경로}")


if __name__ == "__main__":
    main()

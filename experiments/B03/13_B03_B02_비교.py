from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "rag_outputs"
B02_DIR = OUTPUT_DIR / "b02_prefix_v2_full_eval"
B03_DIR = OUTPUT_DIR / "b03_crag_full_eval"
COMPARE_CSV_PATH = OUTPUT_DIR / "b03_b02_compare.csv"
COMPARE_JSON_PATH = OUTPUT_DIR / "b03_b02_compare.json"


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def to_number(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def rows_to_index(rows: list[dict[str, Any]], value_prefix: str) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row["group_name"])
        converted: dict[str, Any] = {"group_name": key}
        for column, value in row.items():
            if column == "group_name":
                continue
            converted[f"{value_prefix}.{column}"] = to_number(value) if column != "question_count" else int(float(value))
        index[key] = converted
    return index


def main() -> None:
    b02_auto = read_csv(B02_DIR / "baseline_eval_summary.csv")
    b02_manual = read_csv(B02_DIR / "baseline_eval_manual_summary.csv")
    b03_auto = read_csv(B03_DIR / "baseline_eval_summary.csv")
    b03_manual = read_csv(B03_DIR / "baseline_eval_manual_summary.csv")

    b02_auto_index = rows_to_index(b02_auto, "b02_auto")
    b02_manual_index = rows_to_index(b02_manual, "b02_manual")
    b03_auto_index = rows_to_index(b03_auto, "b03_auto")
    b03_manual_index = rows_to_index(b03_manual, "b03_manual")

    group_names = sorted(set(b02_auto_index) | set(b02_manual_index) | set(b03_auto_index) | set(b03_manual_index))
    compare_rows: list[dict[str, Any]] = []

    delta_pairs = [
        ("delta.top1_doc_hit_rate", "b03_auto.top1_doc_hit_rate", "b02_auto.top1_doc_hit_rate"),
        ("delta.topk_doc_hit_rate", "b03_auto.topk_doc_hit_rate", "b02_auto.topk_doc_hit_rate"),
        ("delta.rejection_success_rate", "b03_auto.rejection_success_rate", "b02_auto.rejection_success_rate"),
        ("delta.avg_elapsed_sec", "b03_auto.avg_elapsed_sec", "b02_auto.avg_elapsed_sec"),
        ("delta.avg_faithfulness_score", "b03_manual.avg_faithfulness_score", "b02_manual.avg_faithfulness_score"),
        ("delta.avg_completeness_score", "b03_manual.avg_completeness_score", "b02_manual.avg_completeness_score"),
        ("delta.avg_groundedness_score", "b03_manual.avg_groundedness_score", "b02_manual.avg_groundedness_score"),
        ("delta.avg_relevancy_score", "b03_manual.avg_relevancy_score", "b02_manual.avg_relevancy_score"),
        ("delta.avg_manual_eval_score", "b03_manual.avg_manual_eval_score", "b02_manual.avg_manual_eval_score"),
    ]

    for group_name in group_names:
        combined: dict[str, Any] = {"group_name": group_name}
        for source in [b02_auto_index, b02_manual_index, b03_auto_index, b03_manual_index]:
            combined.update(source.get(group_name, {}))

        for delta_name, left_key, right_key in delta_pairs:
            left = combined.get(left_key)
            right = combined.get(right_key)
            combined[delta_name] = round(left - right, 4) if left is not None and right is not None else None
        compare_rows.append(combined)

    compare_rows.sort(key=lambda row: row["group_name"])
    with COMPARE_CSV_PATH.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(compare_rows[0].keys()))
        writer.writeheader()
        writer.writerows(compare_rows)

    COMPARE_JSON_PATH.write_text(json.dumps(compare_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[완료] 비교 파일 저장: {COMPARE_CSV_PATH}")


if __name__ == "__main__":
    main()

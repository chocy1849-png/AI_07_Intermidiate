from __future__ import annotations

import csv
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "rag_outputs"

B01_AUTO_PATH = OUTPUT_DIR / "b01_hybrid_eval" / "baseline_eval_summary.csv"
B01_MANUAL_PATH = OUTPUT_DIR / "b01_hybrid_eval" / "baseline_eval_manual_summary.csv"
B02_AUTO_PATH = OUTPUT_DIR / "b02_prefix_v2_full_eval" / "baseline_eval_summary.csv"
B02_MANUAL_PATH = OUTPUT_DIR / "b02_prefix_v2_full_eval" / "baseline_eval_manual_summary.csv"

COMPARE_CSV_PATH = OUTPUT_DIR / "b02_b01_compare.csv"
COMPARE_JSON_PATH = OUTPUT_DIR / "b02_b01_compare.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: str) -> float | None:
    value = str(value or "").strip()
    if not value:
        return None
    return float(value)


def index_by_group(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["group_name"]: row for row in rows}


def build_compare_rows() -> list[dict[str, object]]:
    b01_auto = index_by_group(read_csv(B01_AUTO_PATH))
    b01_manual = index_by_group(read_csv(B01_MANUAL_PATH))
    b02_auto = index_by_group(read_csv(B02_AUTO_PATH))
    b02_manual = index_by_group(read_csv(B02_MANUAL_PATH))

    all_groups = sorted(set(b01_auto) | set(b01_manual) | set(b02_auto) | set(b02_manual))
    rows: list[dict[str, object]] = []
    for group_name in all_groups:
        row = {"group_name": group_name}
        for prefix, source in [
            ("b01_auto", b01_auto.get(group_name, {})),
            ("b01_manual", b01_manual.get(group_name, {})),
            ("b02_auto", b02_auto.get(group_name, {})),
            ("b02_manual", b02_manual.get(group_name, {})),
        ]:
            for key, value in source.items():
                if key == "group_name":
                    continue
                row[f"{prefix}.{key}"] = value

        for metric in [
            "top1_doc_hit_rate",
            "topk_doc_hit_rate",
            "rejection_success_rate",
            "avg_elapsed_sec",
            "avg_faithfulness_score",
            "avg_completeness_score",
            "avg_groundedness_score",
            "avg_relevancy_score",
            "avg_manual_eval_score",
        ]:
            b01_value = as_float(row.get(f"b01_auto.{metric}", row.get(f"b01_manual.{metric}", "")))
            b02_value = as_float(row.get(f"b02_auto.{metric}", row.get(f"b02_manual.{metric}", "")))
            if b01_value is not None and b02_value is not None:
                row[f"delta.{metric}"] = round(b02_value - b01_value, 4)
            else:
                row[f"delta.{metric}"] = None

        rows.append(row)
    return rows


def main() -> None:
    compare_rows = build_compare_rows()
    write_csv(COMPARE_CSV_PATH, compare_rows)
    COMPARE_JSON_PATH.write_text(json.dumps(compare_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    overall = next(row for row in compare_rows if row["group_name"] == "overall")
    print("[완료] B-01 대비 B-02 비교표를 생성했습니다.")
    print(f"- overall manual delta: {overall.get('delta.avg_manual_eval_score')}")
    print(f"- overall completeness delta: {overall.get('delta.avg_completeness_score')}")
    print(f"- overall top1 hit delta: {overall.get('delta.top1_doc_hit_rate')}")
    print(f"- 결과 CSV: {COMPARE_CSV_PATH}")


if __name__ == "__main__":
    main()

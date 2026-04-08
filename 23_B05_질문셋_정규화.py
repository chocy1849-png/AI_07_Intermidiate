from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from rag_공통 import OUTPUT_DIR


BASE_DIR = Path(__file__).resolve().parent
SOURCE_QUESTION_PATH = BASE_DIR / "evaluation" / "eval_questions_table_v1.txt"
NORMALIZED_TXT_PATH = OUTPUT_DIR / "eval_sets" / "b05_table_eval_questions_v2.txt"
NORMALIZED_CSV_PATH = OUTPUT_DIR / "eval_sets" / "b05_table_eval_questions_v2.csv"
MANIFEST_PATH = OUTPUT_DIR / "eval_sets" / "b05_table_eval_questions_v2_manifest.json"

QUESTION_HEADER_RE = re.compile(r"^(T[ABC]-\d+)$")
FIELD_RE = re.compile(r"^\s*([A-Za-z_]+)\s*:\s*(.+?)\s*$")
GROUP_RE = re.compile(r"^GROUP\s+([A-C])\s*:")

GROUP_TO_TYPE = {
    "A": ("TYPE 5", "table_enrichment_control"),
    "B": ("TYPE 5", "table_enrichment_hwp_only"),
    "C": ("TYPE 5", "table_enrichment_table_plus_text"),
}


def parse_table_question_file(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    current_group = ""

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        group_match = GROUP_RE.match(line)
        if group_match:
            current_group = group_match.group(1)
            continue

        question_match = QUESTION_HEADER_RE.match(line)
        if question_match:
            if current is not None:
                rows.append(current)
            current = {
                "question_id": f"Q{len(rows) + 1:02d}",
                "source_question_id": question_match.group(1),
                "question_index": len(rows) + 1,
                "group_label": f"Group {current_group}" if current_group else "",
            }
            continue

        if current is None:
            continue

        field_match = FIELD_RE.match(raw_line)
        if field_match:
            key = field_match.group(1).strip()
            value = field_match.group(2).strip()
            current[key] = value

    if current is not None:
        rows.append(current)

    for row in rows:
        group_key = row.get("group_label", "").replace("Group ", "")
        type_group, type_label = GROUP_TO_TYPE.get(group_key, ("TYPE 5", "table_enrichment"))
        row["type_group"] = type_group
        row["type_label"] = type_label
        row["scenario_label"] = row.get("group_label", "")
        row.setdefault("question", "")
        row.setdefault("answer_type", "")
        row.setdefault("ground_truth_doc", "")
        row.setdefault("ground_truth_hint", "")
        row.setdefault("eval_focus", "")
        row.setdefault("expected", "")
        row["depends_on"] = ""
        row["depends_on_list"] = []
    return rows


def write_normalized_text(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    current_group = None

    for row in rows:
        if row["scenario_label"] != current_group:
            current_group = row["scenario_label"]
            lines.append(f"TYPE 5: {row['type_label']}")
            lines.append(f"--- {current_group} ---")
        lines.append(row["question_id"])
        for key in [
            "source_question_id",
            "question",
            "answer_type",
            "ground_truth_doc",
            "ground_truth_hint",
            "eval_focus",
            "expected",
            "scenario_label",
        ]:
            value = str(row.get(key, "")).strip()
            if value:
                lines.append(f"{key}: {value}")
        lines.append("")

    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="B-05 표 평가 질문셋을 기존 평가 harness 형식으로 정규화합니다.")
    parser.add_argument("--입력경로", default=str(SOURCE_QUESTION_PATH))
    parser.add_argument("--출력TXT경로", default=str(NORMALIZED_TXT_PATH))
    parser.add_argument("--출력CSV경로", default=str(NORMALIZED_CSV_PATH))
    parser.add_argument("--manifest경로", default=str(MANIFEST_PATH))
    args = parser.parse_args()

    rows = parse_table_question_file(Path(args.입력경로))
    write_normalized_text(Path(args.출력TXT경로), rows)
    write_csv(Path(args.출력CSV경로), rows)

    manifest = {
        "source_question_path": str(args.입력경로),
        "normalized_txt_path": str(args.출력TXT경로),
        "normalized_csv_path": str(args.출력CSV경로),
        "question_count": len(rows),
        "groups": sorted({row["scenario_label"] for row in rows}),
        "comparison_baseline": "B-02",
        "comparison_candidate": "B-05",
    }
    Path(args.manifest경로).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest경로).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[완료] B-05 표 평가 질문셋 정규화가 끝났습니다.")
    print(f"- 질문 수: {len(rows)}")
    print(f"- TXT 경로: {args.출력TXT경로}")
    print(f"- CSV 경로: {args.출력CSV경로}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path

from eval_utils import build_auto_summary, build_manual_summary, read_csv, sort_result_rows, write_csv, write_json


BASE_DIR = Path(__file__).resolve().parent
RAG_OUTPUTS = BASE_DIR / "rag_outputs"

B05_DIR = RAG_OUTPUTS / "b05_group_bc_eval_b05"
B05_2_DIR = RAG_OUTPUTS / "b05_group_bc_eval_b05_2"
OUT_DIR = RAG_OUTPUTS / "b05_group_bc_eval_b05_2_conditional"


def choose_rows(b05_rows: list[dict], b05_2_rows: list[dict]) -> list[dict]:
    b05_2_by_id = {row["question_id"]: row for row in b05_2_rows}
    merged: list[dict] = []
    for row in b05_rows:
        if row.get("answer_type") == "table_plus_text" and row.get("question_id") in b05_2_by_id:
            chosen = dict(b05_2_by_id[row["question_id"]])
            chosen["selected_pipeline"] = "b05_2_for_table_plus_text"
        else:
            chosen = dict(row)
            chosen["selected_pipeline"] = "b05_default"
        merged.append(chosen)
    return sort_result_rows(merged)


def align_rows(rows: list[dict]) -> list[dict]:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    return [{key: row.get(key, "") for key in fieldnames} for row in rows]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    result_rows = choose_rows(
        read_csv(B05_DIR / "baseline_eval_results.csv"),
        read_csv(B05_2_DIR / "baseline_eval_results.csv"),
    )
    manual_rows = choose_rows(
        read_csv(B05_DIR / "baseline_eval_manual_completed.csv"),
        read_csv(B05_2_DIR / "baseline_eval_manual_completed.csv"),
    )
    parsed_rows = choose_rows(
        read_csv(B05_DIR / "baseline_eval_questions_parsed.csv"),
        read_csv(B05_2_DIR / "baseline_eval_questions_parsed.csv"),
    )
    manual_template_rows = choose_rows(
        read_csv(B05_DIR / "baseline_eval_manual_template.csv"),
        read_csv(B05_2_DIR / "baseline_eval_manual_template.csv"),
    )

    write_csv(OUT_DIR / "baseline_eval_results.csv", align_rows(result_rows))
    write_csv(OUT_DIR / "baseline_eval_summary.csv", build_auto_summary(result_rows))
    write_csv(OUT_DIR / "baseline_eval_manual_completed.csv", align_rows(manual_rows))
    write_csv(OUT_DIR / "baseline_eval_manual_summary.csv", build_manual_summary(manual_rows))
    write_csv(OUT_DIR / "baseline_eval_questions_parsed.csv", align_rows(parsed_rows))
    write_csv(OUT_DIR / "baseline_eval_manual_template.csv", align_rows(manual_template_rows))
    write_json(
        OUT_DIR / "baseline_eval_manifest.json",
        {
            "mode": "merged_conditional",
            "base_dir": str(B05_DIR),
            "override_dir": str(B05_2_DIR),
            "rule": "table_factual -> B05, table_plus_text -> B05.2",
            "question_count": len(result_rows),
        },
    )
    print("[완료] B-05.2 조건부 병합 결과 생성이 끝났습니다.")
    print(f"- 출력 디렉토리: {OUT_DIR}")


if __name__ == "__main__":
    main()

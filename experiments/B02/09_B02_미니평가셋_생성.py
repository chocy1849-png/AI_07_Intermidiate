from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "rag_outputs"
DEFAULT_ID_PATH = BASE_DIR / "evaluation" / "day4_mini_eval_question_ids_v1.txt"
DEFAULT_SOURCE_CANDIDATES = [
    OUTPUT_DIR / "model_eval_runs" / "gpt-5-mini" / "baseline_eval_questions_parsed.csv",
    OUTPUT_DIR / "baseline_eval" / "baseline_eval_questions_parsed.csv",
]
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "eval_sets"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_question_ids(path: Path) -> list[str]:
    question_ids: list[str] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            question_ids.append(value)
    return question_ids


def resolve_source_path(explicit_path: str) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"질문 파싱 CSV가 없습니다: {path}")
        return path

    for path in DEFAULT_SOURCE_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("baseline_eval_questions_parsed.csv를 찾지 못했습니다.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 4 고정 mini eval 세트를 생성합니다.")
    parser.add_argument("--질문ID파일", default=str(DEFAULT_ID_PATH), help="question_id 목록 txt 파일")
    parser.add_argument("--원본CSV", default="", help="baseline_eval_questions_parsed.csv 경로")
    parser.add_argument("--출력디렉토리", default=str(DEFAULT_OUTPUT_DIR), help="mini eval 세트 출력 디렉토리")
    args = parser.parse_args()

    question_id_path = Path(args.질문ID파일)
    if not question_id_path.exists():
        raise FileNotFoundError(f"question_id 파일이 없습니다: {question_id_path}")

    source_csv_path = resolve_source_path(args.원본CSV)
    output_dir = Path(args.출력디렉토리)
    output_dir.mkdir(parents=True, exist_ok=True)

    question_ids = load_question_ids(question_id_path)
    source_rows = read_csv(source_csv_path)
    source_index = {row["question_id"]: row for row in source_rows}

    missing_ids = [qid for qid in question_ids if qid not in source_index]
    if missing_ids:
        raise RuntimeError(f"원본 CSV에 없는 question_id가 있습니다: {missing_ids}")

    selected_rows = [source_index[qid] for qid in question_ids]
    output_csv_path = output_dir / "day4_mini_eval_questions_v1.csv"
    manifest_path = output_dir / "day4_mini_eval_questions_v1_manifest.json"

    write_csv(output_csv_path, selected_rows)
    manifest_path.write_text(
        json.dumps(
            {
                "source_csv_path": str(source_csv_path),
                "question_id_path": str(question_id_path),
                "question_count": len(selected_rows),
                "question_ids": question_ids,
                "type_counts": {
                    key: sum(1 for row in selected_rows if row.get("type_group") == key)
                    for key in sorted({row.get("type_group", "") for row in selected_rows if row.get("type_group", "")})
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("[완료] Day 4 mini eval 세트를 생성했습니다.")
    print(f"- question 수: {len(selected_rows)}")
    print(f"- 출력 CSV: {output_csv_path}")
    print(f"- manifest: {manifest_path}")


if __name__ == "__main__":
    main()

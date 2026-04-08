from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "rag_outputs"
A04_MANIFEST_PATH = OUTPUT_DIR / "b04_candidates" / "b04_candidate_manifest.json"
SMOKE_SELECTED_JSON = OUTPUT_DIR / "b04_smoke_selected_top2.json"
MINI_IDS_PATH = BASE_DIR / "evaluation" / "day4_mini_eval_question_ids_v1.txt"
MINI_ROOT_DIR = OUTPUT_DIR / "b04_mini_eval"
MINI_COMPARE_CSV = OUTPUT_DIR / "b04_mini_compare.csv"
MINI_WINNER_JSON = OUTPUT_DIR / "b04_mini_winner.json"


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


def run_python(args: list[str]) -> None:
    subprocess.run([sys.executable, *args], cwd=BASE_DIR, check=True)


def overall_summary(path: Path) -> dict[str, Any]:
    rows = read_csv(path)
    for row in rows:
        if row["group_name"] == "overall":
            return row
    raise RuntimeError(f"overall row not found: {path}")


def to_float(value: Any) -> float:
    if value in (None, "", "None"):
        return 0.0
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="A-04 mini eval orchestrator")
    parser.add_argument("--manifest-path", default=str(A04_MANIFEST_PATH))
    parser.add_argument("--selected-json", default=str(SMOKE_SELECTED_JSON))
    parser.add_argument("--mini-ids-path", default=str(MINI_IDS_PATH))
    parser.add_argument("--response-model", default="gpt-5-mini")
    parser.add_argument("--judge-model", default="gpt-5")
    args = parser.parse_args()

    manifest_rows = {row["candidate_key"]: row for row in json.loads(Path(args.manifest_path).read_text(encoding="utf-8"))}
    selected_payload = json.loads(Path(args.selected_json).read_text(encoding="utf-8"))
    candidate_keys = ["c500_80", *selected_payload["selected_top2_keys"]]
    candidate_keys = list(dict.fromkeys(candidate_keys))

    compare_rows: list[dict[str, Any]] = []

    for candidate_key in candidate_keys:
        candidate = manifest_rows[candidate_key]
        eval_dir = MINI_ROOT_DIR / candidate_key
        run_python(
            [
                "08_A01_Hybrid_평가.py",
                "--질문ID파일",
                str(args.mini_ids_path),
                "--출력디렉토리",
                str(eval_dir),
                "--컬렉션이름",
                candidate["collection_name"],
        "--bm25인덱스경로",
                candidate["bm25_index_path"],
                "--응답모델",
                args.response_model,
            ]
        )
        run_python(
            [
                "15_자동_Judge_ASCII.py",
                "--eval-dir",
                str(eval_dir),
                "--judge-model",
                args.judge_model,
            ]
        )

        auto_summary = overall_summary(eval_dir / "baseline_eval_summary.csv")
        manual_summary = overall_summary(eval_dir / "baseline_eval_manual_summary.csv")
        compare_rows.append(
            {
                "candidate_key": candidate_key,
                "candidate_label": candidate["candidate_label"],
                "mode": candidate["mode"],
                "chunk_size": candidate["chunk_size"],
                "overlap": candidate["overlap"],
                "top1_doc_hit_rate": auto_summary["top1_doc_hit_rate"],
                "topk_doc_hit_rate": auto_summary["topk_doc_hit_rate"],
                "avg_ground_truth_doc_hit_rate": auto_summary["avg_ground_truth_doc_hit_rate"],
                "avg_elapsed_sec": auto_summary["avg_elapsed_sec"],
                "avg_faithfulness_score": manual_summary["avg_faithfulness_score"],
                "avg_completeness_score": manual_summary["avg_completeness_score"],
                "avg_groundedness_score": manual_summary["avg_groundedness_score"],
                "avg_relevancy_score": manual_summary["avg_relevancy_score"],
                "avg_manual_eval_score": manual_summary["avg_manual_eval_score"],
                "eval_dir": str(eval_dir),
            }
        )

    compare_rows.sort(
        key=lambda row: (
            to_float(row["avg_manual_eval_score"]),
            to_float(row["avg_completeness_score"]),
            to_float(row["avg_faithfulness_score"]),
            to_float(row["top1_doc_hit_rate"]),
        ),
        reverse=True,
    )
    winner = compare_rows[0]
    write_csv(MINI_COMPARE_CSV, compare_rows)
    MINI_WINNER_JSON.write_text(json.dumps(winner, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[완료] A-04 mini compare: {MINI_COMPARE_CSV}")
    print(f"[완료] A-04 mini winner: {MINI_WINNER_JSON}")


if __name__ == "__main__":
    main()

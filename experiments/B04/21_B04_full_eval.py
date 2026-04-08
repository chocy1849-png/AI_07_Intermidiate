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
B04_MANIFEST_PATH = OUTPUT_DIR / "b04_candidates" / "b04_candidate_manifest.json"
MINI_WINNER_JSON = OUTPUT_DIR / "b04_mini_winner.json"
FULL_ROOT_DIR = OUTPUT_DIR / "b04_full_eval"
FULL_COMPARE_CSV = OUTPUT_DIR / "b04_full_compare.csv"
BASELINE_B02_DIR = OUTPUT_DIR / "b02_prefix_v2_full_eval"


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


def overall_summary(path: Path) -> dict[str, Any]:
    rows = read_csv(path)
    for row in rows:
        if row["group_name"] == "overall":
            return row
    raise RuntimeError(f"overall row not found: {path}")


def run_python(args: list[str]) -> None:
    subprocess.run([sys.executable, *args], cwd=BASE_DIR, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="B-04 full eval winner run")
    parser.add_argument("--manifest-path", default=str(B04_MANIFEST_PATH))
    parser.add_argument("--winner-json", default=str(MINI_WINNER_JSON))
    parser.add_argument("--response-model", default="gpt-5-mini")
    parser.add_argument("--judge-model", default="gpt-5")
    args = parser.parse_args()

    manifest_rows = {row["candidate_key"]: row for row in json.loads(Path(args.manifest_path).read_text(encoding="utf-8"))}
    winner = json.loads(Path(args.winner_json).read_text(encoding="utf-8"))
    winner_key = winner["candidate_key"]

    compare_rows: list[dict[str, Any]] = []

    baseline_auto = overall_summary(BASELINE_B02_DIR / "baseline_eval_summary.csv")
    baseline_manual = overall_summary(BASELINE_B02_DIR / "baseline_eval_manual_summary.csv")
    compare_rows.append(
        {
            "run_name": "b02_baseline_500_80",
            "candidate_key": "c500_80",
            "candidate_label": "500/80",
            "top1_doc_hit_rate": baseline_auto["top1_doc_hit_rate"],
            "topk_doc_hit_rate": baseline_auto["topk_doc_hit_rate"],
            "avg_ground_truth_doc_hit_rate": baseline_auto["avg_ground_truth_doc_hit_rate"],
            "avg_elapsed_sec": baseline_auto["avg_elapsed_sec"],
            "avg_faithfulness_score": baseline_manual["avg_faithfulness_score"],
            "avg_completeness_score": baseline_manual["avg_completeness_score"],
            "avg_groundedness_score": baseline_manual["avg_groundedness_score"],
            "avg_relevancy_score": baseline_manual["avg_relevancy_score"],
            "avg_manual_eval_score": baseline_manual["avg_manual_eval_score"],
            "eval_dir": str(BASELINE_B02_DIR),
        }
    )

    if winner_key == "c500_80":
        write_csv(FULL_COMPARE_CSV, compare_rows)
        print("[완료] mini winner is baseline c500_80, reused existing B-02 full eval")
        print(f"[완료] full compare: {FULL_COMPARE_CSV}")
        return

    candidate = manifest_rows[winner_key]
    eval_dir = FULL_ROOT_DIR / winner_key
    run_python(
        [
            "08_B01_Hybrid_평가.py",
            "--출력디렉토리",
            str(eval_dir),
            "--컬렉션이름",
            candidate["collection_name"],
            "--BM25인덱스경로",
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
            "run_name": "b04_winner_full",
            "candidate_key": winner_key,
            "candidate_label": candidate["candidate_label"],
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

    write_csv(FULL_COMPARE_CSV, compare_rows)
    print(f"[완료] B-04 full compare: {FULL_COMPARE_CSV}")


if __name__ == "__main__":
    main()

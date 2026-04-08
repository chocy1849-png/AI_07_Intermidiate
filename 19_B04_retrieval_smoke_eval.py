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
SMOKE_IDS_PATH = BASE_DIR / "evaluation" / "day4_smoke_eval_question_ids_v1.txt"
SMOKE_ROOT_DIR = OUTPUT_DIR / "b04_smoke_eval"
SMOKE_COMPARE_CSV = OUTPUT_DIR / "b04_smoke_compare.csv"
SMOKE_SELECTED_JSON = OUTPUT_DIR / "b04_smoke_selected_top2.json"


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
    parser = argparse.ArgumentParser(description="A-04 smoke retrieval eval orchestrator")
    parser.add_argument("--manifest-path", default=str(A04_MANIFEST_PATH))
    parser.add_argument("--smoke-ids-path", default=str(SMOKE_IDS_PATH))
    parser.add_argument("--response-model", default="gpt-5-mini")
    args = parser.parse_args()

    manifest_rows = json.loads(Path(args.manifest_path).read_text(encoding="utf-8"))
    smoke_ids_path = Path(args.smoke_ids_path)
    if not smoke_ids_path.exists():
        raise FileNotFoundError(f"smoke ids not found: {smoke_ids_path}")

    compare_rows: list[dict[str, Any]] = []
    ranked_non_baseline: list[dict[str, Any]] = []

    for candidate in manifest_rows:
        candidate_key = candidate["candidate_key"]
        eval_dir = SMOKE_ROOT_DIR / candidate_key
        run_python(
            [
                "08_A01_Hybrid_평가.py",
                "--질문ID파일",
                str(smoke_ids_path),
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
        overall = overall_summary(eval_dir / "baseline_eval_summary.csv")
        row = {
            "candidate_key": candidate_key,
            "candidate_label": candidate["candidate_label"],
            "mode": candidate["mode"],
            "chunk_size": candidate["chunk_size"],
            "overlap": candidate["overlap"],
            "question_count": overall["question_count"],
            "top1_doc_hit_rate": overall["top1_doc_hit_rate"],
            "topk_doc_hit_rate": overall["topk_doc_hit_rate"],
            "avg_ground_truth_doc_hit_rate": overall["avg_ground_truth_doc_hit_rate"],
            "rejection_success_rate": overall["rejection_success_rate"],
            "avg_elapsed_sec": overall["avg_elapsed_sec"],
            "eval_dir": str(eval_dir),
        }
        compare_rows.append(row)
        if candidate_key != "c500_80":
            ranked_non_baseline.append(row)

    ranked_non_baseline.sort(
        key=lambda row: (
            to_float(row["top1_doc_hit_rate"]),
            to_float(row["topk_doc_hit_rate"]),
            to_float(row["avg_ground_truth_doc_hit_rate"]),
            -abs(int(row["chunk_size"]) - 1200),
            1 if row["mode"] == "struct" else 0,
        ),
        reverse=True,
    )
    top2 = ranked_non_baseline[:2]

    write_csv(SMOKE_COMPARE_CSV, compare_rows)
    SMOKE_SELECTED_JSON.write_text(
        json.dumps(
            {
                "baseline_key": "c500_80",
                "selected_top2_keys": [row["candidate_key"] for row in top2],
                "selected_top2_rows": top2,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[완료] A-04 smoke eval compare: {SMOKE_COMPARE_CSV}")
    print(f"[완료] A-04 smoke selected top2: {SMOKE_SELECTED_JSON}")


if __name__ == "__main__":
    main()

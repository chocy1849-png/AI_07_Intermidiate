from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
RAG_OUTPUTS_DIR = BASE_DIR / "rag_outputs"
BASELINE_EVAL_DIR = RAG_OUTPUTS_DIR / "baseline_eval"
ARCHIVE_ROOT_DIR = RAG_OUTPUTS_DIR / "model_eval_runs"
COMPARE_CSV_PATH = RAG_OUTPUTS_DIR / "generation_model_compare.csv"
COMPARE_JSON_PATH = RAG_OUTPUTS_DIR / "generation_model_compare.json"

EVAL_SCRIPT = BASE_DIR / "04_베이스라인_평가.py"
JUDGE_SCRIPT = BASE_DIR / "05_자동_Judge_채점.py"

TARGET_MODELS = ("gpt-5", "gpt-5-nano")


def csv_읽기(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def csv_저장(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def 실행(command: list[str]) -> None:
    print("[실행]", " ".join(command))
    subprocess.run(command, cwd=BASE_DIR, check=True)


def 모델_폴더명(model_name: str) -> str:
    return model_name.replace("/", "_")


def baseline_eval_아카이브(model_name: str, overwrite: bool = True) -> Path:
    ARCHIVE_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    target_dir = ARCHIVE_ROOT_DIR / 모델_폴더명(model_name)
    if target_dir.exists() and overwrite:
        shutil.rmtree(target_dir)
    shutil.copytree(BASELINE_EVAL_DIR, target_dir)
    return target_dir


def baseline_eval_정상여부() -> bool:
    required = [
        BASELINE_EVAL_DIR / "baseline_eval_manifest.json",
        BASELINE_EVAL_DIR / "baseline_eval_summary.csv",
        BASELINE_EVAL_DIR / "baseline_eval_manual_summary.csv",
        BASELINE_EVAL_DIR / "baseline_eval_manual_completed.csv",
    ]
    return all(path.exists() for path in required)


def 모델별_행_구성(model_name: str, model_dir: Path) -> dict[str, Any]:
    manifest = json.loads((model_dir / "baseline_eval_manifest.json").read_text(encoding="utf-8"))
    auto_summary_rows = csv_읽기(model_dir / "baseline_eval_summary.csv")
    manual_summary_rows = csv_읽기(model_dir / "baseline_eval_manual_summary.csv")

    auto_overall = next(row for row in auto_summary_rows if row["group_name"] == "overall")
    manual_overall = next(row for row in manual_summary_rows if row["group_name"] == "overall")

    return {
        "response_model": model_name,
        "question_count": manifest.get("question_count"),
        "embedding_model": manifest.get("embedding_model"),
        "top_k": manifest.get("top_k"),
        "history_applied_for_follow_up": manifest.get("history_applied_for_follow_up"),
        "top1_doc_hit_rate": auto_overall.get("top1_doc_hit_rate"),
        "topk_doc_hit_rate": auto_overall.get("topk_doc_hit_rate"),
        "avg_ground_truth_doc_hit_rate": auto_overall.get("avg_ground_truth_doc_hit_rate"),
        "rejection_success_rate": auto_overall.get("rejection_success_rate"),
        "avg_elapsed_sec": auto_overall.get("avg_elapsed_sec"),
        "avg_answer_chars": auto_overall.get("avg_answer_chars"),
        "avg_faithfulness_score": manual_overall.get("avg_faithfulness_score"),
        "avg_completeness_score": manual_overall.get("avg_completeness_score"),
        "avg_groundedness_score": manual_overall.get("avg_groundedness_score"),
        "avg_relevancy_score": manual_overall.get("avg_relevancy_score"),
        "avg_manual_eval_score": manual_overall.get("avg_manual_eval_score"),
        "archive_dir": str(model_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="gpt-5-mini / gpt-5 / gpt-5-nano 생성모델 비교 실행")
    parser.add_argument("--상위개수", type=int, default=5, help="평가 시 retrieval top_k")
    parser.add_argument("--force-mini-rerun", action="store_true", help="기존 gpt-5-mini 결과를 재생성")
    args = parser.parse_args()

    archived_dirs: dict[str, Path] = {}

    if not baseline_eval_정상여부():
        raise RuntimeError(
            "현재 baseline_eval 폴더에 mini baseline 결과가 완전하게 없습니다. "
            "먼저 gpt-5-mini baseline + judge를 완료한 뒤 실행하세요."
        )

    current_manifest = json.loads((BASELINE_EVAL_DIR / "baseline_eval_manifest.json").read_text(encoding="utf-8"))
    current_model = current_manifest.get("response_model")
    if current_model != "gpt-5-mini" or args.force_mini_rerun:
        실행(
            [
                sys.executable,
                str(EVAL_SCRIPT),
                "--응답모델",
                "gpt-5-mini",
                "--상위개수",
                str(args.상위개수),
            ]
        )
        실행(
            [
                sys.executable,
                str(JUDGE_SCRIPT),
                "--judge모델",
                "gpt-5",
            ]
        )

    archived_dirs["gpt-5-mini"] = baseline_eval_아카이브("gpt-5-mini")

    for model_name in TARGET_MODELS:
        실행(
            [
                sys.executable,
                str(EVAL_SCRIPT),
                "--응답모델",
                model_name,
                "--상위개수",
                str(args.상위개수),
            ]
        )
        실행(
            [
                sys.executable,
                str(JUDGE_SCRIPT),
                "--judge모델",
                "gpt-5",
            ]
        )
        archived_dirs[model_name] = baseline_eval_아카이브(model_name)

    compare_rows = [모델별_행_구성(model_name, model_dir) for model_name, model_dir in archived_dirs.items()]
    compare_rows = sorted(compare_rows, key=lambda row: float(row["avg_manual_eval_score"]), reverse=True)

    csv_저장(COMPARE_CSV_PATH, compare_rows)
    COMPARE_JSON_PATH.write_text(json.dumps(compare_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[완료] 생성모델 비교 실행이 끝났습니다.")
    print(f"- 비교 CSV: {COMPARE_CSV_PATH}")
    print(f"- 비교 JSON: {COMPARE_JSON_PATH}")
    for row in compare_rows:
        print(
            f"- {row['response_model']}: "
            f"manual={row['avg_manual_eval_score']} "
            f"(F={row['avg_faithfulness_score']}, C={row['avg_completeness_score']}, "
            f"G={row['avg_groundedness_score']}, R={row['avg_relevancy_score']})"
        )


if __name__ == "__main__":
    main()

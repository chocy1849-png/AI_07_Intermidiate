from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from pathlib import Path
from typing import Any

from eval_utils import parse_question_rows, write_csv, write_json
from scenario_b_phase2.answer_type_router import classify_answer_type


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Rule-based answer_type router smoke test.")
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument(
        "--question-set-path",
        default=str(root / "docs" / "planning" / "pm" / "day3_partA_eval_questions_v1.txt"),
    )
    parser.add_argument("--output-root", default=str(root / "rag_outputs" / "phase2_runs"))
    parser.add_argument("--run-prefix", default="p2_answer_type_router_smoke_v1")
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--low-conf-fallback", default="comparison_safe")
    return parser.parse_args()


def _normalized_answer_type(row: dict[str, Any]) -> str:
    answer_type = str(row.get("answer_type", "")).strip().lower()
    if answer_type:
        return answer_type
    if str(row.get("type_group", "")).strip().upper() == "TYPE 4":
        return "rejection"
    return "factual"


def _stratified_pick(rows: list[dict[str, Any]], sample_size: int) -> list[dict[str, Any]]:
    by_type: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        label = _normalized_answer_type(row)
        by_type.setdefault(label, []).append(row)

    picks: list[dict[str, Any]] = []
    types = sorted(by_type.keys())
    if not types:
        return picks

    quota = max(1, sample_size // max(1, len(types)))
    for label in types:
        picks.extend(by_type[label][:quota])
    if len(picks) >= sample_size:
        return picks[:sample_size]

    used_ids = {str(row.get("question_id", "")) for row in picks}
    for row in rows:
        qid = str(row.get("question_id", ""))
        if qid in used_ids:
            continue
        picks.append(row)
        used_ids.add(qid)
        if len(picks) >= sample_size:
            break
    return picks


def _history_for_row(row: dict[str, Any], by_qid: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    depends = row.get("depends_on_list", []) or []
    if not depends:
        return []
    history: list[dict[str, str]] = []
    for dep_id in depends:
        parent = by_qid.get(str(dep_id))
        if not parent:
            continue
        history.append({"role": "user", "content": str(parent.get("question", ""))})
        history.append({"role": "assistant", "content": "(이전 답변 요약)"})
    return history


def main() -> None:
    args = parse_args()
    question_set_path = Path(args.question_set_path).resolve()
    rows = parse_question_rows(question_set_path)
    by_qid = {str(row.get("question_id", "")): row for row in rows}
    sample_rows = _stratified_pick(rows, args.sample_size)

    result_rows: list[dict[str, Any]] = []
    correct = 0
    for row in sample_rows:
        question = str(row.get("question", "")).strip()
        expected_type = _normalized_answer_type(row)
        history = _history_for_row(row, by_qid)
        routed = classify_answer_type(
            question=question,
            history=history,
            confidence_threshold=float(args.confidence_threshold),
            low_conf_fallback=str(args.low_conf_fallback),
        )
        matched = int(routed.answer_type == expected_type)
        correct += matched
        result_rows.append(
            {
                "question_id": row.get("question_id", ""),
                "type_group": row.get("type_group", ""),
                "expected_type": expected_type,
                "predicted_type": routed.answer_type,
                "predicted_route": routed.route,
                "confidence": routed.confidence,
                "signals": " | ".join(routed.signals),
                "reason": routed.reason,
                "question": question,
            }
        )

    run_dir = Path(args.output_root).resolve() / args.run_prefix
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "answer_type_router_smoke.csv"
    write_csv(csv_path, result_rows)

    summary = {
        "question_set_path": str(question_set_path),
        "sample_size": len(sample_rows),
        "exact_match_count": correct,
        "exact_match_rate": round(correct / max(1, len(sample_rows)), 4),
        "confidence_threshold": float(args.confidence_threshold),
        "low_conf_fallback": str(args.low_conf_fallback),
        "csv_path": str(csv_path),
        "counts": {},
    }
    for row in result_rows:
        key = f"{row['expected_type']}->{row['predicted_type']}"
        summary["counts"][key] = int(summary["counts"].get(key, 0)) + 1
    summary_path = run_dir / "answer_type_router_smoke_summary.json"
    write_json(summary_path, summary)
    print(f"[done] csv={csv_path}")
    print(f"[done] summary={summary_path}")


if __name__ == "__main__":
    main()

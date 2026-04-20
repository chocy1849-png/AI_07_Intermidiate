from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from pathlib import Path
from typing import Any

from eval_utils import read_csv, write_csv, write_json


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Answer type router confusion analysis.")
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument(
        "--input-csv",
        default=str(root / "rag_outputs" / "phase2_runs" / "p2_answer_type_router_smoke_v1" / "answer_type_router_smoke.csv"),
    )
    parser.add_argument("--output-root", default=str(root / "rag_outputs" / "phase2_runs"))
    parser.add_argument("--run-prefix", default="p2_answer_type_router_analysis_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv).resolve()
    rows = read_csv(input_csv)
    out_dir = Path(args.output_root).resolve() / args.run_prefix
    out_dir.mkdir(parents=True, exist_ok=True)

    confusion: dict[tuple[str, str], int] = {}
    by_expected: dict[str, int] = {}
    mismatches: list[dict[str, Any]] = []
    correct = 0
    for row in rows:
        expected = str(row.get("expected_type", "")).strip().lower()
        predicted = str(row.get("predicted_type", "")).strip().lower()
        by_expected[expected] = by_expected.get(expected, 0) + 1
        confusion[(expected, predicted)] = confusion.get((expected, predicted), 0) + 1
        if expected == predicted:
            correct += 1
        else:
            mismatches.append(row)

    confusion_rows: list[dict[str, Any]] = []
    for (expected, predicted), count in sorted(confusion.items(), key=lambda item: (-item[1], item[0][0], item[0][1])):
        confusion_rows.append(
            {
                "expected_type": expected,
                "predicted_type": predicted,
                "count": count,
            }
        )
    confusion_csv = out_dir / "answer_type_router_confusion.csv"
    write_csv(confusion_csv, confusion_rows)

    mismatch_csv = out_dir / "answer_type_router_mismatches.csv"
    write_csv(mismatch_csv, mismatches)

    summary = {
        "input_csv": str(input_csv),
        "sample_size": len(rows),
        "exact_match_count": correct,
        "exact_match_rate": round(correct / max(1, len(rows)), 4),
        "by_expected": by_expected,
        "mismatch_count": len(mismatches),
        "confusion_csv": str(confusion_csv),
        "mismatch_csv": str(mismatch_csv),
    }
    summary_json = out_dir / "answer_type_router_analysis_summary.json"
    write_json(summary_json, summary)
    print(f"[done] confusion_csv={confusion_csv}")
    print(f"[done] mismatch_csv={mismatch_csv}")
    print(f"[done] summary_json={summary_json}")


if __name__ == "__main__":
    main()

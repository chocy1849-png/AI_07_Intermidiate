from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RAG_OUTPUTS = BASE_DIR / "rag_outputs"

DEFAULT_MINI_DIR = RAG_OUTPUTS / "b06_adopted_eval_gpt5mini"
DEFAULT_GPT5_DIR = RAG_OUTPUTS / "b06_adopted_eval_gpt5"
DEFAULT_OUT_CSV = RAG_OUTPUTS / "b06_compare.csv"
DEFAULT_OUT_JSON = RAG_OUTPUTS / "b06_compare.json"


def load_summary(eval_dir: Path, filename: str) -> pd.DataFrame:
    df = pd.read_csv(eval_dir / filename)
    for column in df.columns:
        if column != "group_name":
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="B-06 비교 요약 생성")
    parser.add_argument("--mini-dir", default=str(DEFAULT_MINI_DIR))
    parser.add_argument("--gpt5-dir", default=str(DEFAULT_GPT5_DIR))
    parser.add_argument("--out-csv", default=str(DEFAULT_OUT_CSV))
    parser.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    args = parser.parse_args()

    mini_dir = Path(args.mini_dir)
    gpt5_dir = Path(args.gpt5_dir)
    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)

    mini_auto = load_summary(mini_dir, "baseline_eval_summary.csv").add_prefix("mini_").rename(columns={"mini_group_name": "group_name"})
    mini_manual = load_summary(mini_dir, "baseline_eval_manual_summary.csv").add_prefix("mini_manual_").rename(columns={"mini_manual_group_name": "group_name"})
    gpt5_auto = load_summary(gpt5_dir, "baseline_eval_summary.csv").add_prefix("gpt5_").rename(columns={"gpt5_group_name": "group_name"})
    gpt5_manual = load_summary(gpt5_dir, "baseline_eval_manual_summary.csv").add_prefix("gpt5_manual_").rename(columns={"gpt5_manual_group_name": "group_name"})

    compare = mini_auto.merge(mini_manual, on="group_name", how="outer")
    compare = compare.merge(gpt5_auto, on="group_name", how="outer")
    compare = compare.merge(gpt5_manual, on="group_name", how="outer")

    metric_pairs = [
        ("question_count", "delta_question_count", "gpt5_", "mini_"),
        ("top1_doc_hit_rate", "delta_top1_doc_hit_rate", "gpt5_", "mini_"),
        ("topk_doc_hit_rate", "delta_topk_doc_hit_rate", "gpt5_", "mini_"),
        ("rejection_success_rate", "delta_rejection_success_rate", "gpt5_", "mini_"),
        ("avg_elapsed_sec", "delta_avg_elapsed_sec", "gpt5_", "mini_"),
        ("avg_answer_chars", "delta_avg_answer_chars", "gpt5_", "mini_"),
        ("avg_faithfulness_score", "delta_avg_faithfulness_score", "gpt5_manual_", "mini_manual_"),
        ("avg_completeness_score", "delta_avg_completeness_score", "gpt5_manual_", "mini_manual_"),
        ("avg_groundedness_score", "delta_avg_groundedness_score", "gpt5_manual_", "mini_manual_"),
        ("avg_relevancy_score", "delta_avg_relevancy_score", "gpt5_manual_", "mini_manual_"),
        ("avg_manual_eval_score", "delta_avg_manual_eval_score", "gpt5_manual_", "mini_manual_"),
    ]

    for metric, delta_col, left_prefix, right_prefix in metric_pairs:
        compare[delta_col] = compare[f"{left_prefix}{metric}"] - compare[f"{right_prefix}{metric}"]

    compare.to_csv(out_csv, index=False, encoding="utf-8-sig")
    overall_row = compare.loc[compare["group_name"] == "overall"]
    overall = overall_row.iloc[0].to_dict() if len(overall_row) else {}
    payload = {
        "mini_dir": str(mini_dir),
        "gpt5_dir": str(gpt5_dir),
        "overall": overall,
        "rows": compare.to_dict(orient="records"),
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[완료] B-06 비교 요약을 생성했습니다.")
    print(f"- 비교 CSV: {out_csv}")
    print(f"- 비교 JSON: {out_json}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RAG_OUTPUTS = BASE_DIR / "rag_outputs"

B02_DIR = RAG_OUTPUTS / "b02_full_eval_parallel"
COND_DIR = RAG_OUTPUTS / "b05_1_conditional_eval"
OUT_CSV = RAG_OUTPUTS / "b05_1_compare.csv"
OUT_JSON = RAG_OUTPUTS / "b05_1_compare.json"


def load_manual_summary(eval_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(eval_dir / "baseline_eval_manual_summary.csv")
    for col in df.columns:
        if col != "group_name":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def main() -> None:
    b02 = load_manual_summary(B02_DIR).add_prefix("b02_").rename(columns={"b02_group_name": "group_name"})
    cond = load_manual_summary(COND_DIR).add_prefix("cond_").rename(columns={"cond_group_name": "group_name"})
    compare = b02.merge(cond, on="group_name", how="outer")

    metric_pairs = [
        ("question_count", "delta_question_count"),
        ("avg_faithfulness_score", "delta_avg_faithfulness_score"),
        ("avg_completeness_score", "delta_avg_completeness_score"),
        ("avg_groundedness_score", "delta_avg_groundedness_score"),
        ("avg_relevancy_score", "delta_avg_relevancy_score"),
        ("avg_manual_eval_score", "delta_avg_manual_eval_score"),
    ]
    for metric, delta_col in metric_pairs:
        compare[delta_col] = compare[f"cond_{metric}"] - compare[f"b02_{metric}"]

    compare.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    overall_row = compare.loc[compare["group_name"] == "overall"]
    overall = overall_row.iloc[0].to_dict() if len(overall_row) else {}
    payload = {
        "b02_dir": str(B02_DIR),
        "conditional_dir": str(COND_DIR),
        "overall": overall,
        "rows": compare.to_dict(orient="records"),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("[완료] B-05.1 비교 요약 생성이 끝났습니다.")
    print(f"- 비교 CSV: {OUT_CSV}")
    print(f"- 비교 JSON: {OUT_JSON}")


if __name__ == "__main__":
    main()

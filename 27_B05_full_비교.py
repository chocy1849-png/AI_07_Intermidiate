from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RAG_OUTPUTS = BASE_DIR / "rag_outputs"

B02_DIR = RAG_OUTPUTS / "b02_prefix_v2_full_eval"
B05_DIR = RAG_OUTPUTS / "b05_full_eval"
OUT_CSV = RAG_OUTPUTS / "b05_full_compare.csv"
OUT_JSON = RAG_OUTPUTS / "b05_full_compare.json"


def load_manual_summary(eval_dir: Path) -> pd.DataFrame:
    path = eval_dir / "baseline_eval_manual_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    metric_cols = [c for c in df.columns if c != "group_name"]
    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def main() -> None:
    b02 = load_manual_summary(B02_DIR).add_prefix("b02_").rename(columns={"b02_group_name": "group_name"})
    b05 = load_manual_summary(B05_DIR).add_prefix("b05_").rename(columns={"b05_group_name": "group_name"})

    compare = b02.merge(b05, on="group_name", how="outer")

    metric_pairs = [
        ("question_count", "delta_question_count"),
        ("avg_faithfulness_score", "delta_avg_faithfulness_score"),
        ("avg_completeness_score", "delta_avg_completeness_score"),
        ("avg_groundedness_score", "delta_avg_groundedness_score"),
        ("avg_relevancy_score", "delta_avg_relevancy_score"),
        ("avg_manual_eval_score", "delta_avg_manual_eval_score"),
    ]
    for metric, delta_col in metric_pairs:
        compare[delta_col] = compare[f"b05_{metric}"] - compare[f"b02_{metric}"]

    compare.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    overall_row = compare.loc[compare["group_name"] == "overall"]
    overall = overall_row.iloc[0].to_dict() if len(overall_row) else {}

    payload = {
        "baseline_dir": str(B02_DIR),
        "b05_dir": str(B05_DIR),
        "overall": overall,
        "rows": compare.to_dict(orient="records"),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("[완료] B-05 full 비교 요약 생성이 끝났습니다.")
    print(f"- 비교 CSV: {OUT_CSV}")
    print(f"- 비교 JSON: {OUT_JSON}")


if __name__ == "__main__":
    main()

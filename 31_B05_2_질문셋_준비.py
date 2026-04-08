from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
EVAL_SET_CSV = BASE_DIR / "rag_outputs" / "eval_sets" / "b05_table_eval_questions_v2.csv"
OUT_TXT = BASE_DIR / "evaluation" / "day4_b05_group_bc_question_ids_v1.txt"
OUT_CSV = BASE_DIR / "rag_outputs" / "eval_sets" / "b05_group_bc_questions_v1.csv"


def main() -> None:
    df = pd.read_csv(EVAL_SET_CSV)
    sub = df.loc[df["group_label"].isin(["Group B", "Group C"])].copy()
    sub = sub.sort_values("question_index").reset_index(drop=True)

    question_ids = sub["question_id"].astype(str).tolist()
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(question_ids), encoding="utf-8")
    sub.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print("[완료] B-05.2 Group B/C 질문셋 준비가 끝났습니다.")
    print(f"- 질문 수: {len(sub)}")
    print(f"- question_id 파일: {OUT_TXT}")
    print(f"- CSV: {OUT_CSV}")


if __name__ == "__main__":
    main()

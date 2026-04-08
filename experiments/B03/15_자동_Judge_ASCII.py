from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_EVAL_DIR = BASE_DIR / "rag_outputs" / "baseline_eval"
ALLOWED_CHAT_MODELS = ("gpt-5-mini", "gpt-5-nano", "gpt-5")


def load_openai_client():
    import os

    load_dotenv(BASE_DIR / ".env", override=False)
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip()

    if not api_key:
        raise RuntimeError(".env 파일에 OPENAI_API_KEY가 없습니다.")
    if not base_url:
        base_url = "https://api.openai.com/v1"
    elif not re.match(r"^https?://", base_url):
        raise RuntimeError("OPENAI_BASE_URL은 http:// 또는 https://로 시작해야 합니다.")

    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url)


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


def to_number(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def average(values: list[float | None]) -> float | None:
    valid = [x for x in values if x is not None]
    if not valid:
        return None
    return round(sum(valid) / len(valid), 4)


def extract_json(text: str) -> dict[str, Any]:
    cleaned = str(text).strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError("JSON 객체를 찾지 못했습니다.")
    return json.loads(match.group(0))


def judge_prompt(row: dict[str, Any]) -> tuple[str, str]:
    system_prompt = (
        "당신은 공공 제안요청서 RAG 답변 평가자다.\n"
        "주어진 질문, 검색 문맥, 모델 답변, 정답 힌트를 보고 4개 항목을 1~5점으로 채점하라.\n"
        "항목:\n"
        "- faithfulness_score: 문맥 밖 내용을 끼워넣지 않았는가\n"
        "- completeness_score: 질문이 요구한 핵심 요소를 빠뜨리지 않았는가\n"
        "- groundedness_score: 답변이 검색 문맥에 직접 근거하는가\n"
        "- relevancy_score: 질문에 직접 답하고 있는가\n"
        "반드시 JSON 객체만 반환하라."
    )
    user_prompt = (
        f"question_id: {row.get('question_id', '')}\n"
        f"type_group: {row.get('type_group', '')}\n"
        f"answer_type: {row.get('answer_type', '')}\n"
        f"question: {row.get('question', '')}\n"
        f"ground_truth_hint: {row.get('ground_truth_hint', '')}\n"
        f"expected: {row.get('expected', '')}\n\n"
        f"[검색 문맥]\n{row.get('retrieval_context', '')}\n\n"
        f"[모델 답변]\n{row.get('answer_text', '')}\n\n"
        "아래 형식의 JSON만 반환하라.\n"
        "{\n"
        '  "faithfulness_score": 1,\n'
        '  "completeness_score": 1,\n'
        '  "groundedness_score": 1,\n'
        '  "relevancy_score": 1,\n'
        '  "evaluator_note": "짧은 평가 메모"\n'
        "}"
    )
    return system_prompt, user_prompt


def judge_row(client, judge_model: str, row: dict[str, Any]) -> dict[str, Any]:
    system_prompt, user_prompt = judge_prompt(row)
    response = client.responses.create(
        model=judge_model,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
        ],
    )
    payload = extract_json(response.output_text)
    return {
        "faithfulness_score": int(payload["faithfulness_score"]),
        "completeness_score": int(payload["completeness_score"]),
        "groundedness_score": int(payload["groundedness_score"]),
        "relevancy_score": int(payload["relevancy_score"]),
        "evaluator_note": str(payload.get("evaluator_note", "")).strip(),
    }


def build_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    group_specs: list[tuple[str, list[dict[str, Any]]]] = [("overall", rows)]
    for type_group in sorted({row["type_group"] for row in rows if row.get("type_group")}):
        group_specs.append((type_group, [row for row in rows if row.get("type_group") == type_group]))
    for answer_type in sorted({row["answer_type"] for row in rows if row.get("answer_type")}):
        group_specs.append((f"answer_type:{answer_type}", [row for row in rows if row.get("answer_type") == answer_type]))

    summary_rows: list[dict[str, Any]] = []
    for label, group_rows in group_specs:
        summary_rows.append(
            {
                "group_name": label,
                "question_count": len(group_rows),
                "avg_faithfulness_score": average([to_number(row.get("faithfulness_score")) for row in group_rows]),
                "avg_completeness_score": average([to_number(row.get("completeness_score")) for row in group_rows]),
                "avg_groundedness_score": average([to_number(row.get("groundedness_score")) for row in group_rows]),
                "avg_relevancy_score": average([to_number(row.get("relevancy_score")) for row in group_rows]),
                "avg_manual_eval_score": average(
                    [
                        average(
                            [
                                to_number(row.get("faithfulness_score")),
                                to_number(row.get("completeness_score")),
                                to_number(row.get("groundedness_score")),
                                to_number(row.get("relevancy_score")),
                            ]
                        )
                        for row in group_rows
                    ]
                ),
            }
        )
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="ASCII judge wrapper for evaluation directories")
    parser.add_argument("--eval-dir", default=str(DEFAULT_EVAL_DIR))
    parser.add_argument("--judge-model", default="gpt-5", choices=ALLOWED_CHAT_MODELS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    results_csv = eval_dir / "baseline_eval_results.csv"
    manifest_json = eval_dir / "baseline_eval_manifest.json"
    completed_csv = eval_dir / "baseline_eval_manual_completed.csv"
    summary_csv = eval_dir / "baseline_eval_manual_summary.csv"
    judge_manifest_json = eval_dir / "baseline_eval_manual_judge_manifest.json"

    if not results_csv.exists():
        raise FileNotFoundError(f"평가 결과 CSV가 없습니다: {results_csv}")
    if not manifest_json.exists():
        raise FileNotFoundError(f"평가 manifest가 없습니다: {manifest_json}")

    result_rows = read_csv(results_csv)
    existing_rows = read_csv(completed_csv) if args.resume and completed_csv.exists() else []
    existing_index = {str(row.get("question_id", "")).strip(): row for row in existing_rows}

    client = load_openai_client()
    output_rows: list[dict[str, Any]] = []

    for row in result_rows:
        question_id = str(row.get("question_id", "")).strip()
        existing = existing_index.get(question_id)
        if existing and all(str(existing.get(key, "")).strip() for key in ("faithfulness_score", "completeness_score", "groundedness_score", "relevancy_score")):
            output_rows.append(existing)
            continue

        started_at = time.time()
        judged = judge_row(client=client, judge_model=args.judge_model, row=row)
        new_row = dict(row)
        new_row.update(judged)
        new_row["judge_model"] = args.judge_model
        new_row["judge_elapsed_sec"] = round(time.time() - started_at, 2)
        output_rows.append(new_row)
        write_csv(completed_csv, output_rows)
        print(
            f"[채점완료] {question_id} | "
            f"F={new_row['faithfulness_score']} C={new_row['completeness_score']} "
            f"G={new_row['groundedness_score']} R={new_row['relevancy_score']} | "
            f"{new_row['judge_elapsed_sec']}초"
        )

    summary_rows = build_summary(output_rows)
    write_csv(summary_csv, summary_rows)
    judge_manifest_json.write_text(
        json.dumps(
            {
                "source_results_csv": str(results_csv),
                "source_manifest": str(manifest_json),
                "question_count": len(output_rows),
                "judge_model": args.judge_model,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    overall = next((row for row in summary_rows if row["group_name"] == "overall"), None)
    print("[완료] 자동 judge 채점이 끝났습니다.")
    if overall is not None:
        print(f"- 질문 수: {overall['question_count']}")
        print(f"- overall avg_faithfulness_score: {overall['avg_faithfulness_score']}")
        print(f"- overall avg_completeness_score: {overall['avg_completeness_score']}")
        print(f"- overall avg_groundedness_score: {overall['avg_groundedness_score']}")
        print(f"- overall avg_relevancy_score: {overall['avg_relevancy_score']}")
        print(f"- overall avg_manual_eval_score: {overall['avg_manual_eval_score']}")


if __name__ == "__main__":
    main()

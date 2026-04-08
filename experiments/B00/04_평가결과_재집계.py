from __future__ import annotations

import csv
import difflib
import json
import re
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
QUESTION_SET_PATH = BASE_DIR / "evaluation" / "day3_partA_eval_questions_v1.txt"
EVAL_DIR = BASE_DIR / "rag_outputs" / "baseline_eval"
RESULTS_CSV_PATH = EVAL_DIR / "baseline_eval_results.csv"
QUESTIONS_CSV_PATH = EVAL_DIR / "baseline_eval_questions_parsed.csv"
SUMMARY_CSV_PATH = EVAL_DIR / "baseline_eval_summary.csv"
MANUAL_TEMPLATE_CSV_PATH = EVAL_DIR / "baseline_eval_manual_template.csv"
RESULTS_JSONL_PATH = EVAL_DIR / "baseline_eval_results.jsonl"

QUESTION_HEADER_RE = re.compile(r"^Q(?P<num>\d+)(?: \[(?P<turn>\d+)턴])?$")
FIELD_RE = re.compile(r"^\s*([A-Za-z_]+)\s*:\s*(.+?)\s*$")
TYPE_RE = re.compile(r"^TYPE\s+(?P<type_num>\d+)\s*:\s*(?P<label>.+)$")
SCENARIO_RE = re.compile(r"^---\s*(?P<label>시나리오.+?)\s*---$")

REJECTION_PATTERNS = [
    r"문서에.*정보가 없습니다",
    r"제공된 문서.*정보가 없습니다",
    r"제공된 문서.*없습니다",
    r"문맥에 없습니다",
    r"확인되지 않습니다",
    r"문서에서 확인되지 않습니다",
    r"기재되어 있지 않습니다",
    r"명시되어 있지 않습니다",
    r"자료가 없습니다",
    r"정보가 없습니다",
    r"문서.*없",
]


def 질문셋_파싱(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    current_type_num = ""
    current_type_label = ""
    current_scenario = ""

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        type_match = TYPE_RE.match(line)
        if type_match:
            current_type_num = f"TYPE {type_match.group('type_num')}"
            current_type_label = type_match.group("label").strip()
            current_scenario = ""
            continue

        scenario_match = SCENARIO_RE.match(line)
        if scenario_match:
            current_scenario = scenario_match.group("label").strip()
            continue

        question_match = QUESTION_HEADER_RE.match(line)
        if question_match:
            if current is not None:
                rows.append(current)
            qnum = int(question_match.group("num"))
            current = {
                "question_id": f"Q{qnum:02d}",
                "question_index": qnum,
                "turn_index": int(question_match.group("turn")) if question_match.group("turn") else None,
                "type_group": current_type_num,
                "type_label": current_type_label,
                "scenario_label": current_scenario,
            }
            continue

        if current is None:
            continue

        field_match = FIELD_RE.match(raw_line)
        if field_match:
            key = field_match.group(1).strip()
            value = field_match.group(2).strip()
            current[key] = value

    if current is not None:
        rows.append(current)

    for row in rows:
        row.setdefault("question", "")
        row.setdefault("answer_type", "")
        row.setdefault("ground_truth_doc", "")
        row.setdefault("ground_truth_docs", "")
        row.setdefault("ground_truth_hint", "")
        row.setdefault("eval_focus", "")
        row.setdefault("depends_on", "")
        row.setdefault("expected", "")
        if not row["answer_type"] and (row["type_group"] == "TYPE 4" or row["expected"]):
            row["answer_type"] = "rejection"
        row["depends_on_list"] = [
            item.strip()
            for item in str(row.get("depends_on", "")).split(",")
            if item.strip() and item.strip() != "-"
        ]

    return rows


def 정규화(text: str) -> str:
    value = str(text or "").lower().strip()
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"[^0-9a-z가-힣._-]+", "", value)
    return value


def 유사일치(target: str, candidate: str) -> bool:
    if not target or not candidate:
        return False
    if target in candidate or candidate in target:
        return True
    return difflib.SequenceMatcher(None, target, candidate).ratio() >= 0.86


def 정답문서_목록(question_row: dict[str, Any]) -> list[str]:
    if question_row.get("ground_truth_doc"):
        return [str(question_row["ground_truth_doc"]).strip()]
    if question_row.get("ground_truth_docs"):
        return [item.strip() for item in str(question_row["ground_truth_docs"]).split("+") if item.strip()]
    return []


def 정답문서_적중률(ground_truth_docs: list[str], retrieved_files: list[str]) -> tuple[int | None, int | None, float | None]:
    if not ground_truth_docs:
        return None, None, None

    normalized_retrieved = [정규화(x) for x in retrieved_files]
    normalized_ground_truth = [정규화(x) for x in ground_truth_docs]

    top1_hit = 0
    if normalized_retrieved:
        top1_hit = int(any(유사일치(target, normalized_retrieved[0]) for target in normalized_ground_truth))

    matched_count = 0
    for target in normalized_ground_truth:
        if any(유사일치(target, candidate) for candidate in normalized_retrieved):
            matched_count += 1

    topk_hit = int(matched_count > 0)
    hit_rate = matched_count / len(normalized_ground_truth)
    return top1_hit, topk_hit, hit_rate


def 거절응답_감지(answer: str) -> int:
    text = str(answer or "")
    return int(any(re.search(pattern, text) for pattern in REJECTION_PATTERNS))


def csv_읽기(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def csv_저장(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def 숫자값(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def 평균(values: list[float | None]) -> float | None:
    valid = [x for x in values if x is not None]
    if not valid:
        return None
    return round(sum(valid) / len(valid), 4)


def main() -> None:
    if not QUESTION_SET_PATH.exists():
        raise FileNotFoundError(f"질문셋 파일이 없습니다: {QUESTION_SET_PATH}")
    if not RESULTS_CSV_PATH.exists():
        raise FileNotFoundError(f"기존 평가 결과가 없습니다: {RESULTS_CSV_PATH}")

    question_rows = 질문셋_파싱(QUESTION_SET_PATH)
    questions_by_id = {row["question_id"]: row for row in question_rows}
    result_rows = csv_읽기(RESULTS_CSV_PATH)

    updated_rows: list[dict[str, Any]] = []
    for row in result_rows:
        question_row = questions_by_id.get(row["question_id"])
        if question_row is None:
            updated_rows.append(row)
            continue

        row["question_index"] = question_row["question_index"]
        row["type_group"] = question_row["type_group"]
        row["type_label"] = question_row["type_label"]
        row["scenario_label"] = question_row["scenario_label"]
        row["turn_index"] = question_row["turn_index"] or ""
        row["answer_type"] = question_row["answer_type"]
        row["depends_on"] = question_row.get("depends_on", "")
        row["ground_truth_doc"] = question_row.get("ground_truth_doc", "")
        row["ground_truth_docs"] = question_row.get("ground_truth_docs", "")
        row["ground_truth_hint"] = question_row.get("ground_truth_hint", "")
        row["expected"] = question_row.get("expected", "")
        row["eval_focus"] = question_row.get("eval_focus", "")

        retrieved_files = [
            item.strip()
            for item in str(row.get("retrieved_source_files", "")).split(" | ")
            if item.strip()
        ]
        ground_truth_docs = 정답문서_목록(question_row)
        top1_doc_hit, topk_doc_hit, doc_hit_rate = 정답문서_적중률(ground_truth_docs, retrieved_files)

        row["top1_doc_hit"] = "" if top1_doc_hit is None else top1_doc_hit
        row["topk_doc_hit"] = "" if topk_doc_hit is None else topk_doc_hit
        row["ground_truth_doc_hit_rate"] = "" if doc_hit_rate is None else round(doc_hit_rate, 4)

        rejection_expected = int(question_row["answer_type"] == "rejection" or bool(question_row.get("expected")))
        rejection_detected = 거절응답_감지(row.get("answer_text", ""))
        row["rejection_expected"] = rejection_expected
        row["rejection_detected"] = rejection_detected
        row["rejection_success"] = rejection_detected if rejection_expected else ""

        updated_rows.append(row)

    parsed_question_rows = [
        {
            "question_id": row["question_id"],
            "question_index": row["question_index"],
            "type_group": row["type_group"],
            "type_label": row["type_label"],
            "scenario_label": row["scenario_label"],
            "turn_index": row["turn_index"] or "",
            "answer_type": row["answer_type"],
            "question": row["question"],
            "depends_on": row.get("depends_on", ""),
            "ground_truth_doc": row.get("ground_truth_doc", ""),
            "ground_truth_docs": row.get("ground_truth_docs", ""),
            "ground_truth_hint": row.get("ground_truth_hint", ""),
            "expected": row.get("expected", ""),
            "eval_focus": row.get("eval_focus", ""),
        }
        for row in question_rows
    ]

    group_specs: list[tuple[str, list[dict[str, Any]]]] = [("overall", updated_rows)]
    for type_group in sorted({row["type_group"] for row in updated_rows if row["type_group"]}):
        group_specs.append((type_group, [row for row in updated_rows if row["type_group"] == type_group]))
    for answer_type in sorted({row["answer_type"] for row in updated_rows if row["answer_type"]}):
        group_specs.append((f"answer_type:{answer_type}", [row for row in updated_rows if row["answer_type"] == answer_type]))

    summary_rows: list[dict[str, Any]] = []
    for label, rows in group_specs:
        summary_rows.append(
            {
                "group_name": label,
                "question_count": len(rows),
                "top1_doc_hit_rate": 평균([숫자값(row.get("top1_doc_hit")) for row in rows]),
                "topk_doc_hit_rate": 평균([숫자값(row.get("topk_doc_hit")) for row in rows]),
                "avg_ground_truth_doc_hit_rate": 평균([숫자값(row.get("ground_truth_doc_hit_rate")) for row in rows]),
                "rejection_success_rate": 평균([숫자값(row.get("rejection_success")) for row in rows]),
                "avg_elapsed_sec": 평균([숫자값(row.get("elapsed_sec")) for row in rows]),
                "avg_answer_chars": 평균([숫자값(row.get("answer_chars")) for row in rows]),
            }
        )

    manual_template_rows: list[dict[str, Any]] = []
    for row in updated_rows:
        manual_row = dict(row)
        manual_row.setdefault("faithfulness_score", "")
        manual_row.setdefault("completeness_score", "")
        manual_row.setdefault("groundedness_score", "")
        manual_row.setdefault("relevancy_score", "")
        manual_row.setdefault("evaluator_note", "")
        manual_template_rows.append(manual_row)

    csv_저장(QUESTIONS_CSV_PATH, parsed_question_rows)
    csv_저장(RESULTS_CSV_PATH, updated_rows)
    csv_저장(SUMMARY_CSV_PATH, summary_rows)
    csv_저장(MANUAL_TEMPLATE_CSV_PATH, manual_template_rows)
    RESULTS_JSONL_PATH.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in updated_rows),
        encoding="utf-8",
    )

    overall = next((row for row in summary_rows if row["group_name"] == "overall"), None)
    print("[완료] 기존 평가 결과 재집계를 마쳤습니다.")
    if overall is not None:
        print(f"- 질문 수: {overall['question_count']}")
        print(f"- overall top1_doc_hit_rate: {overall['top1_doc_hit_rate']}")
        print(f"- overall topk_doc_hit_rate: {overall['topk_doc_hit_rate']}")
        print(f"- overall rejection_success_rate: {overall['rejection_success_rate']}")
    print(f"- 결과 CSV: {RESULTS_CSV_PATH}")
    print(f"- 요약 CSV: {SUMMARY_CSV_PATH}")


if __name__ == "__main__":
    main()

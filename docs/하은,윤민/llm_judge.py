"""
LLM Judge 평가 엔진 (설계안 기준 v2)

- 축소셋 20문항을 실제 RAG 파이프라인으로 실행
- 멀티턴 depends_on 지원
- GPT Judge로 4지표 채점
  - Faithfulness
  - Completeness
  - Groundedness
  - Relevancy

사용법:
    python scripts/llm_judge.py \
        --입력 data/llm_judge/judge_subset_20.json \
        --출력경로 data/llm_judge/results

출력:
    judge_detail.csv
    judge_summary.csv
    judge_report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_JUDGE_INPUT = "judge_subset_20.json"


RAG_SEARCH_MODE = "하이브리드"
RAG_TOP_K = 10


JUDGE_SYSTEM_PROMPT = """\
당신은 정부 RFP(제안요청서) 기반 QA 시스템의 전문 평가자입니다.
반드시 제공된 질문, 정답 힌트, 기대 동작, 검색 문맥, 모델 응답만 보고 평가하세요.

평가 항목:
1. faithfulness_score: 검색 문맥에 없는 사실을 꾸며내지 않았는가
2. completeness_score: 질문이 요구한 핵심 요소를 빠뜨리지 않았는가
3. groundedness_score: 답변이 검색 문맥에 직접 근거하고 있는가
4. relevancy_score: 질문에 직접 답하고 있는가

채점 규칙:
- 각 점수는 1~5의 정수만 사용합니다.
- 거부형(rejection) 질문은 문서에 없는 정보를 추정하지 않고 적절히 거부하면 높은 점수를 줍니다.
- 수치, 날짜, 기관명, URL 등 핵심 사실이 틀리면 낮게 평가합니다.
- ground_truth_hint는 정답의 핵심 기준이며, expected가 있으면 반드시 함께 고려합니다.
- 출력은 반드시 JSON 객체 하나만 반환합니다.

반드시 아래 형식만 반환하세요.
{
  "faithfulness_score": 1,
  "completeness_score": 1,
  "groundedness_score": 1,
  "relevancy_score": 1,
  "evaluator_note": "짧은 평가 메모"
}
"""


JUDGE_USER_TEMPLATE = """\
[문항 정보]
- id: {item_id}
- judge_type: {judge_type}
- category: {category}
- difficulty: {difficulty}
- document_name: {document_name}
- document_names: {document_names}
- depends_on: {depends_on}

[질문]
{question}

[정답 힌트]
{ground_truth_hint}

[기대 동작]
{expected}

[평가 포인트]
{eval_focus}

[검색 문맥]
{retrieval_context}

[모델 응답]
{rag_answer}
"""


_FILE_NAME_CACHE: dict[str, str] = {}


def _normalize_fname(name: str) -> str:
    name = re.sub(r"\s+", " ", name).strip()
    return re.sub(r"\s+\.(hwp|pdf|xlsx?)$", r".\1", name, flags=re.I)


def _nospace(name: str) -> str:
    return re.sub(r"\s+", "", name)


def _resolve_file_name(document_name: str) -> str:
    if not document_name:
        return ""
    if document_name in _FILE_NAME_CACHE:
        return _FILE_NAME_CACHE[document_name]

    from src.db.parsed_store import load_parsed_documents

    stored_names = [d["metadata"].get("file_name", "") for d in load_parsed_documents()]

    if document_name in stored_names:
        _FILE_NAME_CACHE[document_name] = document_name
        return document_name

    doc_norm = _normalize_fname(document_name)
    for stored in stored_names:
        if _normalize_fname(stored) == doc_norm:
            _FILE_NAME_CACHE[document_name] = stored
            return stored

    doc_ns = _nospace(document_name)
    for stored in stored_names:
        stored_ns = _nospace(stored)
        if doc_ns in stored_ns or stored_ns in doc_ns:
            _FILE_NAME_CACHE[document_name] = stored
            return stored

    _FILE_NAME_CACHE[document_name] = ""
    return ""


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _score_int(value: Any) -> int:
    try:
        score = int(float(value))
    except (TypeError, ValueError):
        return 1
    return max(1, min(5, score))


def _normalize_depends_on(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if "," in text:
            return [part.strip() for part in text.split(",") if part.strip()]
        return [text]
    return []


def _build_chat_history(
    items_by_id: dict[str, dict[str, Any]],
    answers_by_id: dict[str, dict[str, Any]],
    depends_on_ids: list[str],
) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    for dep_id in depends_on_ids:
        dep_item = items_by_id.get(dep_id)
        dep_answer = answers_by_id.get(dep_id)
        if not dep_item or not dep_answer:
            continue
        history.append({"role": "user", "content": dep_item["question"]})
        history.append({"role": "assistant", "content": dep_answer["rag_answer"]})
    return history


def _build_retrieval_context(retrieved_docs: list[dict[str, Any]]) -> str:
    if not retrieved_docs:
        return "검색 문맥 없음"

    # 생성 단계와 동일하게 최대 10개 문서를 Judge 문맥으로 제공
    blocks: list[str] = []
    for idx, doc in enumerate(retrieved_docs[:10], start=1):
        meta = doc.get("metadata", {})
        blocks.append(
            "\n".join(
                [
                    f"[검색 결과 {idx}]",
                    f"- file_name: {meta.get('file_name', '')}",
                    f"- agency: {meta.get('agency', '')}",
                    f"- chunk_id: {doc.get('chunk_id', '')}",
                    f"- score: {round(max(0.0, 1 - doc.get('distance', 1.0)), 4)}",
                    doc.get("text", "")[:600],
                ]
            )
        )
    return "\n\n".join(blocks)


def _serialize_document_names(item: dict[str, Any]) -> str:
    names = item.get("document_names") or []
    if isinstance(names, list) and names:
        return ", ".join(str(name) for name in names)
    return ""


def _build_query(item: dict[str, Any]) -> str:
    query = str(item["question"]).strip()
    choices = item.get("choices") or []
    if choices:
        choices_text = "\n".join(f"  {choice}" for choice in choices)
        query = (
            f"{query}\n\n"
            f"[선택지]\n{choices_text}\n\n"
            "반드시 위 선택지에 근거해 답하세요."
        )
    return query


def _run_rag(item: dict[str, Any], chat_history: list[dict[str, str]]) -> dict[str, Any]:
    import sys
    from pathlib import Path as _Path

    root = PROJECT_ROOT
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from dotenv import load_dotenv

    load_dotenv(root / ".env")

    from rag_pipeline import answer_query

    document_name = str(item.get("document_name", "")).strip()
    filters = None
    if document_name:
        resolved_name = _resolve_file_name(document_name)
        if resolved_name:
            filters = {"file_name": resolved_name}

    result = answer_query(
        query=_build_query(item),
        search_mode=str(item.get("search_mode") or RAG_SEARCH_MODE),
        filters=filters,
        top_k=int(item.get("top_k") or RAG_TOP_K),
        chat_history=chat_history,
        eval_mode=bool(item.get("choices")),
    )

    return {
        "rag_answer": result.get("answer", ""),
        "retrieved_docs": result.get("retrieved_docs", []),
        "filters": result.get("debug", {}).get("filters", {}),
    }


def _judge_prompt_inputs(
    item: dict[str, Any],
    rag_answer: str,
    retrieval_context: str,
) -> str:
    return JUDGE_USER_TEMPLATE.format(
        item_id=item.get("id", ""),
        judge_type=item.get("judge_type", ""),
        category=item.get("category", ""),
        difficulty=item.get("difficulty", ""),
        document_name=item.get("document_name", ""),
        document_names=_serialize_document_names(item),
        depends_on=", ".join(_normalize_depends_on(item.get("depends_on"))) or "-",
        question=item.get("question", ""),
        ground_truth_hint=item.get("ground_truth_hint", "") or item.get("answer", ""),
        expected=item.get("expected", "") or "-",
        eval_focus=item.get("eval_focus", "") or "-",
        retrieval_context=retrieval_context,
        rag_answer=rag_answer,
    )


def _call_judge_model(
    item: dict[str, Any],
    rag_answer: str,
    retrieval_context: str,
    model: str,
) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI()
    request = {
        "model": model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _judge_prompt_inputs(
                    item=item,
                    rag_answer=rag_answer,
                    retrieval_context=retrieval_context,
                ),
            },
        ],
        "response_format": {"type": "json_object"},
    }
    # gpt-5 계열은 현재 temperature=0 명시를 지원하지 않아 기본값으로 호출한다.
    if not str(model).startswith("gpt-5"):
        request["temperature"] = 0

    response = client.chat.completions.create(**request)

    raw = response.choices[0].message.content or "{}"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = {
            "faithfulness_score": 1,
            "completeness_score": 1,
            "groundedness_score": 1,
            "relevancy_score": 1,
            "evaluator_note": f"JSON 파싱 실패: {raw[:120]}",
        }

    return {
        "faithfulness_score": _score_int(payload.get("faithfulness_score")),
        "completeness_score": _score_int(payload.get("completeness_score")),
        "groundedness_score": _score_int(payload.get("groundedness_score")),
        "relevancy_score": _score_int(payload.get("relevancy_score")),
        "evaluator_note": str(payload.get("evaluator_note", "")).strip(),
        "raw": raw,
    }


def _compute_verdict(item: dict[str, Any], judged: dict[str, Any]) -> tuple[float, str]:
    scores = [
        judged["faithfulness_score"],
        judged["completeness_score"],
        judged["groundedness_score"],
        judged["relevancy_score"],
    ]
    avg_score = _mean([float(score) for score in scores])
    judge_type = str(item.get("judge_type", "")).strip()

    if judge_type == "rejection":
        if min(
            judged["faithfulness_score"],
            judged["groundedness_score"],
            judged["relevancy_score"],
        ) <= 3:
            return avg_score, "fail"
        if avg_score >= 4.0 and judged["faithfulness_score"] >= 4:
            return avg_score, "pass"
        return avg_score, "review"

    if avg_score >= 4.0 and judged["faithfulness_score"] >= 4:
        return avg_score, "pass"
    if avg_score < 3.0 or judged["faithfulness_score"] <= 2 or judged["relevancy_score"] <= 2:
        return avg_score, "fail"
    return avg_score, "review"


def run_judge(items: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
    items_by_id = {str(item["id"]): item for item in items}
    answers_by_id: dict[str, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []

    for index, item in enumerate(items, start=1):
        started = time.time()
        item_id = str(item["id"])
        print(f"  [{index:02d}/{len(items)}] {item_id} 실행 중...", end=" ", flush=True)

        depends_on_ids = _normalize_depends_on(item.get("depends_on"))
        chat_history = _build_chat_history(items_by_id, answers_by_id, depends_on_ids)

        try:
            rag_result = _run_rag(item, chat_history=chat_history)
        except Exception as exc:
            print(f"[RAG 오류] {exc}")
            rag_result = {"rag_answer": f"[RAG 오류: {exc}]", "retrieved_docs": [], "filters": {}}

        rag_answer = rag_result["rag_answer"]
        retrieval_context = _build_retrieval_context(rag_result["retrieved_docs"])

        try:
            judged = _call_judge_model(
                item=item,
                rag_answer=rag_answer,
                retrieval_context=retrieval_context,
                model=model,
            )
        except Exception as exc:
            print(f"[Judge 오류] {exc}")
            judged = {
                "faithfulness_score": 1,
                "completeness_score": 1,
                "groundedness_score": 1,
                "relevancy_score": 1,
                "evaluator_note": f"Judge 호출 오류: {exc}",
                "raw": "",
            }

        avg_score, verdict = _compute_verdict(item, judged)
        elapsed = round(time.time() - started, 2)

        answers_by_id[item_id] = {
            "rag_answer": rag_answer,
        }

        print(f"{verdict.upper()} ({elapsed}s)")

        results.append(
            {
                "id": item_id,
                "judge_type": item.get("judge_type", ""),
                "category": item.get("category", ""),
                "difficulty": item.get("difficulty", ""),
                "document_name": item.get("document_name", ""),
                "document_names": _serialize_document_names(item),
                "depends_on": ", ".join(depends_on_ids),
                "question": item.get("question", ""),
                "ground_truth_hint": item.get("ground_truth_hint", "") or item.get("answer", ""),
                "expected": item.get("expected", ""),
                "eval_focus": item.get("eval_focus", ""),
                "rag_answer": rag_answer,
                "retrieval_context": retrieval_context[:6000],
                "faithfulness_score": judged["faithfulness_score"],
                "completeness_score": judged["completeness_score"],
                "groundedness_score": judged["groundedness_score"],
                "relevancy_score": judged["relevancy_score"],
                "avg_score": avg_score,
                "verdict": verdict,
                "evaluator_note": judged["evaluator_note"],
                "elapsed_sec": elapsed,
            }
        )

    return results


def aggregate(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[tuple[str, list[dict[str, Any]]]] = [("overall", results)]

    for judge_type in sorted({row["judge_type"] for row in results if row.get("judge_type")}):
        groups.append((f"judge_type:{judge_type}", [row for row in results if row.get("judge_type") == judge_type]))

    for difficulty in sorted({row["difficulty"] for row in results if row.get("difficulty")}):
        groups.append((f"difficulty:{difficulty}", [row for row in results if row.get("difficulty") == difficulty]))

    summary_rows: list[dict[str, Any]] = []
    for label, rows in groups:
        summary_rows.append(
            {
                "group": label,
                "total": len(rows),
                "pass_count": sum(1 for row in rows if row["verdict"] == "pass"),
                "review_count": sum(1 for row in rows if row["verdict"] == "review"),
                "fail_count": sum(1 for row in rows if row["verdict"] == "fail"),
                "avg_faithfulness_score": _mean([float(row["faithfulness_score"]) for row in rows]),
                "avg_completeness_score": _mean([float(row["completeness_score"]) for row in rows]),
                "avg_groundedness_score": _mean([float(row["groundedness_score"]) for row in rows]),
                "avg_relevancy_score": _mean([float(row["relevancy_score"]) for row in rows]),
                "avg_score": _mean([float(row["avg_score"]) for row in rows]),
            }
        )
    return summary_rows


def build_report(results: list[dict[str, Any]], summary: list[dict[str, Any]], model: str) -> str:
    overall = next(row for row in summary if row["group"] == "overall")
    low_rows = [row for row in results if row["verdict"] != "pass"]

    lines = [
        "# LLM Judge 평가 보고서",
        "",
        f"> 평가일: {time.strftime('%Y-%m-%d')}  ",
        f"> 모델: {model}  ",
        f"> 문항 수: {overall['total']}개  ",
        "",
        "---",
        "",
        "## 1. 전체 결과",
        "",
        f"- Pass: {overall['pass_count']}개",
        f"- Review: {overall['review_count']}개",
        f"- Fail: {overall['fail_count']}개",
        f"- Avg Score: {overall['avg_score']:.2f}",
        "",
        "| 지표 | 평균 점수 |",
        "|------|-----------|",
        f"| Faithfulness | {overall['avg_faithfulness_score']:.2f} |",
        f"| Completeness | {overall['avg_completeness_score']:.2f} |",
        f"| Groundedness | {overall['avg_groundedness_score']:.2f} |",
        f"| Relevancy | {overall['avg_relevancy_score']:.2f} |",
        "",
        "---",
        "",
        "## 2. 유형별 집계",
        "",
        "| 그룹 | Pass | Review | Fail | Avg |",
        "|------|------|--------|------|-----|",
    ]

    for row in summary[1:]:
        lines.append(
            f"| {row['group']} | {row['pass_count']} | {row['review_count']} | "
            f"{row['fail_count']} | {row['avg_score']:.2f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 3. 재검토/실패 문항",
        "",
    ]

    if not low_rows:
        lines.append("재검토 또는 실패 문항 없음")
    else:
        for row in low_rows:
            lines += [
                f"### {row['id']} ({row['verdict']})",
                f"- 질문: {row['question']}",
                f"- 정답 힌트: {row['ground_truth_hint']}",
                f"- RAG 응답: {row['rag_answer'][:300]}",
                f"- 점수: F {row['faithfulness_score']} / C {row['completeness_score']} / "
                f"G {row['groundedness_score']} / R {row['relevancy_score']}",
                f"- 메모: {row['evaluator_note']}",
                "",
            ]

    lines += [
        "---",
        "",
        "## 4. 해석",
        "",
        "- 이 결과는 Exact Match 대체가 아니라 서술형 품질 진단용이다.",
        "- Review/Fail 문항은 검색 실패인지, 응답 생성 실패인지, Judge 기준상 문제인지 분리해서 다시 보면 된다.",
        "- 멀티턴과 거부응답은 동일 질문셋을 파이프라인 비교용으로 재사용할 수 있다.",
    ]
    return "\n".join(lines)


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[저장] {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="설계안 기준 LLM Judge 평가 엔진")
    parser.add_argument("--입력", default="data/llm_judge/judge_subset_20.json")
    parser.add_argument("--출력경로", default="data/llm_judge/results")
    parser.add_argument("--모델", default="gpt-5")
    parser.add_argument("--제한", type=int, default=0, help="0이면 전체")
    args = parser.parse_args()

    root = PROJECT_ROOT
    input_path = root / args.입력
    output_dir = root / args.출력경로

    data = json.loads(input_path.read_text(encoding="utf-8"))
    items = data["items"]
    if args.제한 > 0:
        items = items[:args.제한]

    print(f"[로드] {len(items)}문항 | 모델: {args.모델}")
    print("=" * 60)

    results = run_judge(items, model=args.모델)
    summary = aggregate(results)
    report = build_report(results, summary, model=args.모델)

    print("\n" + "=" * 60)
    print("[집계 결과]")
    for row in summary:
        if row["group"] == "overall" or row["group"].startswith("judge_type:"):
            print(
                f"  {row['group']:24s} | pass:{row['pass_count']} review:{row['review_count']} "
                f"fail:{row['fail_count']} avg:{row['avg_score']:.2f}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(output_dir / "judge_detail.csv", results)
    save_csv(output_dir / "judge_summary.csv", summary)
    (output_dir / "judge_report.md").write_text(report, encoding="utf-8")
    print(f"[저장] {output_dir / 'judge_report.md'}")
    print(f"\n[완료] {len(results)}문항 LLM Judge 평가")


def cli_main() -> None:
    parser = argparse.ArgumentParser(description="설계안 기준 LLM Judge 평가 엔진")
    parser.add_argument(
        "--input",
        default=DEFAULT_JUDGE_INPUT,
        help="평가 입력 JSON 경로(프로젝트 루트 기준 상대경로 또는 절대경로)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/llm_judge/results",
        help="결과 저장 디렉토리(프로젝트 루트 기준 상대경로 또는 절대경로)",
    )
    parser.add_argument("--model", default="gpt-5", help="Judge 모델명")
    parser.add_argument("--limit", type=int, default=0, help="0이면 전체")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="입력/출력/개수만 검증하고 실제 RAG+Judge 실행은 건너뜀",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    data = json.loads(input_path.read_text(encoding="utf-8"))
    items = data["items"]
    if args.limit > 0:
        items = items[: args.limit]

    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "dry-run",
                    "input_path": str(input_path),
                    "output_dir": str(output_dir),
                    "model": args.model,
                    "item_count": len(items),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    print(f"[로드] {len(items)}문항 | 모델: {args.model}")
    print("=" * 60)

    results = run_judge(items, model=args.model)
    summary = aggregate(results)
    report = build_report(results, summary, model=args.model)

    print("\n" + "=" * 60)
    print("[집계 결과]")
    for row in summary:
        if row["group"] == "overall" or row["group"].startswith("judge_type:"):
            print(
                f"  {row['group']:24s} | pass:{row['pass_count']} review:{row['review_count']} "
                f"fail:{row['fail_count']} avg:{row['avg_score']:.2f}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(output_dir / "judge_detail.csv", results)
    save_csv(output_dir / "judge_summary.csv", summary)
    (output_dir / "judge_report.md").write_text(report, encoding="utf-8")
    print(f"[저장] {output_dir / 'judge_report.md'}")
    print(f"\n[완료] {len(results)}문항 LLM Judge 평가")


if __name__ == "__main__":
    cli_main()

from __future__ import annotations

import argparse
import difflib
import json
import re
import time
from pathlib import Path
from typing import Any

from rag_공통 import (
    ALLOWED_CHAT_MODELS,
    ALLOWED_EMBEDDING_MODELS,
    CHROMA_DIR,
    Chroma_컬렉션명_검증,
    DEFAULT_CHAT_MODEL,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    OpenAI_클라이언트_가져오기,
    OUTPUT_DIR,
    RESULTS_DIR,
    csv_저장,
    디렉토리_준비,
)


BASE_DIR = Path(__file__).resolve().parent
QUESTION_SET_PATH = BASE_DIR / "evaluation" / "day3_partA_eval_questions_v1.txt"
EVAL_DIR = OUTPUT_DIR / "baseline_eval"

QUESTION_HEADER_RE = re.compile(r"^Q(?P<num>\d+)(?: \[(?P<turn>\d+)턴\])?$")
FIELD_RE = re.compile(r"^\s*([A-Za-z_]+)\s*:\s*(.+?)\s*$")
TYPE_RE = re.compile(r"^TYPE\s+(?P<type_num>\d+)\s*:\s*(?P<label>.+)$")
SCENARIO_RE = re.compile(r"^---\s*(?P<label>시나리오.+?)\s*---$")


def 크로마_모듈_가져오기():
    try:
        import chromadb
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "chromadb 패키지가 설치되어 있지 않습니다. "
            "requirements.txt를 사용해 필요한 패키지를 먼저 설치하세요."
        ) from exc
    return chromadb


def 질의_임베딩(client, model_name: str, question: str) -> list[float]:
    response = client.embeddings.create(model=model_name, input=[question])
    return response.data[0].embedding


def 검색_문맥_구성(
    documents: list[str],
    metadatas: list[dict[str, Any]],
    distances: list[float],
) -> str:
    blocks: list[str] = []
    for index, (document, metadata, distance) in enumerate(zip(documents, metadatas, distances), start=1):
        blocks.append(
            "\n".join(
                [
                    f"[검색 결과 {index}]",
                    f"- 사업명: {metadata.get('사업명', '정보 없음')}",
                    f"- 발주 기관: {metadata.get('발주 기관', '정보 없음')}",
                    f"- 파일명: {metadata.get('source_file_name', '정보 없음')}",
                    f"- 청크 ID: {metadata.get('chunk_id', '정보 없음')}",
                    f"- 거리값: {distance}",
                    document,
                ]
            )
        )
    return "\n\n".join(blocks)


def 나이브_RAG_응답_생성(
    client,
    model_name: str,
    question: str,
    retrieval_context: str,
) -> str:
    system_prompt = (
        "당신은 정부 제안요청서(RFP) 기반 요약 도우미다.\n"
        "반드시 제공된 검색 문맥에 근거해서만 답변한다.\n"
        "문맥에 없는 사실은 추정하지 않는다.\n"
        "답변은 한국어로 작성하고, 불필요한 장식 없이 핵심만 정리한다.\n"
        "질문이 특정 사업의 요약을 요구하면 목적, 범위, 주요 요구사항, 일정/예산, 발주기관을 우선 정리한다.\n"
        "문맥에 일정이나 예산이 없으면 없다고 명시한다.\n"
        "마지막에는 참고한 문서명과 청크 ID를 간단히 정리한다."
    )
    user_prompt = (
        f"질문:\n{question}\n\n"
        f"검색 문맥:\n{retrieval_context}\n\n"
        "출력 형식:\n"
        "1. 한줄 요약\n"
        "2. 핵심 내용\n"
        "3. 주요 요구사항\n"
        "4. 일정/예산/발주기관\n"
        "5. 참고 근거"
    )

    response = client.responses.create(
        model=model_name,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_prompt}],
            },
        ],
    )
    return response.output_text.strip()


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
            part.strip()
            for part in str(row.get("depends_on", "")).split(",")
            if part.strip() and part.strip() != "-"
        ]

    return rows


def 정규화(text: str) -> str:
    value = str(text or "").lower().strip()
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"[^0-9a-z가-힣._-]+", "", value)
    return value


def 정답문서목록(row: dict[str, Any]) -> list[str]:
    if row.get("ground_truth_doc"):
        return [str(row["ground_truth_doc"]).strip()]
    if row.get("ground_truth_docs"):
        return [part.strip() for part in str(row["ground_truth_docs"]).split("+") if part.strip()]
    return []


def 정답문서_적중률(ground_truth_docs: list[str], retrieved_files: list[str]) -> tuple[int | None, int | None, float | None]:
    if not ground_truth_docs:
        return None, None, None

    normalized_retrieved = [정규화(x) for x in retrieved_files]
    normalized_ground_truth = [정규화(x) for x in ground_truth_docs]

    def matched(target: str, candidate: str) -> bool:
        if not (target and candidate):
            return False
        if target in candidate or candidate in target:
            return True
        return difflib.SequenceMatcher(None, target, candidate).ratio() >= 0.86

    top1_hit = 0
    if normalized_retrieved:
        top1_hit = int(any(matched(target, normalized_retrieved[0]) for target in normalized_ground_truth))

    matched_count = 0
    for target in normalized_ground_truth:
        if any(matched(target, candidate) for candidate in normalized_retrieved):
            matched_count += 1

    topk_hit = int(matched_count > 0)
    hit_rate = matched_count / len(normalized_ground_truth)
    return top1_hit, topk_hit, hit_rate


def 거부응답_탐지(answer: str) -> int:
    rejection_patterns = [
        r"문서에 해당 정보가 없습니다",
        r"문서에 없습니다",
        r"제공된 문서에 해당 정보가 없습니다",
        r"제공된 문서에 없습니다",
        r"문맥에 없습니다",
        r"확인할 수 없습니다",
        r"확인되지 않습니다",
        r"문서에서 확인되지 않습니다",
        r"문서상 확인되지 않습니다",
        r"기재되어 있지 않습니다",
        r"명시되어 있지 않습니다",
        r"알 수 없습니다",
        r"자료가 없습니다",
        r"정보가 없습니다",
        r"낙찰 업체.*없",
        r"수주.*업체.*없",
        r"제공된 문서.*없",
    ]
    text = str(answer or "")
    return int(any(re.search(pattern, text) for pattern in rejection_patterns))


def 히스토리_문자열(questions_by_id: dict[str, dict[str, Any]], answers_by_id: dict[str, str], depends_on_list: list[str]) -> str:
    parts: list[str] = []
    for qid in depends_on_list:
        question_row = questions_by_id.get(qid)
        answer_text = answers_by_id.get(qid, "")
        if not question_row:
            continue
        parts.append(f"사용자: {question_row.get('question', '')}")
        if answer_text:
            parts.append(f"도우미: {answer_text}")
    return "\n".join(parts).strip()


def 결과경로(base_name: str, extension: str) -> Path:
    디렉토리_준비(EVAL_DIR)
    return EVAL_DIR / f"{base_name}{extension}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Day3 평가 질문셋 45개를 사용해 현재 Naive RAG 베이스라인을 일괄 평가합니다."
    )
    parser.add_argument(
        "--질문셋경로",
        default=str(QUESTION_SET_PATH),
        help="평가 질문셋 TXT 경로",
    )
    parser.add_argument(
        "--크로마경로",
        default=str(CHROMA_DIR),
        help="Chroma 영속 저장 경로",
    )
    parser.add_argument(
        "--컬렉션이름",
        default=DEFAULT_COLLECTION_NAME,
        help="Chroma 컬렉션 이름",
    )
    parser.add_argument(
        "--임베딩모델",
        default=DEFAULT_EMBEDDING_MODEL,
        choices=ALLOWED_EMBEDDING_MODELS,
        help="질문 임베딩에 사용할 모델",
    )
    parser.add_argument(
        "--응답모델",
        default=DEFAULT_CHAT_MODEL,
        choices=ALLOWED_CHAT_MODELS,
        help="최종 응답 생성에 사용할 모델",
    )
    parser.add_argument(
        "--상위개수",
        type=int,
        default=5,
        help="검색할 상위 청크 개수",
    )
    parser.add_argument(
        "--질문개수제한",
        type=int,
        default=0,
        help="0이면 전체 질문 실행, 양수면 앞에서부터 일부만 실행",
    )
    parser.add_argument(
        "--후속질문히스토리적용",
        action="store_true",
        help="Type 3 follow-up 평가 시 이전 Q/A를 retrieval query에도 붙여서 실행",
    )
    args = parser.parse_args()

    question_set_path = Path(args.질문셋경로)
    if not question_set_path.exists():
        raise FileNotFoundError(f"질문셋 파일을 찾을 수 없습니다: {question_set_path}")

    컬렉션이름 = Chroma_컬렉션명_검증(args.컬렉션이름)
    question_rows = 질문셋_파싱(question_set_path)
    if args.질문개수제한 > 0:
        question_rows = question_rows[: args.질문개수제한]

    if not question_rows:
        raise RuntimeError("질문셋 파싱 결과가 비어 있습니다.")

    questions_by_id = {row["question_id"]: row for row in question_rows}

    client = OpenAI_클라이언트_가져오기()
    chromadb = 크로마_모듈_가져오기()
    chroma_client = chromadb.PersistentClient(path=args.크로마경로)
    collection = chroma_client.get_collection(name=컬렉션이름)

    answers_by_id: dict[str, str] = {}
    result_rows: list[dict[str, Any]] = []

    for row in question_rows:
        question_id = row["question_id"]
        history_text = 히스토리_문자열(
            questions_by_id=questions_by_id,
            answers_by_id=answers_by_id,
            depends_on_list=row["depends_on_list"],
        )

        retrieval_query = row["question"]
        if args.후속질문히스토리적용 and history_text:
            retrieval_query = f"이전 대화:\n{history_text}\n\n현재 질문:\n{row['question']}"

        started_at = time.time()

        query_embedding = 질의_임베딩(
            client=client,
            model_name=args.임베딩모델,
            question=retrieval_query,
        )
        query_result = collection.query(
            query_embeddings=[query_embedding],
            n_results=args.상위개수,
            include=["documents", "metadatas", "distances"],
        )

        documents = query_result["documents"][0]
        metadatas = query_result["metadatas"][0]
        distances = query_result["distances"][0]
        retrieved_files = [metadata.get("source_file_name", "") for metadata in metadatas]
        retrieval_context = 검색_문맥_구성(
            documents=documents,
            metadatas=metadatas,
            distances=distances,
        )

        answer_prompt_question = row["question"]
        if args.후속질문히스토리적용 and history_text:
            answer_prompt_question = f"이전 대화:\n{history_text}\n\n현재 질문:\n{row['question']}"

        answer = 나이브_RAG_응답_생성(
            client=client,
            model_name=args.응답모델,
            question=answer_prompt_question,
            retrieval_context=retrieval_context,
        )
        elapsed_sec = round(time.time() - started_at, 2)

        answers_by_id[question_id] = answer

        ground_truth_docs = 정답문서목록(row)
        top1_doc_hit, topk_doc_hit, doc_hit_rate = 정답문서_적중률(
            ground_truth_docs=ground_truth_docs,
            retrieved_files=retrieved_files,
        )
        rejection_detected = 거부응답_탐지(answer)
        rejection_expected = int(row.get("answer_type") == "rejection" or bool(row.get("expected")))
        rejection_success = None
        if rejection_expected:
            rejection_success = rejection_detected

        result_rows.append(
            {
                "question_id": question_id,
                "question_index": row["question_index"],
                "type_group": row["type_group"],
                "type_label": row["type_label"],
                "scenario_label": row["scenario_label"],
                "turn_index": row["turn_index"],
                "answer_type": row.get("answer_type", ""),
                "question": row["question"],
                "depends_on": row.get("depends_on", ""),
                "history_available": int(bool(history_text)),
                "history_applied_to_query": int(args.후속질문히스토리적용 and bool(history_text)),
                "history_text": history_text,
                "retrieval_query": retrieval_query,
                "ground_truth_doc": row.get("ground_truth_doc", ""),
                "ground_truth_docs": row.get("ground_truth_docs", ""),
                "ground_truth_hint": row.get("ground_truth_hint", ""),
                "expected": row.get("expected", ""),
                "eval_focus": row.get("eval_focus", ""),
                "top1_source_file": retrieved_files[0] if retrieved_files else "",
                "retrieved_source_files": " | ".join(retrieved_files),
                "retrieved_chunk_ids": " | ".join([metadata.get("chunk_id", "") for metadata in metadatas]),
                "top1_doc_hit": top1_doc_hit,
                "topk_doc_hit": topk_doc_hit,
                "ground_truth_doc_hit_rate": round(doc_hit_rate, 4) if doc_hit_rate is not None else None,
                "rejection_expected": rejection_expected,
                "rejection_detected": rejection_detected,
                "rejection_success": rejection_success,
                "elapsed_sec": elapsed_sec,
                "answer_chars": len(answer),
                "answer_text": answer,
            }
        )

        print(
            f"[완료] {question_id} | {row['answer_type'] or row['type_group']} | "
            f"top1_hit={top1_doc_hit} | topk_hit={topk_doc_hit} | {elapsed_sec}초"
        )

    parsed_question_rows = [
        {
            "question_id": row["question_id"],
            "question_index": row["question_index"],
            "type_group": row["type_group"],
            "type_label": row["type_label"],
            "scenario_label": row["scenario_label"],
            "turn_index": row["turn_index"],
            "answer_type": row.get("answer_type", ""),
            "question": row.get("question", ""),
            "depends_on": row.get("depends_on", ""),
            "ground_truth_doc": row.get("ground_truth_doc", ""),
            "ground_truth_docs": row.get("ground_truth_docs", ""),
            "ground_truth_hint": row.get("ground_truth_hint", ""),
            "expected": row.get("expected", ""),
            "eval_focus": row.get("eval_focus", ""),
        }
        for row in question_rows
    ]

    summary_rows: list[dict[str, Any]] = []
    group_keys = [
        ("overall", result_rows),
    ]

    for type_group in sorted({row["type_group"] for row in result_rows}):
        group_keys.append((type_group, [row for row in result_rows if row["type_group"] == type_group]))

    for answer_type in sorted({row["answer_type"] for row in result_rows if row["answer_type"]}):
        group_keys.append((f"answer_type:{answer_type}", [row for row in result_rows if row["answer_type"] == answer_type]))

    for label, rows in group_keys:
        top1_values = [row["top1_doc_hit"] for row in rows if row["top1_doc_hit"] is not None]
        topk_values = [row["topk_doc_hit"] for row in rows if row["topk_doc_hit"] is not None]
        hit_rate_values = [row["ground_truth_doc_hit_rate"] for row in rows if row["ground_truth_doc_hit_rate"] is not None]
        rejection_values = [row["rejection_success"] for row in rows if row["rejection_success"] is not None]

        summary_rows.append(
            {
                "group_name": label,
                "question_count": len(rows),
                "top1_doc_hit_rate": round(sum(top1_values) / len(top1_values), 4) if top1_values else None,
                "topk_doc_hit_rate": round(sum(topk_values) / len(topk_values), 4) if topk_values else None,
                "avg_ground_truth_doc_hit_rate": round(sum(hit_rate_values) / len(hit_rate_values), 4) if hit_rate_values else None,
                "rejection_success_rate": round(sum(rejection_values) / len(rejection_values), 4) if rejection_values else None,
                "avg_elapsed_sec": round(sum(row["elapsed_sec"] for row in rows) / len(rows), 2) if rows else None,
                "avg_answer_chars": round(sum(row["answer_chars"] for row in rows) / len(rows), 2) if rows else None,
            }
        )

    manual_template_rows = []
    for row in result_rows:
        manual_template = dict(row)
        manual_template["faithfulness_score"] = ""
        manual_template["completeness_score"] = ""
        manual_template["groundedness_score"] = ""
        manual_template["relevancy_score"] = ""
        manual_template["evaluator_note"] = ""
        manual_template_rows.append(manual_template)

    judge_guide_text = (
        "평가 항목(1~5점)\n"
        "- Faithfulness: 답변이 검색 문맥에 근거해 사실을 왜곡 없이 설명하는가\n"
        "- Completeness: 질문이 요구한 핵심 요소를 빠짐없이 다뤘는가\n"
        "- Groundedness: 답변이 실제 검색 문맥과 정합적인가\n"
        "- Relevancy: 질문에 직접적으로 답했는가\n"
        "\n"
        "Type 4는 문서에 없는 내용을 적절히 거부했는지 함께 확인한다.\n"
    )

    csv_저장(결과경로("baseline_eval_questions_parsed", ".csv"), parsed_question_rows)
    csv_저장(결과경로("baseline_eval_results", ".csv"), result_rows)
    csv_저장(결과경로("baseline_eval_summary", ".csv"), summary_rows)
    csv_저장(결과경로("baseline_eval_manual_template", ".csv"), manual_template_rows)
    결과경로("baseline_eval_results", ".jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in result_rows),
        encoding="utf-8",
    )
    결과경로("baseline_eval_judge_guide", ".txt").write_text(judge_guide_text, encoding="utf-8")
    결과경로("baseline_eval_manifest", ".json").write_text(
        json.dumps(
            {
                "question_set_path": str(question_set_path),
                "question_count": len(question_rows),
                "collection_name": 컬렉션이름,
                "embedding_model": args.임베딩모델,
                "response_model": args.응답모델,
                "top_k": args.상위개수,
                "history_applied_for_follow_up": args.후속질문히스토리적용,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("[완료] 베이스라인 평가가 끝났습니다.")
    print(f"- 질문 수: {len(question_rows)}")
    print(f"- 결과 CSV: {결과경로('baseline_eval_results', '.csv')}")
    print(f"- 요약 CSV: {결과경로('baseline_eval_summary', '.csv')}")
    print(f"- 수동 평가 템플릿: {결과경로('baseline_eval_manual_template', '.csv')}")


if __name__ == "__main__":
    main()

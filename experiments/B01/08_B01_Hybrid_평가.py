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
    OUTPUT_DIR,
    OpenAI_클라이언트_가져오기,
    csv_저장,
)
from rag_bm25 import BM25_검색, BM25_인덱스_불러오기


BASE_DIR = Path(__file__).resolve().parent
QUESTION_SET_PATH = BASE_DIR / "evaluation" / "day3_partA_eval_questions_v1.txt"
BM25_INDEX_PATH = OUTPUT_DIR / "bm25_index_b01.pkl"
EVAL_DIR = OUTPUT_DIR / "b01_hybrid_eval"

QUESTION_HEADER_RE = re.compile(r"^Q(?P<num>\d+)(?: \[(?P<turn>\d+)턴])?$")
FIELD_RE = re.compile(r"^\s*([A-Za-z_]+)\s*:\s*(.+?)\s*$")
TYPE_RE = re.compile(r"^TYPE\s+(?P<type_num>\d+)\s*:\s*(?P<label>.+)$")
SCENARIO_RE = re.compile(r"^---\s*(?P<label>시나리오.+?)\s*---$")


def 크로마_모듈_가져오기():
    try:
        import chromadb
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "chromadb 패키지가 설치되어 있지 않습니다. requirements.txt를 먼저 설치하세요."
        ) from exc
    return chromadb


def 질의_임베딩(client, model_name: str, question: str) -> list[float]:
    response = client.embeddings.create(model=model_name, input=[question])
    return response.data[0].embedding


def 검색_문맥_구성(results: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for index, row in enumerate(results, start=1):
        metadata = row["metadata"]
        blocks.append(
            "\n".join(
                [
                    f"[검색 결과 {index}]",
                    f"- 사업명: {metadata.get('사업명', '정보 없음')}",
                    f"- 발주 기관: {metadata.get('발주 기관', '정보 없음')}",
                    f"- 파일명: {metadata.get('source_file_name', '정보 없음')}",
                    f"- 청크 ID: {metadata.get('chunk_id', '정보 없음')}",
                    f"- fusion_score: {round(float(row['fusion_score']), 6)}",
                    row["document"],
                ]
            )
        )
    return "\n\n".join(blocks)


def 베이스라인_RAG_응답_생성(client, model_name: str, question: str, retrieval_context: str) -> str:
    system_prompt = (
        "당신은 정부 제안요청서(RFP) 기반 요약 도우미다.\n"
        "반드시 제공된 검색 문맥만 근거로 답한다.\n"
        "문맥에 없는 사실은 추정하지 않는다.\n"
        "질문이 특정 사업 요약을 요구하면 목적, 범위, 주요 요구사항, 일정/예산, 발주기관을 우선 정리한다.\n"
        "문맥에 일정이나 예산이 없으면 없다고 명시한다.\n"
        "마지막에는 참고 문서명과 청크 ID를 간단히 정리한다."
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
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
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


def 정답문서_목록(row: dict[str, Any]) -> list[str]:
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


def 거절응답_감지(answer: str) -> int:
    rejection_patterns = [
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


def 벡터_검색(chroma_collection, query_embedding: list[float], candidate_k: int) -> list[dict[str, Any]]:
    query_result = chroma_collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_k,
        include=["documents", "metadatas", "distances"],
    )
    documents = query_result["documents"][0]
    metadatas = query_result["metadatas"][0]
    distances = query_result["distances"][0]
    rows: list[dict[str, Any]] = []
    for rank, (document, metadata, distance) in enumerate(zip(documents, metadatas, distances), start=1):
        rows.append(
            {
                "rank": rank,
                "score": float(distance),
                "chunk_id": metadata.get("chunk_id", ""),
                "document": document,
                "metadata": dict(metadata),
            }
        )
    return rows


def 가중_RRF_결합(
    vector_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    final_k: int,
    vector_weight: float,
    bm25_weight: float,
    rrf_k: int,
) -> list[dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}

    for result_type, results, weight in [
        ("vector", vector_results, vector_weight),
        ("bm25", bm25_results, bm25_weight),
    ]:
        for rank, row in enumerate(results, start=1):
            chunk_id = row["chunk_id"]
            if chunk_id not in fused:
                fused[chunk_id] = {
                    "chunk_id": chunk_id,
                    "document": row["document"],
                    "metadata": row["metadata"],
                    "fusion_score": 0.0,
                    "vector_rank": "",
                    "bm25_rank": "",
                    "vector_distance": "",
                    "bm25_score": "",
                }
            fused[chunk_id]["fusion_score"] += weight * (1.0 / (rrf_k + rank))
            if result_type == "vector":
                fused[chunk_id]["vector_rank"] = rank
                fused[chunk_id]["vector_distance"] = row["score"]
            else:
                fused[chunk_id]["bm25_rank"] = rank
                fused[chunk_id]["bm25_score"] = row["score"]

    ranked = sorted(fused.values(), key=lambda item: item["fusion_score"], reverse=True)
    return ranked[:final_k]


def 결과경로(base_name: str, extension: str) -> Path:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    return EVAL_DIR / f"{base_name}{extension}"


def question_id_set_load(path: Path) -> set[str]:
    question_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            question_ids.add(value)
    return question_ids


def main() -> None:
    global EVAL_DIR
    parser = argparse.ArgumentParser(description="B-01 Hybrid(BM25+Vector) 평가를 실행합니다.")
    parser.add_argument("--질문셋경로", default=str(QUESTION_SET_PATH), help="평가 질문셋 경로")
    parser.add_argument("--크로마경로", default=str(CHROMA_DIR), help="Chroma DB 경로")
    parser.add_argument("--컬렉션이름", default=DEFAULT_COLLECTION_NAME, help="Chroma 컬렉션 이름")
    parser.add_argument("--임베딩모델", default=DEFAULT_EMBEDDING_MODEL, choices=ALLOWED_EMBEDDING_MODELS, help="질문 임베딩 모델")
    parser.add_argument("--응답모델", default=DEFAULT_CHAT_MODEL, choices=ALLOWED_CHAT_MODELS, help="답변 생성 모델")
    parser.add_argument("--BM25인덱스경로", default=str(BM25_INDEX_PATH), help="BM25 인덱스 pickle 경로")
    parser.add_argument("--최종상위개수", type=int, default=5, help="최종 fusion 결과에서 사용할 개수")
    parser.add_argument("--후보개수", type=int, default=10, help="vector/BM25 각각에서 가져올 후보 개수")
    parser.add_argument("--vector가중치", type=float, default=0.7, help="RRF 결합 시 vector 가중치")
    parser.add_argument("--bm25가중치", type=float, default=0.3, help="RRF 결합 시 BM25 가중치")
    parser.add_argument("--rrf_k", type=int, default=60, help="RRF 보정 상수")
    parser.add_argument("--질문개수제한", type=int, default=0, help="0이면 전체 질문, 양수면 앞에서부터 일부만 실행")
    parser.add_argument("--후속질문히스토리적용", action="store_true", help="follow-up 질문에 이전 Q/A를 retrieval_query에 포함")
    parser.add_argument("--질문ID파일", default="", help="실행할 question_id 목록 파일(txt, 한 줄에 하나)")
    parser.add_argument("--출력디렉토리", default=str(EVAL_DIR), help="평가 산출물 출력 디렉토리")
    args = parser.parse_args()
    EVAL_DIR = Path(args.출력디렉토리)

    question_set_path = Path(args.질문셋경로)
    if not question_set_path.exists():
        raise FileNotFoundError(f"질문셋 파일이 없습니다: {question_set_path}")

    bm25_index_path = Path(args.BM25인덱스경로)
    if not bm25_index_path.exists():
        raise FileNotFoundError(f"BM25 인덱스 파일이 없습니다: {bm25_index_path}")

    collection_name = Chroma_컬렉션명_검증(args.컬렉션이름)
    question_rows = 질문셋_파싱(question_set_path)
    if args.질문ID파일:
        question_id_path = Path(args.질문ID파일)
        if not question_id_path.exists():
            raise FileNotFoundError(f"question_id 파일이 없습니다: {question_id_path}")
        selected_ids = question_id_set_load(question_id_path)
        question_rows = [row for row in question_rows if row["question_id"] in selected_ids]
    if args.질문개수제한 > 0:
        question_rows = question_rows[: args.질문개수제한]
    if not question_rows:
        raise RuntimeError("질문셋이 비어 있습니다.")

    questions_by_id = {row["question_id"]: row for row in question_rows}
    answers_by_id: dict[str, str] = {}

    client = OpenAI_클라이언트_가져오기()
    chromadb = 크로마_모듈_가져오기()
    chroma_client = chromadb.PersistentClient(path=args.크로마경로)
    collection = chroma_client.get_collection(name=collection_name)
    bm25_index_payload = BM25_인덱스_불러오기(bm25_index_path)

    result_rows: list[dict[str, Any]] = []
    preview_rows: list[dict[str, Any]] = []

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
        query_embedding = 질의_임베딩(client=client, model_name=args.임베딩모델, question=retrieval_query)
        vector_results = 벡터_검색(collection, query_embedding, args.후보개수)
        bm25_results = BM25_검색(bm25_index_payload, retrieval_query, args.후보개수)
        fused_results = 가중_RRF_결합(
            vector_results=vector_results,
            bm25_results=bm25_results,
            final_k=args.최종상위개수,
            vector_weight=args.vector가중치,
            bm25_weight=args.bm25가중치,
            rrf_k=args.rrf_k,
        )
        retrieval_context = 검색_문맥_구성(fused_results)

        answer_prompt_question = row["question"]
        if args.후속질문히스토리적용 and history_text:
            answer_prompt_question = f"이전 대화:\n{history_text}\n\n현재 질문:\n{row['question']}"

        answer = 베이스라인_RAG_응답_생성(
            client=client,
            model_name=args.응답모델,
            question=answer_prompt_question,
            retrieval_context=retrieval_context,
        )
        elapsed_sec = round(time.time() - started_at, 2)
        answers_by_id[question_id] = answer

        retrieved_files = [item["metadata"].get("source_file_name", "") for item in fused_results]
        ground_truth_docs = 정답문서_목록(row)
        top1_doc_hit, topk_doc_hit, doc_hit_rate = 정답문서_적중률(ground_truth_docs, retrieved_files)
        rejection_detected = 거절응답_감지(answer)
        rejection_expected = int(row.get("answer_type") == "rejection" or bool(row.get("expected")))
        rejection_success = rejection_detected if rejection_expected else None

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
                "retrieval_mode": "hybrid_rrf",
                "vector_weight": args.vector가중치,
                "bm25_weight": args.bm25가중치,
                "candidate_k": args.후보개수,
                "top_k": args.최종상위개수,
                "top1_source_file": retrieved_files[0] if retrieved_files else "",
                "retrieved_source_files": " | ".join(retrieved_files),
                "retrieved_chunk_ids": " | ".join([item["metadata"].get("chunk_id", "") for item in fused_results]),
                "top1_doc_hit": top1_doc_hit,
                "topk_doc_hit": topk_doc_hit,
                "ground_truth_doc_hit_rate": round(doc_hit_rate, 4) if doc_hit_rate is not None else None,
                "rejection_expected": rejection_expected,
                "rejection_detected": rejection_detected,
                "rejection_success": rejection_success,
                "elapsed_sec": elapsed_sec,
                "answer_chars": len(answer),
                "retrieval_context": retrieval_context,
                "answer_text": answer,
            }
        )

        for rank, item in enumerate(fused_results, start=1):
            preview_rows.append(
                {
                    "question_id": question_id,
                    "rank": rank,
                    "chunk_id": item["chunk_id"],
                    "source_file_name": item["metadata"].get("source_file_name", ""),
                    "fusion_score": round(float(item["fusion_score"]), 6),
                    "vector_rank": item["vector_rank"],
                    "bm25_rank": item["bm25_rank"],
                    "vector_distance": item["vector_distance"],
                    "bm25_score": item["bm25_score"],
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
    group_specs: list[tuple[str, list[dict[str, Any]]]] = [("overall", result_rows)]

    for type_group in sorted({row["type_group"] for row in result_rows if row["type_group"]}):
        group_specs.append((type_group, [row for row in result_rows if row["type_group"] == type_group]))
    for answer_type in sorted({row["answer_type"] for row in result_rows if row["answer_type"]}):
        group_specs.append((f"answer_type:{answer_type}", [row for row in result_rows if row["answer_type"] == answer_type]))

    for label, rows in group_specs:
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

    manual_template_rows: list[dict[str, Any]] = []
    for row in result_rows:
        manual_row = dict(row)
        manual_row["faithfulness_score"] = ""
        manual_row["completeness_score"] = ""
        manual_row["groundedness_score"] = ""
        manual_row["relevancy_score"] = ""
        manual_row["evaluator_note"] = ""
        manual_template_rows.append(manual_row)

    judge_guide_text = (
        "평가 항목(1~5점)\n"
        "- Faithfulness: 답변이 검색 문맥에 없는 사실을 추가하지 않았는가\n"
        "- Completeness: 질문이 요구한 핵심 요소를 빠뜨리지 않았는가\n"
        "- Groundedness: 답변이 실제 검색 문맥에 직접 근거하는가\n"
        "- Relevancy: 질문에 직접 답하고 있는가\n"
        "\n"
        "Type 4는 문서에 없는 내용을 적절히 거절했는지 확인한다.\n"
    )

    csv_저장(결과경로("baseline_eval_questions_parsed", ".csv"), parsed_question_rows)
    csv_저장(결과경로("baseline_eval_results", ".csv"), result_rows)
    csv_저장(결과경로("baseline_eval_summary", ".csv"), summary_rows)
    csv_저장(결과경로("baseline_eval_manual_template", ".csv"), manual_template_rows)
    csv_저장(결과경로("b01_hybrid_preview", ".csv"), preview_rows)
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
                "collection_name": collection_name,
                "embedding_model": args.임베딩모델,
                "response_model": args.응답모델,
                "top_k": args.최종상위개수,
                "history_applied_for_follow_up": args.후속질문히스토리적용,
                "retrieval_mode": "hybrid_rrf",
                "vector_weight": args.vector가중치,
                "bm25_weight": args.bm25가중치,
                "candidate_k": args.후보개수,
                "bm25_index_path": str(bm25_index_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("[완료] B-01 Hybrid 평가가 끝났습니다.")
    print(f"- 질문 수: {len(question_rows)}")
    print(f"- 결과 CSV: {결과경로('baseline_eval_results', '.csv')}")
    print(f"- 요약 CSV: {결과경로('baseline_eval_summary', '.csv')}")
    print(f"- 수동 평가 템플릿: {결과경로('baseline_eval_manual_template', '.csv')}")


if __name__ == "__main__":
    main()

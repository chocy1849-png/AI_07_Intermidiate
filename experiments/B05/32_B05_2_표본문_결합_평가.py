from __future__ import annotations

import argparse
import difflib
import json
import os
import pickle
import re
import time
from pathlib import Path
from typing import Any

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

from eval_utils import parse_question_rows, read_csv, write_csv


BASE_DIR = Path(__file__).resolve().parent
QUESTION_SET_PATH = BASE_DIR / "rag_outputs" / "eval_sets" / "b05_table_eval_questions_v2.txt"
QUESTION_ID_FILE = BASE_DIR / "evaluation" / "day4_b05_group_bc_question_ids_v1.txt"
CHROMA_DIR = BASE_DIR / "rag_outputs" / "chroma_db"
COLLECTION_NAME = "rfp_contextual_chunks_v2_b05_table"
BM25_INDEX_PATH = BASE_DIR / "rag_outputs" / "bm25_index_b05.pkl"
OUTPUT_DIR = BASE_DIR / "rag_outputs" / "b05_2_group_bc_eval"


def load_openai_client() -> OpenAI:
    load_dotenv(BASE_DIR / ".env", override=False)
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip() or "https://api.openai.com/v1"
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 없습니다.")
    if not re.match(r"^https?://", base_url):
        raise RuntimeError("OPENAI_BASE_URL은 http:// 또는 https:// 로 시작해야 합니다.")
    return OpenAI(api_key=api_key, base_url=base_url)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[0-9a-zA-Z가-힣]+", str(text or "").lower())


def load_bm25_index(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        return pickle.load(file)


def embed_query(client: OpenAI, model_name: str, question: str) -> list[float]:
    response = client.embeddings.create(model=model_name, input=[question])
    return response.data[0].embedding


def vector_search(collection, query_embedding: list[float], candidate_k: int, where: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    rows: list[dict[str, Any]] = []
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]
    for rank, (document, metadata, distance) in enumerate(zip(documents, metadatas, distances), start=1):
        rows.append(
            {
                "rank": rank,
                "score": float(distance),
                "chunk_id": metadata.get("chunk_id", ""),
                "document": document,
                "metadata": metadata,
            }
        )
    return rows


def bm25_search(index_payload: dict[str, Any], query: str, top_k: int) -> list[dict[str, Any]]:
    tokenized_query = tokenize(query)
    scores = index_payload["model"].get_scores(tokenized_query)
    chunk_rows = index_payload["chunk_rows"]
    ranked_pairs = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
    rows: list[dict[str, Any]] = []
    for rank, (row_index, score) in enumerate(ranked_pairs[:top_k], start=1):
        row = chunk_rows[row_index]
        rows.append(
            {
                "rank": rank,
                "score": float(score),
                "chunk_id": row.get("chunk_id", ""),
                "document": row.get("contextual_chunk_text", ""),
                "metadata": row,
            }
        )
    return rows


def weighted_rrf_fuse(
    vector_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    *,
    final_k: int,
    vector_weight: float,
    bm25_weight: float,
    rrf_k: int,
) -> list[dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}

    for result in vector_results:
        chunk_id = result["chunk_id"]
        row = fused.setdefault(
            chunk_id,
            {
                "chunk_id": chunk_id,
                "document": result["document"],
                "metadata": result["metadata"],
                "vector_rank": None,
                "bm25_rank": None,
                "vector_distance": None,
                "bm25_score": None,
                "fusion_score": 0.0,
            },
        )
        row["vector_rank"] = result["rank"]
        row["vector_distance"] = result["score"]
        row["fusion_score"] += vector_weight * (1.0 / (rrf_k + result["rank"]))

    for result in bm25_results:
        chunk_id = result["chunk_id"]
        row = fused.setdefault(
            chunk_id,
            {
                "chunk_id": chunk_id,
                "document": result["document"],
                "metadata": result["metadata"],
                "vector_rank": None,
                "bm25_rank": None,
                "vector_distance": None,
                "bm25_score": None,
                "fusion_score": 0.0,
            },
        )
        row["bm25_rank"] = result["rank"]
        row["bm25_score"] = result["score"]
        row["fusion_score"] += bm25_weight * (1.0 / (rrf_k + result["rank"]))

    rows = sorted(fused.values(), key=lambda row: row["fusion_score"], reverse=True)
    return rows[:final_k]


def is_table_chunk(row: dict[str, Any]) -> bool:
    chunk_id = str(row.get("chunk_id", ""))
    metadata = row.get("metadata", {}) or {}
    return "__table_" in chunk_id or str(metadata.get("chunk_role", "")).startswith("표")


def select_table_docs(fused_results: list[dict[str, Any]], max_docs: int = 2) -> list[str]:
    docs: list[str] = []
    for row in fused_results:
        if not is_table_chunk(row):
            continue
        source_file = str((row.get("metadata", {}) or {}).get("source_file_name", "")).strip()
        if source_file and source_file not in docs:
            docs.append(source_file)
        if len(docs) >= max_docs:
            break
    return docs


def select_best_table_rows(fused_results: list[dict[str, Any]], max_docs: int = 1) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_docs: set[str] = set()
    for row in fused_results:
        if not is_table_chunk(row):
            continue
        source_file = str((row.get("metadata", {}) or {}).get("source_file_name", "")).strip()
        if not source_file or source_file in seen_docs:
            continue
        seen_docs.add(source_file)
        selected.append(row)
        if len(selected) >= max_docs:
            break
    return selected


def collect_same_doc_body_chunks(
    collection,
    query_embedding: list[float],
    source_file_name: str,
    *,
    body_k: int,
) -> list[dict[str, Any]]:
    rows = vector_search(
        collection,
        query_embedding,
        candidate_k=max(body_k * 4, 8),
        where={"source_file_name": source_file_name},
    )
    filtered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        chunk_id = row["chunk_id"]
        if chunk_id in seen:
            continue
        if is_table_chunk(row):
            continue
        seen.add(chunk_id)
        filtered.append(
            {
                "chunk_id": row["chunk_id"],
                "document": row["document"],
                "metadata": row["metadata"],
                "vector_rank": row["rank"],
                "bm25_rank": None,
                "vector_distance": row["score"],
                "bm25_score": None,
                "fusion_score": None,
            }
        )
        if len(filtered) >= body_k:
            break
    return filtered


def augment_with_body_chunks(
    collection,
    query_embedding: list[float],
    question_row: dict[str, Any],
    pool_results: list[dict[str, Any]],
    primary_results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    answer_type = str(question_row.get("answer_type", ""))
    if answer_type != "table_plus_text":
        return primary_results, [], []

    selected_table_rows = select_best_table_rows(pool_results, max_docs=1)
    selected_table_docs = [
        str((row.get("metadata", {}) or {}).get("source_file_name", "")).strip()
        for row in selected_table_rows
        if str((row.get("metadata", {}) or {}).get("source_file_name", "")).strip()
    ]
    if not selected_table_rows:
        return primary_results, [], []

    body_k = 2 if answer_type == "table_plus_text" else 1
    body_rows_by_doc: dict[str, list[dict[str, Any]]] = {}
    added_chunk_ids: list[str] = []
    for source_file_name in selected_table_docs:
        body_rows = collect_same_doc_body_chunks(
            collection,
            query_embedding,
            source_file_name,
            body_k=body_k,
        )
        body_rows_by_doc[source_file_name] = body_rows

    final_rows: list[dict[str, Any]] = list(primary_results)
    seen_chunk_ids = {row["chunk_id"] for row in primary_results}
    for table_row in selected_table_rows:
        if table_row["chunk_id"] not in seen_chunk_ids:
            final_rows.append(table_row)
            seen_chunk_ids.add(table_row["chunk_id"])

    ordered_rows: list[dict[str, Any]] = []
    inserted_docs: set[str] = set()
    for row in final_rows:
        ordered_rows.append(row)
        source_file_name = str((row.get("metadata", {}) or {}).get("source_file_name", "")).strip()
        if source_file_name in inserted_docs:
            continue
        if source_file_name not in body_rows_by_doc:
            continue
        if not is_table_chunk(row):
            continue
        for body_row in body_rows_by_doc[source_file_name]:
            if body_row["chunk_id"] in seen_chunk_ids:
                continue
            ordered_rows.append(body_row)
            seen_chunk_ids.add(body_row["chunk_id"])
            added_chunk_ids.append(body_row["chunk_id"])
        inserted_docs.add(source_file_name)

    return ordered_rows[:7], selected_table_docs, added_chunk_ids


def build_retrieval_context(results: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for index, row in enumerate(results, start=1):
        metadata = row["metadata"]
        blocks.append(
            "\n".join(
                [
                    f"[검색 결과 {index}]",
                    f"- 사업명: {metadata.get('사업명', metadata.get('사업명', '정보 없음'))}",
                    f"- 발주기관: {metadata.get('발주 기관', '정보 없음')}",
                    f"- 파일명: {metadata.get('source_file_name', '정보 없음')}",
                    f"- 청크 ID: {metadata.get('chunk_id', '정보 없음')}",
                    f"- 청크 역할: {metadata.get('chunk_role', '일반')}",
                    row["document"],
                ]
            )
        )
    return "\n\n".join(blocks)


def generate_answer(client: OpenAI, model_name: str, question: str, retrieval_context: str) -> str:
    system_prompt = (
        "당신은 정부 제안요청서(RFP) 기반 요약 도우미다.\n"
        "반드시 제공된 검색 문맥만 근거로 답한다.\n"
        "문맥에 없는 사실은 추정하지 않는다.\n"
        "질문이 특정 사업 요약을 요구하면 목적, 범위, 주요 요구사항, 일정/예산, 발주기관을 우선 정리한다.\n"
        "문맥에 일정이나 예산이 없으면 없다고 명시한다.\n"
        "마지막에 참고 문서명과 청크 ID를 간단히 정리한다."
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


def normalize_doc_name(text: str) -> str:
    value = str(text or "").lower().strip()
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"[^0-9a-z가-힣._-]+", "", value)
    return value


def extract_ground_truth_docs(row: dict[str, Any]) -> list[str]:
    if row.get("ground_truth_doc"):
        return [str(row["ground_truth_doc"]).strip()]
    if row.get("ground_truth_docs"):
        return [part.strip() for part in str(row["ground_truth_docs"]).split("+") if part.strip()]
    return []


def compute_doc_hits(ground_truth_docs: list[str], retrieved_files: list[str]) -> tuple[int | None, int | None, float | None]:
    if not ground_truth_docs:
        return None, None, None
    normalized_retrieved = [normalize_doc_name(x) for x in retrieved_files]
    normalized_ground_truth = [normalize_doc_name(x) for x in ground_truth_docs]

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


def detect_rejection(answer: str) -> int:
    patterns = [
        r"문서.*정보가 없습니다",
        r"문맥에 없습니다",
        r"확인되지 않습니다",
        r"문서에서 확인되지 않습니다",
        r"기재되어 있지 않습니다",
        r"명시되어 있지 않습니다",
        r"자료가 없습니다",
        r"정보가 없습니다",
    ]
    text = str(answer or "")
    return int(any(re.search(pattern, text) for pattern in patterns))


def build_summary(result_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    group_specs: list[tuple[str, list[dict[str, Any]]]] = [("overall", result_rows)]
    for type_group in sorted({row["type_group"] for row in result_rows if row.get("type_group")}):
        group_specs.append((type_group, [row for row in result_rows if row.get("type_group") == type_group]))
    for answer_type in sorted({row["answer_type"] for row in result_rows if row.get("answer_type")}):
        group_specs.append((f"answer_type:{answer_type}", [row for row in result_rows if row.get("answer_type") == answer_type]))

    summary_rows: list[dict[str, Any]] = []
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
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="B-05.2 표+본문 결합 retrieval 평가")
    parser.add_argument("--question-set-path", default=str(QUESTION_SET_PATH))
    parser.add_argument("--question-id-file", default=str(QUESTION_ID_FILE))
    parser.add_argument("--collection-name", default=COLLECTION_NAME)
    parser.add_argument("--bm25-index-path", default=str(BM25_INDEX_PATH))
    parser.add_argument("--response-model", default="gpt-5-mini", choices=("gpt-5-mini", "gpt-5-nano", "gpt-5"))
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--vector-weight", type=float, default=0.7)
    parser.add_argument("--bm25-weight", type=float, default=0.3)
    parser.add_argument("--rrf-k", type=int, default=60)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = load_openai_client()
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_collection(args.collection_name)
    bm25_index = load_bm25_index(Path(args.bm25_index_path))

    question_rows = parse_question_rows(Path(args.question_set_path))
    selected_ids = {
        line.strip()
        for line in Path(args.question_id_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    question_rows = [row for row in question_rows if row["question_id"] in selected_ids]
    question_rows = sorted(question_rows, key=lambda row: row["question_index"])

    result_rows: list[dict[str, Any]] = []
    preview_rows: list[dict[str, Any]] = []
    manual_template_rows: list[dict[str, Any]] = []

    for row in question_rows:
        question_id = row["question_id"]
        query = row["question"]
        started_at = time.time()

        query_embedding = embed_query(client, args.embedding_model, query)
        vector_results = vector_search(collection, query_embedding, args.candidate_k)
        bm25_results = bm25_search(bm25_index, query, args.candidate_k)
        fused_pool = weighted_rrf_fuse(
            vector_results,
            bm25_results,
            final_k=max(args.candidate_k, args.top_k + 5),
            vector_weight=args.vector_weight,
            bm25_weight=args.bm25_weight,
            rrf_k=args.rrf_k,
        )
        primary_results = fused_pool[: args.top_k]
        final_results, selected_table_docs, added_body_chunk_ids = augment_with_body_chunks(
            collection,
            query_embedding,
            row,
            fused_pool,
            primary_results,
        )
        retrieval_context = build_retrieval_context(final_results)
        answer = generate_answer(client, args.response_model, query, retrieval_context)
        elapsed_sec = round(time.time() - started_at, 2)

        retrieved_files = [item["metadata"].get("source_file_name", "") for item in final_results]
        ground_truth_docs = extract_ground_truth_docs(row)
        top1_doc_hit, topk_doc_hit, doc_hit_rate = compute_doc_hits(ground_truth_docs, retrieved_files)
        rejection_detected = detect_rejection(answer)
        rejection_expected = int(row.get("answer_type") == "rejection" or bool(row.get("expected")))
        rejection_success = rejection_detected if rejection_expected else None

        result_row = {
            "question_id": question_id,
            "question_index": row["question_index"],
            "type_group": row["type_group"],
            "type_label": row["type_label"],
            "scenario_label": row["scenario_label"],
            "turn_index": row["turn_index"],
            "answer_type": row.get("answer_type", ""),
            "question": row["question"],
            "depends_on": row.get("depends_on", ""),
            "ground_truth_doc": row.get("ground_truth_doc", ""),
            "ground_truth_docs": row.get("ground_truth_docs", ""),
            "ground_truth_hint": row.get("ground_truth_hint", ""),
            "expected": row.get("expected", ""),
            "eval_focus": row.get("eval_focus", ""),
            "retrieval_mode": "b05_2_table_body_augmentation",
            "vector_weight": args.vector_weight,
            "bm25_weight": args.bm25_weight,
            "candidate_k": args.candidate_k,
            "top_k": len(final_results),
            "top1_source_file": retrieved_files[0] if retrieved_files else "",
            "retrieved_source_files": " | ".join(retrieved_files),
            "retrieved_chunk_ids": " | ".join([item["metadata"].get("chunk_id", "") for item in final_results]),
            "selected_table_docs": " | ".join(selected_table_docs),
            "added_body_chunk_ids": " | ".join(added_body_chunk_ids),
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
        result_rows.append(result_row)

        manual_row = dict(result_row)
        manual_row["faithfulness_score"] = ""
        manual_row["completeness_score"] = ""
        manual_row["groundedness_score"] = ""
        manual_row["relevancy_score"] = ""
        manual_row["evaluator_note"] = ""
        manual_template_rows.append(manual_row)

        for rank, item in enumerate(final_results, start=1):
            preview_rows.append(
                {
                    "question_id": question_id,
                    "rank": rank,
                    "chunk_id": item["metadata"].get("chunk_id", ""),
                    "source_file_name": item["metadata"].get("source_file_name", ""),
                    "chunk_role": item["metadata"].get("chunk_role", ""),
                    "table_source": item["metadata"].get("table_source", ""),
                    "selected_pipeline": "b05_2",
                    "was_table_chunk": int(is_table_chunk(item)),
                    "added_by_same_doc_augmentation": int(item["metadata"].get("chunk_id", "") in added_body_chunk_ids),
                }
            )

        print(
            f"[완료] {question_id} | {row['scenario_label']} | "
            f"table_docs={len(selected_table_docs)} | body_added={len(added_body_chunk_ids)} | {elapsed_sec}초"
        )

    write_csv(output_dir / "baseline_eval_questions_parsed.csv", question_rows)
    write_csv(output_dir / "baseline_eval_results.csv", result_rows)
    write_csv(output_dir / "baseline_eval_summary.csv", build_summary(result_rows))
    write_csv(output_dir / "baseline_eval_manual_template.csv", manual_template_rows)
    write_csv(output_dir / "b05_2_preview.csv", preview_rows)
    (output_dir / "baseline_eval_results.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in result_rows),
        encoding="utf-8",
    )
    (output_dir / "baseline_eval_judge_guide.txt").write_text(
        "15_자동_Judge_ASCII.py 를 동일하게 사용합니다.",
        encoding="utf-8",
    )
    (output_dir / "baseline_eval_manifest.json").write_text(
        json.dumps(
            {
                "question_set_path": str(args.question_set_path),
                "question_id_file": str(args.question_id_file),
                "question_count": len(question_rows),
                "collection_name": args.collection_name,
                "embedding_model": args.embedding_model,
                "response_model": args.response_model,
                "top_k": args.top_k,
                "candidate_k": args.candidate_k,
                "vector_weight": args.vector_weight,
                "bm25_weight": args.bm25_weight,
                "rrf_k": args.rrf_k,
                "retrieval_mode": "b05_2_table_body_augmentation",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("[완료] B-05.2 표+본문 결합 평가가 끝났습니다.")
    print(f"- 질문 수: {len(question_rows)}")
    print(f"- 결과 CSV: {output_dir / 'baseline_eval_results.csv'}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import difflib
import json
import pickle
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
QUESTION_SET_PATH = BASE_DIR / "evaluation" / "day3_partA_eval_questions_v1.txt"
OUTPUT_DIR = BASE_DIR / "rag_outputs"
DEFAULT_EVAL_DIR = OUTPUT_DIR / "b03_crag_eval"
DEFAULT_BM25_INDEX_PATH = OUTPUT_DIR / "bm25_index_b02.pkl"
DEFAULT_CHROMA_DIR = BASE_DIR / "rag_outputs" / "chroma_db"
DEFAULT_COLLECTION_NAME = "rfp_contextual_chunks_v2_b02"
ALLOWED_CHAT_MODELS = ("gpt-5-mini", "gpt-5-nano", "gpt-5")
ALLOWED_EMBEDDING_MODELS = ("text-embedding-3-small",)
DEFAULT_CHAT_MODEL = "gpt-5-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

QUESTION_HEADER_RE = re.compile(r"^Q(?P<num>\d+)(?: \[(?P<turn>\d+)[^\]]*\])?$")
FIELD_RE = re.compile(r"^\s*([A-Za-z_]+)\s*:\s*(.+?)\s*$")
TYPE_RE = re.compile(r"^TYPE\s+(?P<type_num>\d+)\s*:\s*(?P<label>.+)$")
SCENARIO_RE = re.compile(r"^---\s*(?P<label>.+?)\s*---$")


def validate_collection_name(name: str) -> str:
    value = str(name or "").strip()
    if re.match(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{1,510}[A-Za-z0-9])?$", value):
        return value
    raise ValueError(
        "Chroma 컬렉션 이름은 3~512자이며 영문/숫자/.-_ 만 사용할 수 있습니다. "
        f"현재 값: {value}"
    )


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
        raise RuntimeError("OPENAI_BASE_URL은 http:// 또는 https:// 로 시작해야 합니다.")

    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError("openai 패키지가 없습니다. requirements.txt를 먼저 설치해야 합니다.") from exc

    return OpenAI(api_key=api_key, base_url=base_url)


def load_chromadb():
    try:
        import chromadb
    except ModuleNotFoundError as exc:
        raise RuntimeError("chromadb 패키지가 없습니다. requirements.txt를 먼저 설치해야 합니다.") from exc
    return chromadb


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_pickle(path: Path) -> Any:
    with path.open("rb") as file:
        return pickle.load(file)


def load_question_id_set(path: Path) -> set[str]:
    question_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            question_ids.add(value)
    return question_ids


def parse_questions(path: Path) -> list[dict[str, Any]]:
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
            current[field_match.group(1).strip()] = field_match.group(2).strip()

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


def question_embedding(client, model_name: str, question: str) -> list[float]:
    response = client.embeddings.create(model=model_name, input=[question])
    return response.data[0].embedding


def bm25_tokenize(text: str) -> list[str]:
    return re.findall(r"[0-9A-Za-z가-힣]+", str(text or "").lower())


def vector_search(collection, query_embedding: list[float], candidate_k: int) -> list[dict[str, Any]]:
    query_result = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_k,
        include=["documents", "metadatas", "distances"],
    )
    rows: list[dict[str, Any]] = []
    for rank, (document, metadata, distance) in enumerate(
        zip(query_result["documents"][0], query_result["metadatas"][0], query_result["distances"][0]),
        start=1,
    ):
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


def bm25_search(index_payload: dict[str, Any], query: str, candidate_k: int) -> list[dict[str, Any]]:
    model = index_payload["model"]
    chunk_rows = index_payload["chunk_rows"]
    scores = model.get_scores(bm25_tokenize(query))
    ranked_pairs = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)

    rows: list[dict[str, Any]] = []
    for rank, (row_index, score) in enumerate(ranked_pairs[:candidate_k], start=1):
        row = chunk_rows[row_index]
        rows.append(
            {
                "rank": rank,
                "score": float(score),
                "chunk_id": row.get("chunk_id", ""),
                "document": row.get("contextual_chunk_text", ""),
                "metadata": {
                    key: row.get(key)
                    for key in [
                        "chunk_id",
                        "document_id",
                        "source_file_name",
                        "source_path",
                        "source_extension",
                        "budget_text",
                        "budget_amount_krw",
                        "bid_deadline",
                        "period_raw",
                        "period_days",
                        "period_months",
                        "contract_method",
                        "bid_method",
                        "purpose_summary",
                        "document_type",
                        "domain_tags",
                        "section_title",
                        "section_path",
                        "chunk_role",
                        "chunk_role_tags",
                        "has_table",
                        "has_budget_signal",
                        "has_schedule_signal",
                        "has_contract_signal",
                        "mentioned_systems",
                    ]
                },
            }
        )
    return rows


def fuse_rrf(
    vector_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    vector_weight: float,
    bm25_weight: float,
    rrf_k: int,
) -> list[dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}

    for source_name, results, weight in [
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
            if source_name == "vector":
                fused[chunk_id]["vector_rank"] = rank
                fused[chunk_id]["vector_distance"] = row["score"]
            else:
                fused[chunk_id]["bm25_rank"] = rank
                fused[chunk_id]["bm25_score"] = row["score"]

    return sorted(fused.values(), key=lambda item: item["fusion_score"], reverse=True)


def detect_question_profile(row: dict[str, Any], question: str) -> dict[str, Any]:
    q = str(question or "")
    answer_type = str(row.get("answer_type", ""))
    return {
        "budget": bool(re.search(r"(예산|금액|사업비|기초금액|추정금액|소요예산|얼마)", q)),
        "schedule": bool(re.search(r"(기간|일정|마감|언제|며칠|개월|일 이내|사업기간|수행기간|계약기간)", q)),
        "contract": bool(re.search(r"(계약방식|입찰방식|계약|입찰|협상|낙찰)", q)),
        "purpose": bool(re.search(r"(목적|배경|필요성|왜|목표)", q)),
        "compare": answer_type == "comparison" or bool(re.search(r"(비교|차이|각각|모두|어느|어떤 기관|이상|이하|공통|서로)", q)),
        "rejection": answer_type == "rejection",
        "follow_up": answer_type == "follow_up" or bool(row.get("depends_on_list")),
        "answer_type": answer_type,
    }


def field_bonus(result: dict[str, Any], profile: dict[str, Any], question: str) -> float:
    metadata = result["metadata"]
    bonus = 0.0
    question_text = str(question or "")
    metadata_text = " ".join(
        [
            str(metadata.get("source_file_name", "")),
            str(metadata.get("section_title", "")),
            str(metadata.get("chunk_role", "")),
            str(metadata.get("chunk_role_tags", "")),
            str(metadata.get("purpose_summary", "")),
            str(metadata.get("period_raw", "")),
            str(metadata.get("contract_method", "")),
            str(metadata.get("bid_method", "")),
            str(metadata.get("budget_text", "")),
        ]
    )

    if profile["budget"]:
        if int(metadata.get("has_budget_signal", 0) or 0):
            bonus += 0.0045
        if "예산" in metadata_text:
            bonus += 0.0020
    if profile["schedule"]:
        if int(metadata.get("has_schedule_signal", 0) or 0):
            bonus += 0.0045
        if any(token in metadata_text for token in ["기간", "일정", "마감"]):
            bonus += 0.0020
    if profile["contract"]:
        if int(metadata.get("has_contract_signal", 0) or 0):
            bonus += 0.0045
        if any(token in metadata_text for token in ["계약", "입찰", "협상"]):
            bonus += 0.0020
    if profile["purpose"] and any(token in metadata_text for token in ["목적", "배경", "필요"]):
        bonus += 0.0040
    if int(metadata.get("has_table", 0) or 0) and (profile["budget"] or profile["schedule"]):
        bonus += 0.0015

    overlap_terms = set(re.findall(r"[0-9A-Za-z가-힣]+", question_text)) & set(re.findall(r"[0-9A-Za-z가-힣]+", metadata_text))
    if overlap_terms:
        bonus += min(0.0030, len(overlap_terms) * 0.0005)

    return bonus


def rerank_with_profile(candidates: list[dict[str, Any]], profile: dict[str, Any], question: str) -> list[dict[str, Any]]:
    reranked: list[dict[str, Any]] = []
    for row in candidates:
        adjusted = float(row["fusion_score"]) + field_bonus(row, profile, question)
        new_row = dict(row)
        new_row["adjusted_score"] = round(adjusted, 6)
        reranked.append(new_row)
    reranked.sort(key=lambda item: item["adjusted_score"], reverse=True)
    return reranked


def candidate_summary_block(candidates: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for rank, row in enumerate(candidates, start=1):
        metadata = row["metadata"]
        excerpt = re.sub(r"\s+", " ", str(row["document"]).replace("\n", " ")).strip()[:420]
        blocks.append(
            "\n".join(
                [
                    f"[후보 {rank}]",
                    f"- source_file_name: {metadata.get('source_file_name', '')}",
                    f"- chunk_id: {metadata.get('chunk_id', '')}",
                    f"- section_title: {metadata.get('section_title', '')}",
                    f"- chunk_role: {metadata.get('chunk_role', '')}",
                    f"- budget_text: {metadata.get('budget_text', '')}",
                    f"- period_raw: {metadata.get('period_raw', '')}",
                    f"- contract_method: {metadata.get('contract_method', '')}",
                    f"- purpose_summary: {metadata.get('purpose_summary', '')}",
                    f"- adjusted_score: {row.get('adjusted_score', row.get('fusion_score', ''))}",
                    f"- excerpt: {excerpt}",
                ]
            )
        )
    return "\n\n".join(blocks)


def extract_json(text: str) -> dict[str, Any]:
    cleaned = str(text).strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError("JSON 객체를 찾지 못했습니다.")
    return json.loads(match.group(0))


def crag_evaluate(client, evaluator_model: str, question: str, profile: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    profile_text = ", ".join([key for key in ["budget", "schedule", "contract", "purpose", "compare", "follow_up"] if profile[key]]) or "general"
    system_prompt = (
        "당신은 한국어 RAG 검색 평가기다.\n"
        "질문과 검색 후보를 보고 현재 후보만으로 답변 가능한지 판단하라.\n"
        "판정 기준:\n"
        "- CORRECT: 질문에 직접 답할 근거가 충분하다\n"
        "- AMBIGUOUS: 일부는 맞지만 부족하거나 덜 직접적이다\n"
        "- INCORRECT: 질문과 무관하거나 직접 근거가 없다\n"
        "반드시 JSON 객체만 반환하라.\n"
        "{"
        "\"judgment\":\"CORRECT|AMBIGUOUS|INCORRECT\","
        "\"reason\":\"짧은 이유\","
        "\"relevant_ranks\":[1,2],"
        "\"need_second_pass\":true,"
        "\"rewrite_query\":\"필요 시 보강 검색 질의\","
        "\"focus_aspects\":[\"예산\",\"기간\"]"
        "}"
    )
    user_prompt = (
        f"[질문]\n{question}\n\n"
        f"[질문 프로필]\n{profile_text}\n\n"
        f"[검색 후보]\n{candidate_summary_block(candidates)}\n\n"
        "relevant_ranks에는 실제로 근거가 되는 후보 번호만 최대 4개까지 넣어라.\n"
        "AMBIGUOUS일 때만 rewrite_query를 채워라.\n"
        "질문이 비교형이면 서로 다른 문서를 함께 답해야 하는지 고려하라."
    )
    response = client.responses.create(
        model=evaluator_model,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
        ],
    )
    payload = extract_json(response.output_text)
    judgment = str(payload.get("judgment", "AMBIGUOUS")).strip().upper()
    if judgment not in {"CORRECT", "AMBIGUOUS", "INCORRECT"}:
        judgment = "AMBIGUOUS"

    relevant_ranks: list[int] = []
    for value in payload.get("relevant_ranks", []):
        try:
            rank = int(value)
        except Exception:
            continue
        if 1 <= rank <= len(candidates) and rank not in relevant_ranks:
            relevant_ranks.append(rank)

    focus_aspects = [str(item).strip() for item in payload.get("focus_aspects", []) if str(item).strip()]
    return {
        "judgment": judgment,
        "reason": str(payload.get("reason", "")).strip(),
        "relevant_ranks": relevant_ranks,
        "need_second_pass": bool(payload.get("need_second_pass", judgment == "AMBIGUOUS")),
        "rewrite_query": str(payload.get("rewrite_query", "")).strip(),
        "focus_aspects": focus_aspects,
        "raw_text": response.output_text,
    }


def build_second_pass_query(question: str, profile: dict[str, Any], rewrite_query: str) -> str:
    if rewrite_query:
        return rewrite_query

    hints: list[str] = []
    if profile["budget"]:
        hints.extend(["예산", "사업비", "금액"])
    if profile["schedule"]:
        hints.extend(["사업기간", "수행기간", "마감"])
    if profile["contract"]:
        hints.extend(["입찰방식", "계약방식", "협상"])
    if profile["purpose"]:
        hints.extend(["사업목적", "추진배경"])
    if profile["compare"]:
        hints.extend(["비교 대상 사업", "모든 관련 문서"])

    hint_text = " ".join(dict.fromkeys(hints))
    return f"{question} {hint_text}".strip()


def unique_by_chunk_id(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in candidates:
        chunk_id = str(row.get("chunk_id", ""))
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        deduped.append(row)
    return deduped


def select_diverse_candidates(candidates: list[dict[str, Any]], top_k: int, compare_mode: bool) -> list[dict[str, Any]]:
    if not compare_mode:
        return candidates[:top_k]

    selected: list[dict[str, Any]] = []
    seen_files: set[str] = set()
    for row in candidates:
        source_file = str(row["metadata"].get("source_file_name", ""))
        if source_file and source_file not in seen_files:
            selected.append(row)
            seen_files.add(source_file)
        if len(selected) >= top_k:
            return selected

    for row in candidates:
        if row in selected:
            continue
        selected.append(row)
        if len(selected) >= top_k:
            break
    return selected


def refine_candidates(initial_candidates: list[dict[str, Any]], profile: dict[str, Any], crag_result: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
    if crag_result["relevant_ranks"]:
        selected = [
            initial_candidates[rank - 1]
            for rank in crag_result["relevant_ranks"]
            if 1 <= rank <= len(initial_candidates)
        ]
    else:
        selected = initial_candidates
    selected = unique_by_chunk_id(selected)
    compare_top_k = max(top_k, 4) if profile["compare"] else top_k
    return select_diverse_candidates(selected, compare_top_k, profile["compare"])


def build_retrieval_context(results: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for index, row in enumerate(results, start=1):
        metadata = row["metadata"]
        blocks.append(
            "\n".join(
                [
                    f"[검색 결과 {index}]",
                    f"- 파일명: {metadata.get('source_file_name', '정보 없음')}",
                    f"- 청크 ID: {metadata.get('chunk_id', '정보 없음')}",
                    f"- 섹션 제목: {metadata.get('section_title', '정보 없음')}",
                    f"- 청크 역할: {metadata.get('chunk_role', '정보 없음')}",
                    f"- 예산: {metadata.get('budget_text', '정보 없음')}",
                    f"- 사업기간: {metadata.get('period_raw', '정보 없음')}",
                    f"- 계약방식: {metadata.get('contract_method', '정보 없음')}",
                    row["document"],
                ]
            )
        )
    return "\n\n".join(blocks)


def generate_answer(client, model_name: str, question: str, retrieval_context: str) -> str:
    system_prompt = (
        "당신은 공공 제안요청서(RFP) 기반 질의응답 도우미다.\n"
        "반드시 제공된 검색 문맥만 근거로 답하라.\n"
        "문맥에 없는 사실은 추정하지 않는다.\n"
        "질문이 사업 요약을 요구하면 목적, 범위, 주요 요구사항, 일정/예산, 발주기관을 우선 정리한다.\n"
        "비교형 질문이면 문맥에 실제로 등장한 사업만 비교하고, 부족한 항목은 문맥에 없다고 명시한다.\n"
        "마지막에는 참고 근거로 파일명과 청크 ID를 간단히 적는다."
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


def no_info_answer() -> str:
    return (
        "1. 한줄 요약\n"
        "제공된 문서에서 해당 정보를 확인할 수 없습니다.\n\n"
        "2. 핵심 내용\n"
        "- 현재 검색된 근거만으로는 질문에 직접 답할 수 있는 문맥을 찾지 못했습니다.\n\n"
        "3. 주요 요구사항\n"
        "- 문맥에 없음\n\n"
        "4. 일정/예산/발주기관\n"
        "- 일정: 문맥에 없음\n"
        "- 예산: 문맥에 없음\n"
        "- 발주기관: 문맥에 없음\n\n"
        "5. 참고 근거\n"
        "- 직접 근거가 되는 청크를 찾지 못했습니다."
    )


def normalize_text(text: str) -> str:
    value = str(text or "").lower().strip()
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"[^0-9a-z가-힣_-]+", "", value)
    return value


def ground_truth_docs(row: dict[str, Any]) -> list[str]:
    if row.get("ground_truth_doc"):
        return [str(row["ground_truth_doc"]).strip()]
    if row.get("ground_truth_docs"):
        return [part.strip() for part in str(row["ground_truth_docs"]).split("+") if part.strip()]
    return []


def match_doc_name(target: str, candidate: str) -> bool:
    if not target or not candidate:
        return False
    if target in candidate or candidate in target:
        return True
    return difflib.SequenceMatcher(None, target, candidate).ratio() >= 0.86


def compute_doc_hits(gold_docs: list[str], retrieved_files: list[str]) -> tuple[int | None, int | None, float | None]:
    if not gold_docs:
        return None, None, None
    normalized_gold = [normalize_text(doc) for doc in gold_docs]
    normalized_retrieved = [normalize_text(doc) for doc in retrieved_files]
    top1_hit = 0
    if normalized_retrieved:
        top1_hit = int(any(match_doc_name(target, normalized_retrieved[0]) for target in normalized_gold))
    matched_count = 0
    for target in normalized_gold:
        if any(match_doc_name(target, candidate) for candidate in normalized_retrieved):
            matched_count += 1
    return top1_hit, int(matched_count > 0), matched_count / len(normalized_gold)


def detect_rejection(answer: str) -> int:
    patterns = [
        r"문서.*정보가 없습니다",
        r"제공된 문서.*정보를 찾을 수 없습니다",
        r"문맥에 없음",
        r"확인할 수 없습니다",
        r"직접 근거가 되는 청크를 찾지 못했습니다",
    ]
    return int(any(re.search(pattern, str(answer or "")) for pattern in patterns))


def build_history_text(questions_by_id: dict[str, dict[str, Any]], answers_by_id: dict[str, str], depends_on_list: list[str]) -> str:
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


def result_paths(eval_dir: Path) -> dict[str, Path]:
    eval_dir.mkdir(parents=True, exist_ok=True)
    return {
        "questions_csv": eval_dir / "baseline_eval_questions_parsed.csv",
        "results_csv": eval_dir / "baseline_eval_results.csv",
        "summary_csv": eval_dir / "baseline_eval_summary.csv",
        "manual_template_csv": eval_dir / "baseline_eval_manual_template.csv",
        "preview_csv": eval_dir / "b03_crag_preview.csv",
        "results_jsonl": eval_dir / "baseline_eval_results.jsonl",
        "judge_guide_txt": eval_dir / "baseline_eval_judge_guide.txt",
        "manifest_json": eval_dir / "baseline_eval_manifest.json",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="B-03 Hybrid + Contextual + CRAG 평가를 실행합니다.")
    parser.add_argument("--질문셋경로", default=str(QUESTION_SET_PATH))
    parser.add_argument("--질문ID파일", default="")
    parser.add_argument("--질문개수제한", type=int, default=0)
    parser.add_argument("--출력디렉토리", default=str(DEFAULT_EVAL_DIR))
    parser.add_argument("--크로마경로", default=str(DEFAULT_CHROMA_DIR))
    parser.add_argument("--컬렉션이름", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--임베딩모델", default=DEFAULT_EMBEDDING_MODEL, choices=ALLOWED_EMBEDDING_MODELS)
    parser.add_argument("--응답모델", default=DEFAULT_CHAT_MODEL, choices=ALLOWED_CHAT_MODELS)
    parser.add_argument("--평가기모델", default=DEFAULT_CHAT_MODEL, choices=ALLOWED_CHAT_MODELS)
    parser.add_argument("--bm25인덱스경로", default=str(DEFAULT_BM25_INDEX_PATH))
    parser.add_argument("--후속질문히스토리적용", action="store_true")
    parser.add_argument("--후보개수", type=int, default=10)
    parser.add_argument("--평가기입력개수", type=int, default=6)
    parser.add_argument("--최종상위개수", type=int, default=3)
    parser.add_argument("--비교형최종상위개수", type=int, default=4)
    parser.add_argument("--vector가중치", type=float, default=0.7)
    parser.add_argument("--bm25가중치", type=float, default=0.3)
    parser.add_argument("--2차vector가중치", dest="second_vector_weight", type=float, default=0.55)
    parser.add_argument("--2차bm25가중치", dest="second_bm25_weight", type=float, default=0.45)
    parser.add_argument("--rrf_k", type=int, default=60)
    args = parser.parse_args()

    question_set_path = Path(args.질문셋경로)
    bm25_index_path = Path(args.bm25인덱스경로)
    if not question_set_path.exists():
        raise FileNotFoundError(f"질문셋 파일이 없습니다: {question_set_path}")
    if not bm25_index_path.exists():
        raise FileNotFoundError(f"BM25 인덱스 파일이 없습니다: {bm25_index_path}")

    eval_dir = Path(args.출력디렉토리)
    paths = result_paths(eval_dir)
    collection_name = validate_collection_name(args.컬렉션이름)

    questions = parse_questions(question_set_path)
    if args.질문ID파일:
        selected_ids = load_question_id_set(Path(args.질문ID파일))
        questions = [row for row in questions if row["question_id"] in selected_ids]
    if args.질문개수제한 > 0:
        questions = questions[: args.질문개수제한]
    if not questions:
        raise RuntimeError("평가할 질문이 없습니다.")

    client = load_openai_client()
    chromadb = load_chromadb()
    chroma_client = chromadb.PersistentClient(path=args.크로마경로)
    collection = chroma_client.get_collection(name=collection_name)
    bm25_index_payload = read_pickle(bm25_index_path)

    questions_by_id = {row["question_id"]: row for row in questions}
    answers_by_id: dict[str, str] = {}
    result_rows: list[dict[str, Any]] = []
    preview_rows: list[dict[str, Any]] = []

    for row in questions:
        question_id = row["question_id"]
        history_text = build_history_text(questions_by_id, answers_by_id, row["depends_on_list"])
        retrieval_query = row["question"]
        if args.후속질문히스토리적용 and history_text:
            retrieval_query = f"이전 대화\n{history_text}\n\n현재 질문:\n{row['question']}"

        started_at = time.time()
        profile = detect_question_profile(row, row["question"])
        query_embedding = question_embedding(client, args.임베딩모델, retrieval_query)

        vector_results = vector_search(collection, query_embedding, args.후보개수)
        bm25_results = bm25_search(bm25_index_payload, retrieval_query, args.후보개수)
        fused_results = fuse_rrf(vector_results, bm25_results, args.vector가중치, args.bm25가중치, args.rrf_k)
        initial_ranked = rerank_with_profile(fused_results, profile, row["question"])
        evaluator_candidates = initial_ranked[: args.평가기입력개수]
        crag_result = crag_evaluate(client, args.평가기모델, row["question"], profile, evaluator_candidates)

        second_pass_used = 0
        second_pass_query = ""
        final_candidates: list[dict[str, Any]] = []

        if crag_result["judgment"] == "INCORRECT":
            answer = no_info_answer()
        else:
            refined = refine_candidates(evaluator_candidates, profile, crag_result, args.최종상위개수)
            candidate_pool_for_answer = initial_ranked
            if crag_result["need_second_pass"]:
                second_pass_used = 1
                second_pass_query = build_second_pass_query(row["question"], profile, crag_result["rewrite_query"])
                second_embedding = question_embedding(client, args.임베딩모델, second_pass_query)
                vector_second = vector_search(collection, second_embedding, max(args.후보개수, 12))
                bm25_second = bm25_search(bm25_index_payload, second_pass_query, max(args.후보개수, 12))
                fused_second = fuse_rrf(vector_second, bm25_second, args.second_vector_weight, args.second_bm25_weight, args.rrf_k)
                second_ranked = rerank_with_profile(fused_second, profile, row["question"])
                merged = unique_by_chunk_id(refined + second_ranked + initial_ranked)
                candidate_pool_for_answer = rerank_with_profile(merged, profile, row["question"])
            else:
                merged = unique_by_chunk_id(refined + initial_ranked)
                candidate_pool_for_answer = rerank_with_profile(merged, profile, row["question"])

            final_k = args.비교형최종상위개수 if profile["compare"] else args.최종상위개수
            final_candidates = select_diverse_candidates(candidate_pool_for_answer, final_k, profile["compare"])
            retrieval_context = build_retrieval_context(final_candidates)
            answer_question = row["question"]
            if args.후속질문히스토리적용 and history_text:
                answer_question = f"이전 대화\n{history_text}\n\n현재 질문:\n{row['question']}"
            answer = generate_answer(client, args.응답모델, answer_question, retrieval_context)

        elapsed_sec = round(time.time() - started_at, 2)
        answers_by_id[question_id] = answer
        retrieved_files = [item["metadata"].get("source_file_name", "") for item in final_candidates]
        gold_docs = ground_truth_docs(row)
        top1_doc_hit, topk_doc_hit, doc_hit_rate = compute_doc_hits(gold_docs, retrieved_files)
        rejection_expected = int(row.get("answer_type") == "rejection" or bool(row.get("expected")))
        rejection_detected = detect_rejection(answer)
        rejection_success = rejection_detected if rejection_expected else None
        retrieval_context = build_retrieval_context(final_candidates) if final_candidates else ""

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
                "retrieval_mode": "hybrid_contextual_crag",
                "vector_weight": args.vector가중치,
                "bm25_weight": args.bm25가중치,
                "candidate_k": args.후보개수,
                "top_k": len(final_candidates),
                "top1_source_file": retrieved_files[0] if retrieved_files else "",
                "retrieved_source_files": " | ".join(retrieved_files),
                "retrieved_chunk_ids": " | ".join([item["metadata"].get("chunk_id", "") for item in final_candidates]),
                "top1_doc_hit": top1_doc_hit,
                "topk_doc_hit": topk_doc_hit,
                "ground_truth_doc_hit_rate": round(doc_hit_rate, 4) if doc_hit_rate is not None else None,
                "rejection_expected": rejection_expected,
                "rejection_detected": rejection_detected,
                "rejection_success": rejection_success,
                "elapsed_sec": elapsed_sec,
                "answer_chars": len(answer),
                "crag_judgment": crag_result["judgment"],
                "crag_reason": crag_result["reason"],
                "crag_relevant_ranks": "|".join(str(rank) for rank in crag_result["relevant_ranks"]),
                "crag_need_second_pass": second_pass_used,
                "crag_second_pass_query": second_pass_query,
                "crag_focus_aspects": " | ".join(crag_result["focus_aspects"]),
                "retrieval_context": retrieval_context,
                "answer_text": answer,
            }
        )

        for stage_name, rows_for_preview in [("initial", evaluator_candidates), ("final", final_candidates)]:
            for rank, item in enumerate(rows_for_preview, start=1):
                preview_rows.append(
                    {
                        "question_id": question_id,
                        "stage": stage_name,
                        "crag_judgment": crag_result["judgment"],
                        "rank": rank,
                        "chunk_id": item["metadata"].get("chunk_id", ""),
                        "source_file_name": item["metadata"].get("source_file_name", ""),
                        "section_title": item["metadata"].get("section_title", ""),
                        "chunk_role": item["metadata"].get("chunk_role", ""),
                        "fusion_score": round(float(item.get("fusion_score", 0.0)), 6),
                        "adjusted_score": item.get("adjusted_score", ""),
                        "vector_rank": item.get("vector_rank", ""),
                        "bm25_rank": item.get("bm25_rank", ""),
                    }
                )

        print(
            f"[완료] {question_id} | {row.get('answer_type') or row.get('type_group')} | "
            f"judgment={crag_result['judgment']} | second_pass={second_pass_used} | "
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
        for row in questions
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
        "- Faithfulness: 답변이 검색 문맥에 없는 사실을 추가하지 않는가\n"
        "- Completeness: 질문이 요구한 핵심 요소를 빠뜨리지 않았는가\n"
        "- Groundedness: 답변이 검색 문맥에 직접 근거하는가\n"
        "- Relevancy: 질문에 직접 답하고 있는가\n\n"
        "Type 4는 문서에 없는 정보를 잘 거절하는지 확인한다.\n"
    )

    write_csv(paths["questions_csv"], parsed_question_rows)
    write_csv(paths["results_csv"], result_rows)
    write_csv(paths["summary_csv"], summary_rows)
    write_csv(paths["manual_template_csv"], manual_template_rows)
    write_csv(paths["preview_csv"], preview_rows)
    paths["results_jsonl"].write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in result_rows), encoding="utf-8")
    paths["judge_guide_txt"].write_text(judge_guide_text, encoding="utf-8")
    paths["manifest_json"].write_text(
        json.dumps(
            {
                "question_set_path": str(question_set_path),
                "question_count": len(questions),
                "collection_name": collection_name,
                "embedding_model": args.임베딩모델,
                "response_model": args.응답모델,
                "evaluator_model": args.평가기모델,
                "top_k": args.최종상위개수,
                "compare_top_k": args.비교형최종상위개수,
                "history_applied_for_follow_up": args.후속질문히스토리적용,
                "retrieval_mode": "hybrid_contextual_crag",
                "vector_weight": args.vector가중치,
                "bm25_weight": args.bm25가중치,
                "second_vector_weight": args.second_vector_weight,
                "second_bm25_weight": args.second_bm25_weight,
                "candidate_k": args.후보개수,
                "bm25_index_path": str(bm25_index_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[완료] B-03 평가 산출물을 저장했습니다: {eval_dir}")


if __name__ == "__main__":
    main()

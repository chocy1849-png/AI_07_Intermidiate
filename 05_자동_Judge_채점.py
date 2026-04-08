from __future__ import annotations

import argparse
import csv
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
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    OpenAI_클라이언트_가져오기,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_EVAL_DIR = BASE_DIR / "rag_outputs" / "baseline_eval"


def 크로마_모듈_가져오기():
    try:
        import chromadb
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "chromadb 패키지가 설치되어 있지 않습니다. requirements.txt를 먼저 설치하세요."
        ) from exc
    return chromadb


def csv_읽기(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def csv_저장(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
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


def 재검색_문맥_가져오기(
    client,
    chroma_collection,
    embedding_model: str,
    retrieval_query: str,
    top_k: int,
) -> str:
    query_embedding = 질의_임베딩(client=client, model_name=embedding_model, question=retrieval_query)
    query_result = chroma_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    documents = query_result["documents"][0]
    metadatas = query_result["metadatas"][0]
    distances = query_result["distances"][0]
    return 검색_문맥_구성(documents=documents, metadatas=metadatas, distances=distances)


def json_블록_추출(text: str) -> dict[str, Any]:
    cleaned = str(text).strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError("JSON 객체를 찾지 못했습니다.")
    return json.loads(match.group(0))


def judge_입력_프롬프트(row: dict[str, Any], retrieval_context: str) -> tuple[str, str]:
    system_prompt = (
        "당신은 정부 제안서 RAG 평가자다.\n"
        "반드시 제공된 질문, 검색 문맥, 모델 답변, 정답 힌트만 보고 채점한다.\n"
        "점수는 1~5 정수만 사용한다.\n"
        "출력은 반드시 JSON 객체 하나만 반환한다.\n"
        "평가 항목:\n"
        "- faithfulness_score: 검색 문맥에 없는 사실을 꾸며내지 않았는가\n"
        "- completeness_score: 질문이 요구한 핵심 요소를 빠뜨리지 않았는가\n"
        "- groundedness_score: 답변이 검색 문맥에 직접 근거하고 있는가\n"
        "- relevancy_score: 질문에 직접 답하고 있는가\n"
        "거절형(rejection) 질문은 문서에 없는 정보를 추정하지 않고 거절하면 높은 점수를 준다.\n"
        "evaluator_note는 1~2문장으로 핵심 이유만 적는다."
    )

    user_prompt = (
        f"question_id: {row.get('question_id', '')}\n"
        f"type_group: {row.get('type_group', '')}\n"
        f"answer_type: {row.get('answer_type', '')}\n"
        f"question: {row.get('question', '')}\n"
        f"ground_truth_hint: {row.get('ground_truth_hint', '')}\n"
        f"expected: {row.get('expected', '')}\n\n"
        f"[검색 문맥]\n{retrieval_context}\n\n"
        f"[모델 답변]\n{row.get('answer_text', '')}\n\n"
        "다음 형식의 JSON만 반환하라.\n"
        "{\n"
        '  "faithfulness_score": 1,\n'
        '  "completeness_score": 1,\n'
        '  "groundedness_score": 1,\n'
        '  "relevancy_score": 1,\n'
        '  "evaluator_note": "짧은 평가 메모"\n'
        "}"
    )
    return system_prompt, user_prompt


def judge_채점(client, judge_model: str, row: dict[str, Any], retrieval_context: str) -> dict[str, Any]:
    system_prompt, user_prompt = judge_입력_프롬프트(row=row, retrieval_context=retrieval_context)
    response = client.responses.create(
        model=judge_model,
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
    payload = json_블록_추출(response.output_text)
    return {
        "faithfulness_score": int(payload["faithfulness_score"]),
        "completeness_score": int(payload["completeness_score"]),
        "groundedness_score": int(payload["groundedness_score"]),
        "relevancy_score": int(payload["relevancy_score"]),
        "evaluator_note": str(payload.get("evaluator_note", "")).strip(),
    }


def 완료행_인덱스(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = str(row.get("question_id", "")).strip()
        if qid:
            index[qid] = row
    return index


def summary_행_생성(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
                "avg_faithfulness_score": 평균([숫자값(row.get("faithfulness_score")) for row in group_rows]),
                "avg_completeness_score": 평균([숫자값(row.get("completeness_score")) for row in group_rows]),
                "avg_groundedness_score": 평균([숫자값(row.get("groundedness_score")) for row in group_rows]),
                "avg_relevancy_score": 평균([숫자값(row.get("relevancy_score")) for row in group_rows]),
                "avg_manual_eval_score": 평균(
                    [
                        평균(
                            [
                                숫자값(row.get("faithfulness_score")),
                                숫자값(row.get("completeness_score")),
                                숫자값(row.get("groundedness_score")),
                                숫자값(row.get("relevancy_score")),
                            ]
                        )
                        for row in group_rows
                    ]
                ),
            }
        )
    return summary_rows


def 평가경로_구성(eval_dir: Path) -> dict[str, Path]:
    return {
        "results_csv": eval_dir / "baseline_eval_results.csv",
        "manifest_json": eval_dir / "baseline_eval_manifest.json",
        "completed_csv": eval_dir / "baseline_eval_manual_completed.csv",
        "summary_csv": eval_dir / "baseline_eval_manual_summary.csv",
        "judge_manifest_json": eval_dir / "baseline_eval_manual_judge_manifest.json",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="baseline_eval 결과를 gpt-5 judge로 자동 채점합니다.")
    parser.add_argument("--평가디렉토리", default=str(DEFAULT_EVAL_DIR), help="평가 산출물 디렉토리")
    parser.add_argument("--judge모델", default="gpt-5", choices=ALLOWED_CHAT_MODELS, help="자동 채점에 사용할 모델")
    parser.add_argument("--임베딩모델", default="", choices=("",) + ALLOWED_EMBEDDING_MODELS, help="문맥 재구성 시 사용할 임베딩 모델. 비우면 manifest 값 사용")
    parser.add_argument("--크로마경로", default=str(CHROMA_DIR), help="Chroma DB 경로")
    parser.add_argument("--컬렉션이름", default="", help="Chroma 컬렉션 이름. 비우면 manifest 값 사용")
    parser.add_argument("--resume", action="store_true", help="기존 completed 파일이 있으면 이어서 채점")
    args = parser.parse_args()

    eval_dir = Path(args.평가디렉토리)
    paths = 평가경로_구성(eval_dir)
    results_csv_path = paths["results_csv"]
    manifest_path = paths["manifest_json"]
    completed_csv_path = paths["completed_csv"]
    summary_csv_path = paths["summary_csv"]
    judge_manifest_json_path = paths["judge_manifest_json"]

    if not results_csv_path.exists():
        raise FileNotFoundError(f"평가 결과 CSV가 없습니다: {results_csv_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"평가 manifest가 없습니다: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    collection_name = Chroma_컬렉션명_검증(args.컬렉션이름 or manifest.get("collection_name") or DEFAULT_COLLECTION_NAME)
    embedding_model = args.임베딩모델 or manifest.get("embedding_model") or DEFAULT_EMBEDDING_MODEL
    top_k = int(manifest.get("top_k") or 5)

    result_rows = csv_읽기(results_csv_path)
    completed_rows: list[dict[str, Any]] = []
    if args.resume and completed_csv_path.exists():
        completed_rows = csv_읽기(completed_csv_path)
    completed_index = 완료행_인덱스(completed_rows)

    client = OpenAI_클라이언트_가져오기()
    chromadb = 크로마_모듈_가져오기()
    chroma_client = chromadb.PersistentClient(path=args.크로마경로)
    collection = chroma_client.get_collection(name=collection_name)

    output_rows: list[dict[str, Any]] = []
    for row in result_rows:
        question_id = str(row.get("question_id", "")).strip()
        existing = completed_index.get(question_id)
        if existing and all(str(existing.get(key, "")).strip() for key in ("faithfulness_score", "completeness_score", "groundedness_score", "relevancy_score")):
            output_rows.append(existing)
            continue

        retrieval_query = str(row.get("retrieval_query") or row.get("question") or "").strip()
        if not retrieval_query:
            raise RuntimeError(f"retrieval_query가 비어 있습니다: {question_id}")

        started_at = time.time()
        retrieval_context = str(row.get("retrieval_context") or "").strip()
        if not retrieval_context:
            retrieval_context = 재검색_문맥_가져오기(
                client=client,
                chroma_collection=collection,
                embedding_model=embedding_model,
                retrieval_query=retrieval_query,
                top_k=top_k,
            )
        judged = judge_채점(
            client=client,
            judge_model=args.judge모델,
            row=row,
            retrieval_context=retrieval_context,
        )

        new_row = dict(row)
        new_row.update(judged)
        new_row["judge_model"] = args.judge모델
        new_row["judge_elapsed_sec"] = round(time.time() - started_at, 2)
        output_rows.append(new_row)

        csv_저장(completed_csv_path, output_rows)
        print(
            f"[채점완료] {question_id} | "
            f"F={new_row['faithfulness_score']} C={new_row['completeness_score']} "
            f"G={new_row['groundedness_score']} R={new_row['relevancy_score']} | "
            f"{new_row['judge_elapsed_sec']}초"
        )

    summary_rows = summary_행_생성(output_rows)
    csv_저장(summary_csv_path, summary_rows)
    judge_manifest_json_path.write_text(
        json.dumps(
            {
                "source_results_csv": str(results_csv_path),
                "source_manifest": str(manifest_path),
                "question_count": len(output_rows),
                "judge_model": args.judge모델,
                "embedding_model_for_context_rebuild": embedding_model,
                "collection_name": collection_name,
                "top_k": top_k,
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
    print(f"- completed CSV: {completed_csv_path}")
    print(f"- summary CSV: {summary_csv_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
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
    결과파일_경로,
)


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ChromaDB similarity search와 GPT 응답으로 Naive RAG 베이스라인을 실행합니다."
    )
    parser.add_argument(
        "--질문",
        default="",
        help="검색 및 요약에 사용할 질문",
    )
    parser.add_argument(
        "--상위개수",
        type=int,
        default=5,
        help="검색할 상위 청크 개수",
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
        help="최종 응답 생성에 사용할 GPT 모델",
    )
    args = parser.parse_args()
    컬렉션이름 = Chroma_컬렉션명_검증(args.컬렉션이름)

    question = args.질문.strip()
    if not question:
        question = input("질문을 입력하세요: ").strip()
    if not question:
        raise RuntimeError("질문이 비어 있습니다.")

    client = OpenAI_클라이언트_가져오기()
    chromadb = 크로마_모듈_가져오기()
    chroma_client = chromadb.PersistentClient(path=args.크로마경로)
    collection = chroma_client.get_collection(name=컬렉션이름)

    query_embedding = 질의_임베딩(
        client=client,
        model_name=args.임베딩모델,
        question=question,
    )
    query_result = collection.query(
        query_embeddings=[query_embedding],
        n_results=args.상위개수,
        include=["documents", "metadatas", "distances"],
    )

    documents = query_result["documents"][0]
    metadatas = query_result["metadatas"][0]
    distances = query_result["distances"][0]

    retrieval_context = 검색_문맥_구성(
        documents=documents,
        metadatas=metadatas,
        distances=distances,
    )
    answer = 나이브_RAG_응답_생성(
        client=client,
        model_name=args.응답모델,
        question=question,
        retrieval_context=retrieval_context,
    )

    result_payload = {
        "질문": question,
        "임베딩모델": args.임베딩모델,
        "응답모델": args.응답모델,
        "상위개수": args.상위개수,
        "검색결과": [
            {
                "순위": index + 1,
                "거리값": distances[index],
                "메타데이터": metadatas[index],
                "문서": documents[index],
            }
            for index in range(len(documents))
        ],
        "응답": answer,
    }

    result_path = 결과파일_경로(base_name="naive_rag_결과")
    result_path.write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[질문]")
    print(question)
    print("\n[응답]")
    print(answer)
    print(f"\n[저장 경로] {result_path}")


if __name__ == "__main__":
    main()

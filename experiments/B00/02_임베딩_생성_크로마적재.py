from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rag_공통 import (
    ALLOWED_EMBEDDING_MODELS,
    CHROMA_DIR,
    CHUNK_JSONL_PATH,
    Chroma_컬렉션명_검증,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    OpenAI_클라이언트_가져오기,
    OUTPUT_DIR,
    csv_저장,
    jsonl_불러오기,
    크로마_메타데이터값,
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


def 임베딩_생성(client, model_name: str, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=model_name, input=texts)
    return [item.embedding for item in response.data]


def 크로마_메타데이터_구성(row: dict[str, Any]) -> dict[str, Any]:
    selected_keys = [
        "chunk_id",
        "document_id",
        "chunk_index",
        "source_file_name",
        "source_path",
        "source_extension",
        "사업명",
        "발주 기관",
        "공고 번호",
        "공개 일자",
        "파일형식",
        "raw_chunk_chars",
        "contextual_chunk_chars",
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
    ]
    return {
        key: 크로마_메타데이터값(row.get(key))
        for key in selected_keys
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Contextual chunk에 임베딩을 생성하고 ChromaDB에 적재합니다."
    )
    parser.add_argument(
        "--청크경로",
        default=str(CHUNK_JSONL_PATH),
        help="컨텍스트 청크 JSONL 경로",
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
        help="사용할 임베딩 모델",
    )
    parser.add_argument(
        "--배치크기",
        type=int,
        default=64,
        help="임베딩 생성 배치 크기",
    )
    parser.add_argument(
        "--기존컬렉션초기화",
        action="store_true",
        help="같은 이름의 컬렉션이 있으면 삭제 후 다시 적재",
    )
    args = parser.parse_args()
    컬렉션이름 = Chroma_컬렉션명_검증(args.컬렉션이름)

    chunk_rows = jsonl_불러오기(path=Path(args.청크경로))
    if not chunk_rows:
        raise RuntimeError("청크 데이터가 비어 있습니다. 먼저 01_컨텍스트_청킹.py를 실행하세요.")

    client = OpenAI_클라이언트_가져오기()
    chromadb = 크로마_모듈_가져오기()
    chroma_client = chromadb.PersistentClient(path=args.크로마경로)

    if args.기존컬렉션초기화:
        try:
            chroma_client.delete_collection(name=컬렉션이름)
        except Exception:
            pass

    collection = chroma_client.get_or_create_collection(
        name=컬렉션이름,
        metadata={
            "설명": "정부 제안서 Contextual Retrieval용 Chroma 컬렉션",
            "임베딩모델": args.임베딩모델,
        },
    )

    total_rows = len(chunk_rows)
    for batch_start in range(0, total_rows, args.배치크기):
        batch_rows = chunk_rows[batch_start : batch_start + args.배치크기]
        texts = [row["contextual_chunk_text"] for row in batch_rows]
        embeddings = 임베딩_생성(client=client, model_name=args.임베딩모델, texts=texts)

        collection.upsert(
            ids=[row["chunk_id"] for row in batch_rows],
            documents=texts,
            metadatas=[크로마_메타데이터_구성(row) for row in batch_rows],
            embeddings=embeddings,
        )
        print(f"[진행] {min(batch_start + len(batch_rows), total_rows)}/{total_rows} 청크 적재 완료")

    summary_rows = [
        {
            "컬렉션이름": 컬렉션이름,
            "임베딩모델": args.임베딩모델,
            "적재청크수": collection.count(),
            "크로마경로": args.크로마경로,
        }
    ]
    csv_저장(path=OUTPUT_DIR / "chroma_적재_요약.csv", rows=summary_rows)

    manifest_path = OUTPUT_DIR / "embedding_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "컬렉션이름": 컬렉션이름,
                "임베딩모델": args.임베딩모델,
                "적재청크수": collection.count(),
                "크로마경로": str(args.크로마경로),
                "청크경로": str(args.청크경로),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("[완료] 임베딩 생성 및 Chroma 적재가 끝났습니다.")
    print(f"- 컬렉션 이름: {컬렉션이름}")
    print(f"- 임베딩 모델: {args.임베딩모델}")
    print(f"- 적재 청크 수: {collection.count()}")
    print(f"- Chroma 경로: {args.크로마경로}")


if __name__ == "__main__":
    main()

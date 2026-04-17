"""전체 문서 임베딩 생성 & ChromaDB 적재 자동화 스크립트.

실행:
    python scripts/embed_and_index.py            # 전체 재인덱싱
    python scripts/embed_and_index.py --new      # ChromaDB에 없는 파일만 추가

특징:
    - upsert 방식이라 중복 실행해도 데이터 중복 없음
    - OpenAI API 요금 절감을 위해 배치 50청크 단위로 처리
    - ChromaDB가 None 값을 거부하므로 metadata 자동 sanitize
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunking.chunker import chunk_document
from src.db.parsed_store import load_parsed_documents
from src.db.vector_store import add_documents, count, get_indexed_ids
from src.embedding.embedder import get_embedder

CHUNK_SIZE = 1000
OVERLAP = 200
STRATEGY = "langchain_recursive"
EMBED_BATCH = 50


def _sanitize_metadata(metadata: dict) -> dict:
    """ChromaDB는 None, list, dict 값을 거부하므로 안전한 타입으로 변환한다."""
    result = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)):
            result[k] = v
        elif v is None:
            result[k] = ""
        else:
            result[k] = str(v)
    return result




def embed_and_index(new_only: bool = False) -> None:
    documents = load_parsed_documents()
    embedder = get_embedder()
    existing_ids = get_indexed_ids() if new_only else set()

    all_chunks: list[dict] = []
    for doc in documents:
        chunks = chunk_document(
            doc["text"],
            metadata=doc["metadata"],
            chunk_size=CHUNK_SIZE,
            overlap=OVERLAP,
            strategy=STRATEGY,
        )
        for chunk in chunks:
            if new_only and chunk["chunk_id"] in existing_ids:
                continue
            chunk["metadata"] = _sanitize_metadata(chunk["metadata"])
            all_chunks.append(chunk)

    if not all_chunks:
        print("새로 추가할 청크 없음.")
        return

    print(f"임베딩할 청크: {len(all_chunks)}개")

    for offset in range(0, len(all_chunks), EMBED_BATCH):
        batch = all_chunks[offset : offset + EMBED_BATCH]
        texts = [c["text"] for c in batch]
        embeddings = embedder.embed(texts)
        add_documents(batch, embeddings)
        done = min(offset + EMBED_BATCH, len(all_chunks))
        print(f"  [{done}/{len(all_chunks)}] 적재 완료")

    print(f"\nChromaDB 총 문서 수: {count()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="문서 임베딩 & ChromaDB 적재")
    parser.add_argument("--new", action="store_true", help="신규 문서만 추가")
    args = parser.parse_args()
    embed_and_index(new_only=args.new)

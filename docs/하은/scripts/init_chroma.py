"""ChromaDB Collection 초기화 스크립트.

실행:
    python scripts/init_chroma.py
"""
from __future__ import annotations

from pathlib import Path

import chromadb


CHROMA_PATH = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "rfp_documents"


def init_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


if __name__ == "__main__":
    collection = init_collection()
    print(f"Collection '{COLLECTION_NAME}' 준비 완료")
    print(f"저장 경로: {CHROMA_PATH}")
    print(f"현재 문서 수: {collection.count()}")

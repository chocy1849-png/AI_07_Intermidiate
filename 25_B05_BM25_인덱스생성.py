from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_공통 import OUTPUT_DIR, jsonl_불러오기
from rag_bm25 import BM25_인덱스_구성, BM25_인덱스_저장


DEFAULT_CHUNK_PATH = OUTPUT_DIR / "b05_table_enriched_chunks.jsonl"
DEFAULT_INDEX_PATH = OUTPUT_DIR / "bm25_index_b05.pkl"
DEFAULT_MANIFEST_PATH = OUTPUT_DIR / "bm25_index_b05_manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="B-05용 BM25 인덱스를 생성합니다.")
    parser.add_argument("--청크경로", default=str(DEFAULT_CHUNK_PATH))
    parser.add_argument("--출력경로", default=str(DEFAULT_INDEX_PATH))
    parser.add_argument("--manifest경로", default=str(DEFAULT_MANIFEST_PATH))
    args = parser.parse_args()

    chunk_rows = jsonl_불러오기(Path(args.청크경로))
    if not chunk_rows:
        raise RuntimeError("B-05 청크 데이터가 비어 있습니다.")

    payload = BM25_인덱스_구성(chunk_rows)
    BM25_인덱스_저장(Path(args.출력경로), payload)

    manifest = {
        "chunk_path": str(args.청크경로),
        "index_path": str(args.출력경로),
        "chunk_count": len(chunk_rows),
        "unique_document_count": len({row.get('document_id', '') for row in chunk_rows}),
        "stage": "B-05",
        "text_field": "contextual_chunk_text",
    }
    Path(args.manifest경로).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest경로).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[완료] B-05용 BM25 인덱스 생성이 끝났습니다.")
    print(f"- 청크 수: {manifest['chunk_count']}")
    print(f"- 인덱스 경로: {args.출력경로}")


if __name__ == "__main__":
    main()

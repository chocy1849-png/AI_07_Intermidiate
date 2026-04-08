from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_공통 import CHUNK_JSONL_PATH, OUTPUT_DIR, jsonl_불러오기
from rag_bm25 import BM25_인덱스_구성, BM25_인덱스_저장


DEFAULT_BM25_INDEX_PATH = OUTPUT_DIR / "bm25_index_b01.pkl"
DEFAULT_BM25_MANIFEST_PATH = OUTPUT_DIR / "bm25_index_b01_manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="B-01용 BM25 인덱스를 생성합니다.")
    parser.add_argument("--청크경로", default=str(CHUNK_JSONL_PATH), help="contextual_chunks.jsonl 경로")
    parser.add_argument("--출력경로", default=str(DEFAULT_BM25_INDEX_PATH), help="BM25 인덱스 pickle 경로")
    parser.add_argument("--manifest경로", default=str(DEFAULT_BM25_MANIFEST_PATH), help="BM25 인덱스 manifest 경로")
    args = parser.parse_args()

    chunk_path = Path(args.청크경로)
    output_path = Path(args.출력경로)
    manifest_path = Path(args.manifest경로)

    if not chunk_path.exists():
        raise FileNotFoundError(f"청크 파일이 없습니다: {chunk_path}")

    chunk_rows = jsonl_불러오기(chunk_path)
    if not chunk_rows:
        raise RuntimeError("청크 데이터가 비어 있습니다.")

    index_payload = BM25_인덱스_구성(chunk_rows)
    BM25_인덱스_저장(output_path, index_payload)

    manifest = {
        "chunk_path": str(chunk_path),
        "index_path": str(output_path),
        "chunk_count": len(chunk_rows),
        "unique_document_count": len({row.get("document_id", "") for row in chunk_rows}),
        "text_field": "contextual_chunk_text",
        "tokenizer": "regex_[0-9a-zA-Z가-힣]+",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[완료] B-01용 BM25 인덱스 생성이 끝났습니다.")
    print(f"- 청크 수: {manifest['chunk_count']}")
    print(f"- 문서 수: {manifest['unique_document_count']}")
    print(f"- 인덱스 경로: {output_path}")
    print(f"- manifest 경로: {manifest_path}")


if __name__ == "__main__":
    main()

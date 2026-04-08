from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "rag_outputs"
B04_MANIFEST_PATH = OUTPUT_DIR / "b04_candidates" / "b04_candidate_manifest.json"
DEFAULT_CHROMA_DIR = BASE_DIR / "rag_outputs" / "chroma_db"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def load_openai_client():
    import os

    load_dotenv(BASE_DIR / ".env", override=False)
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in .env")
    if not base_url:
        base_url = "https://api.openai.com/v1"
    elif not re.match(r"^https?://", base_url):
        raise RuntimeError(f"invalid OPENAI_BASE_URL: {base_url}")

    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url)


def load_chromadb():
    import chromadb

    return chromadb


def validate_collection_name(name: str) -> str:
    value = str(name or "").strip()
    if re.match(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{1,510}[A-Za-z0-9])?$", value):
        return value
    raise ValueError(f"invalid collection name: {value}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def bm25_tokenize(text: str) -> list[str]:
    return re.findall(r"[0-9A-Za-z가-힣]+", str(text or "").lower())


def build_bm25_payload(chunk_rows: list[dict[str, Any]]) -> dict[str, Any]:
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [bm25_tokenize(row.get("contextual_chunk_text", "")) for row in chunk_rows]
    model = BM25Okapi(tokenized_corpus)
    return {
        "tokenized_corpus": tokenized_corpus,
        "chunk_rows": chunk_rows,
        "model": model,
    }


def chroma_metadata(row: dict[str, Any]) -> dict[str, Any]:
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
        "mentioned_systems",
    ]
    metadata: dict[str, Any] = {}
    for key in selected_keys:
        value = row.get(key)
        if value is None:
            metadata[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            metadata[key] = value
        else:
            metadata[key] = json.dumps(value, ensure_ascii=False)
    return metadata


def embed_texts(client, model_name: str, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=model_name, input=texts)
    return [item.embedding for item in response.data]


def main() -> None:
    parser = argparse.ArgumentParser(description="B-04 candidate embedding and Chroma indexing")
    parser.add_argument("--manifest-path", default=str(B04_MANIFEST_PATH))
    parser.add_argument("--candidate-key", default="")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chroma-dir", default=str(DEFAULT_CHROMA_DIR))
    parser.add_argument("--reset-collection", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"candidate manifest not found: {manifest_path}")

    candidate_rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.candidate_key:
        candidate_rows = [row for row in candidate_rows if row["candidate_key"] == args.candidate_key]
    if not candidate_rows:
        raise RuntimeError("no candidate rows selected")

    client = load_openai_client()
    chromadb = load_chromadb()
    chroma_client = chromadb.PersistentClient(path=args.chroma_dir)

    for candidate in candidate_rows:
        chunk_path = Path(candidate["chunk_jsonl_path"])
        chunk_rows = read_jsonl(chunk_path)
        collection_name = validate_collection_name(candidate["collection_name"])
        bm25_index_path = Path(candidate["bm25_index_path"])
        candidate_dir = chunk_path.parent

        if args.reset_collection:
            try:
                chroma_client.delete_collection(name=collection_name)
            except Exception:
                pass

        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": f"B04 chunk candidate {candidate['candidate_key']}",
                "embedding_model": args.embedding_model,
            },
        )

        total_rows = len(chunk_rows)
        for start in range(0, total_rows, args.batch_size):
            batch_rows = chunk_rows[start : start + args.batch_size]
            texts = [row["contextual_chunk_text"] for row in batch_rows]
            embeddings = embed_texts(client, args.embedding_model, texts)
            collection.upsert(
                ids=[row["chunk_id"] for row in batch_rows],
                documents=texts,
                metadatas=[chroma_metadata(row) for row in batch_rows],
                embeddings=embeddings,
            )
            print(f"[진행] {candidate['candidate_key']} {min(start + len(batch_rows), total_rows)}/{total_rows}")

        bm25_payload = build_bm25_payload(chunk_rows)
        bm25_index_path.parent.mkdir(parents=True, exist_ok=True)
        with bm25_index_path.open("wb") as file:
            pickle.dump(bm25_payload, file)

        embed_manifest = {
            "candidate_key": candidate["candidate_key"],
            "collection_name": collection_name,
            "embedding_model": args.embedding_model,
            "chunk_count": collection.count(),
            "chroma_dir": args.chroma_dir,
            "bm25_index_path": str(bm25_index_path),
        }
        (candidate_dir / f"{candidate['candidate_key']}_embedding_manifest.json").write_text(
            json.dumps(embed_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(
            f"[완료] {candidate['candidate_key']} | collection={collection_name} "
            f"| chunk_count={collection.count()} | bm25={bm25_index_path.name}"
        )


if __name__ == "__main__":
    main()

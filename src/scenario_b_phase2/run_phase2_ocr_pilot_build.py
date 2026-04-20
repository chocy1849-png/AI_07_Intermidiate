from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from eval_utils import write_csv, write_json
from rag_bm25 import BM25_인덱스_구성, BM25_인덱스_저장
from scenario_b_phase2.ocr_pilot_builder import (
    build_ocr_pilot_chunks,
    read_jsonl,
    read_target_doc_names,
    select_pilot_documents,
)


def _require_openai() -> Any:
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError("openai package is required for pilot embedding build.") from exc
    return OpenAI


def _require_chromadb() -> Any:
    try:
        import chromadb
    except ModuleNotFoundError as exc:
        raise RuntimeError("chromadb package is required for pilot embedding build.") from exc
    return chromadb


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Build Phase2 OCR pilot corpus/chunks/index")
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument(
        "--processed-documents-path",
        default=str(root / "processed_data" / "processed_documents.jsonl"),
    )
    parser.add_argument("--pilot-doc-count", type=int, default=12)
    parser.add_argument("--min-doc-count", type=int, default=10)
    parser.add_argument("--max-doc-count", type=int, default=15)
    parser.add_argument("--body-chunk-chars", type=int, default=850)
    parser.add_argument("--body-chunk-overlap", type=int, default=120)

    parser.add_argument("--table-eval-path", default=str(root / "rag_outputs" / "eval_sets" / "b05_table_eval_questions_v2.csv"))
    parser.add_argument("--group-bc-path", default=str(root / "rag_outputs" / "eval_sets" / "b05_group_bc_questions_v1.csv"))

    parser.add_argument("--output-dir", default=str(root / "rag_outputs" / "phase2_ocr_pilot"))
    parser.add_argument("--output-jsonl-name", default="phase2_ocr_pilot_chunks.jsonl")
    parser.add_argument("--processed-v2-pilot-name", default="processed_data_v2_pilot.jsonl")
    parser.add_argument("--bm25-index-name", default="bm25_index_phase2_pilot_ocr.pkl")

    parser.add_argument("--build-chroma", action="store_true")
    parser.add_argument("--chroma-dir", default=str(root / "rag_outputs" / "chroma_db"))
    parser.add_argument("--collection-name", default="rfp_contextual_chunks_v2_pilot_ocr")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--reset-collection", action="store_true")
    return parser.parse_args()


def _to_simple_metadata(row: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in row.items():
        if key == "contextual_chunk_text":
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            output[key] = value if value is not None else ""
        else:
            output[key] = str(value)
    return output


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_chroma_collection(
    *,
    project_root: Path,
    chroma_dir: Path,
    collection_name: str,
    embedding_model: str,
    embedding_batch_size: int,
    reset_collection: bool,
    chunk_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    load_dotenv(project_root / ".env", override=False)
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for --build-chroma.")
    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip() or "https://api.openai.com/v1"

    OpenAI = _require_openai()
    chromadb = _require_chromadb()

    client = OpenAI(api_key=api_key, base_url=base_url)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
    if reset_collection:
        try:
            chroma_client.delete_collection(collection_name)
        except Exception:  # noqa: BLE001
            pass
    collection = chroma_client.get_or_create_collection(name=collection_name)

    for start in range(0, len(chunk_rows), max(1, embedding_batch_size)):
        batch = chunk_rows[start : start + max(1, embedding_batch_size)]
        texts = [str(row.get("contextual_chunk_text", "")) for row in batch]
        response = client.embeddings.create(model=embedding_model, input=texts)
        vectors = [item.embedding for item in response.data]
        collection.add(
            ids=[str(row.get("chunk_id", "")) for row in batch],
            documents=texts,
            metadatas=[_to_simple_metadata(row) for row in batch],
            embeddings=vectors,
        )

    return {
        "collection_name": collection_name,
        "chroma_dir": str(chroma_dir),
        "embedding_model": embedding_model,
        "chunk_count": len(chunk_rows),
    }


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    processed_documents_path = Path(args.processed_documents_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    documents = read_jsonl(processed_documents_path)
    target_doc_names = read_target_doc_names(
        [Path(args.table_eval_path).resolve(), Path(args.group_bc_path).resolve()]
    )
    selected_docs = select_pilot_documents(
        documents,
        required_doc_names=target_doc_names,
        pilot_doc_count=args.pilot_doc_count,
        min_doc_count=args.min_doc_count,
        max_doc_count=args.max_doc_count,
    )
    built = build_ocr_pilot_chunks(
        selected_docs,
        body_chunk_chars=max(300, args.body_chunk_chars),
        body_chunk_overlap=max(0, args.body_chunk_overlap),
    )

    chunks_jsonl_path = output_dir / args.output_jsonl_name
    processed_v2_path = project_root / "processed_data" / args.processed_v2_pilot_name
    bm25_path = project_root / "rag_outputs" / args.bm25_index_name
    _write_jsonl(chunks_jsonl_path, built.chunk_rows)
    _write_jsonl(processed_v2_path, built.selected_documents)

    index_payload = BM25_인덱스_구성(built.chunk_rows)
    BM25_인덱스_저장(bm25_path, index_payload)

    chunk_summary_rows = [
        {
            "pilot_doc_count": len(built.selected_documents),
            "pilot_chunk_count": len(built.chunk_rows),
            "table_plus_body_chunk_count": sum(
                1 for row in built.chunk_rows if str(row.get("chunk_type", "")) == "table_plus_body"
            ),
            "body_chunk_count": sum(1 for row in built.chunk_rows if str(row.get("chunk_type", "")) == "body"),
            "high_priority_chunk_count": sum(1 for row in built.chunk_rows if int(row.get("is_high_priority", 0) or 0) == 1),
        }
    ]
    write_csv(output_dir / "phase2_ocr_pilot_summary.csv", chunk_summary_rows)
    write_csv(
        output_dir / "phase2_ocr_pilot_selected_docs.csv",
        [
            {
                "document_id": row.get("document_id", ""),
                "source_file_name": row.get("source_file_name", ""),
                "source_extension": row.get("source_extension", ""),
                "table_markers": (row.get("metadata", {}) or {}).get("table_markers", 0),
                "figure_markers": (row.get("metadata", {}) or {}).get("figure_markers", 0),
                "ocr_candidate_count": (row.get("metadata", {}) or {}).get("ocr_candidate_count", 0),
            }
            for row in built.selected_documents
        ],
    )

    chroma_result: dict[str, Any] = {}
    if args.build_chroma:
        chroma_result = _build_chroma_collection(
            project_root=project_root,
            chroma_dir=Path(args.chroma_dir).resolve(),
            collection_name=args.collection_name,
            embedding_model=args.embedding_model,
            embedding_batch_size=args.embedding_batch_size,
            reset_collection=args.reset_collection,
            chunk_rows=built.chunk_rows,
        )

    write_json(
        output_dir / "phase2_ocr_pilot_manifest.json",
        {
            "project_root": str(project_root),
            "processed_documents_path": str(processed_documents_path),
            "table_eval_path": str(Path(args.table_eval_path).resolve()),
            "group_bc_path": str(Path(args.group_bc_path).resolve()),
            "pilot_doc_count": len(built.selected_documents),
            "pilot_chunk_count": len(built.chunk_rows),
            "chunks_jsonl_path": str(chunks_jsonl_path),
            "processed_data_v2_pilot_path": str(processed_v2_path),
            "bm25_index_path": str(bm25_path),
            "chroma": chroma_result,
        },
    )

    print(f"[done] phase2 ocr pilot built: {output_dir}")
    print(f"[done] selected_docs={len(built.selected_documents)} chunks={len(built.chunk_rows)}")
    print(f"[done] bm25_index={bm25_path}")
    if chroma_result:
        print(f"[done] chroma_collection={chroma_result.get('collection_name')}")


if __name__ == "__main__":
    main()

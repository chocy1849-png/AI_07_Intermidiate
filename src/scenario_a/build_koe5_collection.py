from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from scenario_a.common_pipeline import PipelinePaths, PipelineSettings, ScenarioACommonPipeline


def tokenize(text: str) -> list[str]:
    return re.findall(r"[0-9A-Za-z가-힣]+", str(text or "").lower())


def safe_metadata(row: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, bool):
            output[key] = value
        elif isinstance(value, (int, float, str)) or value is None:
            output[key] = "" if value is None else value
        else:
            output[key] = json.dumps(value, ensure_ascii=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Scenario A HF embedding collection")
    parser.add_argument("--embedding-backend", default="koe5", choices=["koe5", "bge_m3"])
    parser.add_argument("--chunk-jsonl", default="")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    pipeline = ScenarioACommonPipeline(
        PipelinePaths(project_root=project_root),
        PipelineSettings(embedding_backend_key=args.embedding_backend),
    )
    chunk_jsonl = Path(args.chunk_jsonl) if args.chunk_jsonl else (project_root / "rag_outputs" / "b02_prefix_v2_chunks.jsonl")
    if not chunk_jsonl.exists():
        raise FileNotFoundError(f"Chunk JSONL not found: {chunk_jsonl}")

    rows = [json.loads(line) for line in chunk_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    backend = pipeline.embedding_backend
    config = backend.config

    chromadb = __import__("chromadb")
    client = chromadb.PersistentClient(path=str(pipeline.paths.chroma_dir))
    try:
        client.delete_collection(config.collection_name)
    except Exception:  # noqa: BLE001
        pass
    collection = client.get_or_create_collection(config.collection_name)

    ids = [str(row["chunk_id"]) for row in rows]
    documents = [str(row.get("contextual_chunk_text", "")) for row in rows]
    metadatas = [safe_metadata(row) for row in rows]

    success_count = 0
    failure_count = 0
    for start in range(0, len(rows), args.batch_size):
        batch_ids = ids[start : start + args.batch_size]
        batch_docs = documents[start : start + args.batch_size]
        batch_meta = metadatas[start : start + args.batch_size]
        try:
            embeddings = backend.embed_documents(batch_docs)
            collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_meta, embeddings=embeddings)
            success_count += len(batch_ids)
        except Exception:  # noqa: BLE001
            failure_count += len(batch_ids)

    bm25_tokens = [tokenize(doc) for doc in documents]
    bm25_payload = {"tokenized_corpus": bm25_tokens, "chunk_rows": rows, "model": BM25Okapi(bm25_tokens)}
    bm25_path = pipeline.resolve_bm25_index_path(args.embedding_backend)
    bm25_path.parent.mkdir(parents=True, exist_ok=True)
    with bm25_path.open("wb") as file:
        pickle.dump(bm25_payload, file)

    sample_queries = [
        "국민연금공단 이러닝시스템 운영 용역의 사업 예산은 얼마야?",
        "사업기간과 계약방식을 알려줘.",
        "고려대학교와 광주과학기술원 사업을 비교해줘.",
    ]
    examples = []
    md_lines = [f"# {args.embedding_backend} retrieval sanity", ""]
    for query in sample_queries:
        temp = ScenarioACommonPipeline(
            PipelinePaths(project_root=project_root),
            PipelineSettings(embedding_backend_key=args.embedding_backend),
        )
        result = temp.retrieve({"answer_type": "factual", "depends_on_list": [], "question_id": "SANITY"}, query)
        sources = [cand.metadata.get("source_file_name", "") for cand in result.candidates[:3]]
        examples.append({"query": query, "top_sources": sources})
        md_lines.append(f"- query: {query}")
        for source in sources:
            md_lines.append(f"  - {source}")

    report = {
        "embedding_backend": args.embedding_backend,
        "embedding_model": config.model_name,
        "collection_name": config.collection_name,
        "bm25_index_path": str(bm25_path),
        "chunk_count": len(rows),
        "embedding_success_count": success_count,
        "embedding_failure_count": failure_count,
        "vector_dimension": len(backend.embed_queries(['테스트'])[0]),
        "source_document_count": len({row.get('document_id', '') for row in rows}),
        "bm25_document_count": len(rows),
        "sanity_examples": examples,
    }

    output_dir = project_root / "rag_outputs" / "scenario_a_step2b"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "koE5_collection_build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "koE5_retrieval_sanity.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(output_dir / "koE5_collection_build_report.json")


if __name__ == "__main__":
    main()

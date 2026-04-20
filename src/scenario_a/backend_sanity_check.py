from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from pathlib import Path

from scenario_a.common_pipeline import PipelinePaths, PipelineSettings, ScenarioACommonPipeline


SAMPLE_QUERIES = [
    "국민연금공단이 발주한 2024년 이러닝시스템 운영 용역의 사업 예산은 얼마야?",
    "고려대학교와 광주과학기술원의 사업 차이를 비교해줘.",
    "이 사업의 계약방식과 수행기간을 알려줘.",
]


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "rag_outputs" / "scenario_a_preflight"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    md_lines = ["# Scenario A Backend Sanity Examples", ""]

    for backend_key in ["openai_text_embedding_3_small", "koe5", "bge_m3"]:
        pipeline = ScenarioACommonPipeline(
            PipelinePaths(project_root=project_root),
            PipelineSettings(embedding_backend_key=backend_key),
        )
        config = pipeline.load_embedding_config(backend_key)
        backend = pipeline.embedding_backend
        vector = backend.embed_queries([SAMPLE_QUERIES[0]])[0]

        record = {
            "backend_key": backend_key,
            "backend_name": config.backend_name,
            "embedding_model": config.model_name,
            "chroma_dir": str(pipeline.paths.chroma_dir),
            "collection_name": config.collection_name,
            "bm25_index_path": str(pipeline.resolve_bm25_index_path(backend_key)),
            "query_vector_dim": len(vector),
            "retrieval_status": "not_run",
            "examples": [],
        }

        try:
            for query in SAMPLE_QUERIES:
                temp = ScenarioACommonPipeline(
                    PipelinePaths(project_root=project_root),
                    PipelineSettings(embedding_backend_key=backend_key),
                )
                result = temp.retrieve(
                    {"answer_type": "factual", "depends_on_list": [], "question_id": "SANITY"},
                    query,
                )
                record["examples"].append(
                    {
                        "query": query,
                        "route": result.route,
                        "resolved_chroma_dir": str(temp.paths.chroma_dir),
                        "top_sources": [cand.metadata.get("source_file_name", "") for cand in result.candidates[:3]],
                    }
                )
                record["retrieval_status"] = "ok"
        except Exception as exc:  # noqa: BLE001
            record["retrieval_status"] = "not_ready"
            record["retrieval_error"] = str(exc)

        records.append(record)

        md_lines.append(f"## {backend_key}")
        md_lines.append("")
        md_lines.append(f"- backend: `{config.backend_name}`")
        md_lines.append(f"- model: `{config.model_name}`")
        md_lines.append(f"- collection: `{config.collection_name}`")
        md_lines.append(f"- bm25: `{pipeline.resolve_bm25_index_path(backend_key)}`")
        md_lines.append(f"- query_vector_dim: `{len(vector)}`")
        md_lines.append(f"- retrieval_status: `{record['retrieval_status']}`")
        if record["retrieval_status"] == "ok":
            md_lines.append("")
            for example in record["examples"]:
                md_lines.append(f"- query: {example['query']}")
                for source in example["top_sources"]:
                    md_lines.append(f"  - {source}")
        else:
            md_lines.append(f"- note: {record.get('retrieval_error', '')}")
        md_lines.append("")

    (output_dir / "scenario_a_backend_sanity_check.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "scenario_a_backend_sanity_examples.md").write_text(
        "\n".join(md_lines),
        encoding="utf-8",
    )
    print(output_dir / "scenario_a_backend_sanity_check.json")


if __name__ == "__main__":
    main()

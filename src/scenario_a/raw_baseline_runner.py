from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

from eval_utils import build_auto_summary, build_manual_summary, parse_question_rows, sort_result_rows, write_csv, write_json

from scenario_a.common_pipeline import PipelinePaths, PipelineSettings, ScenarioACommonPipeline
from scenario_a.judge import judge_row


RAW_BASELINE_COLLECTION = "rfp_contextual_chunks_v1"
RAW_BASELINE_EMBEDDING_MODEL = "text-embedding-3-small"


def _require_chromadb() -> Any:
    try:
        import chromadb
    except ModuleNotFoundError as exc:
        raise RuntimeError("chromadb package is required for raw baseline retrieval.") from exc
    return chromadb


def read_question_rows(question_set_path: Path) -> list[dict[str, Any]]:
    return parse_question_rows(question_set_path)


def build_raw_context(
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
                    str(document),
                ]
            )
        )
    return "\n\n".join(blocks)


def build_raw_system_instruction() -> str:
    return "\n".join(
        [
            "당신은 정부 제안요청서(RFP) 기반 요약 도우미다.",
            "반드시 제공된 검색 문맥에 근거해서만 답변한다.",
            "문맥에 없는 사실은 추정하지 않는다.",
            "질문이 특정 사업을 가리키면 그 사업 범위를 벗어나지 않는다.",
            "질문이 요구한 핵심만 간결하게 정리한다.",
            "문맥에 일정이나 예산이 없으면 없다고 명시한다.",
        ]
    )


def compute_doc_hits(question_row: dict[str, Any], source_docs: list[str]) -> tuple[float | None, float | None, float | None]:
    def normalize(value: str) -> str:
        return re.sub(r"\s+", "", str(value or "").lower())

    gt_doc = str(question_row.get("ground_truth_doc", "")).strip()
    gt_docs = [x.strip() for x in str(question_row.get("ground_truth_docs", "")).split("|") if x.strip()]
    targets = [*([gt_doc] if gt_doc else []), *gt_docs]
    if not targets:
        return None, None, None

    source_norm = [normalize(x) for x in source_docs]
    target_norm = [normalize(x) for x in targets]
    top1_hit = 1.0 if source_norm and any(t in source_norm[0] or source_norm[0] in t for t in target_norm) else 0.0
    topk_hit = 1.0 if any(any(t in src or src in t for t in target_norm) for src in source_norm) else 0.0
    hit_count = sum(1 for target in target_norm if any(target in src or src in target for src in source_norm))
    hit_rate = hit_count / len(target_norm) if target_norm else None
    return top1_hit, topk_hit, hit_rate


def create_collection(chroma_dir: Path, collection_name: str) -> Any:
    chromadb = _require_chromadb()
    client = chromadb.PersistentClient(path=str(chroma_dir))
    return client.get_collection(collection_name)


def embed_question(client: Any, question: str) -> list[float]:
    response = client.embeddings.create(model=RAW_BASELINE_EMBEDDING_MODEL, input=[question])
    return response.data[0].embedding


def answer_one(
    pipeline: ScenarioACommonPipeline,
    adapter: Any,
    collection: Any,
    question_row: dict[str, Any],
    *,
    top_k: int,
) -> dict[str, Any]:
    question = str(question_row.get("question", "")).strip()
    started = time.time()
    query_embedding = embed_question(pipeline.openai_client, question)
    query_result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    documents = query_result["documents"][0]
    metadatas = query_result["metadatas"][0]
    distances = query_result["distances"][0]
    retrieval_context = build_raw_context(documents, metadatas, distances)
    answer_text = adapter.generate(
        system_instruction=build_raw_system_instruction(),
        question=question,
        context_text=retrieval_context,
        history=[],
    )
    elapsed = round(time.time() - started, 2)
    source_docs = [str(metadata.get("source_file_name", "")).strip() for metadata in metadatas]
    top1_hit, topk_hit, hit_rate = compute_doc_hits(question_row, source_docs)
    return {
        "question_id": question_row.get("question_id", ""),
        "question_index": question_row.get("question_index", ""),
        "type_group": question_row.get("type_group", ""),
        "answer_type": question_row.get("answer_type", ""),
        "question": question,
        "ground_truth_doc": question_row.get("ground_truth_doc", ""),
        "ground_truth_docs": question_row.get("ground_truth_docs", ""),
        "ground_truth_hint": question_row.get("ground_truth_hint", ""),
        "expected": question_row.get("expected", ""),
        "eval_focus": question_row.get("eval_focus", ""),
        "selected_pipeline": "raw_baseline_b00",
        "embedding_backend": "openai_text_embedding_3_small",
        "model_key": adapter.config.model_key,
        "answer_text": answer_text,
        "answer_chars": len(answer_text),
        "elapsed_sec": elapsed,
        "retrieval_context": retrieval_context,
        "source_docs": " | ".join(source_docs),
        "top1_doc_hit": top1_hit,
        "topk_doc_hit": topk_hit,
        "ground_truth_doc_hit_rate": hit_rate,
    }


def write_compare(output_root: Path, rows: list[dict[str, Any]]) -> None:
    compare_csv = output_root / "scenario_a_gemma4_raw_vs_stage1_compare.csv"
    compare_json = output_root / "scenario_a_gemma4_raw_vs_stage1_compare.json"
    write_csv(compare_csv, rows)
    write_json(compare_json, {"rows": rows})


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run Gemma4 on the pre-enhancement raw baseline (B00-style) pipeline.")
    parser.add_argument("--question-set-path", default=str(root / "docs" / "planning" / "pm" / "day3_partA_eval_questions_v1.txt"))
    parser.add_argument("--judge-model", default="gpt-5")
    parser.add_argument("--model-key", default="gemma4_e4b")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        default=str(root / "rag_outputs" / "scenario_a_raw_baseline_gemma4_e4b"),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = ScenarioACommonPipeline(
        PipelinePaths(project_root=root),
        PipelineSettings(embedding_backend_key="openai_text_embedding_3_small"),
    )
    adapter = pipeline.create_adapter(args.model_key)
    collection = create_collection(pipeline.paths.chroma_dir, RAW_BASELINE_COLLECTION)
    question_rows = read_question_rows(Path(args.question_set_path))

    result_rows: list[dict[str, Any]] = []
    for row in question_rows:
        result_rows.append(answer_one(pipeline, adapter, collection, row, top_k=args.top_k))

    result_rows = sort_result_rows(result_rows)
    write_csv(output_dir / "baseline_eval_results.csv", result_rows)
    write_csv(output_dir / "baseline_eval_summary.csv", build_auto_summary(result_rows))
    write_json(
        output_dir / "baseline_eval_manifest.json",
        {
            "run_label": "scenario_a_raw_baseline_gemma4_e4b",
            "question_count": len(result_rows),
            "model_key": args.model_key,
            "embedding_backend": "openai_text_embedding_3_small",
            "collection_name": RAW_BASELINE_COLLECTION,
            "bm25_index_path": "",
            "raw_baseline": True,
        },
    )

    completed_rows: list[dict[str, Any]] = []
    for row in result_rows:
        judged = judge_row(pipeline.openai_client, args.judge_model, row)
        merged = dict(row)
        merged.update(judged)
        completed_rows.append(merged)

    write_csv(output_dir / "baseline_eval_manual_completed.csv", completed_rows)
    write_csv(output_dir / "baseline_eval_manual_summary.csv", build_manual_summary(completed_rows))

    compare_specs = [
        ("raw_baseline_gemma4_e4b", "Raw baseline (B00-style) + Gemma4-E4B", output_dir),
        ("stage1_gemma4_e4b", "Stage1 adopted + Gemma4-E4B", root / "rag_outputs" / "scenario_a_baseline_koe5_gemma4_e4b"),
        ("stage1_qwen_a00", "Stage1 A-00 KoE5 + Qwen", root / "rag_outputs" / "scenario_a_baseline_koe5_qwen"),
    ]
    compare_rows: list[dict[str, Any]] = []
    for run_key, display_name, run_dir in compare_specs:
        summary_path = run_dir / "baseline_eval_manual_summary.csv"
        auto_path = run_dir / "baseline_eval_summary.csv"
        if not summary_path.exists():
            compare_rows.append({"run_key": run_key, "display_name": display_name, "status": "missing", "run_dir": str(run_dir)})
            continue
        manual = pd.read_csv(summary_path)
        auto = pd.read_csv(auto_path) if auto_path.exists() else pd.DataFrame()
        overall_manual = manual.loc[manual["group_name"] == "overall"].iloc[0].to_dict()
        overall_auto = auto.loc[auto["group_name"] == "overall"].iloc[0].to_dict() if not auto.empty else {}
        compare_rows.append(
            {
                "run_key": run_key,
                "display_name": display_name,
                "status": "ok",
                "run_dir": str(run_dir),
                "avg_manual_eval_score": overall_manual.get("avg_manual_eval_score"),
                "avg_faithfulness_score": overall_manual.get("avg_faithfulness_score"),
                "avg_completeness_score": overall_manual.get("avg_completeness_score"),
                "avg_groundedness_score": overall_manual.get("avg_groundedness_score"),
                "avg_relevancy_score": overall_manual.get("avg_relevancy_score"),
                "avg_elapsed_sec": overall_auto.get("avg_elapsed_sec"),
            }
        )
    write_compare(output_dir, compare_rows)
    print(output_dir)


if __name__ == "__main__":
    main()

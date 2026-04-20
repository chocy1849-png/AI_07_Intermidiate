from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path
from typing import Any

from scenario_a.common_pipeline import PipelinePaths, PipelineSettings
from scenario_b_phase2.experiment_config import load_phase2_experiment_bundle
from scenario_b_phase2.phase2_eval import evaluate_phase2, read_eval_rows
from scenario_b_phase2.phase2_pipeline import Phase2Options, ScenarioBPhase2Pipeline


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="RAG Phase2 evaluation runner (B-02/B-03a based)")
    parser.add_argument(
        "--project-root",
        default=str(root),
        help="Project root path containing config/, src/, rag_outputs/, docs/.",
    )
    parser.add_argument(
        "--experiment-config",
        default=str(root / "config" / "phase2_experiments.yaml"),
        help="Phase2 experiment yaml config path.",
    )
    parser.add_argument("--eval-set-key", default="", help="Evaluation set key from phase2_experiments.yaml.")
    parser.add_argument("--experiment-key", default="", help="Experiment key from phase2_experiments.yaml.")
    parser.add_argument(
        "--question-set-path",
        default="",
        help="Question set file (.txt or .csv).",
    )
    parser.add_argument("--question-id-file", default="", help="Optional question-id filter file.")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory for eval artifacts.",
    )
    parser.add_argument("--chroma-dir", default="", help="Optional chroma directory override.")
    parser.add_argument("--embedding-backend", default="openai_text_embedding_3_small")
    parser.add_argument("--model-key", default="gpt5mini_api")
    parser.add_argument("--routing-model", default="gpt-5-mini")
    parser.add_argument("--judge-model", default="gpt-5")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--run-label", default="")

    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--crag-top-n", type=int, default=5)
    parser.add_argument("--vector-weight", type=float, default=0.7)
    parser.add_argument("--bm25-weight", type=float, default=0.3)

    parser.add_argument("--disable-query-expansion", action="store_true")
    parser.add_argument("--disable-normalized-bm25", action="store_true")
    parser.add_argument("--disable-metadata-bonus-v2", action="store_true")
    parser.add_argument("--disable-table-body-pairing", action="store_true")
    parser.add_argument("--disable-soft-crag-lite", action="store_true")

    parser.add_argument("--expansion-query-limit", type=int, default=1)
    parser.add_argument("--expansion-query-weight", type=float, default=0.35)
    parser.add_argument("--normalized-bm25-weight", type=float, default=0.35)
    parser.add_argument("--soft-crag-top-n", type=int, default=6)
    parser.add_argument("--soft-crag-score-weight", type=float, default=0.045)
    parser.add_argument("--soft-crag-keep-k", type=int, default=3)

    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def _apply_experiment_options(args: argparse.Namespace, options: dict[str, Any]) -> Phase2Options:
    metadata_flag = options.get("enable_metadata_aware_retrieval", options.get("enable_metadata_bonus_v2", True))
    return Phase2Options(
        enable_controlled_query_expansion=(
            False
            if args.disable_query_expansion
            else bool(options.get("enable_controlled_query_expansion", True))
        ),
        enable_normalized_bm25=(
            False
            if args.disable_normalized_bm25
            else bool(options.get("enable_normalized_bm25", True))
        ),
        enable_metadata_aware_retrieval=(
            False
            if args.disable_metadata_bonus_v2
            else bool(metadata_flag)
        ),
        enable_metadata_bonus_v2=None,
        enable_table_body_pairing=(
            False
            if args.disable_table_body_pairing
            else bool(options.get("enable_table_body_pairing", True))
        ),
        enable_soft_crag_lite=(
            False
            if args.disable_soft_crag_lite
            else bool(options.get("enable_soft_crag_lite", True))
        ),
        expansion_query_limit=int(options.get("expansion_query_limit", args.expansion_query_limit)),
        expansion_query_weight=float(options.get("expansion_query_weight", args.expansion_query_weight)),
        normalized_bm25_weight=float(options.get("normalized_bm25_weight", args.normalized_bm25_weight)),
        soft_crag_top_n=int(options.get("soft_crag_top_n", args.soft_crag_top_n)),
        soft_crag_score_weight=float(options.get("soft_crag_score_weight", args.soft_crag_score_weight)),
        soft_crag_keep_k=int(options.get("soft_crag_keep_k", args.soft_crag_keep_k)),
        metadata_boost_scale=float(options.get("metadata_boost_scale", 1.0)),
        metadata_disable_for_rejection=bool(options.get("metadata_disable_for_rejection", False)),
        metadata_scope_mode=str(options.get("metadata_scope_mode", "all")),
        normalized_bm25_mode=str(options.get("normalized_bm25_mode", "all")),
        enable_comparison_evidence_helper=bool(options.get("enable_comparison_evidence_helper", False)),
        comparison_helper_doc_bonus=float(options.get("comparison_helper_doc_bonus", 0.0045)),
        comparison_helper_axis_bonus=float(options.get("comparison_helper_axis_bonus", 0.0015)),
        comparison_helper_max_per_doc=int(options.get("comparison_helper_max_per_doc", 2)),
        enable_b03_legacy_crag_parity=bool(options.get("enable_b03_legacy_crag_parity", True)),
        b03_evaluator_top_n=int(options.get("b03_evaluator_top_n", 6)),
        b03_second_pass_vector_weight=float(options.get("b03_second_pass_vector_weight", 0.55)),
        b03_second_pass_bm25_weight=float(options.get("b03_second_pass_bm25_weight", 0.45)),
    )


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    config_path = Path(args.experiment_config).resolve()
    bundle = (
        load_phase2_experiment_bundle(config_path, project_root)
        if config_path.exists()
        else None
    )

    default_question_set = project_root / "docs" / "planning" / "pm" / "day3_partA_eval_questions_v1.txt"
    if args.eval_set_key and bundle is not None:
        _, question_set_path = bundle.resolve_eval_set(args.eval_set_key)
        question_id_file = bundle.resolve_question_id_file(args.eval_set_key)
    else:
        question_set_path = Path(args.question_set_path).resolve() if args.question_set_path else default_question_set.resolve()
        question_id_file = None

    if args.question_id_file:
        question_id_file = Path(args.question_id_file).resolve()

    experiment_options: dict[str, Any] = {}
    if args.experiment_key and bundle is not None:
        experiment_options = bundle.resolve_experiment_options(args.experiment_key)

    run_label = args.run_label or args.experiment_key or "phase2_default_run"
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (project_root / "rag_outputs" / "phase2_runs" / run_label).resolve()
    )

    settings = PipelineSettings(
        embedding_backend_key=args.embedding_backend,
        routing_model=args.routing_model,
        candidate_k=args.candidate_k,
        top_k=args.top_k,
        crag_top_n=args.crag_top_n,
        vector_weight=args.vector_weight,
        bm25_weight=args.bm25_weight,
        factual_or_comparison_route="b03a",
        default_route="b02",
        rejection_route="b02",
        follow_up_route="b02",
    )
    options = _apply_experiment_options(args, experiment_options)
    pipeline = ScenarioBPhase2Pipeline(
        PipelinePaths(
            project_root=project_root,
            chroma_dir=Path(args.chroma_dir).resolve() if args.chroma_dir else None,
        ),
        settings=settings,
        options=options,
    )
    adapter = pipeline.create_adapter(args.model_key)

    question_rows = read_eval_rows(
        question_set_path,
        question_id_file=question_id_file,
        shard_count=max(1, args.shard_count),
        shard_index=max(0, args.shard_index),
    )
    judge_model = None if args.no_judge else args.judge_model

    result = evaluate_phase2(
        pipeline,
        adapter,
        question_rows,
        output_dir,
        judge_model=judge_model,
        run_label=run_label,
        extra_manifest={
            "project_root": str(project_root),
            "chroma_dir": str(Path(args.chroma_dir).resolve()) if args.chroma_dir else "",
            "phase2_experiment_config": str(config_path) if config_path.exists() else "",
            "eval_set_key": args.eval_set_key,
            "experiment_key": args.experiment_key,
            "question_set_path": str(question_set_path),
            "question_id_file": str(question_id_file) if question_id_file else "",
            "shard_count": max(1, args.shard_count),
            "shard_index": max(0, args.shard_index),
            "phase2_options": {
                "enable_controlled_query_expansion": options.enable_controlled_query_expansion,
                "enable_normalized_bm25": options.enable_normalized_bm25,
                "enable_metadata_aware_retrieval": options.enable_metadata_aware_retrieval,
                "enable_table_body_pairing": options.enable_table_body_pairing,
                "enable_soft_crag_lite": options.enable_soft_crag_lite,
                "expansion_query_limit": options.expansion_query_limit,
                "expansion_query_weight": options.expansion_query_weight,
                "normalized_bm25_weight": options.normalized_bm25_weight,
                "soft_crag_top_n": options.soft_crag_top_n,
                "soft_crag_score_weight": options.soft_crag_score_weight,
                "soft_crag_keep_k": options.soft_crag_keep_k,
                "metadata_boost_scale": options.metadata_boost_scale,
                "metadata_disable_for_rejection": options.metadata_disable_for_rejection,
                "metadata_scope_mode": options.metadata_scope_mode,
                "normalized_bm25_mode": options.normalized_bm25_mode,
                "enable_comparison_evidence_helper": options.enable_comparison_evidence_helper,
                "comparison_helper_doc_bonus": options.comparison_helper_doc_bonus,
                "comparison_helper_axis_bonus": options.comparison_helper_axis_bonus,
                "comparison_helper_max_per_doc": options.comparison_helper_max_per_doc,
                "enable_b03_legacy_crag_parity": options.enable_b03_legacy_crag_parity,
                "b03_evaluator_top_n": options.b03_evaluator_top_n,
                "b03_second_pass_vector_weight": options.b03_second_pass_vector_weight,
                "b03_second_pass_bm25_weight": options.b03_second_pass_bm25_weight,
            },
        },
    )

    print(f"[done] phase2 eval finished: {output_dir}")
    print(f"[done] questions: {len(result['result_rows'])}")
    if result["manual_rows"]:
        print(f"[done] manual judged rows: {len(result['manual_rows'])}")
    print(f"[manifest] {json.dumps(result['manifest'], ensure_ascii=False)}")


if __name__ == "__main__":
    main()

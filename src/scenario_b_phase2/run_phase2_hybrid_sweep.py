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

from eval_utils import read_csv, write_csv, write_json
from scenario_a.common_pipeline import PipelinePaths, PipelineSettings
from scenario_b_phase2.experiment_config import load_phase2_experiment_bundle
from scenario_b_phase2.phase2_eval import evaluate_phase2, read_eval_rows
from scenario_b_phase2.phase2_pipeline import Phase2Options, ScenarioBPhase2Pipeline
from scenario_b_phase2.retrieval_metrics import build_phase2_compare_row


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Phase2 hybrid weight sweep runner")
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument("--experiment-config", default=str(root / "config" / "phase2_experiments.yaml"))
    parser.add_argument("--experiment-key", default="combined")
    parser.add_argument("--eval-set-key", default="full_45")
    parser.add_argument("--question-set-path", default="")
    parser.add_argument("--question-id-file", default="")
    parser.add_argument("--output-root", default=str(root / "rag_outputs" / "phase2_runs" / "hybrid_weight_sweep"))
    parser.add_argument("--run-prefix", default="p0_weight_sweep")
    parser.add_argument("--chroma-dir", default="", help="Optional chroma directory override.")

    parser.add_argument("--weights", default="0.6:0.4,0.7:0.3,0.8:0.2")
    parser.add_argument("--embedding-backend", default="openai_text_embedding_3_small")
    parser.add_argument("--model-key", default="gpt5mini_api")
    parser.add_argument("--routing-model", default="gpt-5-mini")
    parser.add_argument("--judge-model", default="gpt-5")
    parser.add_argument("--no-judge", action="store_true")

    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--crag-top-n", type=int, default=5)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def _parse_weights(raw: str) -> list[tuple[float, float]]:
    output: list[tuple[float, float]] = []
    for chunk in str(raw or "").split(","):
        part = chunk.strip()
        if not part:
            continue
        left, right = [x.strip() for x in part.split(":", 1)]
        output.append((float(left), float(right)))
    if not output:
        raise ValueError("No valid weight pair was provided.")
    return output


def _resolve_question_paths(
    *,
    project_root: Path,
    bundle: Any,
    eval_set_key: str,
    question_set_path: str,
    question_id_file: str,
) -> tuple[Path, Path | None]:
    default_question_set = project_root / "docs" / "planning" / "pm" / "day3_partA_eval_questions_v1.txt"
    if question_set_path:
        qs = Path(question_set_path).resolve()
        qid = Path(question_id_file).resolve() if question_id_file else None
        return qs, qid
    if bundle is not None and eval_set_key:
        _, qs = bundle.resolve_eval_set(eval_set_key)
        qid = bundle.resolve_question_id_file(eval_set_key)
        if question_id_file:
            qid = Path(question_id_file).resolve()
        return qs, qid
    return default_question_set.resolve(), (Path(question_id_file).resolve() if question_id_file else None)


def _phase2_options(base_options: dict[str, Any]) -> Phase2Options:
    metadata_flag = base_options.get("enable_metadata_aware_retrieval", base_options.get("enable_metadata_bonus_v2", True))
    return Phase2Options(
        enable_controlled_query_expansion=bool(base_options.get("enable_controlled_query_expansion", True)),
        enable_normalized_bm25=bool(base_options.get("enable_normalized_bm25", True)),
        enable_metadata_aware_retrieval=bool(metadata_flag),
        enable_metadata_bonus_v2=None,
        enable_table_body_pairing=bool(base_options.get("enable_table_body_pairing", True)),
        enable_soft_crag_lite=bool(base_options.get("enable_soft_crag_lite", True)),
        expansion_query_limit=int(base_options.get("expansion_query_limit", 1)),
        expansion_query_weight=float(base_options.get("expansion_query_weight", 0.35)),
        normalized_bm25_weight=float(base_options.get("normalized_bm25_weight", 0.35)),
        soft_crag_top_n=int(base_options.get("soft_crag_top_n", 6)),
        soft_crag_score_weight=float(base_options.get("soft_crag_score_weight", 0.045)),
        soft_crag_keep_k=int(base_options.get("soft_crag_keep_k", 3)),
        metadata_boost_scale=float(base_options.get("metadata_boost_scale", 1.0)),
        metadata_disable_for_rejection=bool(base_options.get("metadata_disable_for_rejection", False)),
        metadata_scope_mode=str(base_options.get("metadata_scope_mode", "all")),
        normalized_bm25_mode=str(base_options.get("normalized_bm25_mode", "all")),
    )


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.experiment_config).resolve()
    bundle = load_phase2_experiment_bundle(config_path, project_root) if config_path.exists() else None
    question_set_path, question_id_file = _resolve_question_paths(
        project_root=project_root,
        bundle=bundle,
        eval_set_key=args.eval_set_key,
        question_set_path=args.question_set_path,
        question_id_file=args.question_id_file,
    )

    base_options: dict[str, Any] = {}
    if bundle is not None and args.experiment_key:
        base_options = bundle.resolve_experiment_options(args.experiment_key)

    question_rows = read_eval_rows(
        question_set_path,
        question_id_file=question_id_file,
        shard_count=max(1, args.shard_count),
        shard_index=max(0, args.shard_index),
    )
    weight_pairs = _parse_weights(args.weights)
    compare_rows: list[dict[str, Any]] = []
    run_manifests: list[dict[str, Any]] = []
    judge_model = None if args.no_judge else args.judge_model

    for vector_weight, bm25_weight in weight_pairs:
        tag = f"vw{int(vector_weight * 10)}_bw{int(bm25_weight * 10)}"
        run_label = f"{args.run_prefix}_{tag}"
        run_output_dir = output_root / run_label

        settings = PipelineSettings(
            embedding_backend_key=args.embedding_backend,
            routing_model=args.routing_model,
            candidate_k=args.candidate_k,
            top_k=args.top_k,
            crag_top_n=args.crag_top_n,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            factual_or_comparison_route="b03a",
            default_route="b02",
            rejection_route="b02",
            follow_up_route="b02",
        )
        options = _phase2_options(base_options)
        pipeline = ScenarioBPhase2Pipeline(
            PipelinePaths(
                project_root=project_root,
                chroma_dir=Path(args.chroma_dir).resolve() if args.chroma_dir else None,
            ),
            settings=settings,
            options=options,
        )
        adapter = pipeline.create_adapter(args.model_key)

        result = evaluate_phase2(
            pipeline,
            adapter,
            question_rows,
            run_output_dir,
            judge_model=judge_model,
            run_label=run_label,
            extra_manifest={
                "project_root": str(project_root),
                "chroma_dir": str(Path(args.chroma_dir).resolve()) if args.chroma_dir else "",
                "phase2_experiment_config": str(config_path) if config_path.exists() else "",
                "experiment_key": args.experiment_key,
                "eval_set_key": args.eval_set_key,
                "question_set_path": str(question_set_path),
                "question_id_file": str(question_id_file) if question_id_file else "",
                "vector_weight": vector_weight,
                "bm25_weight": bm25_weight,
                "shard_count": max(1, args.shard_count),
                "shard_index": max(0, args.shard_index),
            },
        )
        run_manifests.append(result["manifest"])

        auto_summary_rows = read_csv(run_output_dir / "phase2_eval_auto_summary.csv")
        coverage_summary_rows = read_csv(run_output_dir / "phase2_eval_coverage_summary.csv")
        manual_path = run_output_dir / "phase2_eval_manual_summary.csv"
        manual_summary_rows = read_csv(manual_path) if manual_path.exists() else []
        compare_rows.append(
            build_phase2_compare_row(
                run_label=run_label,
                auto_summary_rows=auto_summary_rows,
                manual_summary_rows=manual_summary_rows,
                coverage_summary_rows=coverage_summary_rows,
            )
        )

    compare_path = output_root / "hybrid_sweep_compare.csv"
    write_csv(compare_path, compare_rows)
    write_json(
        output_root / "hybrid_sweep_manifest.json",
        {
            "output_root": str(output_root),
            "project_root": str(project_root),
            "question_set_path": str(question_set_path),
            "question_id_file": str(question_id_file) if question_id_file else "",
            "experiment_config": str(config_path) if config_path.exists() else "",
            "experiment_key": args.experiment_key,
            "eval_set_key": args.eval_set_key,
            "weights": [{"vector_weight": x, "bm25_weight": y} for x, y in weight_pairs],
            "run_manifests": run_manifests,
        },
    )

    print(f"[done] sweep finished: {output_root}")
    print(f"[done] compare csv: {compare_path}")
    print(f"[manifest] {json.dumps({'runs': len(run_manifests), 'output_root': str(output_root)}, ensure_ascii=False)}")


if __name__ == "__main__":
    main()

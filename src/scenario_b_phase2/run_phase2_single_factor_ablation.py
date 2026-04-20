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

from eval_utils import average, read_csv, safe_float, write_csv, write_json
from scenario_a.common_pipeline import PipelinePaths, PipelineSettings
from scenario_b_phase2.experiment_config import load_phase2_experiment_bundle
from scenario_b_phase2.phase2_eval import evaluate_phase2, read_eval_rows
from scenario_b_phase2.phase2_pipeline import Phase2Options, ScenarioBPhase2Pipeline


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Phase2 single-factor ablation runner (full_45)")
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument("--experiment-config", default=str(root / "config" / "phase2_experiments.yaml"))
    parser.add_argument("--eval-set-key", default="full_45")
    parser.add_argument("--question-set-path", default="")
    parser.add_argument("--question-id-file", default="")
    parser.add_argument("--output-root", default=str(root / "rag_outputs" / "phase2_runs"))
    parser.add_argument("--run-prefix", default="p0_single_factor_full45")
    parser.add_argument("--chroma-dir", default="")

    parser.add_argument("--embedding-backend", default="openai_text_embedding_3_small")
    parser.add_argument("--model-key", default="gpt5mini_api")
    parser.add_argument("--routing-model", default="gpt-5-mini")
    parser.add_argument("--judge-model", default="gpt-5")
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--crag-top-n", type=int, default=5)
    parser.add_argument("--vector-weight", type=float, default=0.7)
    parser.add_argument("--bm25-weight", type=float, default=0.3)
    parser.add_argument("--no-judge", action="store_true")

    parser.add_argument(
        "--baseline-manual-summary",
        default=str(root / "rag_outputs" / "b06_adopted_eval_gpt5mini" / "baseline_eval_manual_summary.csv"),
    )
    parser.add_argument(
        "--baseline-auto-summary",
        default=str(root / "rag_outputs" / "b06_adopted_eval_gpt5mini" / "baseline_eval_summary.csv"),
    )
    parser.add_argument(
        "--baseline-manual-completed",
        default=str(root / "rag_outputs" / "b06_adopted_eval_gpt5mini" / "baseline_eval_manual_completed.csv"),
    )
    return parser.parse_args()


def _resolve_question_paths(
    *,
    project_root: Path,
    bundle: Any | None,
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


def _to_float(value: Any) -> float | None:
    return safe_float(value)


def _pick_group_value(rows: list[dict[str, Any]], group_name: str, keys: list[str]) -> float | None:
    target = next((row for row in rows if str(row.get("group_name", "")) == group_name), None)
    if target is None:
        return None
    for key in keys:
        value = _to_float(target.get(key))
        if value is not None:
            return value
    return None


def _manual_mean(row: dict[str, Any]) -> float | None:
    return average(
        [
            _to_float(row.get("faithfulness_score")),
            _to_float(row.get("completeness_score")),
            _to_float(row.get("groundedness_score")),
            _to_float(row.get("relevancy_score")),
        ]
    )


def _build_phase2_options(base: dict[str, Any], *, enable_metadata: bool, enable_normalized_bm25: bool) -> Phase2Options:
    return Phase2Options(
        enable_controlled_query_expansion=False,
        enable_normalized_bm25=enable_normalized_bm25,
        enable_metadata_aware_retrieval=enable_metadata,
        enable_metadata_bonus_v2=None,
        enable_table_body_pairing=False,
        enable_soft_crag_lite=False,
        expansion_query_limit=int(base.get("expansion_query_limit", 1)),
        expansion_query_weight=float(base.get("expansion_query_weight", 0.35)),
        normalized_bm25_weight=float(base.get("normalized_bm25_weight", 0.35)),
        soft_crag_top_n=int(base.get("soft_crag_top_n", 6)),
        soft_crag_score_weight=float(base.get("soft_crag_score_weight", 0.045)),
        soft_crag_keep_k=int(base.get("soft_crag_keep_k", 3)),
    )


def _collect_degradation_rows(
    baseline_rows: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    baseline_by_qid = {str(row.get("question_id", "")).strip(): row for row in baseline_rows}
    degradation_rows: list[dict[str, Any]] = []
    overall_degraded = 0
    overall_severe_degraded = 0
    type2_degraded = 0
    rejection_worsened = 0

    for row in run_rows:
        qid = str(row.get("question_id", "")).strip()
        baseline_row = baseline_by_qid.get(qid)
        if baseline_row is None:
            continue

        base_mean = _manual_mean(baseline_row)
        run_mean = _manual_mean(row)
        delta = None if (base_mean is None or run_mean is None) else round(run_mean - base_mean, 4)
        degraded = int(delta is not None and delta < 0)
        severe_degraded = int(delta is not None and delta <= -0.5)
        is_type2 = str(row.get("type_group", "")).strip() == "TYPE 2"
        type2_degraded_flag = int(is_type2 and degraded)

        base_rej = _to_float(baseline_row.get("rejection_success"))
        run_rej = _to_float(row.get("rejection_success"))
        rejection_worsened_flag = int(base_rej == 1.0 and run_rej == 0.0)

        overall_degraded += degraded
        overall_severe_degraded += severe_degraded
        type2_degraded += type2_degraded_flag
        rejection_worsened += rejection_worsened_flag

        degradation_rows.append(
            {
                "question_id": qid,
                "type_group": row.get("type_group", ""),
                "answer_type": row.get("answer_type", ""),
                "baseline_manual_mean": base_mean,
                "run_manual_mean": run_mean,
                "delta_manual_mean": delta,
                "degraded": degraded,
                "severe_degraded": severe_degraded,
                "type2_degraded": type2_degraded_flag,
                "baseline_rejection_success": base_rej,
                "run_rejection_success": run_rej,
                "rejection_worsened": rejection_worsened_flag,
            }
        )

    counters = {
        "overall_degraded_case_count": overall_degraded,
        "overall_severe_degraded_case_count": overall_severe_degraded,
        "type2_degraded_case_count": type2_degraded,
        "rejection_worsened_case_count": rejection_worsened,
    }
    return degradation_rows, counters


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

    question_rows = read_eval_rows(
        question_set_path,
        question_id_file=question_id_file,
        shard_count=1,
        shard_index=0,
    )
    judge_model = None if args.no_judge else args.judge_model

    baseline_manual_summary_path = Path(args.baseline_manual_summary).resolve()
    baseline_auto_summary_path = Path(args.baseline_auto_summary).resolve()
    baseline_manual_completed_path = Path(args.baseline_manual_completed).resolve()
    for path in [baseline_manual_summary_path, baseline_auto_summary_path, baseline_manual_completed_path]:
        if not path.exists():
            raise FileNotFoundError(f"Baseline file not found: {path}")

    baseline_manual_summary_rows = read_csv(baseline_manual_summary_path)
    baseline_auto_summary_rows = read_csv(baseline_auto_summary_path)
    baseline_manual_completed_rows = read_csv(baseline_manual_completed_path)

    baseline_metrics = {
        "manual_mean": _pick_group_value(baseline_manual_summary_rows, "overall", ["avg_manual_eval_score", "manual_mean"]),
        "type2_manual_mean": _pick_group_value(baseline_manual_summary_rows, "TYPE 2", ["avg_manual_eval_score", "manual_mean"]),
        "type4_manual_mean": _pick_group_value(baseline_manual_summary_rows, "TYPE 4", ["avg_manual_eval_score", "manual_mean"]),
        "rejection_success_rate": _pick_group_value(
            baseline_auto_summary_rows,
            "TYPE 4",
            ["rejection_success_rate"],
        ),
        "latency_sec": _pick_group_value(baseline_auto_summary_rows, "overall", ["avg_elapsed_sec", "latency_sec"]),
    }

    base_option_row = bundle.resolve_experiment_options("combined") if bundle is not None else {}
    profiles = [
        {
            "run_key": "metadata_only",
            "experiment_key": "metadata_boost",
            "options": _build_phase2_options(base_option_row, enable_metadata=True, enable_normalized_bm25=False),
        },
        {
            "run_key": "normalized_bm25_only",
            "experiment_key": "normalized_bm25",
            "options": _build_phase2_options(base_option_row, enable_metadata=False, enable_normalized_bm25=True),
        },
    ]

    compare_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for profile in profiles:
        run_key = profile["run_key"]
        run_label = f"{args.run_prefix}_{run_key}"
        run_output_dir = output_root / run_label
        run_output_dir.mkdir(parents=True, exist_ok=True)

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

        pipeline = ScenarioBPhase2Pipeline(
            PipelinePaths(
                project_root=project_root,
                chroma_dir=Path(args.chroma_dir).resolve() if args.chroma_dir else None,
            ),
            settings=settings,
            options=profile["options"],
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
                "eval_set_key": args.eval_set_key,
                "question_set_path": str(question_set_path),
                "question_id_file": str(question_id_file) if question_id_file else "",
                "experiment_key": profile["experiment_key"],
                "single_factor_run_key": run_key,
                "single_factor_options": {
                    "enable_controlled_query_expansion": profile["options"].enable_controlled_query_expansion,
                    "enable_normalized_bm25": profile["options"].enable_normalized_bm25,
                    "enable_metadata_aware_retrieval": profile["options"].enable_metadata_aware_retrieval,
                    "enable_table_body_pairing": profile["options"].enable_table_body_pairing,
                    "enable_soft_crag_lite": profile["options"].enable_soft_crag_lite,
                },
            },
        )
        manifest_rows.append(result["manifest"])

        manual_summary_rows = read_csv(run_output_dir / "phase2_eval_manual_summary.csv")
        auto_summary_rows = read_csv(run_output_dir / "phase2_eval_auto_summary.csv")
        manual_completed_rows = read_csv(run_output_dir / "phase2_eval_manual_completed.csv")

        run_metrics = {
            "manual_mean": _pick_group_value(manual_summary_rows, "overall", ["avg_manual_eval_score", "manual_mean"]),
            "type2_manual_mean": _pick_group_value(manual_summary_rows, "TYPE 2", ["avg_manual_eval_score", "manual_mean"]),
            "type4_manual_mean": _pick_group_value(manual_summary_rows, "TYPE 4", ["avg_manual_eval_score", "manual_mean"]),
            "rejection_success_rate": _pick_group_value(auto_summary_rows, "TYPE 4", ["rejection_success_rate"]),
            "latency_sec": _pick_group_value(auto_summary_rows, "overall", ["avg_elapsed_sec", "latency_sec"]),
        }

        degradation_rows, counters = _collect_degradation_rows(
            baseline_rows=baseline_manual_completed_rows,
            run_rows=manual_completed_rows,
        )
        detail_path = run_output_dir / "phase2_single_factor_degradation_cases.csv"
        write_csv(detail_path, degradation_rows)

        compare_row = {
            "run_label": run_label,
            "experiment_key": profile["experiment_key"],
            "question_count": len(manual_completed_rows),
            "overall_manual_mean": run_metrics["manual_mean"],
            "type2_manual_mean": run_metrics["type2_manual_mean"],
            "type4_manual_mean": run_metrics["type4_manual_mean"],
            "type4_rejection_success_rate": run_metrics["rejection_success_rate"],
            "latency_sec": run_metrics["latency_sec"],
            "delta_overall_manual_mean_vs_b06": (
                round((run_metrics["manual_mean"] or 0.0) - (baseline_metrics["manual_mean"] or 0.0), 4)
                if run_metrics["manual_mean"] is not None and baseline_metrics["manual_mean"] is not None
                else None
            ),
            "delta_type2_manual_mean_vs_b06": (
                round((run_metrics["type2_manual_mean"] or 0.0) - (baseline_metrics["type2_manual_mean"] or 0.0), 4)
                if run_metrics["type2_manual_mean"] is not None and baseline_metrics["type2_manual_mean"] is not None
                else None
            ),
            "delta_type4_rejection_success_vs_b06": (
                round((run_metrics["rejection_success_rate"] or 0.0) - (baseline_metrics["rejection_success_rate"] or 0.0), 4)
                if run_metrics["rejection_success_rate"] is not None and baseline_metrics["rejection_success_rate"] is not None
                else None
            ),
            "delta_latency_sec_vs_b06": (
                round((run_metrics["latency_sec"] or 0.0) - (baseline_metrics["latency_sec"] or 0.0), 2)
                if run_metrics["latency_sec"] is not None and baseline_metrics["latency_sec"] is not None
                else None
            ),
            **counters,
            "degradation_case_file": str(detail_path),
        }
        compare_rows.append(compare_row)

        print(
            f"[run={run_key}] overall={run_metrics['manual_mean']} | "
            f"type2={run_metrics['type2_manual_mean']} | "
            f"type4_rej={run_metrics['rejection_success_rate']} | "
            f"latency={run_metrics['latency_sec']} | "
            f"delta={compare_row['delta_overall_manual_mean_vs_b06']}"
        )

    compare_csv_path = output_root / "single_factor_ablation_full45_compare.csv"
    write_csv(compare_csv_path, compare_rows)

    should_hold_combined = any(
        (row.get("delta_overall_manual_mean_vs_b06") is not None and float(row["delta_overall_manual_mean_vs_b06"]) < 0)
        or (row.get("delta_type4_rejection_success_vs_b06") is not None and float(row["delta_type4_rejection_success_vs_b06"]) < 0)
        for row in compare_rows
    )

    report_payload = {
        "project_root": str(project_root),
        "output_root": str(output_root),
        "question_set_path": str(question_set_path),
        "question_id_file": str(question_id_file) if question_id_file else "",
        "baseline_paths": {
            "manual_summary": str(baseline_manual_summary_path),
            "auto_summary": str(baseline_auto_summary_path),
            "manual_completed": str(baseline_manual_completed_path),
        },
        "baseline_metrics": baseline_metrics,
        "compare_csv": str(compare_csv_path),
        "run_manifests": manifest_rows,
        "hold_combined": should_hold_combined,
        "hold_reason": (
            "At least one single-factor run degraded overall manual mean or type4 rejection success vs B-06 baseline."
            if should_hold_combined
            else "No significant degradation detected in single-factor runs."
        ),
    }
    write_json(output_root / "single_factor_ablation_full45_report.json", report_payload)

    print(f"[done] compare_csv={compare_csv_path}")
    print(f"[done] hold_combined={should_hold_combined}")
    print(f"[baseline] {json.dumps(baseline_metrics, ensure_ascii=False)}")


if __name__ == "__main__":
    main()

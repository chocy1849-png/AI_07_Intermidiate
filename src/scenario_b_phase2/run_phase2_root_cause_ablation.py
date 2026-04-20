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
    parser = argparse.ArgumentParser(description="Phase2 root-cause decomposition runner (full_45)")
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument("--experiment-config", default=str(root / "config" / "phase2_experiments.yaml"))
    parser.add_argument("--eval-set-key", default="full_45")
    parser.add_argument("--question-set-path", default="")
    parser.add_argument("--question-id-file", default="")
    parser.add_argument("--output-root", default=str(root / "rag_outputs" / "phase2_runs"))
    parser.add_argument("--run-prefix", default="p0_root_cause_full45")
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


def _collect_degradation_rows(
    baseline_rows: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    baseline_by_qid = {str(row.get("question_id", "")).strip(): row for row in baseline_rows}
    degradation_rows: list[dict[str, Any]] = []
    degraded_count = 0
    for row in run_rows:
        qid = str(row.get("question_id", "")).strip()
        baseline_row = baseline_by_qid.get(qid)
        if baseline_row is None:
            continue
        base_mean = _manual_mean(baseline_row)
        run_mean = _manual_mean(row)
        delta = None if (base_mean is None or run_mean is None) else round(run_mean - base_mean, 4)
        degraded = int(delta is not None and delta < 0)
        degraded_count += degraded
        degradation_rows.append(
            {
                "question_id": qid,
                "type_group": row.get("type_group", ""),
                "answer_type": row.get("answer_type", ""),
                "baseline_manual_mean": base_mean,
                "run_manual_mean": run_mean,
                "delta_manual_mean": delta,
                "degraded": degraded,
            }
        )
    return degradation_rows, degraded_count


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


def _build_condition_table_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "condition": "generator_model",
            "b06_baseline": "gpt-5-mini",
            "phase2_baseline_aa": "gpt-5-mini",
            "match": "Y",
            "note": "Same model id.",
        },
        {
            "condition": "generation_prompt_template",
            "b06_baseline": "Legacy B01/B03 fixed long-format template",
            "phase2_baseline_aa": "Legacy B01/B03 fixed long-format template",
            "match": "Y",
            "note": "Route-specific parity prompt applied.",
        },
        {
            "condition": "max_new_tokens",
            "b06_baseline": "Not explicitly bounded in legacy OpenAI responses call",
            "phase2_baseline_aa": "Not explicitly bounded in OpenAI parity path",
            "match": "Y",
            "note": "Paritied for OpenAI generator path.",
        },
        {
            "condition": "stop_sequences",
            "b06_baseline": "None",
            "phase2_baseline_aa": "None",
            "match": "Y",
            "note": "No explicit stop sequence in both.",
        },
        {
            "condition": "context_assembly",
            "b06_baseline": "Route-specific context blocks (legacy B01/B03 formatter)",
            "phase2_baseline_aa": "Route-specific context blocks (legacy B01/B03 formatter)",
            "match": "Y",
            "note": "Context fields aligned by route.",
        },
        {
            "condition": "B03 branch top_k",
            "b06_baseline": "B02 route: 5, B03 route: 3~4",
            "phase2_baseline_aa": "B02 route: 5, B03 route: 3~4",
            "match": "Y",
            "note": "B03 comparison=4 / non-comparison=3 aligned.",
        },
        {
            "condition": "routing_rule",
            "b06_baseline": "factual/comparison->b03, follow_up/rejection/dependency->b02",
            "phase2_baseline_aa": "factual/comparison->b03a, follow_up/rejection->b02",
            "match": "Y",
            "note": "Semantically aligned.",
        },
        {
            "condition": "controlled_query_expansion",
            "b06_baseline": "Off",
            "phase2_baseline_aa": "Off",
            "match": "Y",
            "note": "",
        },
        {
            "condition": "metadata_aware_retrieval",
            "b06_baseline": "Off",
            "phase2_baseline_aa": "Off",
            "match": "Y",
            "note": "",
        },
        {
            "condition": "normalized_bm25",
            "b06_baseline": "Off",
            "phase2_baseline_aa": "Off",
            "match": "Y",
            "note": "",
        },
        {
            "condition": "answer_formatting_policy",
            "b06_baseline": "5-section output format (요약/핵심/요구사항/일정예산기관/근거)",
            "phase2_baseline_aa": "5-section output format (요약/핵심/요구사항/일정예산기관/근거)",
            "match": "Y",
            "note": "B02/B03 route prompt policy matched.",
        },
        {
            "condition": "soft_crag_lite",
            "b06_baseline": "Off",
            "phase2_baseline_aa": "Off",
            "match": "Y",
            "note": "Hard CRAG branch remains active on b03a.",
        },
    ]
    return rows


def _run_profile(
    *,
    args: argparse.Namespace,
    project_root: Path,
    question_rows: list[dict[str, Any]],
    question_set_path: Path,
    question_id_file: Path | None,
    judge_model: str | None,
    profile: dict[str, Any],
    output_root: Path,
    b06_metrics: dict[str, float | None],
    baseline_manual_completed_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_label = f"{args.run_prefix}_{profile['run_key']}"
    run_dir = output_root / run_label
    run_dir.mkdir(parents=True, exist_ok=True)

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
        run_dir,
        judge_model=judge_model,
        run_label=run_label,
        extra_manifest={
            "project_root": str(project_root),
            "eval_set_key": args.eval_set_key,
            "question_set_path": str(question_set_path),
            "question_id_file": str(question_id_file) if question_id_file else "",
            "single_factor_experiment_key": profile["experiment_key"],
            "phase2_options": {
                "enable_controlled_query_expansion": profile["options"].enable_controlled_query_expansion,
                "enable_normalized_bm25": profile["options"].enable_normalized_bm25,
                "enable_metadata_aware_retrieval": profile["options"].enable_metadata_aware_retrieval,
                "enable_table_body_pairing": profile["options"].enable_table_body_pairing,
                "enable_soft_crag_lite": profile["options"].enable_soft_crag_lite,
                "metadata_boost_scale": profile["options"].metadata_boost_scale,
                "metadata_disable_for_rejection": profile["options"].metadata_disable_for_rejection,
                "metadata_scope_mode": profile["options"].metadata_scope_mode,
                "normalized_bm25_mode": profile["options"].normalized_bm25_mode,
            },
        },
    )
    manifest = result["manifest"]

    manual_summary = read_csv(run_dir / "phase2_eval_manual_summary.csv")
    auto_summary = read_csv(run_dir / "phase2_eval_auto_summary.csv")
    manual_completed = read_csv(run_dir / "phase2_eval_manual_completed.csv")
    degradation_rows, degradation_count = _collect_degradation_rows(baseline_manual_completed_rows, manual_completed)
    degradation_path = run_dir / "phase2_eval_degradation_vs_b06.csv"
    write_csv(degradation_path, degradation_rows)

    run_metrics = {
        "avg_answer_chars": _pick_group_value(auto_summary, "overall", ["avg_answer_chars"]),
        "latency_sec": _pick_group_value(auto_summary, "overall", ["avg_elapsed_sec", "latency_sec"]),
        "overall_manual_mean": _pick_group_value(manual_summary, "overall", ["avg_manual_eval_score", "manual_mean"]),
        "type2_manual_mean": _pick_group_value(manual_summary, "TYPE 2", ["avg_manual_eval_score", "manual_mean"]),
        "type4_rejection_success": _pick_group_value(auto_summary, "TYPE 4", ["rejection_success_rate"]),
    }

    compare_row = {
        "run_label": run_label,
        "experiment_key": profile["experiment_key"],
        "avg_answer_chars": run_metrics["avg_answer_chars"],
        "overall_manual_mean": run_metrics["overall_manual_mean"],
        "type2_manual_mean": run_metrics["type2_manual_mean"],
        "type4_rejection_success": run_metrics["type4_rejection_success"],
        "latency_sec": run_metrics["latency_sec"],
        "degradation_count": degradation_count,
        "delta_avg_answer_chars_vs_b06": (
            round((run_metrics["avg_answer_chars"] or 0.0) - (b06_metrics["avg_answer_chars"] or 0.0), 2)
            if run_metrics["avg_answer_chars"] is not None and b06_metrics["avg_answer_chars"] is not None
            else None
        ),
        "delta_overall_manual_mean_vs_b06": (
            round((run_metrics["overall_manual_mean"] or 0.0) - (b06_metrics["overall_manual_mean"] or 0.0), 4)
            if run_metrics["overall_manual_mean"] is not None and b06_metrics["overall_manual_mean"] is not None
            else None
        ),
        "delta_type2_manual_mean_vs_b06": (
            round((run_metrics["type2_manual_mean"] or 0.0) - (b06_metrics["type2_manual_mean"] or 0.0), 4)
            if run_metrics["type2_manual_mean"] is not None and b06_metrics["type2_manual_mean"] is not None
            else None
        ),
        "delta_type4_rejection_success_vs_b06": (
            round((run_metrics["type4_rejection_success"] or 0.0) - (b06_metrics["type4_rejection_success"] or 0.0), 4)
            if run_metrics["type4_rejection_success"] is not None and b06_metrics["type4_rejection_success"] is not None
            else None
        ),
        "delta_latency_sec_vs_b06": (
            round((run_metrics["latency_sec"] or 0.0) - (b06_metrics["latency_sec"] or 0.0), 2)
            if run_metrics["latency_sec"] is not None and b06_metrics["latency_sec"] is not None
            else None
        ),
        "degradation_file": str(degradation_path),
    }
    print(
        f"[run={profile['run_key']}] chars={run_metrics['avg_answer_chars']} "
        f"overall={run_metrics['overall_manual_mean']} type2={run_metrics['type2_manual_mean']} "
        f"type4_rej={run_metrics['type4_rejection_success']} latency={run_metrics['latency_sec']} "
        f"degradation={degradation_count}"
    )
    return compare_row, manifest


def _evaluate_aa_parity(
    aa_row: dict[str, Any],
    b06_metrics: dict[str, float | None],
) -> dict[str, Any]:
    chars = _to_float(aa_row.get("avg_answer_chars"))
    b06_chars = _to_float(b06_metrics.get("avg_answer_chars"))
    overall_delta = _to_float(aa_row.get("delta_overall_manual_mean_vs_b06"))
    type2_delta = _to_float(aa_row.get("delta_type2_manual_mean_vs_b06"))
    type4_delta = _to_float(aa_row.get("delta_type4_rejection_success_vs_b06"))

    chars_ratio = None
    chars_within_20pct = None
    if chars is not None and b06_chars not in (None, 0):
        chars_ratio = chars / b06_chars
        chars_within_20pct = int(0.8 <= chars_ratio <= 1.2)

    checks = {
        "avg_answer_chars_within_20pct": chars_within_20pct,
        "overall_manual_mean_delta_gte_minus_0_10": int(overall_delta is not None and overall_delta >= -0.10),
        "type2_manual_mean_delta_gte_minus_0_10": int(type2_delta is not None and type2_delta >= -0.10),
        "type4_rejection_success_delta_gte_minus_0_05": int(type4_delta is not None and type4_delta >= -0.05),
    }
    passed = all(value == 1 for value in checks.values())
    return {
        "passed": passed,
        "chars_ratio_vs_b06": round(chars_ratio, 4) if chars_ratio is not None else None,
        "checks": checks,
    }


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    judge_model = None if args.no_judge else args.judge_model

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

    baseline_manual_summary_path = Path(args.baseline_manual_summary).resolve()
    baseline_auto_summary_path = Path(args.baseline_auto_summary).resolve()
    baseline_manual_completed_path = Path(args.baseline_manual_completed).resolve()
    baseline_manual_summary_rows = read_csv(baseline_manual_summary_path)
    baseline_auto_summary_rows = read_csv(baseline_auto_summary_path)
    baseline_manual_completed_rows = read_csv(baseline_manual_completed_path)
    b06_metrics = {
        "avg_answer_chars": _pick_group_value(baseline_auto_summary_rows, "overall", ["avg_answer_chars"]),
        "latency_sec": _pick_group_value(baseline_auto_summary_rows, "overall", ["avg_elapsed_sec", "latency_sec"]),
        "overall_manual_mean": _pick_group_value(baseline_manual_summary_rows, "overall", ["avg_manual_eval_score", "manual_mean"]),
        "type2_manual_mean": _pick_group_value(baseline_manual_summary_rows, "TYPE 2", ["avg_manual_eval_score", "manual_mean"]),
        "type4_rejection_success": _pick_group_value(baseline_auto_summary_rows, "TYPE 4", ["rejection_success_rate"]),
    }

    compare_rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    baseline_profile = {
        "run_key": "phase2_baseline_aa",
        "experiment_key": "baseline_aa",
        "options": Phase2Options(
            enable_controlled_query_expansion=False,
            enable_normalized_bm25=False,
            enable_metadata_aware_retrieval=False,
            enable_table_body_pairing=False,
            enable_soft_crag_lite=False,
            normalized_bm25_mode="all",
            metadata_scope_mode="all",
            metadata_disable_for_rejection=False,
            metadata_boost_scale=1.0,
        ),
    }
    baseline_row, baseline_manifest = _run_profile(
        args=args,
        project_root=project_root,
        question_rows=question_rows,
        question_set_path=question_set_path,
        question_id_file=question_id_file,
        judge_model=judge_model,
        profile=baseline_profile,
        output_root=output_root,
        b06_metrics=b06_metrics,
        baseline_manual_completed_rows=baseline_manual_completed_rows,
    )
    compare_rows.append(baseline_row)
    manifests.append(baseline_manifest)

    aa_parity = _evaluate_aa_parity(baseline_row, b06_metrics)
    if aa_parity["passed"]:
        metadata_profiles = [
            {
                "run_key": "metadata_t4off",
                "experiment_key": "metadata_t4off",
                "options": Phase2Options(
                    enable_controlled_query_expansion=False,
                    enable_normalized_bm25=False,
                    enable_metadata_aware_retrieval=True,
                    enable_table_body_pairing=False,
                    enable_soft_crag_lite=False,
                    metadata_disable_for_rejection=True,
                    metadata_boost_scale=1.0,
                    metadata_scope_mode="all",
                ),
            },
            {
                "run_key": "metadata_t4off_scoped",
                "experiment_key": "metadata_t4off_scoped",
                "options": Phase2Options(
                    enable_controlled_query_expansion=False,
                    enable_normalized_bm25=False,
                    enable_metadata_aware_retrieval=True,
                    enable_table_body_pairing=False,
                    enable_soft_crag_lite=False,
                    metadata_disable_for_rejection=True,
                    metadata_boost_scale=0.5,
                    metadata_scope_mode="comparison_and_explicit_factual",
                ),
            },
            {
                "run_key": "metadata_t4off_half",
                "experiment_key": "metadata_t4off_half",
                "options": Phase2Options(
                    enable_controlled_query_expansion=False,
                    enable_normalized_bm25=False,
                    enable_metadata_aware_retrieval=True,
                    enable_table_body_pairing=False,
                    enable_soft_crag_lite=False,
                    metadata_disable_for_rejection=True,
                    metadata_boost_scale=0.5,
                    metadata_scope_mode="all",
                ),
            },
        ]
        for profile in metadata_profiles:
            row, manifest = _run_profile(
                args=args,
                project_root=project_root,
                question_rows=question_rows,
                question_set_path=question_set_path,
                question_id_file=question_id_file,
                judge_model=judge_model,
                profile=profile,
                output_root=output_root,
                b06_metrics=b06_metrics,
                baseline_manual_completed_rows=baseline_manual_completed_rows,
            )
            compare_rows.append(row)
            manifests.append(manifest)
    else:
        print("[gate] A/A parity failed. Metadata helper re-validation is skipped.")

    compare_csv_path = output_root / "root_cause_single_factor_compare.csv"
    write_csv(compare_csv_path, compare_rows)

    phase2_baseline_row = next((row for row in compare_rows if row.get("experiment_key") == "baseline_aa"), None)
    baseline_side_by_side: list[dict[str, Any]] = [
        {
            "run_label": "b06_baseline_reference",
            **b06_metrics,
        }
    ]
    if phase2_baseline_row is not None:
        baseline_side_by_side.append(
            {
                "run_label": "phase2_baseline_aa",
                "avg_answer_chars": phase2_baseline_row.get("avg_answer_chars"),
                "latency_sec": phase2_baseline_row.get("latency_sec"),
                "overall_manual_mean": phase2_baseline_row.get("overall_manual_mean"),
                "type2_manual_mean": phase2_baseline_row.get("type2_manual_mean"),
                "type4_rejection_success": phase2_baseline_row.get("type4_rejection_success"),
            }
        )
        baseline_side_by_side.append(
            {
                "run_label": "delta_phase2_minus_b06",
                "avg_answer_chars": phase2_baseline_row.get("delta_avg_answer_chars_vs_b06"),
                "latency_sec": phase2_baseline_row.get("delta_latency_sec_vs_b06"),
                "overall_manual_mean": phase2_baseline_row.get("delta_overall_manual_mean_vs_b06"),
                "type2_manual_mean": phase2_baseline_row.get("delta_type2_manual_mean_vs_b06"),
                "type4_rejection_success": phase2_baseline_row.get("delta_type4_rejection_success_vs_b06"),
            }
        )

    baseline_compare_path = output_root / "root_cause_baseline_aa_side_by_side.csv"
    write_csv(baseline_compare_path, baseline_side_by_side)

    condition_compare_path = output_root / "root_cause_condition_compare.csv"
    write_csv(condition_compare_path, _build_condition_table_rows())

    report = {
        "project_root": str(project_root),
        "question_set_path": str(question_set_path),
        "question_count": len(question_rows),
        "aa_parity_check": aa_parity,
        "metadata_revalidation_executed": aa_parity["passed"],
        "b06_reference": {
            "manual_summary": str(baseline_manual_summary_path),
            "auto_summary": str(baseline_auto_summary_path),
            "manual_completed": str(baseline_manual_completed_path),
            "metrics": b06_metrics,
        },
        "outputs": {
            "compare_csv": str(compare_csv_path),
            "baseline_aa_side_by_side_csv": str(baseline_compare_path),
            "condition_compare_csv": str(condition_compare_path),
        },
        "run_manifests": manifests,
    }
    report_path = output_root / "root_cause_analysis_report.json"
    write_json(report_path, report)

    print(f"[done] compare={compare_csv_path}")
    print(f"[done] baseline_side_by_side={baseline_compare_path}")
    print(f"[done] condition_compare={condition_compare_path}")
    print(f"[done] report={report_path}")
    print(f"[report_json] {json.dumps({'runs': len(compare_rows), 'aa_parity_passed': aa_parity['passed']}, ensure_ascii=False)}")


if __name__ == "__main__":
    main()

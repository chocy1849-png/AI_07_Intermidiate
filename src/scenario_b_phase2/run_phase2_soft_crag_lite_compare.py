from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from pathlib import Path
from typing import Any

from eval_utils import average, read_csv, safe_float, write_csv, write_json
from scenario_a.common_pipeline import PipelinePaths, PipelineSettings
from scenario_b_phase2.experiment_config import load_phase2_experiment_bundle
from scenario_b_phase2.phase2_eval import evaluate_phase2, read_eval_rows
from scenario_b_phase2.phase2_pipeline import Phase2Options, ScenarioBPhase2Pipeline


def _read_csv_if_exists(path: Path) -> list[dict[str, Any]]:
    return read_csv(path) if path.exists() else []


def _pick(rows: list[dict[str, Any]], group_name: str, key: str) -> float | None:
    row = next((item for item in rows if str(item.get("group_name", "")) == group_name), None)
    if row is None:
        return None
    return safe_float(row.get(key))


def _manual_mean(row: dict[str, Any]) -> float | None:
    return average(
        [
            safe_float(row.get("faithfulness_score")),
            safe_float(row.get("completeness_score")),
            safe_float(row.get("groundedness_score")),
            safe_float(row.get("relevancy_score")),
        ]
    )


def _manual_mean_by_filter(rows: list[dict[str, Any]], predicate: Any) -> float | None:
    values: list[float] = []
    for row in rows:
        if not predicate(row):
            continue
        mean = _manual_mean(row)
        if mean is not None:
            values.append(mean)
    return average(values)


def _collect_degradation_count(
    baseline_rows: list[dict[str, Any]],
    variant_rows: list[dict[str, Any]],
) -> int:
    baseline_by_qid = {str(row.get("question_id", "")).strip(): row for row in baseline_rows}
    degraded = 0
    for row in variant_rows:
        qid = str(row.get("question_id", "")).strip()
        base_row = baseline_by_qid.get(qid)
        if base_row is None:
            continue
        base_mean = _manual_mean(base_row)
        run_mean = _manual_mean(row)
        if base_mean is None or run_mean is None:
            continue
        if run_mean < base_mean:
            degraded += 1
    return degraded


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Phase2 Soft CRAG-lite compare runner (baseline_v3 candidate fixed)."
    )
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument("--experiment-config", default=str(root / "config" / "phase2_experiments.yaml"))
    parser.add_argument("--output-root", default=str(root / "rag_outputs" / "phase2_runs"))
    parser.add_argument("--run-prefix", default="p2_soft_crag_lite_compare_v1")
    parser.add_argument("--model-key", default="gpt5mini_api")
    parser.add_argument("--routing-model", default="gpt-5-mini")
    parser.add_argument("--judge-model", default="gpt-5")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--crag-top-n", type=int, default=5)
    parser.add_argument("--vector-weight", type=float, default=0.7)
    parser.add_argument("--bm25-weight", type=float, default=0.3)
    parser.add_argument("--embedding-backend", default="openai_text_embedding_3_small_true_table_ocr_v4")
    parser.add_argument("--chroma-dir", default=str(root.parent / "rfp_rag_chroma_db_phase2_true_table_ocr_v4"))
    parser.add_argument("--max-full45-questions", type=int, default=0)
    parser.add_argument("--max-groupbc-questions", type=int, default=0)
    parser.add_argument("--variants", default="baseline_v3_soft_off,soft_crag_targeted")
    return parser.parse_args()


def _baseline_v3_options() -> Phase2Options:
    return Phase2Options(
        enable_controlled_query_expansion=False,
        enable_normalized_bm25=False,
        enable_metadata_aware_retrieval=True,
        enable_metadata_bonus_v2=None,
        enable_table_body_pairing=True,
        enable_soft_crag_lite=False,
        metadata_disable_for_rejection=True,
        metadata_boost_scale=0.5,
        metadata_scope_mode="all",
        normalized_bm25_mode="all",
        enable_comparison_evidence_helper=True,
        comparison_helper_doc_bonus=0.0045,
        comparison_helper_axis_bonus=0.0015,
        comparison_helper_max_per_doc=2,
        enable_groupc_table_plus_text_guard=True,
        groupc_pair_bonus=0.006,
        groupc_parent_bonus=0.003,
        groupc_table_penalty_without_body=0.012,
        enable_question_type_gated_ocr_routing=True,
        enable_structured_evidence_priority=True,
        enable_hybridqa_stage_metrics=True,
        enable_table_factual_exact_answer_mode=True,
        enable_table_factual_alignment_scoring=True,
        table_factual_generic_penalty=0.012,
        enable_answer_type_router=True,
        answer_type_router_force=False,
    )


def _variant_options(variant: str) -> Phase2Options:
    base = _baseline_v3_options()
    if variant == "baseline_v3_soft_off":
        return base
    if variant == "soft_crag_targeted":
        base.enable_soft_crag_lite = True
        base.soft_crag_scope_mode = "targeted"
        base.soft_crag_factual_mode = "off"
        return base
    if variant == "soft_crag_targeted_weak_factual":
        base.enable_soft_crag_lite = True
        base.soft_crag_scope_mode = "targeted"
        base.soft_crag_factual_mode = "weak"
        return base
    raise KeyError(f"Unknown variant: {variant}")


def _run_eval(
    *,
    project_root: Path,
    question_rows: list[dict[str, Any]],
    run_dir: Path,
    run_label: str,
    args: argparse.Namespace,
    options: Phase2Options,
) -> dict[str, Any]:
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
        options=options,
    )
    adapter = pipeline.create_adapter(args.model_key)
    judge_model = None if args.no_judge else args.judge_model
    return evaluate_phase2(
        pipeline,
        adapter,
        question_rows,
        run_dir,
        judge_model=judge_model,
        run_label=run_label,
        extra_manifest={
            "soft_crag_variant": run_label,
            "phase": "soft_crag_lite_compare",
        },
    )


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_root = output_root / args.run_prefix
    run_root.mkdir(parents=True, exist_ok=True)

    bundle = load_phase2_experiment_bundle(Path(args.experiment_config).resolve(), project_root)
    _, full45_path = bundle.resolve_eval_set("full_45")
    _, groupbc_path = bundle.resolve_eval_set("group_bc")

    full45_rows = read_eval_rows(full45_path, question_id_file=None, shard_count=1, shard_index=0)
    groupbc_rows = read_eval_rows(groupbc_path, question_id_file=None, shard_count=1, shard_index=0)
    if args.max_full45_questions > 0:
        full45_rows = full45_rows[: int(args.max_full45_questions)]
    if args.max_groupbc_questions > 0:
        groupbc_rows = groupbc_rows[: int(args.max_groupbc_questions)]

    variants = [item.strip() for item in str(args.variants).split(",") if item.strip()]
    if "baseline_v3_soft_off" not in variants:
        variants = ["baseline_v3_soft_off", *variants]

    run_metrics: dict[str, dict[str, Any]] = {}
    full45_manual_rows_by_variant: dict[str, list[dict[str, Any]]] = {}
    manifests: list[dict[str, Any]] = []
    for variant in variants:
        options = _variant_options(variant)
        full_dir = run_root / f"{variant}_full45"
        groupbc_dir = run_root / f"{variant}_groupbc"

        full_result = _run_eval(
            project_root=project_root,
            question_rows=full45_rows,
            run_dir=full_dir,
            run_label=f"{args.run_prefix}_{variant}_full45",
            args=args,
            options=options,
        )
        group_result = _run_eval(
            project_root=project_root,
            question_rows=groupbc_rows,
            run_dir=groupbc_dir,
            run_label=f"{args.run_prefix}_{variant}_groupbc",
            args=args,
            options=options,
        )
        manifests.extend([full_result["manifest"], group_result["manifest"]])

        full_manual_summary = _read_csv_if_exists(full_dir / "phase2_eval_manual_summary.csv")
        full_auto_summary = _read_csv_if_exists(full_dir / "phase2_eval_auto_summary.csv")
        full_manual_completed = _read_csv_if_exists(full_dir / "phase2_eval_manual_completed.csv")
        group_manual_completed = _read_csv_if_exists(groupbc_dir / "phase2_eval_manual_completed.csv")

        full45_manual_rows_by_variant[variant] = full_manual_completed
        run_metrics[variant] = {
            "overall_manual_mean": _pick(full_manual_summary, "overall", "avg_manual_eval_score"),
            "type2_manual_mean": _pick(full_manual_summary, "TYPE 2", "avg_manual_eval_score"),
            "comparison_manual_mean": _pick(full_manual_summary, "answer_type:comparison", "avg_manual_eval_score"),
            "type4_rejection_success": _pick(full_auto_summary, "TYPE 4", "rejection_success_rate"),
            "latency_sec": _pick(full_auto_summary, "overall", "avg_elapsed_sec"),
            "group_c_manual_mean": _manual_mean_by_filter(
                group_manual_completed,
                lambda row: (
                    str(row.get("group_label", "")).strip().lower() == "group c"
                    or str(row.get("answer_type", "")).strip().lower() == "table_plus_text"
                ),
            ),
        }

    baseline_key = "baseline_v3_soft_off"
    baseline_rows = full45_manual_rows_by_variant.get(baseline_key, [])
    compare_rows: list[dict[str, Any]] = []
    for variant in variants:
        metrics = run_metrics.get(variant, {})
        baseline = run_metrics.get(baseline_key, {})
        degradation_count = 0
        if (
            variant != baseline_key
            and baseline_rows
            and full45_manual_rows_by_variant.get(variant, [])
        ):
            degradation_count = _collect_degradation_count(
                baseline_rows=baseline_rows,
                variant_rows=full45_manual_rows_by_variant.get(variant, []),
            )
        overall_delta = None
        type2_delta = None
        groupc_delta = None
        type4_delta = None
        latency_delta = None
        if metrics.get("overall_manual_mean") is not None and baseline.get("overall_manual_mean") is not None:
            overall_delta = round(float(metrics["overall_manual_mean"]) - float(baseline["overall_manual_mean"]), 4)
        if metrics.get("type2_manual_mean") is not None and baseline.get("type2_manual_mean") is not None:
            type2_delta = round(float(metrics["type2_manual_mean"]) - float(baseline["type2_manual_mean"]), 4)
        if metrics.get("group_c_manual_mean") is not None and baseline.get("group_c_manual_mean") is not None:
            groupc_delta = round(float(metrics["group_c_manual_mean"]) - float(baseline["group_c_manual_mean"]), 4)
        if metrics.get("type4_rejection_success") is not None and baseline.get("type4_rejection_success") is not None:
            type4_delta = round(
                float(metrics["type4_rejection_success"]) - float(baseline["type4_rejection_success"]),
                4,
            )
        if metrics.get("latency_sec") is not None and baseline.get("latency_sec") is not None:
            latency_delta = round(float(metrics["latency_sec"]) - float(baseline["latency_sec"]), 2)

        improved_axis = bool((type2_delta is not None and type2_delta > 0) or (groupc_delta is not None and groupc_delta > 0))
        full45_non_degrade = bool(overall_delta is None or overall_delta >= 0.0)
        type4_non_degrade = bool(type4_delta is None or type4_delta >= 0.0)
        latency_ok = bool(latency_delta is None or latency_delta <= 6.0)
        verdict = "candidate_adopt"
        if not (full45_non_degrade and improved_axis and type4_non_degrade and latency_ok):
            verdict = "hold"
        if variant == baseline_key:
            verdict = "baseline"

        compare_rows.append(
            {
                "variant": variant,
                "full45_overall_manual_mean": metrics.get("overall_manual_mean"),
                "full45_type2_manual_mean": metrics.get("type2_manual_mean"),
                "full45_comparison_manual_mean": metrics.get("comparison_manual_mean"),
                "group_c_manual_mean": metrics.get("group_c_manual_mean"),
                "type4_rejection_success": metrics.get("type4_rejection_success"),
                "latency_sec": metrics.get("latency_sec"),
                "degradation_count": degradation_count,
                "delta_overall_vs_baseline": overall_delta,
                "delta_type2_vs_baseline": type2_delta,
                "delta_group_c_vs_baseline": groupc_delta,
                "delta_type4_vs_baseline": type4_delta,
                "delta_latency_vs_baseline_sec": latency_delta,
                "full45_non_degrade": int(full45_non_degrade),
                "improved_axis": int(improved_axis),
                "type4_non_degrade": int(type4_non_degrade),
                "latency_ok": int(latency_ok),
                "verdict": verdict,
            }
        )

    compare_csv = run_root / "soft_crag_lite_compare.csv"
    write_csv(compare_csv, compare_rows)
    report_json = run_root / "soft_crag_lite_report.json"
    write_json(
        report_json,
        {
            "run_prefix": args.run_prefix,
            "baseline_variant": baseline_key,
            "compare_csv": str(compare_csv),
            "manifests": manifests,
            "variants": compare_rows,
            "notes": {
                "scope_rule": "targeted (comparison/table_plus_text/follow_up), factual/table_factual off or weak variant",
                "gate": "full45_non_degrade AND (type2 OR group_c improvement) AND type4_non_degrade AND latency_ok",
            },
        },
    )
    print(f"[done] compare_csv={compare_csv}")
    print(f"[done] report_json={report_json}")


if __name__ == "__main__":
    main()

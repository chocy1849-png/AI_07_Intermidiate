from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

from eval_utils import average, read_csv, safe_float, write_csv, write_json
from scenario_a.common_pipeline import PipelinePaths, PipelineSettings
from scenario_b_phase2.experiment_config import load_phase2_experiment_bundle
from scenario_b_phase2.phase2_eval import evaluate_phase2, read_eval_rows
from scenario_b_phase2.phase2_pipeline import Phase2Options, ScenarioBPhase2Pipeline


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Compare baseline_v2 vs true_table_ocr_v1 vs true_table_ocr_v2 "
            "(table_15 -> group_bc -> gate -> full_45)."
        )
    )
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument("--experiment-config", default=str(root / "config" / "phase2_experiments.yaml"))
    parser.add_argument("--output-root", default=str(root / "rag_outputs" / "phase2_runs"))
    parser.add_argument("--run-prefix", default="p2_true_table_ocr_v2_compare")
    parser.add_argument("--table-eval-set-key", default="table_15")
    parser.add_argument("--groupbc-eval-set-key", default="group_bc")
    parser.add_argument("--full-eval-set-key", default="full_45")
    parser.add_argument("--embedding-backend-baseline", default="openai_text_embedding_3_small")
    parser.add_argument("--embedding-backend-v1", default="openai_text_embedding_3_small_true_table_ocr")
    parser.add_argument("--embedding-backend-v2", default="openai_text_embedding_3_small_true_table_ocr_v2")
    parser.add_argument("--model-key", default="gpt5mini_api")
    parser.add_argument("--routing-model", default="gpt-5-mini")
    parser.add_argument("--judge-model", default="gpt-5")
    parser.add_argument("--chroma-dir-baseline", default="")
    parser.add_argument("--chroma-dir-v1", default=str(root.parent / "rfp_rag_chroma_db_phase2_true_table_ocr"))
    parser.add_argument("--chroma-dir-v2", default=str(root.parent / "rfp_rag_chroma_db_phase2_true_table_ocr_v2"))
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--crag-top-n", type=int, default=5)
    parser.add_argument("--vector-weight", type=float, default=0.7)
    parser.add_argument("--bm25-weight", type=float, default=0.3)
    parser.add_argument("--subset-gate-table15-delta-min", type=float, default=-0.05)
    parser.add_argument("--subset-gate-groupbc-delta-min", type=float, default=-0.05)
    parser.add_argument("--subset-gate-type4-delta-min", type=float, default=-0.05)
    parser.add_argument("--build-v2-assets", action="store_true")
    parser.add_argument("--reset-v2-collection", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
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


def _build_settings(args: argparse.Namespace, embedding_backend: str) -> PipelineSettings:
    return PipelineSettings(
        embedding_backend_key=embedding_backend,
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


def _build_baseline_v2_options() -> Phase2Options:
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
    )


def _collect_metrics(run_dir: Path) -> dict[str, Any]:
    manual_summary_rows = read_csv(run_dir / "phase2_eval_manual_summary.csv")
    auto_summary_rows = read_csv(run_dir / "phase2_eval_auto_summary.csv")
    coverage_summary_rows = read_csv(run_dir / "phase2_eval_coverage_summary.csv")
    manual_completed_rows = read_csv(run_dir / "phase2_eval_manual_completed.csv")
    return {
        "manual_completed_rows": manual_completed_rows,
        "overall_manual_mean": _pick_group_value(manual_summary_rows, "overall", ["avg_manual_eval_score", "manual_mean"]),
        "type2_manual_mean": _pick_group_value(manual_summary_rows, "TYPE 2", ["avg_manual_eval_score", "manual_mean"]),
        "type4_rejection_success": _pick_group_value(auto_summary_rows, "TYPE 4", ["rejection_success_rate"]),
        "table_plus_body_coverage": _pick_group_value(coverage_summary_rows, "overall", ["table_plus_body_coverage"]),
        "latency_sec": _pick_group_value(auto_summary_rows, "overall", ["avg_elapsed_sec", "latency_sec"]),
    }


def _degradation_count(baseline_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> int:
    baseline_by_qid = {str(row.get("question_id", "")).strip(): row for row in baseline_rows}
    count = 0
    for row in candidate_rows:
        qid = str(row.get("question_id", "")).strip()
        base = baseline_by_qid.get(qid)
        if base is None:
            continue
        base_mean = _manual_mean(base)
        cand_mean = _manual_mean(row)
        if base_mean is None or cand_mean is None:
            continue
        if cand_mean < base_mean:
            count += 1
    return count


def _delta(candidate: float | None, baseline: float | None, digits: int = 4) -> float | None:
    if candidate is None or baseline is None:
        return None
    return round(candidate - baseline, digits)


def _run_eval(
    *,
    args: argparse.Namespace,
    project_root: Path,
    question_set_path: Path,
    question_id_file: Path | None,
    run_dir: Path,
    run_label: str,
    embedding_backend: str,
    chroma_dir: Path | None,
) -> dict[str, Any]:
    if bool(args.reuse_existing):
        manual = run_dir / "phase2_eval_manual_summary.csv"
        auto = run_dir / "phase2_eval_auto_summary.csv"
        coverage = run_dir / "phase2_eval_coverage_summary.csv"
        completed = run_dir / "phase2_eval_manual_completed.csv"
        if manual.exists() and auto.exists() and coverage.exists() and completed.exists():
            return _collect_metrics(run_dir)

    run_dir.mkdir(parents=True, exist_ok=True)
    question_rows = read_eval_rows(question_set_path, question_id_file=question_id_file, shard_count=1, shard_index=0)
    pipeline = ScenarioBPhase2Pipeline(
        PipelinePaths(project_root=project_root, chroma_dir=chroma_dir),
        settings=_build_settings(args, embedding_backend),
        options=_build_baseline_v2_options(),
    )
    adapter = pipeline.create_adapter(args.model_key)
    evaluate_phase2(
        pipeline,
        adapter,
        question_rows,
        run_dir,
        judge_model=args.judge_model,
        run_label=run_label,
        extra_manifest={
            "compare_target": "baseline_v2_vs_true_table_ocr_v1_vs_true_table_ocr_v2",
            "question_set_path": str(question_set_path),
            "question_id_file": str(question_id_file) if question_id_file else "",
            "answer_layer_policy": "b06_exact_with_phase2_baseline_v2_options",
        },
    )
    return _collect_metrics(run_dir)


def _build_v2_assets_if_requested(args: argparse.Namespace, project_root: Path) -> None:
    if not args.build_v2_assets:
        return
    command = [
        sys.executable,
        str(project_root / "src" / "scenario_b_phase2" / "true_hwp_table_ocr_v2_augment.py"),
        "--project-root",
        str(project_root),
    ]
    if args.reset_v2_collection:
        command.append("--reset-collection")
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        raise RuntimeError(
            "true_hwp_table_ocr_v2_augment build failed\n"
            f"command: {' '.join(command)}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


def _append_compare_rows(
    *,
    rows: list[dict[str, Any]],
    eval_set_key: str,
    metrics_by_variant: dict[str, dict[str, Any]],
) -> None:
    baseline = metrics_by_variant["baseline_v2"]
    v1 = metrics_by_variant["true_table_ocr_v1"]

    for variant, metrics in metrics_by_variant.items():
        row = {
            "eval_set_key": eval_set_key,
            "run_variant": variant,
            "overall_manual_mean": metrics["overall_manual_mean"],
            "type2_manual_mean": metrics["type2_manual_mean"],
            "type4_rejection_success": metrics["type4_rejection_success"],
            "table_plus_body_coverage": metrics["table_plus_body_coverage"],
            "latency_sec": metrics["latency_sec"],
            "degradation_count": 0,
            "delta_overall_vs_baseline_v2": _delta(metrics["overall_manual_mean"], baseline["overall_manual_mean"]),
            "delta_overall_vs_true_table_ocr_v1": _delta(metrics["overall_manual_mean"], v1["overall_manual_mean"]),
            "delta_type2_vs_baseline_v2": _delta(metrics["type2_manual_mean"], baseline["type2_manual_mean"]),
            "delta_type4_vs_baseline_v2": _delta(metrics["type4_rejection_success"], baseline["type4_rejection_success"]),
            "delta_coverage_vs_baseline_v2": _delta(metrics["table_plus_body_coverage"], baseline["table_plus_body_coverage"]),
            "delta_latency_vs_baseline_v2": _delta(metrics["latency_sec"], baseline["latency_sec"]),
        }
        if variant != "baseline_v2":
            row["degradation_count"] = _degradation_count(
                baseline_rows=baseline["manual_completed_rows"],
                candidate_rows=metrics["manual_completed_rows"],
            )
        rows.append(row)


def _summarize_full45(compare_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    full_rows = [row for row in compare_rows if str(row.get("eval_set_key", "")) == "full_45"]
    by_variant = {str(row.get("run_variant", "")): row for row in full_rows}
    baseline = by_variant.get("baseline_v2")
    v1 = by_variant.get("true_table_ocr_v1")
    v2 = by_variant.get("true_table_ocr_v2")
    if baseline is None:
        return []

    def _value(row: dict[str, Any] | None, key: str) -> float | None:
        if row is None:
            return None
        return _to_float(row.get(key))

    summary = [
        {
            "run_variant": "baseline_v2",
            "overall_manual_mean": _value(baseline, "overall_manual_mean"),
            "table_15_mean": _value(next((r for r in compare_rows if r["eval_set_key"] == "table_15" and r["run_variant"] == "baseline_v2"), {}), "overall_manual_mean"),
            "group_bc_mean": _value(next((r for r in compare_rows if r["eval_set_key"] == "group_bc" and r["run_variant"] == "baseline_v2"), {}), "overall_manual_mean"),
            "table_plus_body_coverage": _value(baseline, "table_plus_body_coverage"),
            "type4_rejection_success": _value(baseline, "type4_rejection_success"),
            "degradation_count": 0,
            "latency_sec": _value(baseline, "latency_sec"),
        },
        {
            "run_variant": "true_table_ocr_v1",
            "overall_manual_mean": _value(v1, "overall_manual_mean"),
            "table_15_mean": _value(next((r for r in compare_rows if r["eval_set_key"] == "table_15" and r["run_variant"] == "true_table_ocr_v1"), {}), "overall_manual_mean"),
            "group_bc_mean": _value(next((r for r in compare_rows if r["eval_set_key"] == "group_bc" and r["run_variant"] == "true_table_ocr_v1"), {}), "overall_manual_mean"),
            "table_plus_body_coverage": _value(v1, "table_plus_body_coverage"),
            "type4_rejection_success": _value(v1, "type4_rejection_success"),
            "degradation_count": int(_to_float(v1.get("degradation_count")) or 0) if v1 else None,
            "latency_sec": _value(v1, "latency_sec"),
            "delta_vs_baseline_v2": _delta(_value(v1, "overall_manual_mean"), _value(baseline, "overall_manual_mean")),
        },
        {
            "run_variant": "true_table_ocr_v2",
            "overall_manual_mean": _value(v2, "overall_manual_mean"),
            "table_15_mean": _value(next((r for r in compare_rows if r["eval_set_key"] == "table_15" and r["run_variant"] == "true_table_ocr_v2"), {}), "overall_manual_mean"),
            "group_bc_mean": _value(next((r for r in compare_rows if r["eval_set_key"] == "group_bc" and r["run_variant"] == "true_table_ocr_v2"), {}), "overall_manual_mean"),
            "table_plus_body_coverage": _value(v2, "table_plus_body_coverage"),
            "type4_rejection_success": _value(v2, "type4_rejection_success"),
            "degradation_count": int(_to_float(v2.get("degradation_count")) or 0) if v2 else None,
            "latency_sec": _value(v2, "latency_sec"),
            "delta_vs_baseline_v2": _delta(_value(v2, "overall_manual_mean"), _value(baseline, "overall_manual_mean")),
            "delta_vs_true_table_ocr_v1": _delta(_value(v2, "overall_manual_mean"), _value(v1, "overall_manual_mean")),
        },
    ]
    return summary


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    run_root = Path(args.output_root).resolve() / args.run_prefix
    run_root.mkdir(parents=True, exist_ok=True)

    _build_v2_assets_if_requested(args, project_root)

    bundle = load_phase2_experiment_bundle(Path(args.experiment_config).resolve(), project_root)
    eval_order = [args.table_eval_set_key, args.groupbc_eval_set_key]

    baseline_chroma = Path(args.chroma_dir_baseline).resolve() if args.chroma_dir_baseline else None
    v1_chroma = Path(args.chroma_dir_v1).resolve() if args.chroma_dir_v1 else None
    v2_chroma = Path(args.chroma_dir_v2).resolve() if args.chroma_dir_v2 else None

    compare_rows: list[dict[str, Any]] = []
    subset_gate_checks: dict[str, int] = {}
    subset_gate_passed = True

    variants = [
        ("baseline_v2", args.embedding_backend_baseline, baseline_chroma),
        ("true_table_ocr_v1", args.embedding_backend_v1, v1_chroma),
        ("true_table_ocr_v2", args.embedding_backend_v2, v2_chroma),
    ]

    for eval_key in eval_order:
        _, qset = bundle.resolve_eval_set(eval_key)
        qids = bundle.resolve_question_id_file(eval_key)
        metrics_by_variant: dict[str, dict[str, Any]] = {}
        for variant, backend, chroma_dir in variants:
            run_dir = run_root / f"{args.run_prefix}_{eval_key}_{variant}"
            metrics_by_variant[variant] = _run_eval(
                args=args,
                project_root=project_root,
                question_set_path=qset,
                question_id_file=qids,
                run_dir=run_dir,
                run_label=f"{args.run_prefix}_{eval_key}_{variant}",
                embedding_backend=backend,
                chroma_dir=chroma_dir,
            )
        _append_compare_rows(rows=compare_rows, eval_set_key=eval_key, metrics_by_variant=metrics_by_variant)

        baseline = metrics_by_variant["baseline_v2"]
        v2 = metrics_by_variant["true_table_ocr_v2"]
        if eval_key == args.table_eval_set_key:
            delta_table15 = _delta(v2["overall_manual_mean"], baseline["overall_manual_mean"])
            subset_gate_checks["table_15_overall_gate"] = int(
                delta_table15 is not None and delta_table15 >= args.subset_gate_table15_delta_min
            )
            subset_gate_passed = subset_gate_passed and subset_gate_checks["table_15_overall_gate"] == 1
        if eval_key == args.groupbc_eval_set_key:
            delta_groupbc = _delta(v2["overall_manual_mean"], baseline["overall_manual_mean"])
            subset_gate_checks["group_bc_overall_gate"] = int(
                delta_groupbc is not None and delta_groupbc >= args.subset_gate_groupbc_delta_min
            )
            delta_type4 = _delta(v2["type4_rejection_success"], baseline["type4_rejection_success"])
            subset_gate_checks["group_bc_type4_gate"] = int(
                delta_type4 is None or delta_type4 >= args.subset_gate_type4_delta_min
            )
            subset_gate_passed = (
                subset_gate_passed
                and subset_gate_checks["group_bc_overall_gate"] == 1
                and subset_gate_checks["group_bc_type4_gate"] == 1
            )

    if subset_gate_passed:
        eval_key = args.full_eval_set_key
        _, qset = bundle.resolve_eval_set(eval_key)
        qids = bundle.resolve_question_id_file(eval_key)
        metrics_by_variant: dict[str, dict[str, Any]] = {}
        for variant, backend, chroma_dir in variants:
            run_dir = run_root / f"{args.run_prefix}_{eval_key}_{variant}"
            metrics_by_variant[variant] = _run_eval(
                args=args,
                project_root=project_root,
                question_set_path=qset,
                question_id_file=qids,
                run_dir=run_dir,
                run_label=f"{args.run_prefix}_{eval_key}_{variant}",
                embedding_backend=backend,
                chroma_dir=chroma_dir,
            )
        _append_compare_rows(rows=compare_rows, eval_set_key=eval_key, metrics_by_variant=metrics_by_variant)

    compare_csv = run_root / "true_table_ocr_v2_compare.csv"
    write_csv(compare_csv, compare_rows)

    summary_csv = run_root / "true_table_ocr_v2_compare_summary.csv"
    write_csv(summary_csv, _summarize_full45(compare_rows))

    report = {
        "run_prefix": args.run_prefix,
        "baseline_profile": "phase2_baseline_v2 + table_body_pairing=true",
        "compare_variants": ["baseline_v2", "true_table_ocr_v1", "true_table_ocr_v2"],
        "subset_gate": {
            "passed": subset_gate_passed,
            "checks": subset_gate_checks,
            "table15_floor": args.subset_gate_table15_delta_min,
            "groupbc_floor": args.subset_gate_groupbc_delta_min,
            "type4_floor": args.subset_gate_type4_delta_min,
        },
        "compare_csv": str(compare_csv),
        "summary_csv": str(summary_csv),
    }
    report_path = run_root / "true_table_ocr_v2_compare_report.json"
    write_json(report_path, report)

    print(f"[done] compare_csv={compare_csv}")
    print(f"[done] summary_csv={summary_csv}")
    print(f"[done] report={report_path}")
    print(f"[done] subset_gate_passed={subset_gate_passed}")


if __name__ == "__main__":
    main()

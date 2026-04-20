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
        description="Single upgrade: true HWP table OCR augment on phase2_baseline_v2 (answer-layer parity fixed)."
    )
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument("--experiment-config", default=str(root / "config" / "phase2_experiments.yaml"))
    parser.add_argument("--output-root", default=str(root / "rag_outputs" / "phase2_runs"))
    parser.add_argument("--run-prefix", default="p2_true_table_ocr_single_upgrade_v1")
    parser.add_argument("--table-eval-set-key", default="table_15")
    parser.add_argument("--groupbc-eval-set-key", default="group_bc")
    parser.add_argument("--full-eval-set-key", default="full_45")
    parser.add_argument("--embedding-backend-baseline", default="openai_text_embedding_3_small")
    parser.add_argument("--embedding-backend-ocr", default="openai_text_embedding_3_small_true_table_ocr")
    parser.add_argument("--model-key", default="gpt5mini_api")
    parser.add_argument("--routing-model", default="gpt-5-mini")
    parser.add_argument("--judge-model", default="gpt-5")
    parser.add_argument("--chroma-dir-baseline", default="")
    parser.add_argument("--chroma-dir-ocr", default=str(root.parent / "rfp_rag_chroma_db_phase2_true_table_ocr"))
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--crag-top-n", type=int, default=5)
    parser.add_argument("--vector-weight", type=float, default=0.7)
    parser.add_argument("--bm25-weight", type=float, default=0.3)
    parser.add_argument("--subset-gate-overall-delta-min", type=float, default=-0.05)
    parser.add_argument("--subset-gate-type4-delta-min", type=float, default=-0.05)
    parser.add_argument("--build-assets", action="store_true")
    parser.add_argument("--reset-ocr-collection", action="store_true")
    parser.add_argument("--include-section-header-support", action="store_true")
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


def _delta(candidate: float | None, baseline: float | None, digits: int = 4) -> float | None:
    if candidate is None or baseline is None:
        return None
    return round(candidate - baseline, digits)


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
            "single_upgrade_change": "true_hwp_table_ocr_augment_only",
            "question_set_path": str(question_set_path),
            "question_id_file": str(question_id_file) if question_id_file else "",
            "answer_layer_policy": "b06_exact",
        },
    )
    return _collect_metrics(run_dir)


def _build_assets_if_requested(args: argparse.Namespace, project_root: Path) -> None:
    if not args.build_assets:
        return
    command = [
        sys.executable,
        str(project_root / "src" / "scenario_b_phase2" / "true_hwp_table_ocr_augment.py"),
        "--project-root",
        str(project_root),
    ]
    if args.reset_ocr_collection:
        command.append("--reset-collection")
    if args.include_section_header_support:
        command.append("--include-section-header-support")
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        raise RuntimeError(
            "true_hwp_table_ocr_augment build 실패\n"
            f"command: {' '.join(command)}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    run_root = Path(args.output_root).resolve() / args.run_prefix
    run_root.mkdir(parents=True, exist_ok=True)
    _build_assets_if_requested(args, project_root)

    bundle = load_phase2_experiment_bundle(Path(args.experiment_config).resolve(), project_root)
    eval_order = [args.table_eval_set_key, args.groupbc_eval_set_key]
    baseline_chroma = Path(args.chroma_dir_baseline).resolve() if args.chroma_dir_baseline else None
    ocr_chroma = Path(args.chroma_dir_ocr).resolve() if args.chroma_dir_ocr else None
    rows: list[dict[str, Any]] = []
    subset_gate_ok = True
    subset_checks: dict[str, int] = {}

    for eval_key in eval_order:
        _, qset = bundle.resolve_eval_set(eval_key)
        qids = bundle.resolve_question_id_file(eval_key)
        base_dir = run_root / f"{args.run_prefix}_{eval_key}_baseline_v2"
        ocr_dir = run_root / f"{args.run_prefix}_{eval_key}_true_table_ocr"
        base_metrics = _run_eval(
            args=args,
            project_root=project_root,
            question_set_path=qset,
            question_id_file=qids,
            run_dir=base_dir,
            run_label=f"{args.run_prefix}_{eval_key}_baseline_v2",
            embedding_backend=args.embedding_backend_baseline,
            chroma_dir=baseline_chroma,
        )
        ocr_metrics = _run_eval(
            args=args,
            project_root=project_root,
            question_set_path=qset,
            question_id_file=qids,
            run_dir=ocr_dir,
            run_label=f"{args.run_prefix}_{eval_key}_true_table_ocr",
            embedding_backend=args.embedding_backend_ocr,
            chroma_dir=ocr_chroma,
        )
        delta_overall = _delta(ocr_metrics["overall_manual_mean"], base_metrics["overall_manual_mean"])
        delta_type4 = _delta(ocr_metrics["type4_rejection_success"], base_metrics["type4_rejection_success"])
        rows.append(
            {
                "eval_set_key": eval_key,
                "run_variant": "baseline_v2",
                "overall_manual_mean": base_metrics["overall_manual_mean"],
                "type2_manual_mean": base_metrics["type2_manual_mean"],
                "type4_rejection_success": base_metrics["type4_rejection_success"],
                "table_plus_body_coverage": base_metrics["table_plus_body_coverage"],
                "latency_sec": base_metrics["latency_sec"],
                "degradation_count": 0,
                "delta_overall_vs_baseline_v2": 0.0,
            }
        )
        rows.append(
            {
                "eval_set_key": eval_key,
                "run_variant": "true_table_ocr",
                "overall_manual_mean": ocr_metrics["overall_manual_mean"],
                "type2_manual_mean": ocr_metrics["type2_manual_mean"],
                "type4_rejection_success": ocr_metrics["type4_rejection_success"],
                "table_plus_body_coverage": ocr_metrics["table_plus_body_coverage"],
                "latency_sec": ocr_metrics["latency_sec"],
                "degradation_count": _degradation_count(base_metrics["manual_completed_rows"], ocr_metrics["manual_completed_rows"]),
                "delta_overall_vs_baseline_v2": delta_overall,
            }
        )
        check_key = f"{eval_key}_overall_gate"
        subset_checks[check_key] = int(delta_overall is not None and delta_overall >= args.subset_gate_overall_delta_min)
        subset_gate_ok = subset_gate_ok and subset_checks[check_key] == 1
        if eval_key == args.groupbc_eval_set_key:
            subset_checks["group_bc_type4_gate"] = int(delta_type4 is None or delta_type4 >= args.subset_gate_type4_delta_min)
            subset_gate_ok = subset_gate_ok and subset_checks["group_bc_type4_gate"] == 1

    if subset_gate_ok:
        eval_key = args.full_eval_set_key
        _, qset = bundle.resolve_eval_set(eval_key)
        qids = bundle.resolve_question_id_file(eval_key)
        base_dir = run_root / f"{args.run_prefix}_{eval_key}_baseline_v2"
        ocr_dir = run_root / f"{args.run_prefix}_{eval_key}_true_table_ocr"
        base_metrics = _run_eval(
            args=args,
            project_root=project_root,
            question_set_path=qset,
            question_id_file=qids,
            run_dir=base_dir,
            run_label=f"{args.run_prefix}_{eval_key}_baseline_v2",
            embedding_backend=args.embedding_backend_baseline,
            chroma_dir=baseline_chroma,
        )
        ocr_metrics = _run_eval(
            args=args,
            project_root=project_root,
            question_set_path=qset,
            question_id_file=qids,
            run_dir=ocr_dir,
            run_label=f"{args.run_prefix}_{eval_key}_true_table_ocr",
            embedding_backend=args.embedding_backend_ocr,
            chroma_dir=ocr_chroma,
        )
        rows.append(
            {
                "eval_set_key": eval_key,
                "run_variant": "baseline_v2",
                "overall_manual_mean": base_metrics["overall_manual_mean"],
                "type2_manual_mean": base_metrics["type2_manual_mean"],
                "type4_rejection_success": base_metrics["type4_rejection_success"],
                "table_plus_body_coverage": base_metrics["table_plus_body_coverage"],
                "latency_sec": base_metrics["latency_sec"],
                "degradation_count": 0,
                "delta_overall_vs_baseline_v2": 0.0,
            }
        )
        rows.append(
            {
                "eval_set_key": eval_key,
                "run_variant": "true_table_ocr",
                "overall_manual_mean": ocr_metrics["overall_manual_mean"],
                "type2_manual_mean": ocr_metrics["type2_manual_mean"],
                "type4_rejection_success": ocr_metrics["type4_rejection_success"],
                "table_plus_body_coverage": ocr_metrics["table_plus_body_coverage"],
                "latency_sec": ocr_metrics["latency_sec"],
                "degradation_count": _degradation_count(base_metrics["manual_completed_rows"], ocr_metrics["manual_completed_rows"]),
                "delta_overall_vs_baseline_v2": _delta(ocr_metrics["overall_manual_mean"], base_metrics["overall_manual_mean"]),
            }
        )

    compare_csv = run_root / "true_table_ocr_single_upgrade_compare.csv"
    write_csv(compare_csv, rows)
    report_path = run_root / "true_table_ocr_single_upgrade_report.json"
    write_json(
        report_path,
        {
            "run_prefix": args.run_prefix,
            "baseline_profile": "phase2_baseline_v2 + table_body_pairing=true",
            "single_upgrade": "true_hwp_table_ocr_augment_only",
            "subset_gate": {
                "passed": subset_gate_ok,
                "checks": subset_checks,
                "overall_floor": args.subset_gate_overall_delta_min,
                "type4_floor": args.subset_gate_type4_delta_min,
            },
            "compare_csv": str(compare_csv),
        },
    )
    print(f"[done] compare_csv={compare_csv}")
    print(f"[done] report={report_path}")
    print(f"[gate] passed={subset_gate_ok}")


if __name__ == "__main__":
    main()

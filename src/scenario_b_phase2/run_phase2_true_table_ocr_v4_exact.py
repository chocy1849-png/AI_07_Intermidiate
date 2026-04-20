from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import csv
import json
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
        description="OCR v4 exact extraction compare: baseline_v2 vs true_table_ocr_v3 vs true_table_ocr_v4."
    )
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument("--experiment-config", default=str(root / "config" / "phase2_experiments.yaml"))
    parser.add_argument("--output-root", default=str(root / "rag_outputs" / "phase2_runs"))
    parser.add_argument("--run-prefix", default="p2_true_table_ocr_v4_exact_v1")
    parser.add_argument("--table-eval-set-key", default="table_15")
    parser.add_argument("--groupbc-eval-set-key", default="group_bc")
    parser.add_argument("--full-eval-set-key", default="full_45")
    parser.add_argument("--embedding-backend-baseline", default="openai_text_embedding_3_small")
    parser.add_argument("--embedding-backend-v3", default="openai_text_embedding_3_small_true_table_ocr_v3")
    parser.add_argument("--embedding-backend-v4", default="openai_text_embedding_3_small_true_table_ocr_v4")
    parser.add_argument("--model-key", default="gpt5mini_api")
    parser.add_argument("--routing-model", default="gpt-5-mini")
    parser.add_argument("--judge-model", default="gpt-5")
    parser.add_argument("--chroma-dir-baseline", default="")
    parser.add_argument("--chroma-dir-v3", default=str(root.parent / "rfp_rag_chroma_db_phase2_true_table_ocr_v3"))
    parser.add_argument("--chroma-dir-v4", default=str(root.parent / "rfp_rag_chroma_db_phase2_true_table_ocr_v4"))
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--crag-top-n", type=int, default=5)
    parser.add_argument("--vector-weight", type=float, default=0.7)
    parser.add_argument("--bm25-weight", type=float, default=0.3)
    parser.add_argument("--table15-floor", type=float, default=3.7167)
    parser.add_argument("--build-v4-assets", action="store_true")
    parser.add_argument("--reset-v4-collection", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--reuse-v4-assets", action="store_true")
    parser.add_argument("--v4-min-page-score", type=float, default=0.05)
    parser.add_argument("--v4-top-n-pages-per-table", type=int, default=1)
    parser.add_argument("--v4-min-region-match-score", type=float, default=0.12)
    parser.add_argument("--v4-min-ocr-region-score", type=float, default=0.45)
    parser.add_argument("--v4-page-neighbor-window", type=int, default=0)
    parser.add_argument("--v4-pdf-zoom", type=float, default=1.6)
    parser.add_argument("--v4-render-timeout-sec", type=int, default=180)
    parser.add_argument("--v4-groupc-pair-bonus", type=float, default=0.008)
    parser.add_argument("--v4-groupc-parent-bonus", type=float, default=0.004)
    parser.add_argument("--v4-groupc-table-penalty", type=float, default=0.015)
    parser.add_argument("--v4-table-factual-generic-penalty", type=float, default=0.012)
    return parser.parse_args()


def _to_float(value: Any) -> float | None:
    return safe_float(value)


def _manual_mean(row: dict[str, Any]) -> float | None:
    return average(
        [
            _to_float(row.get("faithfulness_score")),
            _to_float(row.get("completeness_score")),
            _to_float(row.get("groundedness_score")),
            _to_float(row.get("relevancy_score")),
        ]
    )


def _pick_group_value(rows: list[dict[str, Any]], group_name: str, keys: list[str]) -> float | None:
    target = next((row for row in rows if str(row.get("group_name", "")) == group_name), None)
    if target is None:
        return None
    for key in keys:
        value = _to_float(target.get(key))
        if value is not None:
            return value
    return None


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


def _baseline_v2_options() -> Phase2Options:
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


def _v3_options(args: argparse.Namespace) -> Phase2Options:
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
        groupc_pair_bonus=float(args.v4_groupc_pair_bonus),
        groupc_parent_bonus=float(args.v4_groupc_parent_bonus),
        groupc_table_penalty_without_body=float(args.v4_groupc_table_penalty),
        enable_question_type_gated_ocr_routing=True,
        enable_structured_evidence_priority=True,
        enable_hybridqa_stage_metrics=True,
    )


def _v4_options(args: argparse.Namespace) -> Phase2Options:
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
        groupc_pair_bonus=float(args.v4_groupc_pair_bonus),
        groupc_parent_bonus=float(args.v4_groupc_parent_bonus),
        groupc_table_penalty_without_body=float(args.v4_groupc_table_penalty),
        enable_question_type_gated_ocr_routing=True,
        enable_structured_evidence_priority=True,
        enable_hybridqa_stage_metrics=True,
        enable_table_factual_exact_answer_mode=True,
        enable_table_factual_alignment_scoring=True,
        table_factual_generic_penalty=float(args.v4_table_factual_generic_penalty),
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
        "table_hit": _pick_group_value(coverage_summary_rows, "overall", ["table_hit"]),
        "body_hit": _pick_group_value(coverage_summary_rows, "overall", ["body_hit"]),
        "pair_hit": _pick_group_value(coverage_summary_rows, "overall", ["pair_hit"]),
        "answer_hit": _pick_group_value(coverage_summary_rows, "overall", ["answer_hit"]),
        "structured_evidence_hit": _pick_group_value(coverage_summary_rows, "overall", ["structured_evidence_hit"]),
        "exact_header_match_hit": _pick_group_value(coverage_summary_rows, "overall", ["exact_header_match_hit"]),
        "exact_row_match_hit": _pick_group_value(coverage_summary_rows, "overall", ["exact_row_match_hit"]),
        "generic_row_pollution_count": _pick_group_value(coverage_summary_rows, "overall", ["generic_row_pollution_count"]),
        "latency_sec": _pick_group_value(auto_summary_rows, "overall", ["avg_elapsed_sec", "latency_sec"]),
    }


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
    options: Phase2Options,
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
        options=options,
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
            "phase": "true_table_ocr_v4_exact_extraction",
            "question_set_path": str(question_set_path),
            "question_id_file": str(question_id_file) if question_id_file else "",
        },
    )
    return _collect_metrics(run_dir)


def _build_v4_assets_if_requested(args: argparse.Namespace, project_root: Path) -> None:
    if not args.build_v4_assets:
        return
    command = [
        sys.executable,
        str(project_root / "src" / "scenario_b_phase2" / "true_hwp_table_ocr_v2_augment.py"),
        "--project-root",
        str(project_root),
        "--output-root",
        str(project_root / "rag_outputs" / "phase2_true_table_ocr_v4_assets"),
        "--chroma-output-dir",
        str(project_root.parent / "rfp_rag_chroma_db_phase2_true_table_ocr_v4"),
        "--embedding-backend-key",
        str(args.embedding_backend_v4),
        "--collection-name",
        "rfp_contextual_chunks_v2_true_table_ocr_v4",
        "--bm25-index-name",
        "bm25_index_phase2_true_table_ocr_v4.pkl",
        "--min-page-score",
        str(args.v4_min_page_score),
        "--top-n-pages-per-table",
        str(args.v4_top_n_pages_per_table),
        "--min-region-match-score",
        str(args.v4_min_region_match_score),
        "--min-ocr-region-score",
        str(args.v4_min_ocr_region_score),
        "--page-neighbor-window",
        str(args.v4_page_neighbor_window),
        "--pdf-zoom",
        str(args.v4_pdf_zoom),
        "--render-timeout-sec",
        str(args.v4_render_timeout_sec),
        "--include-structured-evidence-chunks",
    ]
    if args.reset_v4_collection:
        command.append("--reset-collection")
    if args.reuse_v4_assets:
        command.append("--reuse-existing-assets")
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        raise RuntimeError(
            "v4 asset build failed\n"
            f"command: {' '.join(command)}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


def _write_group_bc_id_files(groupbc_csv: Path, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    group_b_ids: list[str] = []
    group_c_ids: list[str] = []
    with groupbc_csv.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            qid = str(row.get("question_id", "")).strip()
            label = str(row.get("group_label", "")).strip().lower()
            if not qid:
                continue
            if label == "group b":
                group_b_ids.append(qid)
            elif label == "group c":
                group_c_ids.append(qid)

    group_b_path = out_dir / "group_b_question_ids.txt"
    group_c_path = out_dir / "group_c_question_ids.txt"
    group_b_path.write_text("\n".join(group_b_ids) + "\n", encoding="utf-8")
    group_c_path.write_text("\n".join(group_c_ids) + "\n", encoding="utf-8")
    return group_b_path, group_c_path


def _append_rows(
    *,
    rows: list[dict[str, Any]],
    eval_set_key: str,
    metrics_by_variant: dict[str, dict[str, Any]],
) -> None:
    baseline = metrics_by_variant["baseline_v2"]
    v3 = metrics_by_variant["true_table_ocr_v3"]
    for variant, metrics in metrics_by_variant.items():
        row = {
            "eval_set_key": eval_set_key,
            "run_variant": variant,
            "overall_manual_mean": metrics["overall_manual_mean"],
            "type2_manual_mean": metrics["type2_manual_mean"],
            "type4_rejection_success": metrics["type4_rejection_success"],
            "table_plus_body_coverage": metrics["table_plus_body_coverage"],
            "table_hit": metrics["table_hit"],
            "body_hit": metrics["body_hit"],
            "pair_hit": metrics["pair_hit"],
            "answer_hit": metrics["answer_hit"],
            "structured_evidence_hit": metrics["structured_evidence_hit"],
            "exact_header_match_hit": metrics["exact_header_match_hit"],
            "exact_row_match_hit": metrics["exact_row_match_hit"],
            "generic_row_pollution_count": metrics["generic_row_pollution_count"],
            "latency_sec": metrics["latency_sec"],
            "degradation_count": 0,
            "delta_overall_vs_baseline_v2": _delta(metrics["overall_manual_mean"], baseline["overall_manual_mean"]),
            "delta_overall_vs_v3": _delta(metrics["overall_manual_mean"], v3["overall_manual_mean"]),
            "delta_type4_vs_baseline_v2": _delta(metrics["type4_rejection_success"], baseline["type4_rejection_success"]),
            "delta_pair_hit_vs_v3": _delta(metrics["pair_hit"], v3["pair_hit"]),
        }
        if variant != "baseline_v2":
            row["degradation_count"] = _degradation_count(
                baseline_rows=baseline["manual_completed_rows"],
                candidate_rows=metrics["manual_completed_rows"],
            )
        rows.append(row)


def _metric(compare_rows: list[dict[str, Any]], eval_set_key: str, variant: str, key: str) -> float | None:
    row = next((item for item in compare_rows if item["eval_set_key"] == eval_set_key and item["run_variant"] == variant), None)
    return _to_float(row.get(key)) if row else None


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    run_root = Path(args.output_root).resolve() / args.run_prefix
    run_root.mkdir(parents=True, exist_ok=True)

    _build_v4_assets_if_requested(args, project_root)

    bundle = load_phase2_experiment_bundle(Path(args.experiment_config).resolve(), project_root)
    _, groupbc_set_path = bundle.resolve_eval_set(args.groupbc_eval_set_key)
    group_b_ids, group_c_ids = _write_group_bc_id_files(groupbc_set_path, run_root)
    _, table15_set_path = bundle.resolve_eval_set(args.table_eval_set_key)
    _, full45_set_path = bundle.resolve_eval_set(args.full_eval_set_key)

    baseline_chroma = Path(args.chroma_dir_baseline).resolve() if args.chroma_dir_baseline else None
    v3_chroma = Path(args.chroma_dir_v3).resolve() if args.chroma_dir_v3 else None
    v4_chroma = Path(args.chroma_dir_v4).resolve() if args.chroma_dir_v4 else None

    variants = [
        ("baseline_v2", args.embedding_backend_baseline, baseline_chroma, _baseline_v2_options()),
        ("true_table_ocr_v3", args.embedding_backend_v3, v3_chroma, _v3_options(args)),
        ("true_table_ocr_v4", args.embedding_backend_v4, v4_chroma, _v4_options(args)),
    ]

    compare_rows: list[dict[str, Any]] = []

    # 1) table_15
    metrics_by_variant: dict[str, dict[str, Any]] = {}
    for variant, backend, chroma_dir, options in variants:
        run_dir = run_root / f"{args.run_prefix}_table_15_{variant}"
        metrics_by_variant[variant] = _run_eval(
            args=args,
            project_root=project_root,
            question_set_path=table15_set_path,
            question_id_file=None,
            run_dir=run_dir,
            run_label=f"{args.run_prefix}_table_15_{variant}",
            embedding_backend=backend,
            chroma_dir=chroma_dir,
            options=options,
        )
    _append_rows(rows=compare_rows, eval_set_key="table_15", metrics_by_variant=metrics_by_variant)

    # 2) Group B
    metrics_by_variant = {}
    for variant, backend, chroma_dir, options in variants:
        run_dir = run_root / f"{args.run_prefix}_group_b_{variant}"
        metrics_by_variant[variant] = _run_eval(
            args=args,
            project_root=project_root,
            question_set_path=groupbc_set_path,
            question_id_file=group_b_ids,
            run_dir=run_dir,
            run_label=f"{args.run_prefix}_group_b_{variant}",
            embedding_backend=backend,
            chroma_dir=chroma_dir,
            options=options,
        )
    _append_rows(rows=compare_rows, eval_set_key="group_b", metrics_by_variant=metrics_by_variant)

    # 3) Group C
    metrics_by_variant = {}
    for variant, backend, chroma_dir, options in variants:
        run_dir = run_root / f"{args.run_prefix}_group_c_{variant}"
        metrics_by_variant[variant] = _run_eval(
            args=args,
            project_root=project_root,
            question_set_path=groupbc_set_path,
            question_id_file=group_c_ids,
            run_dir=run_dir,
            run_label=f"{args.run_prefix}_group_c_{variant}",
            embedding_backend=backend,
            chroma_dir=chroma_dir,
            options=options,
        )
    _append_rows(rows=compare_rows, eval_set_key="group_c", metrics_by_variant=metrics_by_variant)

    gate_table15 = (_metric(compare_rows, "table_15", "true_table_ocr_v4", "overall_manual_mean") or -999.0) >= float(args.table15_floor)
    gate_group_b = (_metric(compare_rows, "group_b", "true_table_ocr_v4", "overall_manual_mean") or -999.0) >= (
        _metric(compare_rows, "group_b", "true_table_ocr_v3", "overall_manual_mean") or 999.0
    )
    gate_group_c = (_metric(compare_rows, "group_c", "true_table_ocr_v4", "overall_manual_mean") or -999.0) >= (
        _metric(compare_rows, "group_c", "true_table_ocr_v3", "overall_manual_mean") or 999.0
    )
    gate_pass = bool(gate_table15 and gate_group_b and gate_group_c)

    full45_pass = None
    if gate_pass:
        metrics_by_variant = {}
        for variant, backend, chroma_dir, options in variants:
            run_dir = run_root / f"{args.run_prefix}_full_45_{variant}"
            metrics_by_variant[variant] = _run_eval(
                args=args,
                project_root=project_root,
                question_set_path=full45_set_path,
                question_id_file=None,
                run_dir=run_dir,
                run_label=f"{args.run_prefix}_full_45_{variant}",
                embedding_backend=backend,
                chroma_dir=chroma_dir,
                options=options,
            )
        _append_rows(rows=compare_rows, eval_set_key="full_45", metrics_by_variant=metrics_by_variant)

        v4_full = _metric(compare_rows, "full_45", "true_table_ocr_v4", "overall_manual_mean")
        base_full = _metric(compare_rows, "full_45", "baseline_v2", "overall_manual_mean")
        v4_type4 = _metric(compare_rows, "full_45", "true_table_ocr_v4", "type4_rejection_success")
        base_type4 = _metric(compare_rows, "full_45", "baseline_v2", "type4_rejection_success")
        full45_pass = bool(
            v4_full is not None
            and base_full is not None
            and v4_full >= base_full
            and ((v4_type4 is None or base_type4 is None) or v4_type4 >= base_type4)
        )

    compare_csv = run_root / "true_table_ocr_v4_exact_compare.csv"
    write_csv(compare_csv, compare_rows)
    report = {
        "run_prefix": args.run_prefix,
        "baseline": "phase2_baseline_v2",
        "candidates": ["true_table_ocr_v3", "true_table_ocr_v4"],
        "gate_passed": gate_pass,
        "gate_rule": {
            "table_15_floor": f"v4 >= {float(args.table15_floor):.4f}",
            "group_b_vs_v3": "v4 >= v3",
            "group_c_vs_v3": "v4 >= v3",
        },
        "gate_checks": {
            "table15_pass": int(gate_table15),
            "group_b_pass": int(gate_group_b),
            "group_c_pass": int(gate_group_c),
        },
        "full45_pass": full45_pass,
        "files": {
            "group_b_ids": str(group_b_ids),
            "group_c_ids": str(group_c_ids),
            "compare_csv": str(compare_csv),
        },
    }
    report_path = run_root / "true_table_ocr_v4_exact_report.json"
    write_json(report_path, report)
    print(f"[done] compare_csv={compare_csv}")
    print(f"[done] report={report_path}")
    print(f"[done] gate_passed={gate_pass}")


if __name__ == "__main__":
    main()


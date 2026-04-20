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
        description="Phase2 OCR v3 gated routing compare: baseline_v2 vs true_table_ocr_v2.1 vs true_table_ocr_v3."
    )
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument("--experiment-config", default=str(root / "config" / "phase2_experiments.yaml"))
    parser.add_argument("--output-root", default=str(root / "rag_outputs" / "phase2_runs"))
    parser.add_argument("--run-prefix", default="p2_true_table_ocr_v3_gated_v1")
    parser.add_argument("--table-eval-set-key", default="table_15")
    parser.add_argument("--groupbc-eval-set-key", default="group_bc")
    parser.add_argument("--full-eval-set-key", default="full_45")
    parser.add_argument("--embedding-backend-baseline", default="openai_text_embedding_3_small")
    parser.add_argument("--embedding-backend-v21", default="openai_text_embedding_3_small_true_table_ocr_v21")
    parser.add_argument("--embedding-backend-v3", default="openai_text_embedding_3_small_true_table_ocr_v3")
    parser.add_argument("--model-key", default="gpt5mini_api")
    parser.add_argument("--routing-model", default="gpt-5-mini")
    parser.add_argument("--judge-model", default="gpt-5")
    parser.add_argument("--chroma-dir-baseline", default="")
    parser.add_argument("--chroma-dir-v21", default=str(root.parent / "rfp_rag_chroma_db_phase2_true_table_ocr_v21"))
    parser.add_argument("--chroma-dir-v3", default=str(root.parent / "rfp_rag_chroma_db_phase2_true_table_ocr_v3"))
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--crag-top-n", type=int, default=5)
    parser.add_argument("--vector-weight", type=float, default=0.7)
    parser.add_argument("--bm25-weight", type=float, default=0.3)

    parser.add_argument("--build-v3-assets", action="store_true")
    parser.add_argument("--reset-v3-collection", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--reuse-v3-assets", action="store_true")
    parser.add_argument("--v3-min-page-score", type=float, default=0.05)
    parser.add_argument("--v3-top-n-pages-per-table", type=int, default=1)
    parser.add_argument("--v3-min-region-match-score", type=float, default=0.12)
    parser.add_argument("--v3-min-ocr-region-score", type=float, default=0.45)
    parser.add_argument("--v3-page-neighbor-window", type=int, default=0)
    parser.add_argument("--v3-pdf-zoom", type=float, default=1.6)
    parser.add_argument("--v3-render-timeout-sec", type=int, default=180)
    parser.add_argument("--v3-groupc-pair-bonus", type=float, default=0.008)
    parser.add_argument("--v3-groupc-parent-bonus", type=float, default=0.004)
    parser.add_argument("--v3-groupc-table-penalty", type=float, default=0.015)
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


def _v21_options(args: argparse.Namespace) -> Phase2Options:
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
        groupc_pair_bonus=float(args.v3_groupc_pair_bonus),
        groupc_parent_bonus=float(args.v3_groupc_parent_bonus),
        groupc_table_penalty_without_body=float(args.v3_groupc_table_penalty),
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
        groupc_pair_bonus=float(args.v3_groupc_pair_bonus),
        groupc_parent_bonus=float(args.v3_groupc_parent_bonus),
        groupc_table_penalty_without_body=float(args.v3_groupc_table_penalty),
        enable_question_type_gated_ocr_routing=True,
        enable_structured_evidence_priority=True,
        enable_hybridqa_stage_metrics=True,
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
        "dual_doc_coverage": _pick_group_value(coverage_summary_rows, "overall", ["dual_doc_coverage"]),
        "comparison_evidence_coverage": _pick_group_value(coverage_summary_rows, "overall", ["comparison_evidence_coverage"]),
        "table_plus_body_coverage": _pick_group_value(coverage_summary_rows, "overall", ["table_plus_body_coverage"]),
        "table_hit": _pick_group_value(coverage_summary_rows, "overall", ["table_hit"]),
        "body_hit": _pick_group_value(coverage_summary_rows, "overall", ["body_hit"]),
        "pair_hit": _pick_group_value(coverage_summary_rows, "overall", ["pair_hit"]),
        "answer_hit": _pick_group_value(coverage_summary_rows, "overall", ["answer_hit"]),
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
            "phase": "true_table_ocr_v3_gated_routing",
            "question_set_path": str(question_set_path),
            "question_id_file": str(question_id_file) if question_id_file else "",
        },
    )
    return _collect_metrics(run_dir)


def _build_v3_assets_if_requested(args: argparse.Namespace, project_root: Path) -> None:
    if not args.build_v3_assets:
        return
    command = [
        sys.executable,
        str(project_root / "src" / "scenario_b_phase2" / "true_hwp_table_ocr_v2_augment.py"),
        "--project-root",
        str(project_root),
        "--output-root",
        str(project_root / "rag_outputs" / "phase2_true_table_ocr_v3_assets"),
        "--chroma-output-dir",
        str(project_root.parent / "rfp_rag_chroma_db_phase2_true_table_ocr_v3"),
        "--embedding-backend-key",
        str(args.embedding_backend_v3),
        "--collection-name",
        "rfp_contextual_chunks_v2_true_table_ocr_v3",
        "--bm25-index-name",
        "bm25_index_phase2_true_table_ocr_v3.pkl",
        "--min-page-score",
        str(args.v3_min_page_score),
        "--top-n-pages-per-table",
        str(args.v3_top_n_pages_per_table),
        "--min-region-match-score",
        str(args.v3_min_region_match_score),
        "--min-ocr-region-score",
        str(args.v3_min_ocr_region_score),
        "--page-neighbor-window",
        str(args.v3_page_neighbor_window),
        "--pdf-zoom",
        str(args.v3_pdf_zoom),
        "--render-timeout-sec",
        str(args.v3_render_timeout_sec),
        "--include-structured-evidence-chunks",
    ]
    if args.reset_v3_collection:
        command.append("--reset-collection")
    if args.reuse_v3_assets:
        command.append("--reuse-existing-assets")
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        raise RuntimeError(
            "v3 asset build failed\n"
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
    v21 = metrics_by_variant["true_table_ocr_v2_1"]
    for variant, metrics in metrics_by_variant.items():
        row = {
            "eval_set_key": eval_set_key,
            "run_variant": variant,
            "overall_manual_mean": metrics["overall_manual_mean"],
            "type2_manual_mean": metrics["type2_manual_mean"],
            "type4_rejection_success": metrics["type4_rejection_success"],
            "dual_doc_coverage": metrics["dual_doc_coverage"],
            "comparison_evidence_coverage": metrics["comparison_evidence_coverage"],
            "table_plus_body_coverage": metrics["table_plus_body_coverage"],
            "table_hit": metrics["table_hit"],
            "body_hit": metrics["body_hit"],
            "pair_hit": metrics["pair_hit"],
            "answer_hit": metrics["answer_hit"],
            "latency_sec": metrics["latency_sec"],
            "degradation_count": 0,
            "delta_overall_vs_baseline_v2": _delta(metrics["overall_manual_mean"], baseline["overall_manual_mean"]),
            "delta_overall_vs_v2_1": _delta(metrics["overall_manual_mean"], v21["overall_manual_mean"]),
            "delta_type4_vs_baseline_v2": _delta(metrics["type4_rejection_success"], baseline["type4_rejection_success"]),
            "delta_coverage_vs_baseline_v2": _delta(metrics["table_plus_body_coverage"], baseline["table_plus_body_coverage"]),
            "delta_pair_hit_vs_baseline_v2": _delta(metrics["pair_hit"], baseline["pair_hit"]),
        }
        if variant != "baseline_v2":
            row["degradation_count"] = _degradation_count(
                baseline_rows=baseline["manual_completed_rows"],
                candidate_rows=metrics["manual_completed_rows"],
            )
        rows.append(row)


def _load_v3_render_status(project_root: Path) -> dict[str, Any]:
    summary_path = project_root / "rag_outputs" / "phase2_true_table_ocr_v3_assets" / "phase2_true_table_ocr_v2_summary.json"
    if not summary_path.exists():
        return {"exists": 0, "summary_path": str(summary_path)}
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    build_summary = payload.get("build_summary", {}) if isinstance(payload, dict) else {}
    return {
        "exists": 1,
        "summary_path": str(summary_path),
        "com_pdf_bbox_success_rate": _to_float(build_summary.get("com_pdf_bbox_success_rate")),
        "render_stability_gate_passed": int(build_summary.get("render_stability_gate_passed", 0) or 0),
        "render_mode_counts": build_summary.get("render_mode_counts", {}),
        "render_failure_reason_counts": build_summary.get("render_failure_reason_counts", {}),
        "structured_chunk_count": int(build_summary.get("structured_chunk_count", 0) or 0),
        "structured_evidence_files": build_summary.get("structured_evidence_files", {}),
    }


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    run_root = Path(args.output_root).resolve() / args.run_prefix
    run_root.mkdir(parents=True, exist_ok=True)

    _build_v3_assets_if_requested(args, project_root)
    render_status = _load_v3_render_status(project_root)

    bundle = load_phase2_experiment_bundle(Path(args.experiment_config).resolve(), project_root)
    _, groupbc_set_path = bundle.resolve_eval_set(args.groupbc_eval_set_key)
    group_b_ids, group_c_ids = _write_group_bc_id_files(groupbc_set_path, run_root)
    _, table15_set_path = bundle.resolve_eval_set(args.table_eval_set_key)
    _, full45_set_path = bundle.resolve_eval_set(args.full_eval_set_key)

    baseline_chroma = Path(args.chroma_dir_baseline).resolve() if args.chroma_dir_baseline else None
    v21_chroma = Path(args.chroma_dir_v21).resolve() if args.chroma_dir_v21 else None
    v3_chroma = Path(args.chroma_dir_v3).resolve() if args.chroma_dir_v3 else None

    variants = [
        ("baseline_v2", args.embedding_backend_baseline, baseline_chroma, _baseline_v2_options()),
        ("true_table_ocr_v2_1", args.embedding_backend_v21, v21_chroma, _v21_options(args)),
        ("true_table_ocr_v3", args.embedding_backend_v3, v3_chroma, _v3_options(args)),
    ]

    compare_rows: list[dict[str, Any]] = []

    # 1) Group B
    metrics_by_variant: dict[str, dict[str, Any]] = {}
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

    # 2) Group C
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

    # 3) table_15
    metrics_by_variant = {}
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

    def _metric(eval_key: str, variant: str, key: str) -> float | None:
        row = next((item for item in compare_rows if item["eval_set_key"] == eval_key and item["run_variant"] == variant), None)
        return _to_float(row.get(key)) if row else None

    gate_group_b = (_metric("group_b", "true_table_ocr_v3", "overall_manual_mean") or -999.0) >= (
        _metric("group_b", "true_table_ocr_v2_1", "overall_manual_mean") or 999.0
    )
    gate_group_c = (_metric("group_c", "true_table_ocr_v3", "overall_manual_mean") or -999.0) >= (
        _metric("group_c", "baseline_v2", "overall_manual_mean") or 999.0
    )
    gate_table15 = (_metric("table_15", "true_table_ocr_v3", "overall_manual_mean") or -999.0) >= 3.70
    gate_pass = bool(gate_group_b and gate_group_c and gate_table15)

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

    compare_csv = run_root / "true_table_ocr_v3_gated_compare.csv"
    write_csv(compare_csv, compare_rows)

    summary_rows: list[dict[str, Any]] = []
    for variant in ["baseline_v2", "true_table_ocr_v2_1", "true_table_ocr_v3"]:
        summary_rows.append(
            {
                "run_variant": variant,
                "group_b_mean": _metric("group_b", variant, "overall_manual_mean"),
                "group_c_mean": _metric("group_c", variant, "overall_manual_mean"),
                "table_15_mean": _metric("table_15", variant, "overall_manual_mean"),
                "full_45_mean": _metric("full_45", variant, "overall_manual_mean"),
                "full_45_type4_rejection_success": _metric("full_45", variant, "type4_rejection_success"),
                "full_45_pair_hit": _metric("full_45", variant, "pair_hit"),
                "full_45_table_plus_body_coverage": _metric("full_45", variant, "table_plus_body_coverage"),
                "full_45_latency_sec": _metric("full_45", variant, "latency_sec"),
            }
        )
    summary_csv = run_root / "true_table_ocr_v3_gated_summary.csv"
    write_csv(summary_csv, summary_rows)

    report = {
        "run_prefix": args.run_prefix,
        "baseline": "phase2_baseline_v2",
        "candidates": ["true_table_ocr_v2_1", "true_table_ocr_v3"],
        "gate_passed": gate_pass,
        "gate_rule": {
            "group_b_v3_vs_v2_1": "v3 >= v2.1",
            "group_c_v3_vs_baseline_v2": "v3 >= baseline_v2",
            "table_15_v3_floor": "v3 >= 3.70",
        },
        "gate_checks": {
            "group_b_pass": int(gate_group_b),
            "group_c_pass": int(gate_group_c),
            "table15_pass": int(gate_table15),
        },
        "render_path_status_v3": render_status,
        "files": {
            "group_b_ids": str(group_b_ids),
            "group_c_ids": str(group_c_ids),
            "compare_csv": str(compare_csv),
            "summary_csv": str(summary_csv),
        },
    }
    report_path = run_root / "true_table_ocr_v3_gated_report.json"
    write_json(report_path, report)

    print(f"[done] compare_csv={compare_csv}")
    print(f"[done] summary_csv={summary_csv}")
    print(f"[done] report={report_path}")
    print(f"[done] gate_passed={gate_pass}")


if __name__ == "__main__":
    main()


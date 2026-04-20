from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import re
from pathlib import Path
from typing import Any

from eval_utils import average, parse_question_rows, read_csv, safe_float, write_csv, write_json
from scenario_a.common_pipeline import PipelinePaths, PipelineSettings
from scenario_b_phase2.phase2_eval import evaluate_phase2, read_eval_rows
from scenario_b_phase2.phase2_pipeline import Phase2Options, ScenarioBPhase2Pipeline


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Confirm baseline_v1 vs baseline_v2 on full_45.")
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument("--output-root", default=str(root / "rag_outputs" / "phase2_runs"))
    parser.add_argument("--run-prefix", default="p2_baseline_v2_confirm")
    parser.add_argument(
        "--question-set-path",
        default=str(root / "docs" / "planning" / "pm" / "day3_partA_eval_questions_v1.txt"),
    )
    parser.add_argument("--embedding-backend", default="openai_text_embedding_3_small")
    parser.add_argument("--model-key", default="gpt5mini_api")
    parser.add_argument("--routing-model", default="gpt-5-mini")
    parser.add_argument("--judge-model", default="gpt-5")
    parser.add_argument("--chroma-dir", default="")
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--crag-top-n", type=int, default=5)
    parser.add_argument("--vector-weight", type=float, default=0.7)
    parser.add_argument("--bm25-weight", type=float, default=0.3)
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


def _normalize_doc_name(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").lower())


def _dual_doc_coverage_proxy(result_rows: list[dict[str, Any]]) -> float | None:
    target_rows = [
        row
        for row in result_rows
        if str(row.get("answer_type", "")).strip() == "comparison" or str(row.get("type_group", "")).strip() == "TYPE 2"
    ]
    if not target_rows:
        return None
    hit = 0
    for row in target_rows:
        docs = [part.strip() for part in str(row.get("source_docs", "")).split("|") if part.strip()]
        unique = {_normalize_doc_name(doc) for doc in docs if doc.strip()}
        if len(unique) >= 2:
            hit += 1
    return round(hit / len(target_rows), 4)


def _build_settings(args: argparse.Namespace) -> PipelineSettings:
    return PipelineSettings(
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


def _build_options(*, comparison_helper: bool) -> Phase2Options:
    return Phase2Options(
        enable_controlled_query_expansion=False,
        enable_normalized_bm25=False,
        enable_metadata_aware_retrieval=True,
        enable_metadata_bonus_v2=None,
        enable_table_body_pairing=False,
        enable_soft_crag_lite=False,
        metadata_boost_scale=0.5,
        metadata_disable_for_rejection=True,
        metadata_scope_mode="all",
        normalized_bm25_mode="all",
        enable_b03_legacy_crag_parity=True,
        b03_evaluator_top_n=6,
        b03_second_pass_vector_weight=0.55,
        b03_second_pass_bm25_weight=0.45,
        enable_comparison_evidence_helper=comparison_helper,
        comparison_helper_doc_bonus=0.0045,
        comparison_helper_axis_bonus=0.0015,
        comparison_helper_max_per_doc=2,
    )


def _run_eval(
    *,
    args: argparse.Namespace,
    project_root: Path,
    question_set_path: Path,
    run_dir: Path,
    run_label: str,
    options: Phase2Options,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_dir.mkdir(parents=True, exist_ok=True)
    question_rows = read_eval_rows(question_set_path, question_id_file=None, shard_count=1, shard_index=0)
    pipeline = ScenarioBPhase2Pipeline(
        PipelinePaths(project_root=project_root, chroma_dir=Path(args.chroma_dir).resolve() if args.chroma_dir else None),
        settings=_build_settings(args),
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
            "question_set_path": str(question_set_path),
            "single_upgrade_change": "baseline_promotion_check",
            "phase2_options": {
                "enable_controlled_query_expansion": options.enable_controlled_query_expansion,
                "enable_normalized_bm25": options.enable_normalized_bm25,
                "enable_metadata_aware_retrieval": options.enable_metadata_aware_retrieval,
                "enable_table_body_pairing": options.enable_table_body_pairing,
                "enable_soft_crag_lite": options.enable_soft_crag_lite,
                "metadata_boost_scale": options.metadata_boost_scale,
                "metadata_disable_for_rejection": options.metadata_disable_for_rejection,
                "metadata_scope_mode": options.metadata_scope_mode,
                "enable_comparison_evidence_helper": options.enable_comparison_evidence_helper,
                "comparison_helper_doc_bonus": options.comparison_helper_doc_bonus,
                "comparison_helper_axis_bonus": options.comparison_helper_axis_bonus,
                "comparison_helper_max_per_doc": options.comparison_helper_max_per_doc,
            },
        },
    )

    manual_summary = read_csv(run_dir / "phase2_eval_manual_summary.csv")
    auto_summary = read_csv(run_dir / "phase2_eval_auto_summary.csv")
    coverage_summary = read_csv(run_dir / "phase2_eval_coverage_summary.csv")
    manual_rows = read_csv(run_dir / "phase2_eval_manual_completed.csv")
    result_rows = read_csv(run_dir / "phase2_eval_results.csv")

    dual_doc = _pick_group_value(coverage_summary, "overall", ["dual_doc_coverage"])
    proxy = _dual_doc_coverage_proxy(result_rows)
    if dual_doc is None:
        dual_doc = proxy

    metrics = {
        "overall_manual_mean": _pick_group_value(manual_summary, "overall", ["avg_manual_eval_score", "manual_mean"]),
        "type2_manual_mean": _pick_group_value(manual_summary, "TYPE 2", ["avg_manual_eval_score", "manual_mean"]),
        "type4_rejection_success": _pick_group_value(auto_summary, "TYPE 4", ["rejection_success_rate"]),
        "dual_doc_coverage": dual_doc,
        "comparison_evidence_coverage": _pick_group_value(coverage_summary, "overall", ["comparison_evidence_coverage"]),
        "avg_answer_chars": _pick_group_value(auto_summary, "overall", ["avg_answer_chars"]),
        "latency_sec": _pick_group_value(auto_summary, "overall", ["avg_elapsed_sec", "latency_sec"]),
        "dual_doc_coverage_proxy": proxy,
    }
    return metrics, manual_rows


def _degradation_count(baseline_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]]]:
    baseline_by_qid = {str(row.get("question_id", "")).strip(): row for row in baseline_rows}
    count = 0
    detail_rows: list[dict[str, Any]] = []
    for row in candidate_rows:
        qid = str(row.get("question_id", "")).strip()
        base = baseline_by_qid.get(qid)
        if base is None:
            continue
        base_mean = _manual_mean(base)
        cand_mean = _manual_mean(row)
        if base_mean is None or cand_mean is None:
            continue
        delta = round(cand_mean - base_mean, 4)
        if delta < 0:
            count += 1
        detail_rows.append(
            {
                "question_id": qid,
                "type_group": row.get("type_group", ""),
                "answer_type": row.get("answer_type", ""),
                "baseline_v1_manual_mean": base_mean,
                "baseline_v2_manual_mean": cand_mean,
                "delta_manual_mean": delta,
            }
        )
    return count, detail_rows


def _delta(candidate: float | None, baseline: float | None, digits: int = 4) -> float | None:
    if candidate is None or baseline is None:
        return None
    return round(candidate - baseline, digits)


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    run_root = Path(args.output_root).resolve() / args.run_prefix
    run_root.mkdir(parents=True, exist_ok=True)
    question_set_path = Path(args.question_set_path).resolve()
    if not question_set_path.exists():
        raise FileNotFoundError(f"Question set not found: {question_set_path}")
    _ = parse_question_rows(question_set_path)

    v1_metrics, v1_manual_rows = _run_eval(
        args=args,
        project_root=project_root,
        question_set_path=question_set_path,
        run_dir=run_root / f"{args.run_prefix}_baseline_v1",
        run_label=f"{args.run_prefix}_baseline_v1",
        options=_build_options(comparison_helper=False),
    )
    v2_metrics, v2_manual_rows = _run_eval(
        args=args,
        project_root=project_root,
        question_set_path=question_set_path,
        run_dir=run_root / f"{args.run_prefix}_baseline_v2",
        run_label=f"{args.run_prefix}_baseline_v2",
        options=_build_options(comparison_helper=True),
    )

    degradation_count, degradation_rows = _degradation_count(v1_manual_rows, v2_manual_rows)
    degradation_file = run_root / "baseline_v2_confirm_degradation_cases.csv"
    write_csv(degradation_file, degradation_rows)

    compare_rows = [
        {
            "run_variant": "baseline_v1",
            "overall_manual_mean": v1_metrics["overall_manual_mean"],
            "type2_manual_mean": v1_metrics["type2_manual_mean"],
            "type4_rejection_success": v1_metrics["type4_rejection_success"],
            "dual_doc_coverage": v1_metrics["dual_doc_coverage"],
            "comparison_evidence_coverage": v1_metrics["comparison_evidence_coverage"],
            "degradation_count": 0,
            "avg_answer_chars": v1_metrics["avg_answer_chars"],
            "latency_sec": v1_metrics["latency_sec"],
            "delta_overall_manual_mean_vs_baseline_v1": 0.0,
            "delta_type2_manual_mean_vs_baseline_v1": 0.0,
            "delta_type4_rejection_success_vs_baseline_v1": 0.0,
            "delta_dual_doc_coverage_vs_baseline_v1": 0.0,
            "delta_comparison_evidence_coverage_vs_baseline_v1": 0.0,
            "delta_avg_answer_chars_vs_baseline_v1": 0.0,
            "delta_latency_sec_vs_baseline_v1": 0.0,
            "dual_doc_coverage_proxy": v1_metrics["dual_doc_coverage_proxy"],
            "degradation_file": "",
        },
        {
            "run_variant": "baseline_v2",
            "overall_manual_mean": v2_metrics["overall_manual_mean"],
            "type2_manual_mean": v2_metrics["type2_manual_mean"],
            "type4_rejection_success": v2_metrics["type4_rejection_success"],
            "dual_doc_coverage": v2_metrics["dual_doc_coverage"],
            "comparison_evidence_coverage": v2_metrics["comparison_evidence_coverage"],
            "degradation_count": degradation_count,
            "avg_answer_chars": v2_metrics["avg_answer_chars"],
            "latency_sec": v2_metrics["latency_sec"],
            "delta_overall_manual_mean_vs_baseline_v1": _delta(v2_metrics["overall_manual_mean"], v1_metrics["overall_manual_mean"]),
            "delta_type2_manual_mean_vs_baseline_v1": _delta(v2_metrics["type2_manual_mean"], v1_metrics["type2_manual_mean"]),
            "delta_type4_rejection_success_vs_baseline_v1": _delta(
                v2_metrics["type4_rejection_success"], v1_metrics["type4_rejection_success"]
            ),
            "delta_dual_doc_coverage_vs_baseline_v1": _delta(v2_metrics["dual_doc_coverage"], v1_metrics["dual_doc_coverage"]),
            "delta_comparison_evidence_coverage_vs_baseline_v1": _delta(
                v2_metrics["comparison_evidence_coverage"], v1_metrics["comparison_evidence_coverage"]
            ),
            "delta_avg_answer_chars_vs_baseline_v1": _delta(v2_metrics["avg_answer_chars"], v1_metrics["avg_answer_chars"], digits=2),
            "delta_latency_sec_vs_baseline_v1": _delta(v2_metrics["latency_sec"], v1_metrics["latency_sec"], digits=2),
            "dual_doc_coverage_proxy": v2_metrics["dual_doc_coverage_proxy"],
            "degradation_file": str(degradation_file),
        },
    ]
    compare_path = run_root / "baseline_v2_confirm_compare.csv"
    write_csv(compare_path, compare_rows)
    report_path = run_root / "baseline_v2_confirm_report.json"
    write_json(
        report_path,
        {
            "project_root": str(project_root),
            "run_prefix": args.run_prefix,
            "baseline_v1": "b06_exact + metadata_t4off_half",
            "baseline_v2": "b06_exact + metadata_t4off_half + comparison_helper_only",
            "question_set_path": str(question_set_path),
            "compare_csv": str(compare_path),
            "degradation_file": str(degradation_file),
        },
    )

    print(f"[done] compare_csv={compare_path}")
    print(f"[done] report={report_path}")


if __name__ == "__main__":
    main()

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

from eval_utils import average, parse_question_rows, read_csv, safe_float, write_csv, write_json
from scenario_a.common_pipeline import PipelinePaths, PipelineSettings
from scenario_b_phase2.phase2_eval import evaluate_phase2, read_eval_rows
from scenario_b_phase2.phase2_pipeline import Phase2Options, ScenarioBPhase2Pipeline


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Phase2 single-upgrade runner: baseline_v1 vs comparison-helper-only."
    )
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument("--output-root", default=str(root / "rag_outputs" / "phase2_runs"))
    parser.add_argument("--run-prefix", default="p1_comparison_helper_only_v1")

    parser.add_argument(
        "--full-question-set-path",
        default=str(root / "docs" / "planning" / "pm" / "day3_partA_eval_questions_v1.txt"),
    )
    parser.add_argument("--type2-question-id-file", default=str(root / "rag_outputs" / "eval_sets" / "day3_type2_question_ids_v1.txt"))

    parser.add_argument("--embedding-backend", default="openai_text_embedding_3_small")
    parser.add_argument("--model-key", default="gpt5mini_api")
    parser.add_argument("--routing-model", default="gpt-5-mini")
    parser.add_argument("--judge-model", default="gpt-5")
    parser.add_argument("--no-judge", action="store_true")

    parser.add_argument("--chroma-dir", default="")
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--crag-top-n", type=int, default=5)
    parser.add_argument("--vector-weight", type=float, default=0.7)
    parser.add_argument("--bm25-weight", type=float, default=0.3)

    parser.add_argument("--comparison-helper-doc-bonus", type=float, default=0.0045)
    parser.add_argument("--comparison-helper-axis-bonus", type=float, default=0.0015)
    parser.add_argument("--comparison-helper-max-per-doc", type=int, default=2)
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


def _delta(candidate: float | None, baseline: float | None, digits: int = 4) -> float | None:
    if candidate is None or baseline is None:
        return None
    return round(candidate - baseline, digits)


def _build_baseline_v1_options(*, enable_comparison_helper: bool, args: argparse.Namespace) -> Phase2Options:
    return Phase2Options(
        enable_controlled_query_expansion=False,
        enable_normalized_bm25=False,
        enable_metadata_aware_retrieval=True,
        enable_metadata_bonus_v2=None,
        enable_table_body_pairing=False,
        enable_soft_crag_lite=False,
        metadata_disable_for_rejection=True,
        metadata_boost_scale=0.5,
        metadata_scope_mode="all",
        normalized_bm25_mode="all",
        enable_b03_legacy_crag_parity=True,
        b03_evaluator_top_n=6,
        b03_second_pass_vector_weight=0.55,
        b03_second_pass_bm25_weight=0.45,
        enable_comparison_evidence_helper=enable_comparison_helper,
        comparison_helper_doc_bonus=args.comparison_helper_doc_bonus,
        comparison_helper_axis_bonus=args.comparison_helper_axis_bonus,
        comparison_helper_max_per_doc=max(1, args.comparison_helper_max_per_doc),
    )


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


def _collect_metrics(run_dir: Path) -> dict[str, Any]:
    manual_summary_rows = read_csv(run_dir / "phase2_eval_manual_summary.csv")
    auto_summary_rows = read_csv(run_dir / "phase2_eval_auto_summary.csv")
    coverage_summary_rows = read_csv(run_dir / "phase2_eval_coverage_summary.csv")
    manual_completed_rows = read_csv(run_dir / "phase2_eval_manual_completed.csv")
    return {
        "manual_summary_rows": manual_summary_rows,
        "auto_summary_rows": auto_summary_rows,
        "coverage_summary_rows": coverage_summary_rows,
        "manual_completed_rows": manual_completed_rows,
        "overall_manual_mean": _pick_group_value(manual_summary_rows, "overall", ["avg_manual_eval_score", "manual_mean"]),
        "type2_manual_mean": _pick_group_value(manual_summary_rows, "TYPE 2", ["avg_manual_eval_score", "manual_mean"]),
        "type4_rejection_success": _pick_group_value(auto_summary_rows, "TYPE 4", ["rejection_success_rate"]),
        "dual_doc_coverage": _pick_group_value(coverage_summary_rows, "overall", ["dual_doc_coverage"]),
        "comparison_evidence_coverage": _pick_group_value(coverage_summary_rows, "overall", ["comparison_evidence_coverage"]),
        "latency_sec": _pick_group_value(auto_summary_rows, "overall", ["avg_elapsed_sec", "latency_sec"]),
    }


def _collect_degradation(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
    baseline_by_qid = {str(row.get("question_id", "")).strip(): row for row in baseline_rows}
    degraded = 0
    details: list[dict[str, Any]] = []
    improved = 0
    by_group: dict[str, dict[str, float]] = {}

    for row in candidate_rows:
        qid = str(row.get("question_id", "")).strip()
        if not qid or qid not in baseline_by_qid:
            continue
        base_mean = _manual_mean(baseline_by_qid[qid])
        cand_mean = _manual_mean(row)
        if base_mean is None or cand_mean is None:
            continue
        delta_value = round(cand_mean - base_mean, 4)
        type_group = str(row.get("type_group", "")).strip() or "UNKNOWN"
        if delta_value < 0:
            degraded += 1
        if delta_value > 0:
            improved += 1

        state = by_group.setdefault(type_group, {"sum_delta": 0.0, "count": 0.0, "improved": 0.0, "degraded": 0.0})
        state["sum_delta"] += delta_value
        state["count"] += 1
        if delta_value > 0:
            state["improved"] += 1
        if delta_value < 0:
            state["degraded"] += 1

        details.append(
            {
                "question_id": qid,
                "type_group": type_group,
                "answer_type": row.get("answer_type", ""),
                "baseline_manual_mean": base_mean,
                "candidate_manual_mean": cand_mean,
                "delta_manual_mean": delta_value,
            }
        )

    improved_groups: list[dict[str, Any]] = []
    degraded_groups: list[dict[str, Any]] = []
    for group_name, state in by_group.items():
        count = int(state["count"])
        avg_delta = round(state["sum_delta"] / count, 4) if count else 0.0
        row = {
            "type_group": group_name,
            "avg_delta": avg_delta,
            "count": count,
            "improved_count": int(state["improved"]),
            "degraded_count": int(state["degraded"]),
        }
        if avg_delta > 0:
            improved_groups.append(row)
        elif avg_delta < 0:
            degraded_groups.append(row)
    improved_groups.sort(key=lambda item: item["avg_delta"], reverse=True)
    degraded_groups.sort(key=lambda item: item["avg_delta"])

    return degraded, details, {
        "improved_question_count": improved,
        "degraded_question_count": degraded,
        "improved_type_groups": improved_groups,
        "degraded_type_groups": degraded_groups,
    }


def _ensure_type2_question_ids(full_question_set_path: Path, output_path: Path) -> Path:
    rows = parse_question_rows(full_question_set_path)
    ids = [str(row.get("question_id", "")).strip() for row in rows if str(row.get("type_group", "")).strip() == "TYPE 2"]
    ids = [item for item in ids if item]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(ids) + "\n", encoding="utf-8")
    return output_path


def _run_eval(
    *,
    args: argparse.Namespace,
    project_root: Path,
    output_dir: Path,
    question_set_path: Path,
    question_id_file: Path | None,
    options: Phase2Options,
    run_label: str,
    judge_model: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = ScenarioBPhase2Pipeline(
        PipelinePaths(
            project_root=project_root,
            chroma_dir=Path(args.chroma_dir).resolve() if args.chroma_dir else None,
        ),
        settings=_build_settings(args),
        options=options,
    )
    adapter = pipeline.create_adapter(args.model_key)
    question_rows = read_eval_rows(
        question_set_path,
        question_id_file=question_id_file,
        shard_count=1,
        shard_index=0,
    )
    result = evaluate_phase2(
        pipeline,
        adapter,
        question_rows,
        output_dir,
        judge_model=judge_model,
        run_label=run_label,
        extra_manifest={
            "question_set_path": str(question_set_path),
            "question_id_file": str(question_id_file) if question_id_file else "",
            "single_upgrade_change": "comparison_helper_only",
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
    return _collect_metrics(output_dir), result["manifest"]


def _format_group_rows(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    return " | ".join(
        f"{row['type_group']}(avg_delta={row['avg_delta']}, improved={row['improved_count']}, degraded={row['degraded_count']})"
        for row in rows
    )


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_root).resolve()
    run_root = output_root / args.run_prefix
    run_root.mkdir(parents=True, exist_ok=True)

    if args.no_judge:
        raise RuntimeError("manual mean 중심 비교가 필수이므로 --no-judge 는 지원하지 않습니다.")
    judge_model = args.judge_model
    full_question_set_path = Path(args.full_question_set_path).resolve()
    if not full_question_set_path.exists():
        raise FileNotFoundError(f"Question set not found: {full_question_set_path}")

    type2_id_file = _ensure_type2_question_ids(full_question_set_path, Path(args.type2_question_id_file).resolve())

    baseline_options = _build_baseline_v1_options(enable_comparison_helper=False, args=args)
    helper_options = _build_baseline_v1_options(enable_comparison_helper=True, args=args)

    run_manifests: list[dict[str, Any]] = []
    compare_rows: list[dict[str, Any]] = []
    analysis_rows: list[dict[str, Any]] = []

    eval_specs = [
        {"eval_key": "full_45", "question_set_path": full_question_set_path, "question_id_file": None},
        {"eval_key": "type2_subset", "question_set_path": full_question_set_path, "question_id_file": type2_id_file},
    ]

    for spec in eval_specs:
        eval_key = spec["eval_key"]
        question_set_path = spec["question_set_path"]
        question_id_file = spec["question_id_file"]

        baseline_dir = run_root / f"{args.run_prefix}_{eval_key}_baseline_v1"
        helper_dir = run_root / f"{args.run_prefix}_{eval_key}_comparison_helper"

        print(f"[run] {eval_key} baseline_v1")
        baseline_metrics, baseline_manifest = _run_eval(
            args=args,
            project_root=project_root,
            output_dir=baseline_dir,
            question_set_path=question_set_path,
            question_id_file=question_id_file,
            options=baseline_options,
            run_label=f"{args.run_prefix}_{eval_key}_baseline_v1",
            judge_model=judge_model,
        )
        print(f"[run] {eval_key} comparison_helper_only")
        helper_metrics, helper_manifest = _run_eval(
            args=args,
            project_root=project_root,
            output_dir=helper_dir,
            question_set_path=question_set_path,
            question_id_file=question_id_file,
            options=helper_options,
            run_label=f"{args.run_prefix}_{eval_key}_comparison_helper",
            judge_model=judge_model,
        )
        run_manifests.extend([baseline_manifest, helper_manifest])

        degradation_count, degradation_rows, degradation_summary = _collect_degradation(
            baseline_metrics["manual_completed_rows"],
            helper_metrics["manual_completed_rows"],
        )
        degradation_file = run_root / f"{args.run_prefix}_{eval_key}_degradation_cases.csv"
        write_csv(degradation_file, degradation_rows)

        compare_rows.append(
            {
                "eval_key": eval_key,
                "run_variant": "baseline_v1",
                "overall_manual_mean": baseline_metrics["overall_manual_mean"],
                "type2_manual_mean": baseline_metrics["type2_manual_mean"],
                "type4_rejection_success": baseline_metrics["type4_rejection_success"],
                "dual_doc_coverage": baseline_metrics["dual_doc_coverage"],
                "comparison_evidence_coverage": baseline_metrics["comparison_evidence_coverage"],
                "degradation_count": 0,
                "delta_overall_manual_mean_vs_baseline_v1": 0.0,
                "delta_type2_manual_mean_vs_baseline_v1": 0.0,
                "delta_type4_rejection_success_vs_baseline_v1": 0.0,
                "delta_dual_doc_coverage_vs_baseline_v1": 0.0,
                "delta_comparison_evidence_coverage_vs_baseline_v1": 0.0,
                "delta_latency_sec_vs_baseline_v1": 0.0,
                "latency_sec": baseline_metrics["latency_sec"],
                "run_dir": str(baseline_dir),
                "degradation_file": "",
            }
        )
        compare_rows.append(
            {
                "eval_key": eval_key,
                "run_variant": "comparison_helper_only",
                "overall_manual_mean": helper_metrics["overall_manual_mean"],
                "type2_manual_mean": helper_metrics["type2_manual_mean"],
                "type4_rejection_success": helper_metrics["type4_rejection_success"],
                "dual_doc_coverage": helper_metrics["dual_doc_coverage"],
                "comparison_evidence_coverage": helper_metrics["comparison_evidence_coverage"],
                "degradation_count": degradation_count,
                "delta_overall_manual_mean_vs_baseline_v1": _delta(
                    helper_metrics["overall_manual_mean"],
                    baseline_metrics["overall_manual_mean"],
                ),
                "delta_type2_manual_mean_vs_baseline_v1": _delta(
                    helper_metrics["type2_manual_mean"],
                    baseline_metrics["type2_manual_mean"],
                ),
                "delta_type4_rejection_success_vs_baseline_v1": _delta(
                    helper_metrics["type4_rejection_success"],
                    baseline_metrics["type4_rejection_success"],
                ),
                "delta_dual_doc_coverage_vs_baseline_v1": _delta(
                    helper_metrics["dual_doc_coverage"],
                    baseline_metrics["dual_doc_coverage"],
                ),
                "delta_comparison_evidence_coverage_vs_baseline_v1": _delta(
                    helper_metrics["comparison_evidence_coverage"],
                    baseline_metrics["comparison_evidence_coverage"],
                ),
                "delta_latency_sec_vs_baseline_v1": _delta(
                    helper_metrics["latency_sec"],
                    baseline_metrics["latency_sec"],
                    digits=2,
                ),
                "latency_sec": helper_metrics["latency_sec"],
                "run_dir": str(helper_dir),
                "degradation_file": str(degradation_file),
            }
        )
        analysis_rows.append(
            {
                "eval_key": eval_key,
                "improved_question_count": degradation_summary["improved_question_count"],
                "degraded_question_count": degradation_summary["degraded_question_count"],
                "improved_groups": _format_group_rows(degradation_summary["improved_type_groups"]),
                "degraded_groups": _format_group_rows(degradation_summary["degraded_type_groups"]),
                "degradation_file": str(degradation_file),
            }
        )

    compare_csv = run_root / "comparison_helper_single_upgrade_compare.csv"
    analysis_csv = run_root / "comparison_helper_single_upgrade_analysis.csv"
    write_csv(compare_csv, compare_rows)
    write_csv(analysis_csv, analysis_rows)

    full_helper = next(
        (
            row
            for row in compare_rows
            if row.get("eval_key") == "full_45" and row.get("run_variant") == "comparison_helper_only"
        ),
        {},
    )
    verdict = "보류"
    overall_delta = _to_float(full_helper.get("delta_overall_manual_mean_vs_baseline_v1"))
    type2_delta = _to_float(full_helper.get("delta_type2_manual_mean_vs_baseline_v1"))
    type4_delta = _to_float(full_helper.get("delta_type4_rejection_success_vs_baseline_v1"))
    if overall_delta is not None and type2_delta is not None and type4_delta is not None:
        if overall_delta >= 0 and type2_delta >= 0 and type4_delta >= -0.05:
            verdict = "채택"
        elif overall_delta >= -0.05 and type2_delta >= -0.05 and type4_delta >= -0.05:
            verdict = "조건부 채택"
        else:
            verdict = "기각"

    report = {
        "project_root": str(project_root),
        "run_prefix": args.run_prefix,
        "baseline_v1": "b06_exact + metadata_t4off_half",
        "single_upgrade": "comparison helper only",
        "type2_question_id_file": str(type2_id_file),
        "compare_csv": str(compare_csv),
        "analysis_csv": str(analysis_csv),
        "run_manifests": run_manifests,
        "verdict": verdict,
    }
    report_path = run_root / "comparison_helper_single_upgrade_report.json"
    write_json(report_path, report)

    print(f"[done] compare_csv={compare_csv}")
    print(f"[done] analysis_csv={analysis_csv}")
    print(f"[done] report={report_path}")
    print(f"[verdict] {verdict}")


if __name__ == "__main__":
    main()

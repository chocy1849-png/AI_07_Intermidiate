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
from scenario_b_phase2.experiment_config import Phase2ExperimentBundle, load_phase2_experiment_bundle
from scenario_b_phase2.phase2_eval import evaluate_phase2, read_eval_rows
from scenario_b_phase2.phase2_pipeline import Phase2Options, ScenarioBPhase2Pipeline


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Phase2 single-upgrade runner: baseline_v1 vs OCR pilot corpus (one change only)."
    )
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument("--experiment-config", default=str(root / "config" / "phase2_experiments.yaml"))
    parser.add_argument("--output-root", default=str(root / "rag_outputs" / "phase2_runs"))
    parser.add_argument("--run-prefix", default="p1_ocr_single_upgrade")

    parser.add_argument("--table-eval-set-key", default="table_15")
    parser.add_argument("--groupbc-eval-set-key", default="group_bc")
    parser.add_argument("--full-eval-set-key", default="full_45")

    parser.add_argument("--embedding-backend-baseline", default="openai_text_embedding_3_small")
    parser.add_argument("--embedding-backend-ocr", default="openai_text_embedding_3_small_pilot_ocr")
    parser.add_argument("--model-key", default="gpt5mini_api")
    parser.add_argument("--routing-model", default="gpt-5-mini")
    parser.add_argument("--judge-model", default="gpt-5")
    parser.add_argument("--no-judge", action="store_true")

    parser.add_argument("--chroma-dir-baseline", default="")
    parser.add_argument("--chroma-dir-ocr", default=str(root.parent / "rfp_rag_chroma_db_phase2_pilot"))

    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--crag-top-n", type=int, default=5)
    parser.add_argument("--vector-weight", type=float, default=0.7)
    parser.add_argument("--bm25-weight", type=float, default=0.3)

    parser.add_argument("--subset-gate-overall-delta-min", type=float, default=-0.05)
    parser.add_argument("--subset-gate-type4-delta-min", type=float, default=-0.05)
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


def _resolve_eval_paths(bundle: Phase2ExperimentBundle, eval_set_key: str) -> tuple[Path, Path | None]:
    _, question_set_path = bundle.resolve_eval_set(eval_set_key)
    question_id_file = bundle.resolve_question_id_file(eval_set_key)
    return question_set_path, question_id_file


def _build_phase2_baseline_v1_options() -> Phase2Options:
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
    )


def _build_profile_alias_manifest() -> list[dict[str, Any]]:
    return [
        {
            "alias_name": "phase2_baseline_v1",
            "definition": "b06_exact + metadata_t4off_half",
            "metadata_boost_scale": 0.5,
            "metadata_scope_mode": "all",
            "metadata_disable_for_rejection": 1,
        },
        {
            "alias_name": "comparison_safe_shadow",
            "definition": "b06_exact + metadata_t4off",
            "metadata_boost_scale": 1.0,
            "metadata_scope_mode": "all",
            "metadata_disable_for_rejection": 1,
        },
        {
            "alias_name": "reject",
            "definition": "metadata_t4off_scoped",
            "metadata_boost_scale": 0.5,
            "metadata_scope_mode": "comparison_and_explicit_factual",
            "metadata_disable_for_rejection": 1,
        },
    ]


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


def _collect_run_metrics(run_dir: Path) -> dict[str, Any]:
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
        "latency_sec": _pick_group_value(auto_summary_rows, "overall", ["avg_elapsed_sec", "latency_sec"]),
        "table_plus_body_coverage": _pick_group_value(coverage_summary_rows, "overall", ["table_plus_body_coverage"]),
        "question_count": int(
            _to_float(_pick_group_value(auto_summary_rows, "overall", ["question_count"])) or len(manual_completed_rows)
        ),
    }


def _compare_manual_rows(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline_by_qid = {str(row.get("question_id", "")).strip(): row for row in baseline_rows}
    case_rows: list[dict[str, Any]] = []
    group_rollup: dict[str, dict[str, float]] = {}
    improved = 0
    degraded = 0

    for row in candidate_rows:
        qid = str(row.get("question_id", "")).strip()
        if not qid or qid not in baseline_by_qid:
            continue
        base_row = baseline_by_qid[qid]
        base_mean = _manual_mean(base_row)
        cand_mean = _manual_mean(row)
        if base_mean is None or cand_mean is None:
            continue
        delta = round(cand_mean - base_mean, 4)
        type_group = str(row.get("type_group", "")).strip() or "UNKNOWN"
        answer_type = str(row.get("answer_type", "")).strip() or "UNKNOWN"

        if delta > 0:
            improved += 1
        elif delta < 0:
            degraded += 1

        if type_group not in group_rollup:
            group_rollup[type_group] = {
                "sum_delta": 0.0,
                "count": 0.0,
                "improved_count": 0.0,
                "degraded_count": 0.0,
            }
        group_rollup[type_group]["sum_delta"] += delta
        group_rollup[type_group]["count"] += 1
        if delta > 0:
            group_rollup[type_group]["improved_count"] += 1
        elif delta < 0:
            group_rollup[type_group]["degraded_count"] += 1

        case_rows.append(
            {
                "question_id": qid,
                "type_group": type_group,
                "answer_type": answer_type,
                "baseline_manual_mean": base_mean,
                "candidate_manual_mean": cand_mean,
                "delta_manual_mean": delta,
            }
        )

    improved_groups: list[dict[str, Any]] = []
    degraded_groups: list[dict[str, Any]] = []
    for group_name, values in group_rollup.items():
        count = int(values["count"])
        avg_delta = round(values["sum_delta"] / count, 4) if count else 0.0
        row = {
            "type_group": group_name,
            "avg_delta": avg_delta,
            "count": count,
            "improved_count": int(values["improved_count"]),
            "degraded_count": int(values["degraded_count"]),
        }
        if avg_delta > 0:
            improved_groups.append(row)
        elif avg_delta < 0:
            degraded_groups.append(row)

    improved_groups.sort(key=lambda item: item["avg_delta"], reverse=True)
    degraded_groups.sort(key=lambda item: item["avg_delta"])
    summary = {
        "improved_question_count": improved,
        "degraded_question_count": degraded,
        "degradation_count": degraded,
        "improved_type_groups": improved_groups,
        "degraded_type_groups": degraded_groups,
    }
    return case_rows, summary


def _run_one(
    *,
    args: argparse.Namespace,
    project_root: Path,
    bundle: Phase2ExperimentBundle,
    eval_set_key: str,
    run_label: str,
    embedding_backend: str,
    chroma_dir: Path | None,
    options: Phase2Options,
    output_root: Path,
    judge_model: str | None,
    single_upgrade_change: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    question_set_path, question_id_file = _resolve_eval_paths(bundle, eval_set_key)
    question_rows = read_eval_rows(question_set_path, question_id_file=question_id_file, shard_count=1, shard_index=0)
    run_dir = output_root / run_label
    run_dir.mkdir(parents=True, exist_ok=True)

    pipeline = ScenarioBPhase2Pipeline(
        PipelinePaths(project_root=project_root, chroma_dir=chroma_dir),
        settings=_build_settings(args, embedding_backend),
        options=options,
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
            "eval_set_key": eval_set_key,
            "question_set_path": str(question_set_path),
            "question_id_file": str(question_id_file) if question_id_file else "",
            "answer_layer_policy": "b06_exact_parity",
            "single_upgrade_change": single_upgrade_change,
            "phase2_options": {
                "enable_controlled_query_expansion": options.enable_controlled_query_expansion,
                "enable_normalized_bm25": options.enable_normalized_bm25,
                "enable_metadata_aware_retrieval": options.enable_metadata_aware_retrieval,
                "enable_table_body_pairing": options.enable_table_body_pairing,
                "enable_soft_crag_lite": options.enable_soft_crag_lite,
                "metadata_boost_scale": options.metadata_boost_scale,
                "metadata_disable_for_rejection": options.metadata_disable_for_rejection,
                "metadata_scope_mode": options.metadata_scope_mode,
                "normalized_bm25_mode": options.normalized_bm25_mode,
                "enable_b03_legacy_crag_parity": options.enable_b03_legacy_crag_parity,
                "b03_evaluator_top_n": options.b03_evaluator_top_n,
                "b03_second_pass_vector_weight": options.b03_second_pass_vector_weight,
                "b03_second_pass_bm25_weight": options.b03_second_pass_bm25_weight,
            },
        },
    )
    metrics = _collect_run_metrics(run_dir)
    manifest = dict(result["manifest"])
    manifest["run_dir"] = str(run_dir)
    return metrics, manifest


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_count": metrics.get("question_count"),
        "overall_manual_mean": metrics.get("overall_manual_mean"),
        "type2_manual_mean": metrics.get("type2_manual_mean"),
        "type4_rejection_success": metrics.get("type4_rejection_success"),
        "table_plus_body_coverage": metrics.get("table_plus_body_coverage"),
        "latency_sec": metrics.get("latency_sec"),
    }


def _delta(candidate: float | None, baseline: float | None, digits: int = 4) -> float | None:
    if candidate is None or baseline is None:
        return None
    return round(candidate - baseline, digits)


def _subset_gate_passed(
    *,
    table_delta_overall: float | None,
    groupbc_delta_overall: float | None,
    groupbc_delta_type4: float | None,
    overall_floor: float,
    type4_floor: float,
) -> tuple[bool, dict[str, Any]]:
    checks = {
        "table_15_overall_delta": int(table_delta_overall is not None and table_delta_overall >= overall_floor),
        "group_bc_overall_delta": int(groupbc_delta_overall is not None and groupbc_delta_overall >= overall_floor),
    }
    if groupbc_delta_type4 is not None:
        checks["group_bc_type4_delta"] = int(groupbc_delta_type4 >= type4_floor)
    else:
        checks["group_bc_type4_delta"] = 1
    return all(value == 1 for value in checks.values()), checks


def _format_group_summary(groups: list[dict[str, Any]]) -> str:
    if not groups:
        return ""
    parts: list[str] = []
    for row in groups:
        parts.append(
            f"{row['type_group']}(avg_delta={row['avg_delta']}, improved={row['improved_count']}, degraded={row['degraded_count']})"
        )
    return " | ".join(parts)


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_root = output_root / args.run_prefix
    run_root.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.experiment_config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")
    bundle = load_phase2_experiment_bundle(config_path, project_root)
    judge_model = None if args.no_judge else args.judge_model
    if judge_model is None:
        raise RuntimeError("Judge is required for manual mean comparison in this runner. Remove --no-judge.")

    baseline_options = _build_phase2_baseline_v1_options()
    baseline_chroma_dir = Path(args.chroma_dir_baseline).resolve() if args.chroma_dir_baseline else None
    ocr_chroma_dir = Path(args.chroma_dir_ocr).resolve() if args.chroma_dir_ocr else None

    write_csv(run_root / "baseline_profile_aliases.csv", _build_profile_alias_manifest())

    set_results: list[dict[str, Any]] = []
    run_manifests: list[dict[str, Any]] = []
    pair_summaries: dict[str, dict[str, Any]] = {}
    pair_case_files: list[dict[str, Any]] = []

    for eval_set_key in [args.table_eval_set_key, args.groupbc_eval_set_key]:
        baseline_label = f"{args.run_prefix}_{eval_set_key}_baseline_v1"
        ocr_label = f"{args.run_prefix}_{eval_set_key}_ocr_upgrade"
        print(f"[run] eval_set={eval_set_key} baseline_v1")
        baseline_metrics, baseline_manifest = _run_one(
            args=args,
            project_root=project_root,
            bundle=bundle,
            eval_set_key=eval_set_key,
            run_label=baseline_label,
            embedding_backend=args.embedding_backend_baseline,
            chroma_dir=baseline_chroma_dir,
            options=baseline_options,
            output_root=run_root,
            judge_model=judge_model,
            single_upgrade_change="none",
        )
        print(f"[run] eval_set={eval_set_key} ocr_upgrade")
        ocr_metrics, ocr_manifest = _run_one(
            args=args,
            project_root=project_root,
            bundle=bundle,
            eval_set_key=eval_set_key,
            run_label=ocr_label,
            embedding_backend=args.embedding_backend_ocr,
            chroma_dir=ocr_chroma_dir,
            options=baseline_options,
            output_root=run_root,
            judge_model=judge_model,
            single_upgrade_change="ocr_pilot_corpus",
        )
        run_manifests.extend([baseline_manifest, ocr_manifest])

        case_rows, diff_summary = _compare_manual_rows(
            baseline_metrics["manual_completed_rows"],
            ocr_metrics["manual_completed_rows"],
        )
        case_path = run_root / f"{args.run_prefix}_{eval_set_key}_delta_cases.csv"
        write_csv(case_path, case_rows)
        pair_case_files.append({"eval_set_key": eval_set_key, "delta_case_file": str(case_path)})

        delta_overall = _delta(ocr_metrics["overall_manual_mean"], baseline_metrics["overall_manual_mean"])
        delta_type2 = _delta(ocr_metrics["type2_manual_mean"], baseline_metrics["type2_manual_mean"])
        delta_type4 = _delta(ocr_metrics["type4_rejection_success"], baseline_metrics["type4_rejection_success"])
        delta_table_cov = _delta(ocr_metrics["table_plus_body_coverage"], baseline_metrics["table_plus_body_coverage"])
        delta_latency = _delta(ocr_metrics["latency_sec"], baseline_metrics["latency_sec"], digits=2)

        set_results.extend(
            [
                {
                    "eval_set_key": eval_set_key,
                    "run_variant": "baseline_v1",
                    "overall_manual_mean": baseline_metrics["overall_manual_mean"],
                    "type2_manual_mean": baseline_metrics["type2_manual_mean"],
                    "type4_rejection_success": baseline_metrics["type4_rejection_success"],
                    "table_plus_body_coverage": baseline_metrics["table_plus_body_coverage"],
                    "latency_sec": baseline_metrics["latency_sec"],
                    "degradation_count": 0,
                    "delta_overall_vs_baseline_v1": 0.0,
                    "delta_type2_vs_baseline_v1": 0.0,
                    "delta_type4_vs_baseline_v1": 0.0,
                    "delta_table_plus_body_coverage_vs_baseline_v1": 0.0,
                    "delta_latency_vs_baseline_v1": 0.0,
                    "improved_question_count": 0,
                    "degraded_question_count": 0,
                    "improved_question_groups": "",
                    "degraded_question_groups": "",
                    "run_dir": baseline_manifest.get("run_dir", ""),
                },
                {
                    "eval_set_key": eval_set_key,
                    "run_variant": "ocr_upgrade",
                    "overall_manual_mean": ocr_metrics["overall_manual_mean"],
                    "type2_manual_mean": ocr_metrics["type2_manual_mean"],
                    "type4_rejection_success": ocr_metrics["type4_rejection_success"],
                    "table_plus_body_coverage": ocr_metrics["table_plus_body_coverage"],
                    "latency_sec": ocr_metrics["latency_sec"],
                    "degradation_count": diff_summary["degradation_count"],
                    "delta_overall_vs_baseline_v1": delta_overall,
                    "delta_type2_vs_baseline_v1": delta_type2,
                    "delta_type4_vs_baseline_v1": delta_type4,
                    "delta_table_plus_body_coverage_vs_baseline_v1": delta_table_cov,
                    "delta_latency_vs_baseline_v1": delta_latency,
                    "improved_question_count": diff_summary["improved_question_count"],
                    "degraded_question_count": diff_summary["degraded_question_count"],
                    "improved_question_groups": _format_group_summary(diff_summary["improved_type_groups"]),
                    "degraded_question_groups": _format_group_summary(diff_summary["degraded_type_groups"]),
                    "run_dir": ocr_manifest.get("run_dir", ""),
                },
            ]
        )
        pair_summaries[eval_set_key] = {
            "baseline_metrics": _compact_metrics(baseline_metrics),
            "ocr_metrics": _compact_metrics(ocr_metrics),
            "delta_overall": delta_overall,
            "delta_type2": delta_type2,
            "delta_type4": delta_type4,
            "delta_table_plus_body_coverage": delta_table_cov,
            "delta_latency_sec": delta_latency,
            "diff_summary": diff_summary,
            "delta_case_file": str(case_path),
        }

    table_summary = pair_summaries[args.table_eval_set_key]
    groupbc_summary = pair_summaries[args.groupbc_eval_set_key]
    subset_passed, subset_checks = _subset_gate_passed(
        table_delta_overall=table_summary["delta_overall"],
        groupbc_delta_overall=groupbc_summary["delta_overall"],
        groupbc_delta_type4=groupbc_summary["delta_type4"],
        overall_floor=args.subset_gate_overall_delta_min,
        type4_floor=args.subset_gate_type4_delta_min,
    )

    full45_summary: dict[str, Any] | None = None
    if subset_passed:
        eval_set_key = args.full_eval_set_key
        baseline_label = f"{args.run_prefix}_{eval_set_key}_baseline_v1"
        ocr_label = f"{args.run_prefix}_{eval_set_key}_ocr_upgrade"
        print(f"[run] eval_set={eval_set_key} baseline_v1 (gate passed)")
        baseline_metrics, baseline_manifest = _run_one(
            args=args,
            project_root=project_root,
            bundle=bundle,
            eval_set_key=eval_set_key,
            run_label=baseline_label,
            embedding_backend=args.embedding_backend_baseline,
            chroma_dir=baseline_chroma_dir,
            options=baseline_options,
            output_root=run_root,
            judge_model=judge_model,
            single_upgrade_change="none",
        )
        print(f"[run] eval_set={eval_set_key} ocr_upgrade (gate passed)")
        ocr_metrics, ocr_manifest = _run_one(
            args=args,
            project_root=project_root,
            bundle=bundle,
            eval_set_key=eval_set_key,
            run_label=ocr_label,
            embedding_backend=args.embedding_backend_ocr,
            chroma_dir=ocr_chroma_dir,
            options=baseline_options,
            output_root=run_root,
            judge_model=judge_model,
            single_upgrade_change="ocr_pilot_corpus",
        )
        run_manifests.extend([baseline_manifest, ocr_manifest])

        case_rows, diff_summary = _compare_manual_rows(
            baseline_metrics["manual_completed_rows"],
            ocr_metrics["manual_completed_rows"],
        )
        case_path = run_root / f"{args.run_prefix}_{eval_set_key}_delta_cases.csv"
        write_csv(case_path, case_rows)
        pair_case_files.append({"eval_set_key": eval_set_key, "delta_case_file": str(case_path)})

        delta_overall = _delta(ocr_metrics["overall_manual_mean"], baseline_metrics["overall_manual_mean"])
        delta_type2 = _delta(ocr_metrics["type2_manual_mean"], baseline_metrics["type2_manual_mean"])
        delta_type4 = _delta(ocr_metrics["type4_rejection_success"], baseline_metrics["type4_rejection_success"])
        delta_table_cov = _delta(ocr_metrics["table_plus_body_coverage"], baseline_metrics["table_plus_body_coverage"])
        delta_latency = _delta(ocr_metrics["latency_sec"], baseline_metrics["latency_sec"], digits=2)

        set_results.extend(
            [
                {
                    "eval_set_key": eval_set_key,
                    "run_variant": "baseline_v1",
                    "overall_manual_mean": baseline_metrics["overall_manual_mean"],
                    "type2_manual_mean": baseline_metrics["type2_manual_mean"],
                    "type4_rejection_success": baseline_metrics["type4_rejection_success"],
                    "table_plus_body_coverage": baseline_metrics["table_plus_body_coverage"],
                    "latency_sec": baseline_metrics["latency_sec"],
                    "degradation_count": 0,
                    "delta_overall_vs_baseline_v1": 0.0,
                    "delta_type2_vs_baseline_v1": 0.0,
                    "delta_type4_vs_baseline_v1": 0.0,
                    "delta_table_plus_body_coverage_vs_baseline_v1": 0.0,
                    "delta_latency_vs_baseline_v1": 0.0,
                    "improved_question_count": 0,
                    "degraded_question_count": 0,
                    "improved_question_groups": "",
                    "degraded_question_groups": "",
                    "run_dir": baseline_manifest.get("run_dir", ""),
                },
                {
                    "eval_set_key": eval_set_key,
                    "run_variant": "ocr_upgrade",
                    "overall_manual_mean": ocr_metrics["overall_manual_mean"],
                    "type2_manual_mean": ocr_metrics["type2_manual_mean"],
                    "type4_rejection_success": ocr_metrics["type4_rejection_success"],
                    "table_plus_body_coverage": ocr_metrics["table_plus_body_coverage"],
                    "latency_sec": ocr_metrics["latency_sec"],
                    "degradation_count": diff_summary["degradation_count"],
                    "delta_overall_vs_baseline_v1": delta_overall,
                    "delta_type2_vs_baseline_v1": delta_type2,
                    "delta_type4_vs_baseline_v1": delta_type4,
                    "delta_table_plus_body_coverage_vs_baseline_v1": delta_table_cov,
                    "delta_latency_vs_baseline_v1": delta_latency,
                    "improved_question_count": diff_summary["improved_question_count"],
                    "degraded_question_count": diff_summary["degraded_question_count"],
                    "improved_question_groups": _format_group_summary(diff_summary["improved_type_groups"]),
                    "degraded_question_groups": _format_group_summary(diff_summary["degraded_type_groups"]),
                    "run_dir": ocr_manifest.get("run_dir", ""),
                },
            ]
        )
        full45_summary = {
            "baseline_metrics": _compact_metrics(baseline_metrics),
            "ocr_metrics": _compact_metrics(ocr_metrics),
            "delta_overall": delta_overall,
            "delta_type2": delta_type2,
            "delta_type4": delta_type4,
            "delta_table_plus_body_coverage": delta_table_cov,
            "delta_latency_sec": delta_latency,
            "diff_summary": diff_summary,
            "delta_case_file": str(case_path),
        }
    else:
        print("[gate] subset gate failed. full_45 is skipped.")

    per_set_compare_path = run_root / "ocr_single_upgrade_set_compare.csv"
    write_csv(per_set_compare_path, set_results)

    def _pair_metric(eval_set_key: str, key: str, variant: str) -> Any:
        row = next(
            (
                item
                for item in set_results
                if str(item.get("eval_set_key", "")) == eval_set_key and str(item.get("run_variant", "")) == variant
            ),
            None,
        )
        return row.get(key) if row is not None else None

    baseline_summary_row = {
        "run_variant": "baseline_v1",
        "overall_manual_mean": _pair_metric(args.full_eval_set_key, "overall_manual_mean", "baseline_v1"),
        "table_15_mean": _pair_metric(args.table_eval_set_key, "overall_manual_mean", "baseline_v1"),
        "group_bc_mean": _pair_metric(args.groupbc_eval_set_key, "overall_manual_mean", "baseline_v1"),
        "type4_rejection_success": _pair_metric(args.full_eval_set_key, "type4_rejection_success", "baseline_v1"),
        "table_plus_body_coverage": _pair_metric(args.full_eval_set_key, "table_plus_body_coverage", "baseline_v1"),
        "degradation_count": 0,
    }
    ocr_summary_row = {
        "run_variant": "ocr_upgrade",
        "overall_manual_mean": _pair_metric(args.full_eval_set_key, "overall_manual_mean", "ocr_upgrade"),
        "table_15_mean": _pair_metric(args.table_eval_set_key, "overall_manual_mean", "ocr_upgrade"),
        "group_bc_mean": _pair_metric(args.groupbc_eval_set_key, "overall_manual_mean", "ocr_upgrade"),
        "type4_rejection_success": _pair_metric(args.full_eval_set_key, "type4_rejection_success", "ocr_upgrade"),
        "table_plus_body_coverage": _pair_metric(args.full_eval_set_key, "table_plus_body_coverage", "ocr_upgrade"),
        "degradation_count": _pair_metric(args.full_eval_set_key, "degradation_count", "ocr_upgrade"),
    }
    delta_summary_row = {
        "run_variant": "delta_ocr_minus_baseline_v1",
        "overall_manual_mean": _delta(
            _to_float(ocr_summary_row["overall_manual_mean"]),
            _to_float(baseline_summary_row["overall_manual_mean"]),
        ),
        "table_15_mean": _delta(
            _to_float(ocr_summary_row["table_15_mean"]),
            _to_float(baseline_summary_row["table_15_mean"]),
        ),
        "group_bc_mean": _delta(
            _to_float(ocr_summary_row["group_bc_mean"]),
            _to_float(baseline_summary_row["group_bc_mean"]),
        ),
        "type4_rejection_success": _delta(
            _to_float(ocr_summary_row["type4_rejection_success"]),
            _to_float(baseline_summary_row["type4_rejection_success"]),
        ),
        "table_plus_body_coverage": _delta(
            _to_float(ocr_summary_row["table_plus_body_coverage"]),
            _to_float(baseline_summary_row["table_plus_body_coverage"]),
        ),
        "degradation_count": _to_float(ocr_summary_row["degradation_count"]),
    }
    summary_path = run_root / "ocr_single_upgrade_summary.csv"
    write_csv(summary_path, [baseline_summary_row, ocr_summary_row, delta_summary_row])

    verdict = "보류"
    if full45_summary is not None:
        delta_overall = full45_summary["delta_overall"]
        delta_type4 = full45_summary["delta_type4"]
        if delta_overall is not None and delta_type4 is not None and delta_overall >= 0 and delta_type4 >= -0.05:
            verdict = "채택"
        elif delta_overall is not None and delta_type4 is not None and delta_overall >= -0.05 and delta_type4 >= -0.05:
            verdict = "조건부 채택"
        else:
            verdict = "기각"

    report_payload = {
        "project_root": str(project_root),
        "run_prefix": args.run_prefix,
        "baseline_selection": {
            "phase2_baseline_v1": "b06_exact + metadata_t4off_half",
            "comparison_safe_shadow": "b06_exact + metadata_t4off",
            "reject": "metadata_t4off_scoped",
        },
        "single_upgrade_change": "corpus only (B-02 base -> OCR pilot corpus)",
        "fixed_conditions": {
            "answer_layer_parity": "b06_exact",
            "routing": "same as baseline_v1",
            "hybrid_weight": f"{args.vector_weight}:{args.bm25_weight}",
            "metadata_profile": "metadata_t4off_half",
        },
        "subset_gate": {
            "passed": subset_passed,
            "checks": subset_checks,
            "overall_delta_floor": args.subset_gate_overall_delta_min,
            "type4_delta_floor": args.subset_gate_type4_delta_min,
        },
        "pair_summaries": pair_summaries,
        "full45_summary": full45_summary,
        "verdict": verdict,
        "outputs": {
            "set_compare_csv": str(per_set_compare_path),
            "summary_csv": str(summary_path),
            "alias_manifest_csv": str(run_root / "baseline_profile_aliases.csv"),
            "delta_case_files": pair_case_files,
        },
        "run_manifests": run_manifests,
    }
    report_path = run_root / "ocr_single_upgrade_report.json"
    write_json(report_path, report_payload)

    print(f"[done] set_compare_csv={per_set_compare_path}")
    print(f"[done] summary_csv={summary_path}")
    print(f"[done] report={report_path}")
    print(f"[gate] subset_passed={subset_passed}")
    print(f"[verdict] {verdict}")


if __name__ == "__main__":
    main()

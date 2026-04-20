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
from scenario_b_phase2.experiment_config import load_phase2_experiment_bundle
from scenario_a.common_pipeline import PipelinePaths, PipelineSettings
from scenario_b_phase2.phase2_eval import evaluate_phase2, read_eval_rows
from scenario_b_phase2.phase2_pipeline import Phase2Options, ScenarioBPhase2Pipeline


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Single upgrade: table_body_pairing only on baseline_v2.")
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument("--experiment-config", default=str(root / "config" / "phase2_experiments.yaml"))
    parser.add_argument("--output-root", default=str(root / "rag_outputs" / "phase2_runs"))
    parser.add_argument("--run-prefix", default="p2_table_pairing_only_v1")
    parser.add_argument("--table-eval-set-key", default="table_15")
    parser.add_argument("--groupbc-eval-set-key", default="group_bc")
    parser.add_argument("--full-eval-set-key", default="full_45")
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


def _build_baseline_v2_options(*, table_pairing: bool) -> Phase2Options:
    return Phase2Options(
        enable_controlled_query_expansion=False,
        enable_normalized_bm25=False,
        enable_metadata_aware_retrieval=True,
        enable_metadata_bonus_v2=None,
        enable_table_body_pairing=table_pairing,
        enable_soft_crag_lite=False,
        metadata_boost_scale=0.5,
        metadata_disable_for_rejection=True,
        metadata_scope_mode="all",
        normalized_bm25_mode="all",
        enable_b03_legacy_crag_parity=True,
        b03_evaluator_top_n=6,
        b03_second_pass_vector_weight=0.55,
        b03_second_pass_bm25_weight=0.45,
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
        "type4_rejection_success": _pick_group_value(auto_summary_rows, "TYPE 4", ["rejection_success_rate"]),
        "table_plus_body_coverage": _pick_group_value(coverage_summary_rows, "overall", ["table_plus_body_coverage"]),
        "latency_sec": _pick_group_value(auto_summary_rows, "overall", ["avg_elapsed_sec", "latency_sec"]),
    }


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
        delta_value = round(cand_mean - base_mean, 4)
        if delta_value < 0:
            count += 1
        detail_rows.append(
            {
                "question_id": qid,
                "type_group": row.get("type_group", ""),
                "answer_type": row.get("answer_type", ""),
                "baseline_v2_manual_mean": base_mean,
                "table_pairing_manual_mean": cand_mean,
                "delta_manual_mean": delta_value,
            }
        )
    return count, detail_rows


def _run_eval(
    *,
    args: argparse.Namespace,
    project_root: Path,
    question_set_path: Path,
    question_id_file: Path | None,
    run_dir: Path,
    run_label: str,
    options: Phase2Options,
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_dir.mkdir(parents=True, exist_ok=True)
    question_rows = read_eval_rows(question_set_path, question_id_file=question_id_file, shard_count=1, shard_index=0)
    pipeline = ScenarioBPhase2Pipeline(
        PipelinePaths(project_root=project_root, chroma_dir=Path(args.chroma_dir).resolve() if args.chroma_dir else None),
        settings=_build_settings(args),
        options=options,
    )
    adapter = pipeline.create_adapter(args.model_key)
    result = evaluate_phase2(
        pipeline,
        adapter,
        question_rows,
        run_dir,
        judge_model=args.judge_model,
        run_label=run_label,
        extra_manifest={
            "question_set_path": str(question_set_path),
            "question_id_file": str(question_id_file) if question_id_file else "",
            "single_upgrade_change": "table_body_pairing_only",
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
            },
        },
    )
    metrics = _collect_metrics(run_dir)
    return metrics, result["manifest"]


def _delta(candidate: float | None, baseline: float | None, digits: int = 4) -> float | None:
    if candidate is None or baseline is None:
        return None
    return round(candidate - baseline, digits)


def _gate_passed(
    *,
    table_delta_overall: float | None,
    group_delta_overall: float | None,
    group_delta_type4: float | None,
    overall_floor: float,
    type4_floor: float,
) -> tuple[bool, dict[str, Any]]:
    checks = {
        "table_15_overall_delta": int(table_delta_overall is not None and table_delta_overall >= overall_floor),
        "group_bc_overall_delta": int(group_delta_overall is not None and group_delta_overall >= overall_floor),
        "group_bc_type4_delta": int(group_delta_type4 is None or group_delta_type4 >= type4_floor),
    }
    return all(value == 1 for value in checks.values()), checks


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    run_root = Path(args.output_root).resolve() / args.run_prefix
    run_root.mkdir(parents=True, exist_ok=True)

    bundle = load_phase2_experiment_bundle(Path(args.experiment_config).resolve(), project_root)
    eval_paths = {
        "table_15": bundle.resolve_eval_set(args.table_eval_set_key),
        "group_bc": bundle.resolve_eval_set(args.groupbc_eval_set_key),
        "full_45": bundle.resolve_eval_set(args.full_eval_set_key),
    }
    eval_id_files = {
        "table_15": bundle.resolve_question_id_file(args.table_eval_set_key),
        "group_bc": bundle.resolve_question_id_file(args.groupbc_eval_set_key),
        "full_45": bundle.resolve_question_id_file(args.full_eval_set_key),
    }

    baseline_options = _build_baseline_v2_options(table_pairing=False)
    pairing_options = _build_baseline_v2_options(table_pairing=True)

    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    subset_delta_tracker: dict[str, dict[str, float | None]] = {}

    for key in ["table_15", "group_bc"]:
        _, question_set_path = eval_paths[key]
        question_id_file = eval_id_files[key]
        base_dir = run_root / f"{args.run_prefix}_{key}_baseline_v2"
        pair_dir = run_root / f"{args.run_prefix}_{key}_table_pairing"

        base_metrics, base_manifest = _run_eval(
            args=args,
            project_root=project_root,
            question_set_path=question_set_path,
            question_id_file=question_id_file,
            run_dir=base_dir,
            run_label=f"{args.run_prefix}_{key}_baseline_v2",
            options=baseline_options,
        )
        pair_metrics, pair_manifest = _run_eval(
            args=args,
            project_root=project_root,
            question_set_path=question_set_path,
            question_id_file=question_id_file,
            run_dir=pair_dir,
            run_label=f"{args.run_prefix}_{key}_table_pairing",
            options=pairing_options,
        )
        manifests.extend([base_manifest, pair_manifest])

        degradation_count, degradation_rows = _degradation_count(
            base_metrics["manual_completed_rows"],
            pair_metrics["manual_completed_rows"],
        )
        degradation_file = run_root / f"{args.run_prefix}_{key}_degradation_cases.csv"
        write_csv(degradation_file, degradation_rows)

        delta_overall = _delta(pair_metrics["overall_manual_mean"], base_metrics["overall_manual_mean"])
        delta_type4 = _delta(pair_metrics["type4_rejection_success"], base_metrics["type4_rejection_success"])
        delta_table_cov = _delta(pair_metrics["table_plus_body_coverage"], base_metrics["table_plus_body_coverage"])

        rows.extend(
            [
                {
                    "eval_key": key,
                    "run_variant": "baseline_v2",
                    "overall_manual_mean": base_metrics["overall_manual_mean"],
                    "type4_rejection_success": base_metrics["type4_rejection_success"],
                    "table_plus_body_coverage": base_metrics["table_plus_body_coverage"],
                    "degradation_count": 0,
                    "delta_overall_manual_mean_vs_baseline_v2": 0.0,
                    "delta_type4_rejection_success_vs_baseline_v2": 0.0,
                    "delta_table_plus_body_coverage_vs_baseline_v2": 0.0,
                    "latency_sec": base_metrics["latency_sec"],
                    "delta_latency_sec_vs_baseline_v2": 0.0,
                    "degradation_file": "",
                    "run_dir": str(base_dir),
                },
                {
                    "eval_key": key,
                    "run_variant": "table_body_pairing_only",
                    "overall_manual_mean": pair_metrics["overall_manual_mean"],
                    "type4_rejection_success": pair_metrics["type4_rejection_success"],
                    "table_plus_body_coverage": pair_metrics["table_plus_body_coverage"],
                    "degradation_count": degradation_count,
                    "delta_overall_manual_mean_vs_baseline_v2": delta_overall,
                    "delta_type4_rejection_success_vs_baseline_v2": delta_type4,
                    "delta_table_plus_body_coverage_vs_baseline_v2": delta_table_cov,
                    "latency_sec": pair_metrics["latency_sec"],
                    "delta_latency_sec_vs_baseline_v2": _delta(pair_metrics["latency_sec"], base_metrics["latency_sec"], digits=2),
                    "degradation_file": str(degradation_file),
                    "run_dir": str(pair_dir),
                },
            ]
        )
        subset_delta_tracker[key] = {"delta_overall": delta_overall, "delta_type4": delta_type4}

    gate_pass, gate_checks = _gate_passed(
        table_delta_overall=subset_delta_tracker["table_15"]["delta_overall"],
        group_delta_overall=subset_delta_tracker["group_bc"]["delta_overall"],
        group_delta_type4=subset_delta_tracker["group_bc"]["delta_type4"],
        overall_floor=args.subset_gate_overall_delta_min,
        type4_floor=args.subset_gate_type4_delta_min,
    )

    full45_run_done = False
    if gate_pass:
        key = "full_45"
        _, question_set_path = eval_paths[key]
        question_id_file = eval_id_files[key]
        base_dir = run_root / f"{args.run_prefix}_{key}_baseline_v2"
        pair_dir = run_root / f"{args.run_prefix}_{key}_table_pairing"

        base_metrics, base_manifest = _run_eval(
            args=args,
            project_root=project_root,
            question_set_path=question_set_path,
            question_id_file=question_id_file,
            run_dir=base_dir,
            run_label=f"{args.run_prefix}_{key}_baseline_v2",
            options=baseline_options,
        )
        pair_metrics, pair_manifest = _run_eval(
            args=args,
            project_root=project_root,
            question_set_path=question_set_path,
            question_id_file=question_id_file,
            run_dir=pair_dir,
            run_label=f"{args.run_prefix}_{key}_table_pairing",
            options=pairing_options,
        )
        manifests.extend([base_manifest, pair_manifest])

        degradation_count, degradation_rows = _degradation_count(
            base_metrics["manual_completed_rows"],
            pair_metrics["manual_completed_rows"],
        )
        degradation_file = run_root / f"{args.run_prefix}_{key}_degradation_cases.csv"
        write_csv(degradation_file, degradation_rows)

        rows.extend(
            [
                {
                    "eval_key": key,
                    "run_variant": "baseline_v2",
                    "overall_manual_mean": base_metrics["overall_manual_mean"],
                    "type4_rejection_success": base_metrics["type4_rejection_success"],
                    "table_plus_body_coverage": base_metrics["table_plus_body_coverage"],
                    "degradation_count": 0,
                    "delta_overall_manual_mean_vs_baseline_v2": 0.0,
                    "delta_type4_rejection_success_vs_baseline_v2": 0.0,
                    "delta_table_plus_body_coverage_vs_baseline_v2": 0.0,
                    "latency_sec": base_metrics["latency_sec"],
                    "delta_latency_sec_vs_baseline_v2": 0.0,
                    "degradation_file": "",
                    "run_dir": str(base_dir),
                },
                {
                    "eval_key": key,
                    "run_variant": "table_body_pairing_only",
                    "overall_manual_mean": pair_metrics["overall_manual_mean"],
                    "type4_rejection_success": pair_metrics["type4_rejection_success"],
                    "table_plus_body_coverage": pair_metrics["table_plus_body_coverage"],
                    "degradation_count": degradation_count,
                    "delta_overall_manual_mean_vs_baseline_v2": _delta(
                        pair_metrics["overall_manual_mean"], base_metrics["overall_manual_mean"]
                    ),
                    "delta_type4_rejection_success_vs_baseline_v2": _delta(
                        pair_metrics["type4_rejection_success"], base_metrics["type4_rejection_success"]
                    ),
                    "delta_table_plus_body_coverage_vs_baseline_v2": _delta(
                        pair_metrics["table_plus_body_coverage"], base_metrics["table_plus_body_coverage"]
                    ),
                    "latency_sec": pair_metrics["latency_sec"],
                    "delta_latency_sec_vs_baseline_v2": _delta(pair_metrics["latency_sec"], base_metrics["latency_sec"], digits=2),
                    "degradation_file": str(degradation_file),
                    "run_dir": str(pair_dir),
                },
            ]
        )
        full45_run_done = True

    compare_path = run_root / "table_pairing_single_upgrade_compare.csv"
    write_csv(compare_path, rows)

    def _pick(eval_key: str, run_variant: str, key: str) -> Any:
        row = next(
            (
                item
                for item in rows
                if str(item.get("eval_key", "")) == eval_key and str(item.get("run_variant", "")) == run_variant
            ),
            None,
        )
        return row.get(key) if row is not None else None

    summary_rows = [
        {
            "run_variant": "baseline_v2",
            "overall_manual_mean": _pick("full_45", "baseline_v2", "overall_manual_mean"),
            "table_15_mean": _pick("table_15", "baseline_v2", "overall_manual_mean"),
            "group_bc_mean": _pick("group_bc", "baseline_v2", "overall_manual_mean"),
            "table_plus_body_coverage": _pick("full_45", "baseline_v2", "table_plus_body_coverage"),
            "type4_rejection_success": _pick("full_45", "baseline_v2", "type4_rejection_success"),
            "degradation_count": 0,
        },
        {
            "run_variant": "table_body_pairing_only",
            "overall_manual_mean": _pick("full_45", "table_body_pairing_only", "overall_manual_mean"),
            "table_15_mean": _pick("table_15", "table_body_pairing_only", "overall_manual_mean"),
            "group_bc_mean": _pick("group_bc", "table_body_pairing_only", "overall_manual_mean"),
            "table_plus_body_coverage": _pick("full_45", "table_body_pairing_only", "table_plus_body_coverage"),
            "type4_rejection_success": _pick("full_45", "table_body_pairing_only", "type4_rejection_success"),
            "degradation_count": _pick("full_45", "table_body_pairing_only", "degradation_count"),
        },
        {
            "run_variant": "delta_pairing_minus_baseline_v2",
            "overall_manual_mean": _delta(
                _to_float(_pick("full_45", "table_body_pairing_only", "overall_manual_mean")),
                _to_float(_pick("full_45", "baseline_v2", "overall_manual_mean")),
            ),
            "table_15_mean": _delta(
                _to_float(_pick("table_15", "table_body_pairing_only", "overall_manual_mean")),
                _to_float(_pick("table_15", "baseline_v2", "overall_manual_mean")),
            ),
            "group_bc_mean": _delta(
                _to_float(_pick("group_bc", "table_body_pairing_only", "overall_manual_mean")),
                _to_float(_pick("group_bc", "baseline_v2", "overall_manual_mean")),
            ),
            "table_plus_body_coverage": _delta(
                _to_float(_pick("full_45", "table_body_pairing_only", "table_plus_body_coverage")),
                _to_float(_pick("full_45", "baseline_v2", "table_plus_body_coverage")),
            ),
            "type4_rejection_success": _delta(
                _to_float(_pick("full_45", "table_body_pairing_only", "type4_rejection_success")),
                _to_float(_pick("full_45", "baseline_v2", "type4_rejection_success")),
            ),
            "degradation_count": _to_float(_pick("full_45", "table_body_pairing_only", "degradation_count")),
        },
    ]
    summary_path = run_root / "table_pairing_single_upgrade_summary.csv"
    write_csv(summary_path, summary_rows)

    report_path = run_root / "table_pairing_single_upgrade_report.json"
    write_json(
        report_path,
        {
            "project_root": str(project_root),
            "run_prefix": args.run_prefix,
            "baseline_v2": "b06_exact + metadata_t4off_half + comparison_helper_only",
            "single_upgrade": "table_body_pairing_only",
            "fixed_conditions": {
                "answer_layer_parity": "b06_exact",
                "hybrid_weight": f"{args.vector_weight}:{args.bm25_weight}",
                "metadata_profile": "metadata_t4off_half",
                "comparison_helper_only": True,
                "soft_crag_lite": False,
                "follow_up": "no-change",
            },
            "subset_gate": {
                "passed": gate_pass,
                "checks": gate_checks,
                "overall_floor": args.subset_gate_overall_delta_min,
                "type4_floor": args.subset_gate_type4_delta_min,
            },
            "full45_run_done": full45_run_done,
            "compare_csv": str(compare_path),
            "summary_csv": str(summary_path),
            "run_manifests": manifests,
        },
    )

    print(f"[done] compare_csv={compare_path}")
    print(f"[done] summary_csv={summary_path}")
    print(f"[done] report={report_path}")
    print(f"[gate] passed={gate_pass}")


if __name__ == "__main__":
    main()

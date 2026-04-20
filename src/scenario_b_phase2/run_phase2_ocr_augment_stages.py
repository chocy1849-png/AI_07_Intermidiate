from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from eval_utils import average, read_csv, safe_float, write_csv, write_json
from scenario_a.common_pipeline import PipelinePaths, PipelineSettings
from scenario_b_phase2.experiment_config import Phase2ExperimentBundle, load_phase2_experiment_bundle
from scenario_b_phase2.ocr_augment_enrichment import STAGE_SPECS
from scenario_b_phase2.phase2_eval import evaluate_phase2, read_eval_rows
from scenario_b_phase2.phase2_pipeline import Phase2Options, ScenarioBPhase2Pipeline


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Phase2 OCR augment experiment runner: A(structural) / B(+image_ocr) / C(+selected discarded hints)."
    )
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument("--experiment-config", default=str(root / "config" / "phase2_experiments.yaml"))
    parser.add_argument("--output-root", default=str(root / "rag_outputs" / "phase2_runs"))
    parser.add_argument("--run-prefix", default="p2_ocr_augment_stages_v1")
    parser.add_argument("--assets-output-root", default=str(root / "rag_outputs" / "phase2_ocr_aug_assets"))
    parser.add_argument("--chroma-dir-baseline", default="")
    parser.add_argument("--chroma-dir-augment", default=str(root.parent / "rfp_rag_chroma_db_phase2_ocr_aug"))
    parser.add_argument("--build-assets", action="store_true")
    parser.add_argument("--reset-augment-collections", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--table-eval-set-key", default="table_15")
    parser.add_argument("--groupbc-eval-set-key", default="group_bc")
    parser.add_argument("--full-eval-set-key", default="full_45")
    parser.add_argument("--model-key", default="gpt5mini_api")
    parser.add_argument("--routing-model", default="gpt-5-mini")
    parser.add_argument("--judge-model", default="gpt-5")
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


def _build_settings(args: argparse.Namespace, embedding_backend_key: str) -> PipelineSettings:
    return PipelineSettings(
        embedding_backend_key=embedding_backend_key,
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


def _baseline_v2_pairing_options() -> Phase2Options:
    return Phase2Options(
        enable_controlled_query_expansion=False,
        enable_normalized_bm25=False,
        enable_metadata_aware_retrieval=True,
        enable_metadata_bonus_v2=None,
        enable_table_body_pairing=True,
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


def _resolve_eval_paths(bundle: Phase2ExperimentBundle, eval_set_key: str) -> tuple[Path, Path | None]:
    _, question_set_path = bundle.resolve_eval_set(eval_set_key)
    question_id_file = bundle.resolve_question_id_file(eval_set_key)
    return question_set_path, question_id_file


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


def _run_eval(
    *,
    args: argparse.Namespace,
    project_root: Path,
    question_set_path: Path,
    question_id_file: Path | None,
    run_dir: Path,
    run_label: str,
    embedding_backend_key: str,
    chroma_dir: Path | None,
    reuse_existing: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_dir.mkdir(parents=True, exist_ok=True)
    manual_summary_path = run_dir / "phase2_eval_manual_summary.csv"
    manual_completed_path = run_dir / "phase2_eval_manual_completed.csv"
    manifest_path = run_dir / "phase2_eval_manifest.json"
    if reuse_existing and manual_summary_path.exists() and manual_completed_path.exists():
        manifest_payload: dict[str, Any] = {}
        if manifest_path.exists():
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return _collect_metrics(run_dir), manifest_payload

    options = _baseline_v2_pairing_options()
    question_rows = read_eval_rows(question_set_path, question_id_file=question_id_file, shard_count=1, shard_index=0)
    pipeline = ScenarioBPhase2Pipeline(
        PipelinePaths(project_root=project_root, chroma_dir=chroma_dir),
        settings=_build_settings(args, embedding_backend_key),
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
            "answer_layer_policy": "b06_exact",
            "fixed_profile": "phase2_baseline_v2 + table_body_pairing",
            "single_upgrade_change": "ocr_augment_stage",
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
    return _collect_metrics(run_dir), result["manifest"]


def _degradation_count(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> tuple[int, list[dict[str, Any]]]:
    baseline_by_qid = {str(row.get("question_id", "")).strip(): row for row in baseline_rows}
    details: list[dict[str, Any]] = []
    count = 0
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
        if delta < 0:
            count += 1
        details.append(
            {
                "question_id": qid,
                "type_group": row.get("type_group", ""),
                "answer_type": row.get("answer_type", ""),
                "baseline_manual_mean": base_mean,
                "candidate_manual_mean": cand_mean,
                "delta_manual_mean": delta,
            }
        )
    return count, details


def _delta(candidate: float | None, baseline: float | None, digits: int = 4) -> float | None:
    if candidate is None or baseline is None:
        return None
    return round(candidate - baseline, digits)


def _build_assets_if_requested(args: argparse.Namespace, project_root: Path) -> Path:
    assets_root = Path(args.assets_output_root).resolve()
    if not args.build_assets:
        return assets_root
    command = [
        sys.executable,
        str(project_root / "src" / "scenario_b_phase2" / "ocr_augment_enrichment.py"),
        "--project-root",
        str(project_root),
        "--output-root",
        str(assets_root),
        "--chroma-output-dir",
        str(Path(args.chroma_dir_augment).resolve()),
    ]
    if args.reset_augment_collections:
        command.append("--reset-collection")
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="cp949",
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "OCR augment asset build 실패\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return assets_root


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_root).resolve()
    run_root = output_root / args.run_prefix
    run_root.mkdir(parents=True, exist_ok=True)
    assets_root = _build_assets_if_requested(args, project_root)

    bundle = load_phase2_experiment_bundle(Path(args.experiment_config).resolve(), project_root)
    eval_order = [args.table_eval_set_key, args.groupbc_eval_set_key, args.full_eval_set_key]
    stage_order = ["A", "B", "C"]

    baseline_chroma_dir = Path(args.chroma_dir_baseline).resolve() if args.chroma_dir_baseline else None
    augment_chroma_dir = Path(args.chroma_dir_augment).resolve()

    set_compare_rows: list[dict[str, Any]] = []
    stage_set_metrics: dict[str, dict[str, dict[str, Any]]] = {stage: {} for stage in stage_order}
    run_manifests: list[dict[str, Any]] = []

    for eval_key in eval_order:
        question_set_path, question_id_file = _resolve_eval_paths(bundle, eval_key)
        baseline_run_dir = run_root / f"{args.run_prefix}_{eval_key}_baseline_v2"
        baseline_run_label = f"{args.run_prefix}_{eval_key}_baseline_v2"
        baseline_metrics, baseline_manifest = _run_eval(
            args=args,
            project_root=project_root,
            question_set_path=question_set_path,
            question_id_file=question_id_file,
            run_dir=baseline_run_dir,
            run_label=baseline_run_label,
            embedding_backend_key="openai_text_embedding_3_small",
            chroma_dir=baseline_chroma_dir,
            reuse_existing=bool(args.reuse_existing),
        )
        run_manifests.append(baseline_manifest)

        set_compare_rows.append(
            {
                "eval_set_key": eval_key,
                "run_variant": "baseline_v2",
                "overall_manual_mean": baseline_metrics["overall_manual_mean"],
                "type2_manual_mean": baseline_metrics["type2_manual_mean"],
                "type4_rejection_success": baseline_metrics["type4_rejection_success"],
                "table_plus_body_coverage": baseline_metrics["table_plus_body_coverage"],
                "latency_sec": baseline_metrics["latency_sec"],
                "degradation_count": 0,
                "delta_overall_vs_baseline_v2": 0.0,
                "delta_type2_vs_baseline_v2": 0.0,
                "delta_type4_vs_baseline_v2": 0.0,
                "delta_table_plus_body_coverage_vs_baseline_v2": 0.0,
                "delta_latency_vs_baseline_v2": 0.0,
                "degradation_file": "",
                "run_dir": str(baseline_run_dir),
            }
        )

        for stage_key in stage_order:
            stage_spec = STAGE_SPECS[stage_key]
            run_dir = run_root / f"{args.run_prefix}_{eval_key}_stage_{stage_key.lower()}"
            run_label = f"{args.run_prefix}_{eval_key}_stage_{stage_key.lower()}"
            metrics, manifest = _run_eval(
                args=args,
                project_root=project_root,
                question_set_path=question_set_path,
                question_id_file=question_id_file,
                run_dir=run_dir,
                run_label=run_label,
                embedding_backend_key=stage_spec.embedding_backend_key,
                chroma_dir=augment_chroma_dir,
                reuse_existing=bool(args.reuse_existing),
            )
            run_manifests.append(manifest)
            stage_set_metrics[stage_key][eval_key] = metrics

            degradation_count, degradation_rows = _degradation_count(
                baseline_metrics["manual_completed_rows"],
                metrics["manual_completed_rows"],
            )
            degradation_file = run_root / f"{args.run_prefix}_{eval_key}_stage_{stage_key.lower()}_degradation_cases.csv"
            write_csv(degradation_file, degradation_rows)

            set_compare_rows.append(
                {
                    "eval_set_key": eval_key,
                    "run_variant": f"stage_{stage_key.lower()}",
                    "overall_manual_mean": metrics["overall_manual_mean"],
                    "type2_manual_mean": metrics["type2_manual_mean"],
                    "type4_rejection_success": metrics["type4_rejection_success"],
                    "table_plus_body_coverage": metrics["table_plus_body_coverage"],
                    "latency_sec": metrics["latency_sec"],
                    "degradation_count": degradation_count,
                    "delta_overall_vs_baseline_v2": _delta(
                        metrics["overall_manual_mean"], baseline_metrics["overall_manual_mean"]
                    ),
                    "delta_type2_vs_baseline_v2": _delta(
                        metrics["type2_manual_mean"], baseline_metrics["type2_manual_mean"]
                    ),
                    "delta_type4_vs_baseline_v2": _delta(
                        metrics["type4_rejection_success"], baseline_metrics["type4_rejection_success"]
                    ),
                    "delta_table_plus_body_coverage_vs_baseline_v2": _delta(
                        metrics["table_plus_body_coverage"], baseline_metrics["table_plus_body_coverage"]
                    ),
                    "delta_latency_vs_baseline_v2": _delta(
                        metrics["latency_sec"], baseline_metrics["latency_sec"], digits=2
                    ),
                    "degradation_file": str(degradation_file),
                    "run_dir": str(run_dir),
                }
            )

    set_compare_csv = run_root / "ocr_augment_stages_set_compare.csv"
    write_csv(set_compare_csv, set_compare_rows)

    def _find_row(eval_key: str, variant: str) -> dict[str, Any] | None:
        return next(
            (
                row
                for row in set_compare_rows
                if str(row.get("eval_set_key", "")) == eval_key and str(row.get("run_variant", "")) == variant
            ),
            None,
        )

    summary_rows: list[dict[str, Any]] = []
    baseline_full = _find_row(args.full_eval_set_key, "baseline_v2")
    baseline_table = _find_row(args.table_eval_set_key, "baseline_v2")
    baseline_group = _find_row(args.groupbc_eval_set_key, "baseline_v2")

    summary_rows.append(
        {
            "run_variant": "baseline_v2",
            "overall_manual_mean": baseline_full.get("overall_manual_mean") if baseline_full else None,
            "table_15_mean": baseline_table.get("overall_manual_mean") if baseline_table else None,
            "group_bc_mean": baseline_group.get("overall_manual_mean") if baseline_group else None,
            "table_plus_body_coverage": baseline_full.get("table_plus_body_coverage") if baseline_full else None,
            "type4_rejection_success": baseline_full.get("type4_rejection_success") if baseline_full else None,
            "degradation_count": 0,
            "delta_overall_vs_baseline_v2": 0.0,
            "delta_table_15_vs_baseline_v2": 0.0,
            "delta_group_bc_vs_baseline_v2": 0.0,
            "delta_table_plus_body_coverage_vs_baseline_v2": 0.0,
            "delta_type4_vs_baseline_v2": 0.0,
        }
    )

    for stage_key in stage_order:
        variant = f"stage_{stage_key.lower()}"
        full_row = _find_row(args.full_eval_set_key, variant)
        table_row = _find_row(args.table_eval_set_key, variant)
        group_row = _find_row(args.groupbc_eval_set_key, variant)
        summary_rows.append(
            {
                "run_variant": variant,
                "overall_manual_mean": full_row.get("overall_manual_mean") if full_row else None,
                "table_15_mean": table_row.get("overall_manual_mean") if table_row else None,
                "group_bc_mean": group_row.get("overall_manual_mean") if group_row else None,
                "table_plus_body_coverage": full_row.get("table_plus_body_coverage") if full_row else None,
                "type4_rejection_success": full_row.get("type4_rejection_success") if full_row else None,
                "degradation_count": full_row.get("degradation_count") if full_row else None,
                "delta_overall_vs_baseline_v2": _delta(
                    _to_float(full_row.get("overall_manual_mean") if full_row else None),
                    _to_float(baseline_full.get("overall_manual_mean") if baseline_full else None),
                ),
                "delta_table_15_vs_baseline_v2": _delta(
                    _to_float(table_row.get("overall_manual_mean") if table_row else None),
                    _to_float(baseline_table.get("overall_manual_mean") if baseline_table else None),
                ),
                "delta_group_bc_vs_baseline_v2": _delta(
                    _to_float(group_row.get("overall_manual_mean") if group_row else None),
                    _to_float(baseline_group.get("overall_manual_mean") if baseline_group else None),
                ),
                "delta_table_plus_body_coverage_vs_baseline_v2": _delta(
                    _to_float(full_row.get("table_plus_body_coverage") if full_row else None),
                    _to_float(baseline_full.get("table_plus_body_coverage") if baseline_full else None),
                ),
                "delta_type4_vs_baseline_v2": _delta(
                    _to_float(full_row.get("type4_rejection_success") if full_row else None),
                    _to_float(baseline_full.get("type4_rejection_success") if baseline_full else None),
                ),
            }
        )

    summary_csv = run_root / "ocr_augment_stages_summary.csv"
    write_csv(summary_csv, summary_rows)

    report = {
        "project_root": str(project_root),
        "run_prefix": args.run_prefix,
        "baseline_definition": "phase2_baseline_v2 (b06_exact + metadata_t4off_half + comparison_helper_only) + table_body_pairing=true",
        "augment_policy": "replace 금지, augment only",
        "experiment_sequence": [args.table_eval_set_key, args.groupbc_eval_set_key, args.full_eval_set_key],
        "stage_order": stage_order,
        "assets_root": str(assets_root),
        "chroma_dir_baseline": str(baseline_chroma_dir) if baseline_chroma_dir else "",
        "chroma_dir_augment": str(augment_chroma_dir),
        "set_compare_csv": str(set_compare_csv),
        "summary_csv": str(summary_csv),
        "run_manifests": run_manifests,
    }
    report_path = run_root / "ocr_augment_stages_report.json"
    write_json(report_path, report)

    print(f"[done] set_compare_csv={set_compare_csv}")
    print(f"[done] summary_csv={summary_csv}")
    print(f"[done] report={report_path}")


if __name__ == "__main__":
    main()

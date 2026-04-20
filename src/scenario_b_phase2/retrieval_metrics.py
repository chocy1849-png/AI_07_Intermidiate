from __future__ import annotations

from typing import Any

from eval_utils import average, safe_float


def _to_float(value: Any) -> float | None:
    return safe_float(value)


def build_phase2_coverage_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    group_specs: list[tuple[str, list[dict[str, Any]]]] = [("overall", rows)]
    type_groups = sorted({row.get("type_group", "") for row in rows if row.get("type_group")})
    for label in type_groups:
        group_specs.append((label, [row for row in rows if row.get("type_group") == label]))
    answer_types = sorted({row.get("answer_type", "") for row in rows if row.get("answer_type")})
    for label in answer_types:
        group_specs.append((f"answer_type:{label}", [row for row in rows if row.get("answer_type") == label]))

    output: list[dict[str, Any]] = []
    for label, group_rows in group_specs:
        fallback_values = [_to_float(row.get("fallback_triggered")) for row in group_rows]
        weak_comparison_count = sum(
            1
            for row in group_rows
            if str(row.get("answer_type", "")).strip() == "comparison"
            and (_to_float(row.get("dual_doc_coverage")) or 0.0) < 1.0
        )
        weak_table_body_count = sum(
            1
            for row in group_rows
            if (_to_float(row.get("table_plus_body_coverage")) is not None)
            and (_to_float(row.get("table_plus_body_coverage")) or 0.0) < 1.0
        )
        weak_pair_count = sum(
            1
            for row in group_rows
            if (_to_float(row.get("pair_hit")) is not None)
            and (_to_float(row.get("pair_hit")) or 0.0) < 1.0
        )
        weak_body_count = sum(
            1
            for row in group_rows
            if (_to_float(row.get("body_hit")) is not None)
            and (_to_float(row.get("body_hit")) or 0.0) < 1.0
        )
        output.append(
            {
                "group_name": label,
                "question_count": len(group_rows),
                "avg_elapsed_sec": average([_to_float(row.get("elapsed_sec")) for row in group_rows]),
                "avg_source_diversity": average([_to_float(row.get("source_diversity")) for row in group_rows]),
                "dual_doc_coverage": average([_to_float(row.get("dual_doc_coverage")) for row in group_rows]),
                "comparison_evidence_coverage": average(
                    [_to_float(row.get("comparison_evidence_coverage")) for row in group_rows]
                ),
                "table_plus_body_coverage": average([_to_float(row.get("table_plus_body_coverage")) for row in group_rows]),
                "table_hit": average([_to_float(row.get("table_hit")) for row in group_rows]),
                "body_hit": average([_to_float(row.get("body_hit")) for row in group_rows]),
                "pair_hit": average([_to_float(row.get("pair_hit")) for row in group_rows]),
                "answer_hit": average([_to_float(row.get("answer_hit")) for row in group_rows]),
                "structured_evidence_hit": average([_to_float(row.get("structured_evidence_hit")) for row in group_rows]),
                "pairing_score_max": average([_to_float(row.get("pairing_score_max")) for row in group_rows]),
                "exact_header_match_hit": average([_to_float(row.get("exact_header_match_hit")) for row in group_rows]),
                "exact_row_match_hit": average([_to_float(row.get("exact_row_match_hit")) for row in group_rows]),
                "generic_row_pollution_count": sum(
                    int(_to_float(row.get("generic_row_pollution_count")) or 0) for row in group_rows
                ),
                "fallback_ratio": average(fallback_values),
                "query_expansion_usage": average(
                    [_to_float(row.get("controlled_query_expansion_used")) for row in group_rows]
                ),
                "metadata_aware_usage": average([_to_float(row.get("metadata_aware_used")) for row in group_rows]),
                "soft_crag_usage": average([_to_float(row.get("soft_crag_lite_used")) for row in group_rows]),
                "soft_crag_low_confidence_ratio": average(
                    [_to_float(row.get("soft_crag_low_confidence_flag")) for row in group_rows]
                ),
                "soft_crag_downrank_avg": average(
                    [_to_float(row.get("soft_crag_decision_downrank_count")) for row in group_rows]
                ),
                "soft_crag_low_conf_avg": average(
                    [_to_float(row.get("soft_crag_decision_low_conf_count")) for row in group_rows]
                ),
                "soft_crag_duplicate_ratio_avg": average(
                    [_to_float(row.get("soft_crag_duplicate_ratio")) for row in group_rows]
                ),
                "answer_type_router_usage": average(
                    [_to_float(row.get("answer_type_router_used")) for row in group_rows]
                ),
                "weak_comparison_count": weak_comparison_count,
                "weak_table_body_count": weak_table_body_count,
                "weak_pair_count": weak_pair_count,
                "weak_body_count": weak_body_count,
            }
        )
    return output


def build_phase2_compare_row(
    *,
    run_label: str,
    auto_summary_rows: list[dict[str, Any]],
    manual_summary_rows: list[dict[str, Any]],
    coverage_summary_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    def pick(rows: list[dict[str, Any]], group_name: str) -> dict[str, Any]:
        return next((row for row in rows if str(row.get("group_name", "")) == group_name), {})

    def first(row: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in row and row.get(key) not in (None, ""):
                return row.get(key)
        return None

    overall_auto = pick(auto_summary_rows, "overall")
    overall_manual = pick(manual_summary_rows, "overall")
    overall_cov = pick(coverage_summary_rows, "overall")
    type2_cov = pick(coverage_summary_rows, "TYPE 2")
    type2_manual = pick(manual_summary_rows, "TYPE 2")
    type4_manual = pick(manual_summary_rows, "TYPE 4")

    return {
        "run_label": run_label,
        "question_count": first(overall_auto, "question_count"),
        "manual_mean": first(overall_manual, "manual_mean", "avg_manual_eval_score"),
        "faithfulness": first(overall_manual, "faithfulness", "avg_faithfulness_score"),
        "completeness": first(overall_manual, "completeness", "avg_completeness_score"),
        "groundedness": first(overall_manual, "groundedness", "avg_groundedness_score"),
        "relevancy": first(overall_manual, "relevancy", "avg_relevancy_score"),
        "type2_manual_mean": first(type2_manual, "manual_mean", "avg_manual_eval_score"),
        "type4_manual_mean": first(type4_manual, "manual_mean", "avg_manual_eval_score"),
        "latency_sec": first(overall_auto, "avg_elapsed_sec", "latency_sec"),
        "type2_latency_sec": pick(auto_summary_rows, "TYPE 2").get("avg_elapsed_sec"),
        "top1_doc_hit": first(overall_auto, "top1_doc_hit_rate", "avg_top1_doc_hit"),
        "topk_doc_hit": first(overall_auto, "topk_doc_hit_rate", "avg_topk_doc_hit"),
        "doc_hit_rate": first(overall_auto, "avg_ground_truth_doc_hit_rate"),
        "rejection_success_rate": first(
            pick(auto_summary_rows, "TYPE 4"),
            "rejection_success_rate",
        ),
        "dual_doc_coverage": first(overall_cov, "dual_doc_coverage"),
        "comparison_evidence_coverage": first(overall_cov, "comparison_evidence_coverage"),
        "type2_dual_doc_coverage": first(type2_cov, "dual_doc_coverage"),
        "type2_comparison_evidence_coverage": first(type2_cov, "comparison_evidence_coverage"),
        "table_plus_body_coverage": first(overall_cov, "table_plus_body_coverage"),
        "fallback_ratio": first(overall_cov, "fallback_ratio"),
        "weak_comparison_count": first(overall_cov, "weak_comparison_count"),
        "type2_weak_comparison_count": first(type2_cov, "weak_comparison_count"),
        "weak_table_body_count": first(overall_cov, "weak_table_body_count"),
    }

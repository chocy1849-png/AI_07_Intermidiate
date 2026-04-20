from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import csv
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from scenario_b_phase2.improved_auto_grader_v41 import GradingResult, NormalizedQBankItem, grade_answer, load_qbank_v4


VERSION_ORDER = [
    "b00_raw_baseline",
    "b01_hybrid",
    "b02_prefix_v2",
    "b06_exact_stage1",
    "phase2_baseline_v2",
    "baseline_v3_ocr_v4_router_off",
]

DISPLAY_NAMES = {
    "b00_raw_baseline": "B-00 raw baseline",
    "b01_hybrid": "B-01",
    "b02_prefix_v2": "B-02",
    "b06_exact_stage1": "B-06 exact",
    "phase2_baseline_v2": "phase2 baseline_v2",
    "baseline_v3_ocr_v4_router_off": "baseline_v3 (OCR v4, router off)",
}

REFERENCE_JUDGE = {
    "b00_raw_baseline": 3.25,
    "b01_hybrid": 3.20,
    "b02_prefix_v2": 2.925,
    "b06_exact_stage1": 3.325,
    "phase2_baseline_v2": 3.225,
    "baseline_v3_ocr_v4_router_off": 3.525,
}

REGRESSION_CASE_IDS = [
    "DOC005_Q001",  # slot_pair 20/70
    "DOC037_Q002",  # list_set alias
    "DOC019_Q004",  # extra explanation
    "DOC020_Q005",  # extra explanation
    "DOC050_Q001",  # choice number+text
    "DOC046_Q001",  # duration tolerant
    "DOC085_Q001",  # list prefix stripping
]


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _to_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "t", "yes", "y"}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _ratio(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return round(sum(1.0 for row in rows if _to_bool(row.get(key))) / float(len(rows)), 4)


def _filter(rows: list[dict[str, Any]], **kwargs: str) -> list[dict[str, Any]]:
    out = rows
    for key, value in kwargs.items():
        out = [row for row in out if str(row.get(key, "")).strip().lower() == value.lower()]
    return out


def _rank_map(score_map: dict[str, float]) -> dict[str, int]:
    ordered = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    return {key: idx + 1 for idx, (key, _) in enumerate(ordered)}


def _spearman(a: dict[str, float], b: dict[str, float]) -> float:
    ra = _rank_map(a)
    rb = _rank_map(b)
    n = len(a)
    if n <= 1:
        return 0.0
    diff_sq = sum((ra[k] - rb[k]) ** 2 for k in a.keys())
    rho = 1.0 - (6.0 * diff_sq) / (n * (n**2 - 1))
    return round(rho, 4)


def _sign(delta: float, eps: float = 1e-9) -> str:
    if delta > eps:
        return "up"
    if delta < -eps:
        return "down"
    return "flat"


def _pairwise_direction(score_map: dict[str, float]) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for idx in range(len(VERSION_ORDER) - 1):
        left = VERSION_ORDER[idx]
        right = VERSION_ORDER[idx + 1]
        out.append((left, right, _sign(score_map[right] - score_map[left])))
    return out


def _direction_agreement(reference: list[tuple[str, str, str]], candidate: list[tuple[str, str, str]]) -> tuple[int, int]:
    ref_map = {(a, b): s for a, b, s in reference}
    cand_map = {(a, b): s for a, b, s in candidate}
    total = 0
    agree = 0
    for pair, ref_sign in ref_map.items():
        if pair not in cand_map:
            continue
        total += 1
        if ref_sign == cand_map[pair]:
            agree += 1
    return agree, total


def _collect_representative_cases(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        old_fail_new_pass = (not _to_bool(row.get("old_is_correct"))) and _to_bool(row.get("new_tolerant_correct"))
        eval_type = str(row.get("answer_eval_type", "")).lower()
        reason = str(row.get("error_reason", "")).strip()
        model_answer = str(row.get("model_answer", "")).strip()
        partial = float(row.get("partial_score") or 0.0)

        if eval_type == "choice" and old_fail_new_pass:
            if re.search(r"\b[1-9]\s*(번|\.|\))", model_answer) or "choice" in str(row.get("matched_mode", "")):
                buckets["multiple_choice_number_text"].append(row)
        if eval_type in {"number", "currency", "date", "duration"} and old_fail_new_pass:
            buckets["numeric_currency_date_format"].append(row)
        if old_fail_new_pass and reason == "extra_explanation":
            buckets["answer_plus_explanation"].append(row)
        if eval_type in {"list_set", "slot_pair"} and 0 < partial < 1:
            buckets["partial_list_or_pair"].append(row)
        if old_fail_new_pass and reason not in {"wrong_value"} and str(row.get("extracted_answer", "")).strip():
            buckets["extraction_fail_to_success"].append(row)

    out: dict[str, list[dict[str, Any]]] = {}
    for key, items in buckets.items():
        picked: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for row in items:
            token = (str(row.get("version_key", "")), str(row.get("id", "")))
            if token in seen:
                continue
            seen.add(token)
            picked.append(row)
            if len(picked) >= 3:
                break
        out[key] = picked
    return out


def _write_design_notes(path: Path) -> None:
    content = "\n".join(
        [
            "# Auto Grader v4.1 Design Notes",
            "",
            f"- generated_at: `{_now()}`",
            "- 목적: tolerant 완화가 아니라 extraction precision 개선",
            "",
            "## Focused Bottlenecks",
            "",
            "1. short_answer long-form extraction",
            "2. slot_pair extraction",
            "3. list_set canonicalization",
            "4. free_string factual answer mining",
            "",
            "## v4.1 Changes",
            "",
            "- question-aware mode (`numeric_count`, `noun_phrase`, `slot_pair`, `list_set`)",
            "- heading/section stripping before extraction",
            "- short_answer/free_string answer cue mining + 핵심 구문 추출",
            "- slot_pair label-aware extraction (`정량/정성`, `기술/가격`) + noisy fallback 억제",
            "- list_set alias dictionary + prefix/suffix stripping + unordered set coverage",
            "- regression case set(7개) 고정",
            "",
            "## Non-goals (This Round)",
            "",
            "- qbank 수정 없음",
            "- LLM Judge 재실행 없음",
            "- soft crag/router/OCR 추가 실험 없음",
        ]
    )
    path.write_text(content, encoding="utf-8")


def _write_trend_report(path: Path, compare_rows: list[dict[str, Any]]) -> None:
    old_map = {row["version_key"]: float(row["old_auto_overall"]) for row in compare_rows}
    strict_map = {row["version_key"]: float(row["new_strict_auto_overall"]) for row in compare_rows}
    tol_map = {row["version_key"]: float(row["new_tolerant_auto_overall"]) for row in compare_rows}
    judge_map = dict(REFERENCE_JUDGE)

    old_rank = _rank_map(old_map)
    tol_rank = _rank_map(tol_map)
    judge_rank = _rank_map(judge_map)

    old_adj = _pairwise_direction(old_map)
    tol_adj = _pairwise_direction(tol_map)
    judge_adj = _pairwise_direction(judge_map)
    old_agree, old_total = _direction_agreement(judge_adj, old_adj)
    tol_agree, tol_total = _direction_agreement(judge_adj, tol_adj)

    post_pairs = [
        ("b02_prefix_v2", "b06_exact_stage1"),
        ("b06_exact_stage1", "phase2_baseline_v2"),
        ("phase2_baseline_v2", "baseline_v3_ocr_v4_router_off"),
    ]
    judge_post = [(a, b, _sign(judge_map[b] - judge_map[a])) for a, b in post_pairs]
    old_post = [(a, b, _sign(old_map[b] - old_map[a])) for a, b in post_pairs]
    tol_post = [(a, b, _sign(tol_map[b] - tol_map[a])) for a, b in post_pairs]
    old_post_agree, post_total = _direction_agreement(judge_post, old_post)
    tol_post_agree, _ = _direction_agreement(judge_post, tol_post)

    lines = [
        "# Auto Trend vs Reference Judge Report (v4.1)",
        "",
        f"- generated_at: `{_now()}`",
        "- reference judge trend: fixed constants (no judge rerun)",
        "",
        "## Rank Alignment",
        "",
        f"- Spearman(old_auto vs judge_ref): **{_spearman(old_map, judge_map)}**",
        f"- Spearman(new_strict vs judge_ref): **{_spearman(strict_map, judge_map)}**",
        f"- Spearman(new_tolerant vs judge_ref): **{_spearman(tol_map, judge_map)}**",
        "",
        "| version | judge_rank | old_auto_rank | new_tolerant_rank |",
        "|---|---:|---:|---:|",
    ]
    for key in VERSION_ORDER:
        lines.append(f"| {key} | {judge_rank[key]} | {old_rank[key]} | {tol_rank[key]} |")

    lines.extend(
        [
            "",
            "## Short-answer / Intermediate",
            "",
            "| version | old_short | new_tolerant_short | delta_short | old_intermediate | new_tolerant_intermediate | delta_intermediate |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in compare_rows:
        old_short = float(row["old_short_answer"])
        new_short = float(row["new_tolerant_short_answer"])
        old_inter = float(row["old_intermediate"])
        new_inter = float(row["new_tolerant_intermediate"])
        lines.append(
            f"| {row['version_key']} | {old_short:.4f} | {new_short:.4f} | {(new_short-old_short):.4f} | "
            f"{old_inter:.4f} | {new_inter:.4f} | {(new_inter-old_inter):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Direction Alignment",
            "",
            "| pair | judge | old_auto | new_tolerant |",
            "|---|---|---|---|",
        ]
    )
    judge_post_map = {(a, b): s for a, b, s in judge_post}
    old_post_map = {(a, b): s for a, b, s in old_post}
    tol_post_map = {(a, b): s for a, b, s in tol_post}
    for a, b in post_pairs:
        lines.append(f"| {a} -> {b} | {judge_post_map[(a,b)]} | {old_post_map[(a,b)]} | {tol_post_map[(a,b)]} |")

    lines.extend(
        [
            "",
            f"- Pairwise direction agreement: **{old_agree}/{old_total} -> {tol_agree}/{tol_total}**",
            f"- Post-stage(B-02 이후) agreement: **{old_post_agree}/{post_total} -> {tol_post_agree}/{post_total}**",
            "",
            "## Fairness Check",
            "",
            f"- baseline_v3 old_auto: **{old_map['baseline_v3_ocr_v4_router_off']:.4f}**",
            f"- baseline_v3 new_tolerant: **{tol_map['baseline_v3_ocr_v4_router_off']:.4f}**",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_representative_cases(path: Path, buckets: dict[str, list[dict[str, Any]]]) -> None:
    label = {
        "multiple_choice_number_text": "multiple_choice 번호/텍스트 혼용",
        "numeric_currency_date_format": "숫자/금액/날짜 표기 차이",
        "answer_plus_explanation": "정답 포함 + 부연설명",
        "partial_list_or_pair": "목록형/slot_pair 일부 정답",
        "extraction_fail_to_success": "extraction_fail -> extraction_success",
    }
    lines = [
        "# Representative Error Cases v4.1",
        "",
        f"- generated_at: `{_now()}`",
        "- before: old auto",
        "- after: improved auto v4.1",
    ]
    order = [
        "multiple_choice_number_text",
        "numeric_currency_date_format",
        "answer_plus_explanation",
        "partial_list_or_pair",
        "extraction_fail_to_success",
    ]
    for key in order:
        rows = buckets.get(key, [])
        lines.extend(
            [
                "",
                f"## {label[key]}",
                "",
                "| version | id | eval_type | old | strict | tolerant | partial | matched_mode | error_reason |",
                "|---|---|---|---:|---:|---:|---:|---|---|",
            ]
        )
        if not rows:
            lines.append("| (none) | - | - | - | - | - | - | - | - |")
            continue
        for row in rows:
            lines.append(
                f"| {row.get('version_key','')} | {row.get('id','')} | {row.get('answer_eval_type','')} | "
                f"{row.get('old_is_correct','')} | {row.get('new_strict_correct','')} | {row.get('new_tolerant_correct','')} | "
                f"{row.get('partial_score','')} | {row.get('matched_mode','')} | {row.get('error_reason','')} |"
            )
            lines.append(f"- Q: {row.get('question','')}")
            lines.append(f"- GT: {row.get('canonical_expected','')}")
            lines.append(f"- Pred: {row.get('model_answer','')}")
            lines.append(f"- Extracted: {row.get('extracted_answer','')}")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_regression_results(path: Path, all_rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in all_rows:
        key = (str(row.get("version_key", "")), str(row.get("id", "")))
        grouped[key] = row

    lines = [
        "# Regression Case Results v4.1",
        "",
        f"- generated_at: `{_now()}`",
        "- fixed case set:",
        *(f"  - {cid}" for cid in REGRESSION_CASE_IDS),
        "",
    ]

    for cid in REGRESSION_CASE_IDS:
        lines.extend(
            [
                f"## {cid}",
                "",
                "| version | old | strict | tolerant | partial | matched_mode | error_reason | extracted |",
                "|---|---:|---:|---:|---:|---|---|---|",
            ]
        )
        for version in VERSION_ORDER:
            row = grouped.get((version, cid))
            if not row:
                lines.append(f"| {version} | - | - | - | - | - | - | - |")
                continue
            lines.append(
                f"| {version} | {row.get('old_is_correct','')} | {row.get('new_strict_correct','')} | "
                f"{row.get('new_tolerant_correct','')} | {row.get('partial_score','')} | {row.get('matched_mode','')} | "
                f"{row.get('error_reason','')} | {str(row.get('extracted_answer','')).replace('|','/')} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run(downloads_root: Path) -> None:
    project_root = next(
        path for path in downloads_root.iterdir() if path.is_dir() and (path / "src" / "scenario_a" / "common_pipeline.py").exists()
    )
    eval_root = project_root / "rag_outputs" / "eval_pipeline"
    qbank_path = project_root / "src" / "PartA_RFP_AutoGrading_QBank_v4_reviewed.json"
    if not qbank_path.exists():
        raise FileNotFoundError(f"qbank v4 not found: {qbank_path}")

    item_map = load_qbank_v4(qbank_path)
    compare_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []

    for version_key in VERSION_ORDER:
        source = eval_root / version_key / "auto_eval_detail.csv"
        if not source.exists():
            raise FileNotFoundError(f"missing auto_eval_detail: {source}")
        base_rows = _read_csv(source)
        version_rows: list[dict[str, Any]] = []

        for base in base_rows:
            item_id = str(base.get("id", "")).strip()
            item: NormalizedQBankItem | None = item_map.get(item_id)
            if item is None:
                continue
            model_answer = str(base.get("model_answer", ""))
            graded: GradingResult = grade_answer(model_answer, item)
            row = {
                "version_key": version_key,
                "id": item_id,
                "question": item.question,
                "question_type": item.question_type,
                "difficulty": item.difficulty,
                "answer_eval_type": item.answer_eval_type,
                "scoring_mode": item.scoring_mode,
                "canonical_expected": graded.canonical_expected,
                "model_answer": model_answer,
                "old_is_correct": _to_bool(base.get("is_correct")),
                **graded.to_row(),
                "new_strict_correct": graded.strict_correct,
                "new_tolerant_correct": graded.tolerant_correct,
            }
            version_rows.append(row)
            all_rows.append(row)

        detail_path = eval_root / version_key / "auto_eval_detail_improved_v41.csv"
        _write_csv(detail_path, version_rows)

        old_overall = _ratio(version_rows, "old_is_correct")
        strict_overall = _ratio(version_rows, "new_strict_correct")
        tolerant_overall = _ratio(version_rows, "new_tolerant_correct")

        row = {
            "version_key": version_key,
            "display_name": DISPLAY_NAMES[version_key],
            "reference_judge_trend": REFERENCE_JUDGE[version_key],
            "old_auto_overall": old_overall,
            "new_strict_auto_overall": strict_overall,
            "new_tolerant_auto_overall": tolerant_overall,
            "delta_strict_vs_old": round(strict_overall - old_overall, 4),
            "delta_tolerant_vs_old": round(tolerant_overall - old_overall, 4),
            "old_multiple_choice": _ratio(_filter(version_rows, question_type="multiple_choice"), "old_is_correct"),
            "new_strict_multiple_choice": _ratio(_filter(version_rows, question_type="multiple_choice"), "new_strict_correct"),
            "new_tolerant_multiple_choice": _ratio(_filter(version_rows, question_type="multiple_choice"), "new_tolerant_correct"),
            "old_short_answer": _ratio(_filter(version_rows, question_type="short_answer"), "old_is_correct"),
            "new_strict_short_answer": _ratio(_filter(version_rows, question_type="short_answer"), "new_strict_correct"),
            "new_tolerant_short_answer": _ratio(_filter(version_rows, question_type="short_answer"), "new_tolerant_correct"),
            "old_basic": _ratio(_filter(version_rows, difficulty="basic"), "old_is_correct"),
            "new_strict_basic": _ratio(_filter(version_rows, difficulty="basic"), "new_strict_correct"),
            "new_tolerant_basic": _ratio(_filter(version_rows, difficulty="basic"), "new_tolerant_correct"),
            "old_intermediate": _ratio(_filter(version_rows, difficulty="intermediate"), "old_is_correct"),
            "new_strict_intermediate": _ratio(_filter(version_rows, difficulty="intermediate"), "new_strict_correct"),
            "new_tolerant_intermediate": _ratio(_filter(version_rows, difficulty="intermediate"), "new_tolerant_correct"),
            "old_advanced": _ratio(_filter(version_rows, difficulty="advanced"), "old_is_correct"),
            "new_strict_advanced": _ratio(_filter(version_rows, difficulty="advanced"), "new_strict_correct"),
            "new_tolerant_advanced": _ratio(_filter(version_rows, difficulty="advanced"), "new_tolerant_correct"),
        }
        compare_rows.append(row)
        print(
            f"[DONE] {version_key} old={old_overall:.4f} strict={strict_overall:.4f} tolerant={tolerant_overall:.4f}"
        )

    _write_csv(eval_root / "auto_grader_v41_compare.csv", compare_rows)
    _write_design_notes(eval_root / "auto_grader_v41_design_notes.md")
    _write_trend_report(eval_root / "auto_trend_vs_reference_judge_report_v41.md", compare_rows)
    _write_representative_cases(eval_root / "representative_error_cases_v41.md", _collect_representative_cases(all_rows))
    _write_regression_results(eval_root / "regression_case_results_v41.md", all_rows)

    print(
        "[SUMMARY]",
        {
            "generated_at": _now(),
            "versions": len(compare_rows),
            "tolerant_mean": round(statistics.mean([float(r["new_tolerant_auto_overall"]) for r in compare_rows]), 4),
            "output_root": str(eval_root),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run improved auto grader v4.1 using qbank v4 (no judge rerun).")
    parser.add_argument("--downloads-root", default=str(Path.home() / "Downloads"))
    args = parser.parse_args()
    run(Path(args.downloads_root))


if __name__ == "__main__":
    main()

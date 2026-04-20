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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scenario_b_phase2.improved_auto_grader import GradingResult, NormalizedQBankItem, grade_answer, load_qbank_v4


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


@dataclass
class VersionAggregation:
    version_key: str
    old_overall: float
    strict_overall: float
    tolerant_overall: float
    old_multiple_choice: float
    strict_multiple_choice: float
    tolerant_multiple_choice: float
    old_short_answer: float
    strict_short_answer: float
    tolerant_short_answer: float
    old_basic: float
    strict_basic: float
    tolerant_basic: float
    old_intermediate: float
    strict_intermediate: float
    tolerant_intermediate: float
    old_advanced: float
    strict_advanced: float
    tolerant_advanced: float


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


def _aggregate_version(rows: list[dict[str, Any]], version_key: str) -> VersionAggregation:
    def filt(**kwargs: str) -> list[dict[str, Any]]:
        out = rows
        for k, v in kwargs.items():
            out = [row for row in out if str(row.get(k, "")).strip().lower() == v.lower()]
        return out

    return VersionAggregation(
        version_key=version_key,
        old_overall=_ratio(rows, "old_is_correct"),
        strict_overall=_ratio(rows, "new_strict_correct"),
        tolerant_overall=_ratio(rows, "new_tolerant_correct"),
        old_multiple_choice=_ratio(filt(question_type="multiple_choice"), "old_is_correct"),
        strict_multiple_choice=_ratio(filt(question_type="multiple_choice"), "new_strict_correct"),
        tolerant_multiple_choice=_ratio(filt(question_type="multiple_choice"), "new_tolerant_correct"),
        old_short_answer=_ratio(filt(question_type="short_answer"), "old_is_correct"),
        strict_short_answer=_ratio(filt(question_type="short_answer"), "new_strict_correct"),
        tolerant_short_answer=_ratio(filt(question_type="short_answer"), "new_tolerant_correct"),
        old_basic=_ratio(filt(difficulty="basic"), "old_is_correct"),
        strict_basic=_ratio(filt(difficulty="basic"), "new_strict_correct"),
        tolerant_basic=_ratio(filt(difficulty="basic"), "new_tolerant_correct"),
        old_intermediate=_ratio(filt(difficulty="intermediate"), "old_is_correct"),
        strict_intermediate=_ratio(filt(difficulty="intermediate"), "new_strict_correct"),
        tolerant_intermediate=_ratio(filt(difficulty="intermediate"), "new_tolerant_correct"),
        old_advanced=_ratio(filt(difficulty="advanced"), "old_is_correct"),
        strict_advanced=_ratio(filt(difficulty="advanced"), "new_strict_correct"),
        tolerant_advanced=_ratio(filt(difficulty="advanced"), "new_tolerant_correct"),
    )


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
    for i in range(len(VERSION_ORDER) - 1):
        left = VERSION_ORDER[i]
        right = VERSION_ORDER[i + 1]
        delta = score_map[right] - score_map[left]
        out.append((left, right, _sign(delta)))
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


def _collect_representative_cases(all_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in all_rows:
        old_fail_new_pass = (not _to_bool(row.get("old_is_correct"))) and _to_bool(row.get("new_tolerant_correct"))
        eval_type = str(row.get("answer_eval_type", "")).strip().lower()
        model_answer = str(row.get("model_answer", "")).strip()
        matched_mode = str(row.get("matched_mode", "")).strip()
        reason = str(row.get("error_reason", "")).strip()
        partial = float(row.get("partial_score") or 0.0)

        if eval_type == "choice" and old_fail_new_pass:
            if re.search(r"\b[1-9]\s*(번|\.|\))", model_answer) or "choice" in matched_mode:
                categories["multiple_choice_번호_텍스트혼용"].append(row)

        if eval_type in {"number", "currency", "date", "duration"} and old_fail_new_pass:
            categories["숫자_금액_날짜_표기차이"].append(row)

        if old_fail_new_pass and reason == "extra_explanation":
            categories["정답포함_부연설명"].append(row)

        if eval_type in {"list_set", "slot_pair"} and 0 < partial < 1:
            categories["목록형_일부정답"].append(row)

        if old_fail_new_pass and matched_mode not in {"no_match", ""} and str(row.get("extracted_answer", "")).strip():
            categories["extraction_fail_to_success"].append(row)

    # ensure minimum 3 rows per category where possible
    for key in list(categories.keys()):
        seen: set[tuple[str, str]] = set()
        dedup: list[dict[str, Any]] = []
        for row in categories[key]:
            token = (str(row.get("version_key", "")), str(row.get("id", "")))
            if token in seen:
                continue
            seen.add(token)
            dedup.append(row)
            if len(dedup) >= 3:
                break
        categories[key] = dedup
    return categories


def _build_schema_notes_md(path: Path) -> None:
    lines = [
        "# Auto Grader v4 Schema Notes",
        "",
        f"- generated_at: `{_now()}`",
        "- source_of_truth: `PartA_RFP_AutoGrading_QBank_v4_reviewed.json`",
        "",
        "## Loader Normalization",
        "",
        "- `question_type`, `answer_format`, `answer_eval_type`, `canonical_answer`, `answer_aliases`, `choice_index`, `scoring_mode`, `choices`를 정규화합니다.",
        "- `answer_eval_type` 변환:",
        "  - `string -> free_string`",
        "  - `list -> list_set`",
        "  - `number_pair -> slot_pair`",
        "",
        "## Extraction Layer",
        "",
        "- choice: 번호/텍스트/번호+텍스트 혼용 파싱",
        "- number/currency/date/duration: regex 기반 핵심 span 추출 + 정규화",
        "- email/phone/url: 표준 패턴 파싱 후 normalize",
        "- list_set/slot_pair: split + set/pair 기반 비교",
        "- free_string: 첫 문장 추출 + 전체 answer contains/alias 허용",
        "",
        "## Scoring",
        "",
        "- strict: 정규화 exact 또는 구조적 exact",
        "- tolerant: alias/contains/순서무시(set) 허용",
        "- partial_score: list/slot 계열에서 coverage 산출",
        "- error_reason: `wrong_value`, `formatting_only`, `extra_explanation`, `partial_list`, `extraction_fail`, `ok`",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_trend_report(
    path: Path,
    compare_rows: list[dict[str, Any]],
) -> None:
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

    post_b02_pairs = [("b02_prefix_v2", "b06_exact_stage1"), ("b06_exact_stage1", "phase2_baseline_v2"), ("phase2_baseline_v2", "baseline_v3_ocr_v4_router_off")]
    judge_post = [(a, b, _sign(judge_map[b] - judge_map[a])) for a, b in post_b02_pairs]
    old_post = [(a, b, _sign(old_map[b] - old_map[a])) for a, b in post_b02_pairs]
    tol_post = [(a, b, _sign(tol_map[b] - tol_map[a])) for a, b in post_b02_pairs]
    old_post_agree, post_total = _direction_agreement(judge_post, old_post)
    tol_post_agree, _ = _direction_agreement(judge_post, tol_post)

    lines = [
        "# Auto Trend vs Reference Judge (No Re-run)",
        "",
        f"- generated_at: `{_now()}`",
        "- reference judge trend: fixed input values (no re-execution)",
        "- objective: old auto 대비 new strict/new tolerant의 흐름 정렬 개선 여부 확인",
        "",
        "## 1) Rank Alignment",
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
            "## 1-1) Intermediate / Short-Answer Improvement",
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
            "## 2) Monotonic Tendency Alignment (Post B-02)",
            "",
            "| pair | judge_direction | old_auto_direction | new_tolerant_direction |",
            "|---|---|---|---|",
        ]
    )
    judge_post_map = {(a, b): s for a, b, s in judge_post}
    old_post_map = {(a, b): s for a, b, s in old_post}
    tol_post_map = {(a, b): s for a, b, s in tol_post}
    for a, b in post_b02_pairs:
        lines.append(f"| {a} -> {b} | {judge_post_map[(a,b)]} | {old_post_map[(a,b)]} | {tol_post_map[(a,b)]} |")

    lines.extend(
        [
            "",
            f"- Post B-02 direction agreement (old_auto): **{old_post_agree}/{post_total}**",
            f"- Post B-02 direction agreement (new_tolerant): **{tol_post_agree}/{post_total}**",
            "",
            "## 3) Pairwise Direction Agreement (All Adjacent Steps)",
            "",
            f"- old_auto agreement: **{old_agree}/{old_total}**",
            f"- new_tolerant agreement: **{tol_agree}/{tol_total}**",
            "",
            "## 4) Fairness Focus",
            "",
            f"- baseline_v3 old_auto: **{old_map['baseline_v3_ocr_v4_router_off']:.4f}**",
            f"- baseline_v3 new_tolerant: **{tol_map['baseline_v3_ocr_v4_router_off']:.4f}**",
            "- judge reference 상 baseline_v3가 최상위이므로, tolerant auto에서 하위 고정이 완화되었는지 확인합니다.",
            "",
            "## 5) Conclusion (This Round)",
            "",
            f"- rank alignment improvement (Spearman): **{_spearman(old_map, judge_map)} -> {_spearman(tol_map, judge_map)}**",
            f"- adjacent direction agreement: **{old_agree}/{old_total} -> {tol_agree}/{tol_total}**",
            "- 해석: tolerant 점수는 문항 포맷 공정성(표기 차이/부연설명 허용)은 개선했으나, Judge 참조 추세와의 방향 일치는 제한적입니다.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_error_cases_md(path: Path, categories: dict[str, list[dict[str, Any]]]) -> None:
    title_map = {
        "multiple_choice_번호_텍스트혼용": "multiple_choice 번호/텍스트 혼용",
        "숫자_금액_날짜_표기차이": "숫자/금액/날짜 표기 차이",
        "정답포함_부연설명": "정답 포함 + 부연설명",
        "목록형_일부정답": "목록형 일부 정답",
        "extraction_fail_to_success": "extraction fail -> extraction success",
    }
    lines = [
        "# Representative Error Cases (Before/After)",
        "",
        f"- generated_at: `{_now()}`",
        "- before: old auto (`is_correct`)",
        "- after: improved auto (`strict/tolerant/partial`)",
    ]
    for key in [
        "multiple_choice_번호_텍스트혼용",
        "숫자_금액_날짜_표기차이",
        "정답포함_부연설명",
        "목록형_일부정답",
        "extraction_fail_to_success",
    ]:
        rows = categories.get(key, [])
        lines.extend(
            [
                "",
                f"## {title_map[key]}",
                "",
                "| version | id | eval_type | old_is_correct | strict | tolerant | partial | matched_mode | error_reason |",
                "|---|---|---|---:|---:|---:|---:|---|---|",
            ]
        )
        if not rows:
            lines.append("| (none) | - | - | - | - | - | - | - | - |")
            continue
        for row in rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("version_key", "")),
                        str(row.get("id", "")),
                        str(row.get("answer_eval_type", "")),
                        str(row.get("old_is_correct", "")),
                        str(row.get("new_strict_correct", "")),
                        str(row.get("new_tolerant_correct", "")),
                        str(row.get("partial_score", "")),
                        str(row.get("matched_mode", "")),
                        str(row.get("error_reason", "")),
                    ]
                )
                + " |"
            )
            lines.append("")
            lines.append(f"- Q: {row.get('question','')}")
            lines.append(f"- GT: {row.get('canonical_expected','')}")
            lines.append(f"- Pred(raw): {row.get('model_answer','')}")
            lines.append(f"- Extracted: {row.get('extracted_answer','')}")

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
    all_evaluated_rows: list[dict[str, Any]] = []

    for version_key in VERSION_ORDER:
        detail_path = eval_root / version_key / "auto_eval_detail.csv"
        if not detail_path.exists():
            raise FileNotFoundError(f"auto_eval_detail missing: {detail_path}")
        raw_rows = _read_csv(detail_path)
        eval_rows: list[dict[str, Any]] = []
        for row in raw_rows:
            item_id = str(row.get("id", "")).strip()
            if item_id not in item_map:
                continue
            item: NormalizedQBankItem = item_map[item_id]
            model_answer = str(row.get("model_answer", ""))
            graded: GradingResult = grade_answer(model_answer, item)
            merged = {
                "version_key": version_key,
                "id": item_id,
                "question": item.question,
                "question_type": item.question_type,
                "difficulty": item.difficulty,
                "answer_eval_type": item.answer_eval_type,
                "scoring_mode": item.scoring_mode,
                "canonical_expected": graded.canonical_expected,
                "model_answer": model_answer,
                "old_is_correct": _to_bool(row.get("is_correct")),
                **graded.to_row(),
                "new_strict_correct": graded.strict_correct,
                "new_tolerant_correct": graded.tolerant_correct,
            }
            eval_rows.append(merged)
            all_evaluated_rows.append(merged)

        if len(eval_rows) != len(item_map):
            missing = len(item_map) - len(eval_rows)
            if missing > 0:
                print(f"[WARN] {version_key}: missing {missing} ids from qbank v4 mapping.")

        detail_out = eval_root / version_key / "auto_eval_detail_improved_v4.csv"
        _write_csv(detail_out, eval_rows)
        agg = _aggregate_version(eval_rows, version_key)
        compare_rows.append(
            {
                "version_key": version_key,
                "display_name": DISPLAY_NAMES[version_key],
                "reference_judge_trend": REFERENCE_JUDGE[version_key],
                "old_auto_overall": agg.old_overall,
                "new_strict_auto_overall": agg.strict_overall,
                "new_tolerant_auto_overall": agg.tolerant_overall,
                "delta_strict_vs_old": round(agg.strict_overall - agg.old_overall, 4),
                "delta_tolerant_vs_old": round(agg.tolerant_overall - agg.old_overall, 4),
                "old_multiple_choice": agg.old_multiple_choice,
                "new_strict_multiple_choice": agg.strict_multiple_choice,
                "new_tolerant_multiple_choice": agg.tolerant_multiple_choice,
                "old_short_answer": agg.old_short_answer,
                "new_strict_short_answer": agg.strict_short_answer,
                "new_tolerant_short_answer": agg.tolerant_short_answer,
                "old_basic": agg.old_basic,
                "new_strict_basic": agg.strict_basic,
                "new_tolerant_basic": agg.tolerant_basic,
                "old_intermediate": agg.old_intermediate,
                "new_strict_intermediate": agg.strict_intermediate,
                "new_tolerant_intermediate": agg.tolerant_intermediate,
                "old_advanced": agg.old_advanced,
                "new_strict_advanced": agg.strict_advanced,
                "new_tolerant_advanced": agg.tolerant_advanced,
            }
        )
        print(
            f"[DONE] {version_key} old={agg.old_overall:.4f} strict={agg.strict_overall:.4f} tolerant={agg.tolerant_overall:.4f}"
        )

    # write required artifacts
    _write_csv(eval_root / "auto_grader_old_vs_new_compare.csv", compare_rows)
    _build_schema_notes_md(eval_root / "auto_grader_v4_schema_notes.md")
    _build_trend_report(eval_root / "auto_trend_vs_reference_judge_report.md", compare_rows)
    cases = _collect_representative_cases(all_evaluated_rows)
    _build_error_cases_md(eval_root / "representative_error_cases.md", cases)

    # concise execution summary
    tolerant_scores = [float(row["new_tolerant_auto_overall"]) for row in compare_rows]
    print(
        "[SUMMARY]",
        {
            "generated_at": _now(),
            "versions": len(compare_rows),
            "tolerant_mean": round(statistics.mean(tolerant_scores), 4) if tolerant_scores else 0.0,
            "output_root": str(eval_root),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run improved auto grader(v4) on existing auto_eval_detail.csv (no judge rerun).")
    parser.add_argument("--downloads-root", default=str(Path.home() / "Downloads"))
    args = parser.parse_args()
    run(Path(args.downloads_root))


if __name__ == "__main__":
    main()

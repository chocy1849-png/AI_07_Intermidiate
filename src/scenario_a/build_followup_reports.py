from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class RunSpec:
    key: str
    display_name: str
    run_dir: Path


def _read_summary(run_dir: Path, filename: str) -> pd.DataFrame | None:
    path = run_dir / filename
    if not path.exists():
        return None
    return pd.read_csv(path)


def _get_group_row(df: pd.DataFrame | None, group_name: str) -> dict[str, Any]:
    if df is None or df.empty:
        return {}
    sub = df.loc[df["group_name"] == group_name]
    if sub.empty:
        return {}
    row = sub.iloc[0].to_dict()
    return {k: v for k, v in row.items() if pd.notna(v)}


def _collect_run_metrics(spec: RunSpec) -> dict[str, Any]:
    manual = _read_summary(spec.run_dir, "baseline_eval_manual_summary.csv")
    auto = _read_summary(spec.run_dir, "baseline_eval_summary.csv")
    overall_manual = _get_group_row(manual, "overall")
    overall_auto = _get_group_row(auto, "overall")
    type2_manual = _get_group_row(manual, "TYPE 2")
    type4_manual = _get_group_row(manual, "TYPE 4")
    comparison_manual = _get_group_row(manual, "answer_type:comparison")
    rejection_manual = _get_group_row(manual, "answer_type:rejection")
    row: dict[str, Any] = {
        "run_key": spec.key,
        "display_name": spec.display_name,
        "run_dir": str(spec.run_dir),
    }
    for src, prefix in [
        (overall_manual, ""),
        (overall_auto, "auto_"),
        (type2_manual, "type2_"),
        (type4_manual, "type4_"),
        (comparison_manual, "comparison_"),
        (rejection_manual, "rejection_"),
    ]:
        for key, value in src.items():
            if key in {"group_name"}:
                continue
            row[f"{prefix}{key}"] = value
    row["status"] = "ok" if overall_manual else "missing"
    return row


def _write_compare(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _build_examples(run_dir: Path, out_path: Path, title: str) -> None:
    completed_path = run_dir / "baseline_eval_manual_completed.csv"
    if not completed_path.exists():
        out_path.write_text(f"# {title}\n\n- baseline_eval_manual_completed.csv not found.\n", encoding="utf-8")
        return
    df = pd.read_csv(completed_path)
    df["manual_mean"] = df[["faithfulness_score", "completeness_score", "groundedness_score", "relevancy_score"]].mean(axis=1)
    worst = df.sort_values(["manual_mean", "completeness_score", "relevancy_score"], ascending=[True, True, True]).head(3)
    best = df.sort_values(["manual_mean", "faithfulness_score"], ascending=[False, False]).head(3)

    lines = [f"# {title}", ""]
    for section_name, section_df in [("Worst 3", worst), ("Best 3", best)]:
        lines.append(f"## {section_name}")
        lines.append("")
        for row in section_df.itertuples(index=False):
            lines.append(f"### {row.question_id}")
            lines.append(f"- answer_type: {row.answer_type}")
            lines.append(f"- question: {row.question}")
            lines.append(f"- manual_mean: {row.manual_mean:.4f}")
            lines.append(
                f"- scores: faithfulness={row.faithfulness_score}, completeness={row.completeness_score}, groundedness={row.groundedness_score}, relevancy={row.relevancy_score}"
            )
            lines.append(f"- source_docs: {row.source_docs}")
            lines.append(f"- evaluator_note: {row.evaluator_note}")
            answer = str(row.answer_text).replace("\r\n", "\n").strip()
            lines.append("- answer:")
            lines.append("```text")
            lines.append(answer[:1500])
            lines.append("```")
            lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _collect_screening_rows(output_root: Path) -> list[dict[str, Any]]:
    path = output_root / "scenario_a_screening_compare.csv"
    if not path.exists():
        return []
    rows = pd.read_csv(path).to_dict(orient="records")
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        cleaned.append({k: v for k, v in row.items() if pd.notna(v)})
    return cleaned


def _write_summary_markdown(output_root: Path, compare_rows: dict[str, list[dict[str, Any]]]) -> None:
    path = output_root / "scenario_a_ablation_summary.md"
    lines = ["# Scenario A Ablation Summary", ""]

    def _find_run(rows: list[dict[str, Any]], run_key: str) -> dict[str, Any]:
        for row in rows:
            if row.get("run_key") == run_key:
                return row
        return {}

    compare_a = compare_rows.get("Ablation A", [])
    compare_b = compare_rows.get("Ablation B", [])
    compare_c = compare_rows.get("Ablation C", [])
    compare_g = compare_rows.get("Gemma4 Baseline", [])
    screening_g = compare_rows.get("Gemma4 Screening", [])

    a00 = _find_run(compare_a, "koe5_qwen")
    op = _find_run(compare_a, "operational_gpt5mini")
    ceiling = _find_run(compare_a, "operational_gpt5")
    ab_a = _find_run(compare_a, "koe5_gpt5mini")
    ab_b = _find_run(compare_b, "openai_qwen")
    ab_c = _find_run(compare_c, "bgem3_qwen")
    gemma = _find_run(compare_g, "gemma4_e4b")

    lines.extend(
        [
            "## Executive Summary",
            "",
            f"- operational baseline: `B-06 adopted + gpt-5-mini` manual mean `{op.get('avg_manual_eval_score', 'n/a')}`",
            f"- operational ceiling: `B-06 adopted + gpt-5` manual mean `{ceiling.get('avg_manual_eval_score', 'n/a')}`",
            f"- A-00 (`KoE5 + Qwen`) manual mean `{a00.get('avg_manual_eval_score', 'n/a')}`",
            f"- Ablation A (`KoE5 + gpt-5-mini`) manual mean `{ab_a.get('avg_manual_eval_score', 'n/a')}`",
            f"- Ablation B (`OpenAI embedding + Qwen`) manual mean `{ab_b.get('avg_manual_eval_score', 'n/a')}`",
            f"- Ablation C (`bge-m3 + Qwen`) manual mean `{ab_c.get('avg_manual_eval_score', 'n/a')}`",
            f"- Gemma4-E4B (`KoE5 + Gemma4-E4B`) manual mean `{gemma.get('avg_manual_eval_score', 'n/a')}`",
            "",
            "## Interpretation",
            "",
            f"- retrieval/embedding-only swap to `KoE5 + gpt-5-mini` did not recover the gap to the operational baseline. Manual mean dropped to `{ab_a.get('avg_manual_eval_score', 'n/a')}` and comparison completeness fell to `{ab_a.get('comparison_avg_completeness_score', 'n/a')}`.",
            f"- generator-only swap to `OpenAI embedding + Qwen` improved over `A-00` (`{a00.get('avg_manual_eval_score', 'n/a')} -> {ab_b.get('avg_manual_eval_score', 'n/a')}`) but still stayed below the operational baseline.",
            f"- embedding ablation with `bge-m3 + Qwen` is the best pure local-HF variant so far (`{ab_c.get('avg_manual_eval_score', 'n/a')}`), improving `A-00` and especially `TYPE 2` (`{a00.get('type2_avg_manual_eval_score', 'n/a')} -> {ab_c.get('type2_avg_manual_eval_score', 'n/a')}`).",
            f"- real `Gemma4-E4B` outperformed `A-00 KoE5 + Qwen` on overall quality (`{a00.get('avg_manual_eval_score', 'n/a')} -> {gemma.get('avg_manual_eval_score', 'n/a')}`) and is essentially tied with the operational `gpt-5-mini` baseline (`{op.get('avg_manual_eval_score', 'n/a')}`), but at higher latency (`{gemma.get('auto_avg_elapsed_sec', 'n/a')}s`).",
            f"- `TYPE 2/comparison` remains the main bottleneck across all local-HF variants. Best observed `TYPE 2` manual mean in this round is `Gemma4-E4B` at `{gemma.get('type2_avg_manual_eval_score', 'n/a')}`, but completeness is still only `{gemma.get('type2_avg_completeness_score', 'n/a')}`.",
            f"- `TYPE 4/rejection` stayed stable for `Gemma4-E4B` (`{gemma.get('type4_avg_manual_eval_score', 'n/a')}`) and `bge-m3 + Qwen` (`{ab_c.get('type4_avg_manual_eval_score', 'n/a')}`), so the current routing/prompt stack is not collapsing on refusal behavior.",
            "",
            "## FT Decision",
            "",
        ]
    )

    qwen_ft_ok = (
        isinstance(ab_b.get("avg_manual_eval_score"), (int, float))
        and isinstance(ab_c.get("avg_manual_eval_score"), (int, float))
        and ab_b.get("avg_manual_eval_score", 0) >= a00.get("avg_manual_eval_score", 0)
        and ab_c.get("avg_manual_eval_score", 0) >= a00.get("avg_manual_eval_score", 0)
    )
    if qwen_ft_ok:
        lines.append("- recommendation: proceed to Qwen FT. The local-HF path is now stable enough, and both generator-side and embedding-side ablations show recoverable gains over `A-00`.")
    else:
        lines.append("- recommendation: hold FT. The ablation results do not yet support a stable local-HF baseline.")
    lines.extend(
        [
            "- keep `Gemma4-E4B` as a strong inference-only comparison point. It is no longer blocked by runtime support and should stay in the comparison set.",
            "- keep `Gemma4-E2B` only as fallback. It was not needed in this run.",
            "- keep `Gemma4-26B-A4B` in backlog as an inference-only upper reference, not an FT target.",
            "",
        ]
    )

    for title, rows in compare_rows.items():
        lines.append(f"## {title}")
        lines.append("")
        if not rows:
            lines.append("- no rows")
            lines.append("")
            continue
        df = pd.DataFrame(rows)
        if title == "Gemma4 Screening":
            keep_cols = [
                "model_key",
                "screening_backend",
                "smoke_status",
                "mini_status",
                "smoke_avg_latency_sec",
                "mini_avg_latency_sec",
                "smoke_avg_manual_eval_score",
                "mini_avg_manual_eval_score",
                "smoke_rejection_failure_count",
                "mini_rejection_failure_count",
                "skip_reason",
                "reason",
            ]
        else:
            keep_cols = [
                "display_name",
                "status",
                "avg_manual_eval_score",
                "avg_faithfulness_score",
                "avg_completeness_score",
                "avg_groundedness_score",
                "avg_relevancy_score",
                "auto_avg_elapsed_sec",
                "comparison_avg_manual_eval_score",
                "comparison_avg_completeness_score",
                "rejection_avg_manual_eval_score",
                "type2_avg_manual_eval_score",
                "type4_avg_manual_eval_score",
                "skip_reason",
                "reason",
            ]
        keep_cols = [c for c in keep_cols if c in df.columns]
        lines.append(df[keep_cols].to_markdown(index=False))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Build Scenario A follow-up compare/example artifacts")
    parser.add_argument("--output-root", default=str(root / "rag_outputs"))
    args = parser.parse_args()

    output_root = Path(args.output_root)

    operational = RunSpec("operational_gpt5mini", "Operational baseline (B-06 + gpt-5-mini)", output_root / "b06_adopted_eval_gpt5mini")
    ceiling = RunSpec("operational_gpt5", "Operational ceiling (B-06 + gpt-5)", output_root / "b06_adopted_eval_gpt5")
    a00 = RunSpec("koe5_qwen", "A-00 KoE5 + Qwen", output_root / "scenario_a_baseline_koe5_qwen")
    ablation_a = RunSpec("koe5_gpt5mini", "Ablation A KoE5 + gpt-5-mini", output_root / "scenario_a_ablation_koe5_gpt5mini")
    ablation_b = RunSpec("openai_qwen", "Ablation B OpenAI embedding + Qwen", output_root / "scenario_a_ablation_openai_qwen_full")
    ablation_c = RunSpec("bgem3_qwen", "Ablation C bge-m3 + Qwen", output_root / "scenario_a_baseline_bgem3_qwen")
    gemma_e4b = RunSpec("gemma4_e4b", "Gemma4-E4B", output_root / "scenario_a_baseline_koe5_gemma4_e4b")
    gemma_e2b = RunSpec("gemma4_e2b", "Gemma4-E2B (fallback)", output_root / "scenario_a_baseline_koe5_gemma4_e2b")

    compare_a = [_collect_run_metrics(spec) for spec in [operational, ceiling, a00, ablation_a]]
    compare_b = [_collect_run_metrics(spec) for spec in [operational, ceiling, a00, ablation_b]]
    compare_c = [_collect_run_metrics(spec) for spec in [operational, ceiling, a00, ablation_c]]
    compare_gemma = [_collect_run_metrics(spec) for spec in [operational, ceiling, a00, gemma_e4b, gemma_e2b]]
    screening_gemma = _collect_screening_rows(output_root)

    _write_compare(output_root / "scenario_a_ablation_koe5_gpt5mini_compare.csv", compare_a)
    _write_compare(output_root / "scenario_a_ablation_openai_qwen_compare.csv", compare_b)
    _write_compare(output_root / "scenario_a_baseline_bgem3_qwen_compare.csv", compare_c)
    _write_compare(output_root / "scenario_a_baseline_gemma4_compare.csv", compare_gemma)
    _write_compare(output_root / "scenario_a_screening_gemma4_compare.csv", screening_gemma)

    _build_examples(ablation_a.run_dir, output_root / "scenario_a_ablation_koe5_gpt5mini_examples.md", "Ablation A Examples")
    _build_examples(ablation_b.run_dir, output_root / "scenario_a_ablation_openai_qwen_examples.md", "Ablation B Examples")
    _build_examples(ablation_c.run_dir, output_root / "scenario_a_baseline_bgem3_qwen_examples.md", "Ablation C Examples")
    if gemma_e4b.run_dir.exists():
        _build_examples(gemma_e4b.run_dir, output_root / "scenario_a_baseline_gemma4_examples.md", "Gemma4 Examples")
    elif gemma_e2b.run_dir.exists():
        _build_examples(gemma_e2b.run_dir, output_root / "scenario_a_baseline_gemma4_examples.md", "Gemma4 Examples")
    else:
        (output_root / "scenario_a_baseline_gemma4_examples.md").write_text(
            "# Gemma4 Examples\n\n- Gemma 4 baseline output not available yet.\n",
            encoding="utf-8",
        )

    _write_summary_markdown(
        output_root,
        {
            "Ablation A": compare_a,
            "Ablation B": compare_b,
            "Ablation C": compare_c,
            "Gemma4 Baseline": compare_gemma,
            "Gemma4 Screening": screening_gemma,
        },
    )

    print(output_root / "scenario_a_ablation_summary.md")


if __name__ == "__main__":
    main()

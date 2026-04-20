from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

from eval_utils import read_csv, write_csv, write_json

from scenario_a.common_pipeline import PipelinePaths, PipelineSettings, ScenarioACommonPipeline
from scenario_a.eval_runner import read_question_rows, run_eval, screening_pass, write_examples_markdown


def is_model_runnable(pipeline: ScenarioACommonPipeline, model_key: str) -> tuple[bool, str]:
    config = pipeline.load_model_config(model_key)
    if config.model_id.startswith("SET_ME_"):
        return False, "model_id_placeholder"
    if config.runtime == "ollama" and (not config.ollama_model_name or config.ollama_model_name.startswith("SET_ME_")):
        return False, "ollama_model_placeholder"
    return True, ""


def load_existing_screening(output_dir: Path) -> tuple[dict[str, object], dict[str, object]] | None:
    diag_path = output_dir / "screening_diagnostics.json"
    summary_path = output_dir / "baseline_eval_manual_summary.csv"
    if not diag_path.exists() or not summary_path.exists():
        return None
    diagnostics = json.loads(diag_path.read_text(encoding="utf-8"))
    summary_rows = read_csv(summary_path)
    overall = next((row for row in summary_rows if row.get("group_name") == "overall"), {})
    return diagnostics, overall


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Scenario A Step 2A screening runner")
    parser.add_argument("--question-set-path", default=str(root / "docs" / "planning" / "pm" / "day3_partA_eval_questions_v1.txt"))
    parser.add_argument("--smoke-id-file", default=str(root / "docs" / "planning" / "pm" / "day4_smoke_eval_question_ids_v1.txt"))
    parser.add_argument("--mini-id-file", default=str(root / "docs" / "planning" / "pm" / "day4_mini_eval_question_ids_v1.txt"))
    parser.add_argument("--judge-model", default="gpt-5")
    parser.add_argument("--models", nargs="+", default=["qwen", "gemma4_e4b", "exaone"])
    parser.add_argument("--output-root", default=str(root / "rag_outputs"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    smoke_rows = read_question_rows(Path(args.question_set_path), Path(args.smoke_id_file))
    mini_rows = read_question_rows(Path(args.question_set_path), Path(args.mini_id_file))

    compare_rows = []
    examples_lines = ["# Scenario A Screening Examples", ""]

    for model_key in args.models:
        pipeline = ScenarioACommonPipeline(
            PipelinePaths(project_root=root),
            PipelineSettings(embedding_backend_key="openai_text_embedding_3_small"),
        )
        model_dir = output_root / f"scenario_a_screening_{model_key}"
        model_dir.mkdir(parents=True, exist_ok=True)

        runnable, reason = is_model_runnable(pipeline, model_key)
        record = {
            "model_key": model_key,
            "screening_backend": "openai_text_embedding_3_small",
            "smoke_status": "skipped",
            "mini_status": "skipped",
            "skip_reason": reason,
        }

        if not runnable:
            (model_dir / "screening_skip.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            compare_rows.append(record)
            continue

        adapter = pipeline.create_adapter(model_key)
        smoke_dir = model_dir / "smoke"
        try:
            existing_smoke = load_existing_screening(smoke_dir) if args.resume else None
            if existing_smoke is not None:
                smoke_diag, smoke_overall = existing_smoke
                smoke_ok, smoke_reasons = screening_pass(smoke_diag)
            else:
                smoke_output = run_eval(pipeline, adapter, smoke_rows, smoke_dir, judge_model=args.judge_model, run_label="screening_smoke")
                smoke_diag = smoke_output["diagnostics"]
                smoke_ok, smoke_reasons = screening_pass(smoke_diag)
                smoke_overall = next(
                    row
                    for row in read_csv(smoke_dir / "baseline_eval_manual_summary.csv")
                    if row["group_name"] == "overall"
                )
                write_examples_markdown(smoke_dir / "screening_examples.md", f"{model_key} smoke", smoke_diag)
        except Exception as exc:  # noqa: BLE001
            record.update({"smoke_status": "fail", "smoke_fail_reasons": f"runtime_error:{type(exc).__name__}", "skip_reason": str(exc)})
            (model_dir / "screening_runtime_error.json").write_text(
                json.dumps({"stage": "smoke", "model_key": model_key, "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            compare_rows.append(record)
            continue

        record.update(
            {
                "smoke_status": "pass" if smoke_ok else "fail",
                "smoke_question_count": smoke_diag["question_count"],
                "smoke_empty_response_count": smoke_diag["empty_response_count"],
                "smoke_malformed_response_count": smoke_diag["malformed_response_count"],
                "smoke_verbose_response_count": smoke_diag["verbose_response_count"],
                "smoke_speculative_response_count": smoke_diag["speculative_response_count"],
                "smoke_rejection_failure_count": smoke_diag["rejection_failure_count"],
                "smoke_avg_latency_sec": smoke_diag["avg_latency_sec"],
                "smoke_avg_answer_chars": smoke_diag["avg_answer_chars"],
                "smoke_avg_manual_eval_score": smoke_overall.get("avg_manual_eval_score", ""),
                "smoke_fail_reasons": "|".join(smoke_reasons),
            }
        )

        examples_lines.append(f"## {model_key}")
        examples_lines.append("")
        examples_lines.append(f"- smoke_status: {record['smoke_status']}")

        if smoke_ok:
            mini_dir = model_dir / "mini"
            try:
                existing_mini = load_existing_screening(mini_dir) if args.resume else None
                if existing_mini is not None:
                    mini_diag, mini_overall = existing_mini
                    mini_ok, mini_reasons = screening_pass(mini_diag)
                else:
                    mini_output = run_eval(pipeline, adapter, mini_rows, mini_dir, judge_model=args.judge_model, run_label="screening_mini")
                    mini_diag = mini_output["diagnostics"]
                    mini_ok, mini_reasons = screening_pass(mini_diag)
                    mini_overall = next(
                        row
                        for row in read_csv(mini_dir / "baseline_eval_manual_summary.csv")
                        if row["group_name"] == "overall"
                    )
                    write_examples_markdown(mini_dir / "screening_examples.md", f"{model_key} mini", mini_diag)
                record.update(
                    {
                        "mini_status": "pass" if mini_ok else "fail",
                        "mini_question_count": mini_diag["question_count"],
                        "mini_empty_response_count": mini_diag["empty_response_count"],
                        "mini_malformed_response_count": mini_diag["malformed_response_count"],
                        "mini_verbose_response_count": mini_diag["verbose_response_count"],
                        "mini_speculative_response_count": mini_diag["speculative_response_count"],
                        "mini_rejection_failure_count": mini_diag["rejection_failure_count"],
                        "mini_avg_latency_sec": mini_diag["avg_latency_sec"],
                        "mini_avg_answer_chars": mini_diag["avg_answer_chars"],
                        "mini_avg_manual_eval_score": mini_overall.get("avg_manual_eval_score", ""),
                        "mini_fail_reasons": "|".join(mini_reasons),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                record.update(
                    {
                        "mini_status": "fail",
                        "mini_fail_reasons": f"runtime_error:{type(exc).__name__}",
                        "skip_reason": str(exc),
                    }
                )
                (model_dir / "screening_runtime_error.json").write_text(
                    json.dumps({"stage": "mini", "model_key": model_key, "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

        compare_rows.append(record)

    write_csv(output_root / "scenario_a_screening_compare.csv", compare_rows)
    write_json(output_root / "scenario_a_screening_compare.json", {"rows": compare_rows})
    (output_root / "scenario_a_screening_examples.md").write_text("\n".join(examples_lines), encoding="utf-8")
    print(output_root / "scenario_a_screening_compare.csv")


if __name__ == "__main__":
    main()

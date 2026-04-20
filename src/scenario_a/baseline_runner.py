from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import csv
import json
from pathlib import Path

from eval_utils import read_csv, write_json

from scenario_a.common_pipeline import PipelinePaths, PipelineSettings, ScenarioACommonPipeline
from scenario_a.eval_runner import read_question_rows, run_eval


def run_combo(project_root: Path, output_root: Path, embedding_backend: str, model_key: str, question_rows: list[dict[str, str]], judge_model: str) -> dict[str, str]:
    pipeline = ScenarioACommonPipeline(
        PipelinePaths(project_root=project_root),
        PipelineSettings(embedding_backend_key=embedding_backend),
    )
    adapter = pipeline.create_adapter(model_key)
    combo_name = f"{embedding_backend}_{model_key}"
    output_dir = output_root / f"scenario_a_baseline_{combo_name}"
    run_eval(pipeline, adapter, question_rows, output_dir, judge_model=judge_model, run_label=combo_name)
    summary = read_csv(output_dir / "baseline_eval_manual_summary.csv")
    overall = next(row for row in summary if row["group_name"] == "overall")
    return {
        "combo_name": combo_name,
        "embedding_backend": embedding_backend,
        "model_key": model_key,
        "question_count": overall["question_count"],
        "avg_manual_eval_score": overall["avg_manual_eval_score"],
        "avg_faithfulness_score": overall["avg_faithfulness_score"],
        "avg_completeness_score": overall["avg_completeness_score"],
        "avg_groundedness_score": overall["avg_groundedness_score"],
        "avg_relevancy_score": overall["avg_relevancy_score"],
        "output_dir": str(output_dir),
    }


def write_compare_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Scenario A Step 2B baseline runner")
    parser.add_argument("--question-set-path", default=str(root / "docs" / "planning" / "pm" / "day3_partA_eval_questions_v1.txt"))
    parser.add_argument("--mini-id-file", default=str(root / "docs" / "planning" / "pm" / "day4_mini_eval_question_ids_v1.txt"))
    parser.add_argument("--judge-model", default="gpt-5")
    parser.add_argument("--run-mini-first", action="store_true")
    parser.add_argument("--mini-only", action="store_true")
    parser.add_argument("--include-bgem3-qwen", action="store_true")
    parser.add_argument("--models", nargs="+", default=["qwen", "gemma4_e4b", "exaone"])
    parser.add_argument("--output-root", default=str(root / "rag_outputs"))
    args = parser.parse_args()

    full_rows = read_question_rows(Path(args.question_set_path))
    mini_rows = read_question_rows(Path(args.question_set_path), Path(args.mini_id_file))
    combos = [("koe5", model_key) for model_key in args.models]
    if args.include_bgem3_qwen:
        combos.append(("bge_m3", "qwen"))

    compare_rows = []
    output_root = Path(args.output_root)

    for embedding_backend, model_key in combos:
        pipeline = ScenarioACommonPipeline(
            PipelinePaths(project_root=root),
            PipelineSettings(embedding_backend_key=embedding_backend),
        )
        config = pipeline.load_model_config(model_key)
        if config.model_id.startswith("SET_ME_") or (config.runtime == "ollama" and (not config.ollama_model_name or config.ollama_model_name.startswith("SET_ME_"))):
            compare_rows.append(
                {
                    "combo_name": f"{embedding_backend}_{model_key}",
                    "embedding_backend": embedding_backend,
                    "model_key": model_key,
                    "status": "skipped",
                    "reason": "runtime_not_resolved",
                }
            )
            continue

        try:
            if args.run_mini_first:
                mini_output_root = output_root / "_mini_probe"
                mini_row = run_combo(root, mini_output_root, embedding_backend, model_key, mini_rows, args.judge_model)
                mini_row["run_scope"] = "mini"
                compare_rows.append(mini_row)
                if args.mini_only:
                    continue
            full_row = run_combo(root, output_root, embedding_backend, model_key, full_rows, args.judge_model)
            full_row["run_scope"] = "full"
            compare_rows.append(full_row)
        except Exception as exc:  # noqa: BLE001
            compare_rows.append(
                {
                    "combo_name": f"{embedding_backend}_{model_key}",
                    "embedding_backend": embedding_backend,
                    "model_key": model_key,
                    "status": "failed",
                    "reason": str(exc),
                }
            )

    write_compare_csv(output_root / "scenario_a_baseline_compare.csv", compare_rows)
    write_json(output_root / "scenario_a_baseline_compare.json", {"rows": compare_rows})
    print(output_root / "scenario_a_baseline_compare.csv")


if __name__ == "__main__":
    main()

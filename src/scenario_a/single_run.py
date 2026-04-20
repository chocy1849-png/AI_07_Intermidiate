from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from pathlib import Path

from scenario_a.common_pipeline import PipelinePaths, PipelineSettings, ScenarioACommonPipeline
from scenario_a.eval_runner import read_question_rows, run_eval


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run a single Scenario A experiment")
    parser.add_argument("--embedding-backend", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--question-set-path", default=str(root / "docs" / "planning" / "pm" / "day3_partA_eval_questions_v1.txt"))
    parser.add_argument("--question-id-file", default="")
    parser.add_argument("--judge-model", default="gpt-5")
    parser.add_argument("--run-label", default="")
    parser.add_argument("--factual-or-comparison-route", default="")
    parser.add_argument("--default-route", default="")
    parser.add_argument("--rejection-route", default="")
    parser.add_argument("--follow-up-route", default="")
    args = parser.parse_args()

    question_rows = read_question_rows(
        Path(args.question_set_path),
        Path(args.question_id_file) if args.question_id_file else None,
    )
    settings = PipelineSettings(embedding_backend_key=args.embedding_backend)
    if args.factual_or_comparison_route:
        settings.factual_or_comparison_route = args.factual_or_comparison_route
    if args.default_route:
        settings.default_route = args.default_route
    if args.rejection_route:
        settings.rejection_route = args.rejection_route
    if args.follow_up_route:
        settings.follow_up_route = args.follow_up_route

    pipeline = ScenarioACommonPipeline(
        PipelinePaths(project_root=root),
        settings,
    )
    adapter = pipeline.create_adapter(args.model_key)
    output_dir = Path(args.output_dir)
    run_label = args.run_label or f"{args.embedding_backend}_{args.model_key}"
    run_eval(pipeline, adapter, question_rows, output_dir, judge_model=args.judge_model, run_label=run_label)
    print(output_dir)


if __name__ == "__main__":
    main()

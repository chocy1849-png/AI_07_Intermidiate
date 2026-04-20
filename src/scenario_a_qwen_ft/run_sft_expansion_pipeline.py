from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scenario_a_qwen_ft.sft_expansion_pipeline import QwenSFTExpansionBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Scenario A Qwen SFT dataset expansion v1.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--question-root", type=Path, default=None)
    parser.add_argument("--base-output-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--retrieval-profile", type=str, default="bge_m3")
    parser.add_argument("--teacher-model", type=str, default="gpt-5")
    parser.add_argument("--judge-model", type=str, default="gpt-5")
    parser.add_argument("--shard-count", type=int, default=8)
    parser.add_argument("--teacher-workers", type=int, default=8)
    parser.add_argument("--judge-workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    builder = QwenSFTExpansionBuilder(
        project_root=project_root,
        question_root=(args.question_root or (project_root / "rag_outputs" / "qwen_ft_instruction_a")).resolve(),
        output_root=(args.output_root or (project_root / "rag_outputs" / "qwen_ft_instruction_expansion_v1")).resolve(),
        retrieval_profile=args.retrieval_profile,
        teacher_model=args.teacher_model,
        judge_model=args.judge_model,
        shard_count=args.shard_count,
        teacher_workers=args.teacher_workers,
        judge_workers=args.judge_workers,
    )
    if args.base_output_root is not None:
        builder.base_output_root = args.base_output_root.resolve()
    result = builder.run()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

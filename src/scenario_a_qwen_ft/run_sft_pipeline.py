from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scenario_a_qwen_ft.sft_pipeline import QwenSFTDatasetBuilder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Qwen SFT dataset builder.")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument(
        "--question-root",
        type=Path,
        default=ROOT / "rag_outputs" / "qwen_ft_instruction_a",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "rag_outputs" / "qwen_ft_instruction",
    )
    parser.add_argument("--retrieval-profile", default="bge_m3")
    parser.add_argument("--teacher-model", default="gpt-5")
    parser.add_argument("--judge-model", default="gpt-5")
    parser.add_argument("--shard-count", type=int, default=8)
    parser.add_argument("--teacher-workers", type=int, default=4)
    parser.add_argument("--judge-workers", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    builder = QwenSFTDatasetBuilder(
        project_root=args.project_root,
        question_root=args.question_root,
        output_root=args.output_root,
        retrieval_profile=args.retrieval_profile,
        teacher_model=args.teacher_model,
        judge_model=args.judge_model,
        shard_count=args.shard_count,
        teacher_workers=args.teacher_workers,
        judge_workers=args.judge_workers,
    )
    stats = builder.run()
    print("qwen sft pipeline completed")
    print(stats)


if __name__ == "__main__":
    main()

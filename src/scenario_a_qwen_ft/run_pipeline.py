from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scenario_a_qwen_ft.pipeline import QwenFTInstructionAPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run qwen_ft_instruction_A question vetting pipeline.")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pipeline = QwenFTInstructionAPipeline(project_root=args.project_root, output_dir=args.output_dir)
    stats = pipeline.run()
    print("question vetting pipeline completed")
    print(stats)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scenario_a_qwen_ft.finetune_qwen_v3 import run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/gemma4_ft_main_experiment_v1.yaml",
        help="Project-relative config path",
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    config_path = (project_root / args.config).resolve()
    summary = run_training(config_path)
    print(summary)


if __name__ == "__main__":
    main()


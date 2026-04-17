from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import CHUNK_PARAMETER_GRID
from src.db.metadata_store import get_dataset_summary


def build_dataset_report() -> dict:
    summary = get_dataset_summary()
    return {
        "summary": summary,
        "chunk_grid": CHUNK_PARAMETER_GRID,
    }


def load_json_report(report_path: str | Path) -> dict[str, Any]:
    path = Path(report_path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_latest_report(report_dir: str | Path, prefix: str) -> dict[str, Any] | None:
    report_path = get_latest_report_path(report_dir, prefix)
    if report_path is None:
        return None
    return load_json_report(report_path)


def get_latest_report_path(report_dir: str | Path, prefix: str) -> Path | None:
    candidates = sorted(Path(report_dir).glob(f"{prefix}*.json"))
    return candidates[-1] if candidates else None

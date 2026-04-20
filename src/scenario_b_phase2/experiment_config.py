from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _load_yaml() -> Any:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required for loading phase2 experiment config.") from exc
    return yaml


@dataclass(slots=True)
class Phase2ExperimentBundle:
    root: Path
    project: dict[str, Any]
    evaluation_sets: dict[str, dict[str, Any]]
    experiments: dict[str, dict[str, Any]]
    config_path: Path

    def resolve_eval_set(self, eval_set_key: str) -> tuple[str, Path]:
        if eval_set_key not in self.evaluation_sets:
            raise KeyError(f"Unknown eval set key: {eval_set_key}")
        row = self.evaluation_sets[eval_set_key]
        question_path = self.root / str(row.get("question_set_path", "")).strip()
        if not question_path.exists():
            raise FileNotFoundError(f"Question set path not found for eval set {eval_set_key}: {question_path}")
        return eval_set_key, question_path.resolve()

    def resolve_question_id_file(self, eval_set_key: str) -> Path | None:
        row = self.evaluation_sets.get(eval_set_key, {}) or {}
        rel = str(row.get("question_id_file", "")).strip()
        if not rel:
            return None
        path = (self.root / rel).resolve()
        return path if path.exists() else None

    def resolve_experiment_options(self, experiment_key: str) -> dict[str, Any]:
        if experiment_key not in self.experiments:
            raise KeyError(f"Unknown experiment key: {experiment_key}")
        row = self.experiments[experiment_key] or {}
        options = row.get("options", {}) or {}
        return dict(options)

    def resolve_project_default(self, key: str, default: Any = None) -> Any:
        return self.project.get(key, default)


def load_phase2_experiment_bundle(config_path: Path, project_root: Path) -> Phase2ExperimentBundle:
    yaml = _load_yaml()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return Phase2ExperimentBundle(
        root=project_root.resolve(),
        project=payload.get("project", {}) or {},
        evaluation_sets=payload.get("evaluation_sets", {}) or {},
        experiments=payload.get("experiments", {}) or {},
        config_path=config_path.resolve(),
    )

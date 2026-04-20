from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scenario_a.common_pipeline import PipelineSettings
from scenario_b_phase2.phase2_pipeline import Phase2Options


DEFAULT_PHASE2_PROFILE = "phase2_baseline_v2"


def _load_yaml() -> Any:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required for Streamlit QA config loading.") from exc
    return yaml


@dataclass(slots=True)
class LocalModelSpec:
    key: str
    label: str
    adapter_name: str
    model_id: str
    tokenizer_id: str | None = None
    runtime: str = "transformers"
    device_map: str = "auto"
    torch_dtype: str = "auto"
    trust_remote_code: bool = False
    stop_sequences: tuple[str, ...] = ()


def load_phase2_config(project_root: Path) -> dict[str, Any]:
    config_path = (project_root / "config" / "phase2_experiments.yaml").resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Phase2 experiment config not found: {config_path}")
    yaml = _load_yaml()
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def load_phase2_profiles(project_root: Path) -> dict[str, dict[str, Any]]:
    payload = load_phase2_config(project_root)
    profiles = payload.get("experiments", {}) or {}
    if not isinstance(profiles, dict):
        return {}
    return {str(key): dict(value or {}) for key, value in profiles.items()}


def build_phase2_options(options: dict[str, Any]) -> Phase2Options:
    metadata_flag = options.get("enable_metadata_aware_retrieval", options.get("enable_metadata_bonus_v2", True))
    return Phase2Options(
        enable_controlled_query_expansion=bool(options.get("enable_controlled_query_expansion", True)),
        enable_normalized_bm25=bool(options.get("enable_normalized_bm25", True)),
        enable_metadata_aware_retrieval=bool(metadata_flag),
        enable_metadata_bonus_v2=None,
        enable_table_body_pairing=bool(options.get("enable_table_body_pairing", True)),
        enable_soft_crag_lite=bool(options.get("enable_soft_crag_lite", True)),
        expansion_query_limit=int(options.get("expansion_query_limit", 1)),
        expansion_query_weight=float(options.get("expansion_query_weight", 0.35)),
        normalized_bm25_weight=float(options.get("normalized_bm25_weight", 0.35)),
        soft_crag_top_n=int(options.get("soft_crag_top_n", 6)),
        soft_crag_score_weight=float(options.get("soft_crag_score_weight", 0.045)),
        soft_crag_keep_k=int(options.get("soft_crag_keep_k", 3)),
        metadata_boost_scale=float(options.get("metadata_boost_scale", 1.0)),
        metadata_disable_for_rejection=bool(options.get("metadata_disable_for_rejection", False)),
        metadata_scope_mode=str(options.get("metadata_scope_mode", "all")),
        normalized_bm25_mode=str(options.get("normalized_bm25_mode", "all")),
        enable_comparison_evidence_helper=bool(options.get("enable_comparison_evidence_helper", False)),
        comparison_helper_doc_bonus=float(options.get("comparison_helper_doc_bonus", 0.0045)),
        comparison_helper_axis_bonus=float(options.get("comparison_helper_axis_bonus", 0.0015)),
        comparison_helper_max_per_doc=int(options.get("comparison_helper_max_per_doc", 2)),
        enable_b03_legacy_crag_parity=bool(options.get("enable_b03_legacy_crag_parity", True)),
        b03_evaluator_top_n=int(options.get("b03_evaluator_top_n", 6)),
        b03_second_pass_vector_weight=float(options.get("b03_second_pass_vector_weight", 0.55)),
        b03_second_pass_bm25_weight=float(options.get("b03_second_pass_bm25_weight", 0.45)),
    )


def phase2_options_to_dict(options: Phase2Options) -> dict[str, Any]:
    return asdict(options)


def build_pipeline_settings(
    *,
    embedding_backend_key: str,
    routing_model: str,
    candidate_k: int,
    top_k: int,
    crag_top_n: int,
    vector_weight: float,
    bm25_weight: float,
) -> PipelineSettings:
    return PipelineSettings(
        embedding_backend_key=embedding_backend_key,
        routing_model=routing_model,
        candidate_k=max(1, int(candidate_k)),
        top_k=max(1, int(top_k)),
        crag_top_n=max(1, int(crag_top_n)),
        vector_weight=float(vector_weight),
        bm25_weight=float(bm25_weight),
        factual_or_comparison_route="b03a",
        default_route="b02",
        rejection_route="b02",
        follow_up_route="b02",
    )


def load_model_config(project_root: Path) -> dict[str, Any]:
    model_config_path = (project_root / "config" / "scenario_a_models.yaml").resolve()
    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_config_path}")
    yaml = _load_yaml()
    return yaml.safe_load(model_config_path.read_text(encoding="utf-8")) or {}


def load_embedding_backend_keys(project_root: Path) -> list[str]:
    payload = load_model_config(project_root)
    backends = payload.get("embedding_backends", {}) or {}
    return [str(key) for key in backends.keys()]


def load_local_model_specs(project_root: Path) -> dict[str, LocalModelSpec]:
    payload = load_model_config(project_root)
    defaults = payload.get("defaults", {}) or {}
    models = payload.get("models", {}) or {}
    supported = {
        "qwen": ("qwen", "Qwen"),
        "gemma4_e4b": ("gemma4-e4b", "Gemma4-E4B"),
    }
    specs: dict[str, LocalModelSpec] = {}
    for model_key, (ui_key, label) in supported.items():
        if model_key not in models:
            continue
        row = dict(defaults)
        row.update(models.get(model_key, {}) or {})
        specs[ui_key] = LocalModelSpec(
            key=model_key,
            label=label,
            adapter_name=str(row.get("adapter", "")).strip(),
            model_id=str(row.get("model_id", "")).strip(),
            tokenizer_id=(str(row.get("tokenizer_id")).strip() or None) if row.get("tokenizer_id") is not None else None,
            runtime=str(row.get("runtime", "transformers")),
            device_map=str(row.get("device_map", "auto")),
            torch_dtype=str(row.get("torch_dtype", "auto")),
            trust_remote_code=bool(row.get("trust_remote_code", False)),
            stop_sequences=tuple(row.get("stop_sequences", []) or []),
        )
    return specs

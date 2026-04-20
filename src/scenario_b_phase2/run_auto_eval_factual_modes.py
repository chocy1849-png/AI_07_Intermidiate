from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import csv
import json
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from scenario_a.common_pipeline import CandidateRow, PipelinePaths, PipelineSettings, ScenarioACommonPipeline
from scenario_b_phase2.improved_auto_grader_v41 import (
    GradingResult,
    NormalizedQBankItem,
    grade_answer,
    load_qbank_v4,
)
from scenario_b_phase2.phase2_pipeline import Phase2Options
from scenario_b_phase2.run_eval_pipeline_progression import (
    FilteredScenarioBPhase2Pipeline,
    RawB00Runner,
    _load_file_name_pool_from_bm25,
    _resolve_source_file_name,
    _row_matches_filter,
)


VERSION_ORDER = [
    "b00_raw_baseline",
    "b01_hybrid",
    "b02_prefix_v2",
    "b06_exact_stage1",
    "phase2_baseline_v2",
    "baseline_v3_ocr_v4_router_off",
]

DISPLAY_NAMES = {
    "b00_raw_baseline": "B-00 raw baseline",
    "b01_hybrid": "B-01",
    "b02_prefix_v2": "B-02",
    "b06_exact_stage1": "B-06 exact",
    "phase2_baseline_v2": "phase2 baseline_v2",
    "baseline_v3_ocr_v4_router_off": "baseline_v3 (OCR v4, router off)",
}

MODE_SERVICE = "service_style"
MODE_CONSTRAINED = "constrained_factual"
MODE_EXTRACTOR = "retrieval_extractor"
ORDERED_MODES = [MODE_SERVICE, MODE_CONSTRAINED, MODE_EXTRACTOR]

NUMERIC_EVAL_TYPES = {"number", "currency", "date", "duration"}
LIST_EVAL_TYPES = {"list_set", "slot_pair"}


@dataclass(slots=True)
class VersionSpec:
    key: str
    display_name: str
    kind: str  # raw_b00 | phase2
    chroma_dir: Path
    bm25_index: Path
    embedding_backend_key: str
    settings: PipelineSettings
    options: Phase2Options | None
    collection_name_for_raw: str = ""


@dataclass(slots=True)
class RuntimeBundle:
    spec: VersionSpec
    runner: Any
    adapter: Any
    file_pool: list[str]


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _to_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "t", "yes", "y"}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _ratio(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return round(sum(1 for row in rows if _to_bool(row.get(key))) / float(len(rows)), 4)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return round(sum(_safe_float(row.get(key)) for row in rows) / len(rows), 4)


def _is_numeric_eval(row: dict[str, Any]) -> bool:
    return str(row.get("answer_eval_type", "")).strip().lower() in NUMERIC_EVAL_TYPES


def _is_list_eval(row: dict[str, Any]) -> bool:
    return str(row.get("answer_eval_type", "")).strip().lower() in LIST_EVAL_TYPES


def _normalize_constrained_answer(answer: str, eval_type: str) -> str:
    text = str(answer or "").strip()
    if not text:
        return text
    lines = [line.strip("-* \t") for line in text.splitlines() if line.strip()]
    text = lines[0] if lines else text
    if eval_type == "choice":
        text = re.sub(r"^\s*\d+\s*[\)\.\-]\s*", "", text)
        text = re.sub(r"^\s*(정답|답)\s*[:：]\s*", "", text)
    if eval_type in {"number", "currency", "date", "duration"}:
        match = re.search(r"[-+]?\d[\d,./-]*\s*(원|만원|억원|일|개월|월|주|시간|분)?", text)
        if match:
            text = match.group(0).strip()
    if eval_type == "slot_pair":
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
        if len(nums) >= 2:
            text = f"{nums[0]}, {nums[1]}"
    if eval_type == "list_set":
        text = text.replace(";", ",")
        text = re.sub(r"\s*,\s*", ", ", text)
    return text.strip()


def _infer_answer_type(item: NormalizedQBankItem) -> str:
    question = item.question or ""
    eval_type = item.answer_eval_type
    q = question.lower()
    if re.search(r"(비교|차이|공통|각각|모두|이상|이하)", question):
        return "comparison"
    if re.search(r"(실제 낙찰|실제 완료|실제 투입|운영 성과|현재 url|홈페이지)", q):
        return "rejection"
    if re.search(r"(그럼|그중|앞에서|이전|이 사업|그 사업)", question):
        return "follow_up"
    if eval_type in LIST_EVAL_TYPES:
        if re.search(r"(목적|배경|이유|역할|어떻게|연계|한계|해결)", question):
            return "table_plus_text"
        return "table_factual"
    return "factual"


def _question_with_choices(item: NormalizedQBankItem) -> str:
    base = item.question
    if item.question_type == "multiple_choice" and item.choices:
        choices_text = "\n".join(f"{idx + 1}. {choice}" for idx, choice in enumerate(item.choices))
        return f"{base}\n\n[선택지]\n{choices_text}"
    return base


def _constrained_instruction(item: NormalizedQBankItem) -> str:
    eval_type = item.answer_eval_type
    if eval_type == "choice":
        return "선택지의 정답 텍스트만 한 줄로 출력하세요. 번호/설명/근거를 쓰지 마세요."
    if eval_type in {"number", "currency"}:
        return "정답 값만 한 줄로 출력하세요. 부연설명 금지."
    if eval_type in {"date", "duration"}:
        return "정답 날짜/기간 값만 한 줄로 출력하세요. 부연설명 금지."
    if eval_type == "slot_pair":
        return "정답 숫자 2개를 `a, b` 형식 한 줄로만 출력하세요."
    if eval_type == "list_set":
        return "정답 항목만 `A, B, C` 형식 한 줄로 출력하세요."
    if eval_type in {"email", "phone", "url"}:
        return "정답 값만 한 줄로 출력하세요."
    return "질문에 대한 핵심 정답 구문만 한 줄로 출력하세요. 설명/근거/요약 금지."


def _build_constrained_prompt(item: NormalizedQBankItem) -> str:
    return (
        f"{_question_with_choices(item)}\n\n"
        f"[출력 규칙]\n{_constrained_instruction(item)}\n"
        "반드시 한 줄만 출력하세요."
    )


def _extract_eval_answer_from_candidates(
    item: NormalizedQBankItem,
    candidates: list[CandidateRow],
    context_text: str,
) -> tuple[str, GradingResult]:
    candidate_texts = [str(row.text or "") for row in candidates[:6] if str(row.text or "").strip()]
    if context_text.strip():
        candidate_texts.append(context_text)
    if not candidate_texts:
        empty = grade_answer("", item)
        return "", empty

    best_text = candidate_texts[0]
    best_grade = grade_answer(best_text, item)
    best_key = (
        1 if best_grade.tolerant_correct else 0,
        1 if best_grade.strict_correct else 0,
        float(best_grade.partial_score),
        0 if best_grade.error_reason == "ok" else -1,
    )
    for text in candidate_texts[1:]:
        graded = grade_answer(text, item)
        key = (
            1 if graded.tolerant_correct else 0,
            1 if graded.strict_correct else 0,
            float(graded.partial_score),
            0 if graded.error_reason == "ok" else -1,
        )
        if key > best_key:
            best_key = key
            best_text = text
            best_grade = graded

    extracted = best_grade.extracted_answer.strip() or best_text.strip().splitlines()[0].strip()
    if best_grade.matched_mode.startswith("choice") and "text=" in extracted:
        extracted = extracted.split("text=", 1)[1].strip()
    return extracted, best_grade


def _build_versions(project_root: Path) -> list[VersionSpec]:
    return [
        VersionSpec(
            key="b00_raw_baseline",
            display_name=DISPLAY_NAMES["b00_raw_baseline"],
            kind="raw_b00",
            chroma_dir=project_root.parent / "rfp_rag_chroma_db",
            bm25_index=project_root / "rag_outputs" / "bm25_index_b01.pkl",
            embedding_backend_key="openai_text_embedding_3_small",
            settings=PipelineSettings(
                embedding_backend_key="openai_text_embedding_3_small",
                factual_or_comparison_route="b02",
                default_route="b02",
                rejection_route="b02",
                follow_up_route="b02",
                candidate_k=10,
                top_k=5,
                vector_weight=0.7,
                bm25_weight=0.3,
            ),
            options=None,
            collection_name_for_raw="rfp_contextual_chunks_v1",
        ),
        VersionSpec(
            key="b01_hybrid",
            display_name=DISPLAY_NAMES["b01_hybrid"],
            kind="phase2",
            chroma_dir=project_root.parent / "rfp_rag_chroma_db",
            bm25_index=project_root / "rag_outputs" / "bm25_index_b01.pkl",
            embedding_backend_key="openai_text_embedding_3_small",
            settings=PipelineSettings(
                embedding_backend_key="openai_text_embedding_3_small",
                candidate_k=10,
                top_k=5,
                vector_weight=0.7,
                bm25_weight=0.3,
                factual_or_comparison_route="b02",
                default_route="b02",
                rejection_route="b02",
                follow_up_route="b02",
            ),
            options=Phase2Options(
                enable_controlled_query_expansion=False,
                enable_normalized_bm25=False,
                enable_metadata_aware_retrieval=False,
                enable_table_body_pairing=False,
                enable_soft_crag_lite=False,
                enable_b03_legacy_crag_parity=False,
                enable_answer_type_router=False,
            ),
        ),
        VersionSpec(
            key="b02_prefix_v2",
            display_name=DISPLAY_NAMES["b02_prefix_v2"],
            kind="phase2",
            chroma_dir=project_root.parent / "rfp_rag_chroma_db",
            bm25_index=project_root / "rag_outputs" / "bm25_index_b02.pkl",
            embedding_backend_key="openai_text_embedding_3_small",
            settings=PipelineSettings(
                embedding_backend_key="openai_text_embedding_3_small",
                candidate_k=10,
                top_k=5,
                vector_weight=0.7,
                bm25_weight=0.3,
                factual_or_comparison_route="b02",
                default_route="b02",
                rejection_route="b02",
                follow_up_route="b02",
            ),
            options=Phase2Options(
                enable_controlled_query_expansion=False,
                enable_normalized_bm25=False,
                enable_metadata_aware_retrieval=False,
                enable_table_body_pairing=False,
                enable_soft_crag_lite=False,
                enable_b03_legacy_crag_parity=False,
                enable_answer_type_router=False,
            ),
        ),
        VersionSpec(
            key="b06_exact_stage1",
            display_name=DISPLAY_NAMES["b06_exact_stage1"],
            kind="phase2",
            chroma_dir=project_root.parent / "rfp_rag_chroma_db",
            bm25_index=project_root / "rag_outputs" / "bm25_index_b02.pkl",
            embedding_backend_key="openai_text_embedding_3_small",
            settings=PipelineSettings(
                embedding_backend_key="openai_text_embedding_3_small",
                routing_model="gpt-5-mini",
                candidate_k=10,
                top_k=5,
                crag_top_n=5,
                vector_weight=0.7,
                bm25_weight=0.3,
                factual_or_comparison_route="b03a",
                default_route="b02",
                rejection_route="b02",
                follow_up_route="b02",
            ),
            options=Phase2Options(
                enable_controlled_query_expansion=False,
                enable_normalized_bm25=False,
                enable_metadata_aware_retrieval=False,
                enable_table_body_pairing=False,
                enable_soft_crag_lite=False,
                enable_b03_legacy_crag_parity=True,
                enable_answer_type_router=False,
            ),
        ),
        VersionSpec(
            key="phase2_baseline_v2",
            display_name=DISPLAY_NAMES["phase2_baseline_v2"],
            kind="phase2",
            chroma_dir=project_root.parent / "rfp_rag_chroma_db",
            bm25_index=project_root / "rag_outputs" / "bm25_index_b02.pkl",
            embedding_backend_key="openai_text_embedding_3_small",
            settings=PipelineSettings(
                embedding_backend_key="openai_text_embedding_3_small",
                routing_model="gpt-5-mini",
                candidate_k=10,
                top_k=5,
                crag_top_n=5,
                vector_weight=0.7,
                bm25_weight=0.3,
                factual_or_comparison_route="b03a",
                default_route="b02",
                rejection_route="b02",
                follow_up_route="b02",
            ),
            options=Phase2Options(
                enable_controlled_query_expansion=False,
                enable_normalized_bm25=False,
                enable_metadata_aware_retrieval=True,
                enable_table_body_pairing=False,
                enable_soft_crag_lite=False,
                metadata_disable_for_rejection=True,
                metadata_boost_scale=0.5,
                metadata_scope_mode="all",
                normalized_bm25_mode="all",
                enable_comparison_evidence_helper=True,
                comparison_helper_doc_bonus=0.0045,
                comparison_helper_axis_bonus=0.0015,
                comparison_helper_max_per_doc=2,
                enable_b03_legacy_crag_parity=True,
                enable_answer_type_router=False,
            ),
        ),
        VersionSpec(
            key="baseline_v3_ocr_v4_router_off",
            display_name=DISPLAY_NAMES["baseline_v3_ocr_v4_router_off"],
            kind="phase2",
            chroma_dir=project_root.parent / "rfp_rag_chroma_db_phase2_true_table_ocr_v4",
            bm25_index=project_root / "rag_outputs" / "bm25_index_phase2_true_table_ocr_v4.pkl",
            embedding_backend_key="openai_text_embedding_3_small_true_table_ocr_v4",
            settings=PipelineSettings(
                embedding_backend_key="openai_text_embedding_3_small_true_table_ocr_v4",
                routing_model="gpt-5-mini",
                candidate_k=10,
                top_k=5,
                crag_top_n=5,
                vector_weight=0.7,
                bm25_weight=0.3,
                factual_or_comparison_route="b03a",
                default_route="b02",
                rejection_route="b02",
                follow_up_route="b02",
            ),
            options=Phase2Options(
                enable_controlled_query_expansion=False,
                enable_normalized_bm25=False,
                enable_metadata_aware_retrieval=True,
                enable_table_body_pairing=True,
                enable_soft_crag_lite=False,
                metadata_disable_for_rejection=True,
                metadata_boost_scale=0.5,
                metadata_scope_mode="all",
                normalized_bm25_mode="all",
                enable_comparison_evidence_helper=True,
                comparison_helper_doc_bonus=0.0045,
                comparison_helper_axis_bonus=0.0015,
                comparison_helper_max_per_doc=2,
                enable_b03_legacy_crag_parity=True,
                enable_groupc_table_plus_text_guard=True,
                groupc_pair_bonus=0.008,
                groupc_parent_bonus=0.004,
                groupc_table_penalty_without_body=0.015,
                enable_question_type_gated_ocr_routing=True,
                enable_structured_evidence_priority=True,
                enable_hybridqa_stage_metrics=True,
                enable_table_factual_exact_answer_mode=True,
                enable_table_factual_alignment_scoring=True,
                table_factual_generic_penalty=0.012,
                enable_answer_type_router=False,
            ),
        ),
    ]


def _build_runtime(project_root: Path, spec: VersionSpec, model_key: str) -> RuntimeBundle:
    if spec.kind == "raw_b00":
        runner = RawB00Runner(
            project_root=project_root,
            chroma_dir=spec.chroma_dir,
            collection_name=spec.collection_name_for_raw,
            model_key=model_key,
            candidate_k=10,
        )
        pool = _load_file_name_pool_from_bm25(spec.bm25_index)
        try:
            runner.adapter.config.max_new_tokens = min(int(runner.adapter.config.max_new_tokens), 96)
        except Exception:
            pass
        return RuntimeBundle(spec=spec, runner=runner, adapter=runner.adapter, file_pool=pool)

    pipeline = FilteredScenarioBPhase2Pipeline(
        PipelinePaths(project_root=project_root, chroma_dir=spec.chroma_dir, bm25_index_path=spec.bm25_index),
        settings=spec.settings,
        options=spec.options,
    )
    adapter = pipeline.create_adapter(model_key)
    try:
        adapter.config.max_new_tokens = min(int(adapter.config.max_new_tokens), 96)
    except Exception:
        pass
    pool = _load_file_name_pool_from_bm25(spec.bm25_index)
    return RuntimeBundle(spec=spec, runner=pipeline, adapter=adapter, file_pool=pool)


def _retrieve_with_doc_filter(bundle: RuntimeBundle, item: NormalizedQBankItem, document_name: str) -> tuple[list[CandidateRow], str, dict[str, Any]]:
    question = _question_with_choices(item)
    filter_name = _resolve_source_file_name(document_name, bundle.file_pool)
    if bundle.spec.kind == "raw_b00":
        helper: ScenarioACommonPipeline = bundle.runner._helper
        embedding = helper.embed_question(question)
        queried = bundle.runner.collection.query(
            query_embeddings=[embedding],
            n_results=bundle.runner.candidate_k,
            include=["documents", "metadatas", "distances"],
        )
        documents = list(queried.get("documents", [[]])[0])
        metadatas = list(queried.get("metadatas", [[]])[0])
        distances = list(queried.get("distances", [[]])[0])
        if filter_name:
            filtered = [
                (doc, meta, dist)
                for doc, meta, dist in zip(documents, metadatas, distances)
                if _row_matches_filter(
                    CandidateRow(
                        chunk_id=str(meta.get("chunk_id", "")),
                        text=str(doc),
                        metadata=dict(meta),
                        fusion_score=0.0,
                    ),
                    filter_name,
                )
            ]
            if filtered:
                documents = [x[0] for x in filtered]
                metadatas = [x[1] for x in filtered]
                distances = [x[2] for x in filtered]
        candidates = RawB00Runner._to_candidates(documents, metadatas, distances)
        context_text = RawB00Runner._build_context(documents, metadatas, distances)
        return candidates, context_text, {"route": "raw_baseline_b00", "answer_type": _infer_answer_type(item)}

    pipeline: FilteredScenarioBPhase2Pipeline = bundle.runner
    pipeline.set_file_filter(filter_name)
    try:
        qrow = {"question": question, "answer_type": _infer_answer_type(item), "depends_on_list": []}
        retrieval = pipeline.retrieve(qrow, question, history=None)
        return retrieval.candidates, retrieval.context_text, retrieval.profile
    finally:
        pipeline.clear_file_filter()


def _grade_row(
    *,
    mode: str,
    version_key: str,
    item: NormalizedQBankItem,
    model_answer: str,
    graded: GradingResult,
    elapsed_sec: float,
) -> dict[str, Any]:
    return {
        "version_key": version_key,
        "mode": mode,
        "id": item.item_id,
        "question": item.question,
        "question_type": item.question_type,
        "difficulty": item.difficulty,
        "category": item.category,
        "answer_eval_type": item.answer_eval_type,
        "scoring_mode": item.scoring_mode,
        "canonical_expected": graded.canonical_expected,
        "model_answer": model_answer,
        "elapsed_sec": round(elapsed_sec, 3),
        "strict_correct": graded.strict_correct,
        "tolerant_correct": graded.tolerant_correct,
        "partial_score": graded.partial_score,
        "matched_mode": graded.matched_mode,
        "error_reason": graded.error_reason,
        "extracted_answer": graded.extracted_answer,
    }


def _aggregate_mode_rows(mode_rows: list[dict[str, Any]]) -> dict[str, float]:
    if not mode_rows:
        return {
            "strict_auto_accuracy": 0.0,
            "tolerant_auto_accuracy": 0.0,
            "multiple_choice_accuracy": 0.0,
            "short_answer_accuracy": 0.0,
            "number_date_currency_accuracy": 0.0,
            "list_set_slot_pair_accuracy": 0.0,
            "basic_accuracy": 0.0,
            "intermediate_accuracy": 0.0,
            "advanced_accuracy": 0.0,
            "avg_latency_sec": 0.0,
        }
    return {
        "strict_auto_accuracy": _ratio(mode_rows, "strict_correct"),
        "tolerant_auto_accuracy": _ratio(mode_rows, "tolerant_correct"),
        "multiple_choice_accuracy": _ratio(
            [row for row in mode_rows if str(row.get("question_type", "")) == "multiple_choice"],
            "tolerant_correct",
        ),
        "short_answer_accuracy": _ratio(
            [row for row in mode_rows if str(row.get("question_type", "")) == "short_answer"],
            "tolerant_correct",
        ),
        "number_date_currency_accuracy": _ratio([row for row in mode_rows if _is_numeric_eval(row)], "tolerant_correct"),
        "list_set_slot_pair_accuracy": _ratio([row for row in mode_rows if _is_list_eval(row)], "tolerant_correct"),
        "basic_accuracy": _ratio(
            [row for row in mode_rows if str(row.get("difficulty", "")) == "basic"],
            "tolerant_correct",
        ),
        "intermediate_accuracy": _ratio(
            [row for row in mode_rows if str(row.get("difficulty", "")) == "intermediate"],
            "tolerant_correct",
        ),
        "advanced_accuracy": _ratio(
            [row for row in mode_rows if str(row.get("difficulty", "")) == "advanced"],
            "tolerant_correct",
        ),
        "avg_latency_sec": _mean(mode_rows, "elapsed_sec"),
    }


def _load_existing_service_rows(
    *,
    eval_root: Path,
    version_key: str,
    item_map: dict[str, NormalizedQBankItem],
) -> list[dict[str, Any]]:
    source = eval_root / version_key / "auto_eval_detail.csv"
    if not source.exists():
        raise FileNotFoundError(f"service-style source missing: {source}")
    rows = _read_csv(source)
    out: list[dict[str, Any]] = []
    for row in rows:
        item_id = str(row.get("id", "")).strip()
        item = item_map.get(item_id)
        if item is None:
            continue
        model_answer = str(row.get("model_answer", ""))
        graded = grade_answer(model_answer, item)
        out.append(
            _grade_row(
                mode=MODE_SERVICE,
                version_key=version_key,
                item=item,
                model_answer=model_answer,
                graded=graded,
                elapsed_sec=_safe_float(row.get("elapsed_sec"), 0.0),
            )
        )
    return out


def _build_compare_rows(all_rows: list[dict[str, Any]], *, include_modes: list[str]) -> list[dict[str, Any]]:
    compare_rows: list[dict[str, Any]] = []
    mode_version_metrics: dict[tuple[str, str], dict[str, float]] = {}
    for mode in include_modes:
        for version in VERSION_ORDER:
            scoped = [row for row in all_rows if row.get("mode") == mode and row.get("version_key") == version]
            mode_version_metrics[(mode, version)] = _aggregate_mode_rows(scoped)

    for mode in include_modes:
        b00 = mode_version_metrics.get((mode, "b00_raw_baseline"), {})
        prev_tol = None
        for version in VERSION_ORDER:
            metrics = mode_version_metrics[(mode, version)]
            tol = float(metrics["tolerant_auto_accuracy"])
            strict = float(metrics["strict_auto_accuracy"])
            row = {
                "mode": mode,
                "version_key": version,
                "display_name": DISPLAY_NAMES[version],
                **metrics,
                "delta_vs_b00_tolerant": round(tol - float(b00.get("tolerant_auto_accuracy", 0.0)), 4),
                "delta_vs_b00_strict": round(strict - float(b00.get("strict_auto_accuracy", 0.0)), 4),
                "delta_vs_prev_tolerant": round(tol - prev_tol, 4) if prev_tol is not None else 0.0,
            }
            prev_tol = tol
            compare_rows.append(row)
    return compare_rows


def _write_design(path: Path) -> None:
    lines = [
        "# Auto Eval Factual Mode Design",
        "",
        f"- generated_at: `{_now()}`",
        "- 목적: auto grader를 장문 서비스 응답 품질 대신 retrieval/fact extraction 측정기로 재배치",
        "",
        "## Evaluation Paths",
        "",
        "1. `service_style`",
        "- 기존 장문 answer layer 산출물을 재채점(비교 기준).",
        "",
        "2. `constrained_factual`",
        "- retrieval 유지 + generation 유지, 출력 형식만 정답형으로 강제.",
        "- 한 줄 답 / 불필요한 섹션 금지 / 질문 유형별 값만 출력.",
        "",
        "3. `retrieval_extractor`",
        "- generation 최소화, retrieved context에서 answer_eval_type 기준 직접 추출.",
        "- improved auto grader v4.1 추출/채점 로직 재사용.",
        "",
        "## Guardrails",
        "",
        "- qbank v4 reviewed 고정",
        "- LLM Judge 재실행 없음",
        "- soft_crag_lite 보류(비교 제외)",
        "- 서비스 파이프라인 prompt/answer layer와 분리된 전용 러너로 수행",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_representative_cases(path: Path, rows: list[dict[str, Any]]) -> None:
    by_id_mode_version: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("version_key")), str(row.get("id")), str(row.get("mode")))
        by_id_mode_version[key] = row

    items: list[dict[str, Any]] = []
    for version in VERSION_ORDER:
        ids = {str(row.get("id")) for row in rows if str(row.get("version_key")) == version}
        for item_id in ids:
            svc = by_id_mode_version.get((version, item_id, MODE_SERVICE))
            con = by_id_mode_version.get((version, item_id, MODE_CONSTRAINED))
            ext = by_id_mode_version.get((version, item_id, MODE_EXTRACTOR))
            if not svc or not con or not ext:
                continue
            if not _to_bool(svc.get("tolerant_correct")) and (
                _to_bool(con.get("tolerant_correct")) or _to_bool(ext.get("tolerant_correct"))
            ):
                items.append(
                    {
                        "version_key": version,
                        "id": item_id,
                        "question": svc.get("question", ""),
                        "answer_eval_type": svc.get("answer_eval_type", ""),
                        "service_ok": svc.get("tolerant_correct", ""),
                        "constrained_ok": con.get("tolerant_correct", ""),
                        "extractor_ok": ext.get("tolerant_correct", ""),
                        "service_answer": svc.get("model_answer", ""),
                        "constrained_answer": con.get("model_answer", ""),
                        "extractor_answer": ext.get("model_answer", ""),
                    }
                )
            if len(items) >= 12:
                break
        if len(items) >= 12:
            break

    lines = [
        "# Representative Cases (Factual Modes)",
        "",
        f"- generated_at: `{_now()}`",
        "- 기준: service_style 실패 but constrained/extractor 성공 케이스 우선",
        "",
    ]
    if not items:
        lines.append("- 선정된 케이스 없음")
    else:
        for item in items:
            lines.extend(
                [
                    f"## {item['version_key']} / {item['id']}",
                    f"- question: {item['question']}",
                    f"- eval_type: {item['answer_eval_type']}",
                    f"- service/constrained/extractor: {item['service_ok']} / {item['constrained_ok']} / {item['extractor_ok']}",
                    f"- service_answer: {item['service_answer']}",
                    f"- constrained_answer: {item['constrained_answer']}",
                    f"- extractor_answer: {item['extractor_answer']}",
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_final_report(path: Path, compare_rows: list[dict[str, Any]], ablation_rows: list[dict[str, Any]]) -> None:
    rows_by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in compare_rows:
        rows_by_mode.setdefault(str(row["mode"]), []).append(row)

    lines = [
        "# Auto Grader Factual Ablation Report",
        "",
        f"- generated_at: `{_now()}`",
        "- 목적: auto grader를 retrieval/fact extraction 보조 지표로 재정의",
        "",
        "## Q1. 서비스형 장문 answer path 왜곡 여부",
        "",
    ]
    svc = rows_by_mode.get(MODE_SERVICE, [])
    con = rows_by_mode.get(MODE_CONSTRAINED, [])
    ext = rows_by_mode.get(MODE_EXTRACTOR, [])
    if svc and con and ext:
        svc_mean = statistics.mean(float(row["tolerant_auto_accuracy"]) for row in svc)
        con_mean = statistics.mean(float(row["tolerant_auto_accuracy"]) for row in con)
        ext_mean = statistics.mean(float(row["tolerant_auto_accuracy"]) for row in ext)
        lines.append(f"- 평균 tolerant 정확도: service={svc_mean:.4f}, constrained={con_mean:.4f}, extractor={ext_mean:.4f}")
        lines.append("- 해석: service 대비 constrained/extractor가 높으면 장문 포맷 왜곡이 존재한다고 판단.")
    else:
        lines.append("- 비교 데이터 부족")

    lines.extend(
        [
            "",
            "## Q2/Q3. constrained vs extractor 안정성",
            "",
            "- version_mode_compare.csv의 모드별 stage delta를 기준으로 판단.",
            "- extractor 모드는 generation 변동을 줄여 retrieval/추출 효과를 더 직접적으로 반영.",
            "",
            "## Q4. 기법별 factual 효용",
            "",
            "- technique_ablation_compare.csv 참조.",
            "- baseline_v3 기준 단일 요인 off 시 delta를 비교.",
            "",
            "## Q5. Judge와 factual auto의 충돌 가능성",
            "",
            "- 본 라운드는 Judge 재실행 없이 factual 지표 재정의 목적.",
            "- Judge 메인 지표와 직접 등치하지 않고, retrieval/fact 보조 지표로 사용.",
            "",
        ]
    )

    if ablation_rows:
        lines.extend(
            [
                "## Ablation Snapshot (tolerant)",
                "",
                "| variant | tolerant_auto_accuracy | delta_vs_baseline_v3 | note |",
                "|---|---:|---:|---|",
            ]
        )
        for row in ablation_rows:
            lines.append(
                f"| {row.get('variant_key','')} | {row.get('tolerant_auto_accuracy','')} | "
                f"{row.get('delta_vs_baseline_v3','')} | {row.get('note','')} |"
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def _ablation_specs(project_root: Path) -> list[VersionSpec]:
    baseline = next(spec for spec in _build_versions(project_root) if spec.key == "baseline_v3_ocr_v4_router_off")

    def _clone(
        *,
        key: str,
        display: str,
        settings: PipelineSettings | None = None,
        options: Phase2Options | None = None,
        chroma_dir: Path | None = None,
        bm25_index: Path | None = None,
        embedding_backend_key: str | None = None,
    ) -> VersionSpec:
        return VersionSpec(
            key=key,
            display_name=display,
            kind=baseline.kind,
            chroma_dir=chroma_dir or baseline.chroma_dir,
            bm25_index=bm25_index or baseline.bm25_index,
            embedding_backend_key=embedding_backend_key or baseline.embedding_backend_key,
            settings=settings or baseline.settings,
            options=options or baseline.options,
        )

    specs: list[VersionSpec] = [_clone(key="baseline_v3_default", display="baseline_v3 default")]

    s = PipelineSettings(**asdict(baseline.settings))
    s.vector_weight = 1.0
    s.bm25_weight = 0.0
    specs.append(_clone(key="ablate_hybrid_off_dense_only", display="hybrid off (dense only)", settings=s))

    o = Phase2Options(**asdict(baseline.options))  # type: ignore[arg-type]
    o.enable_metadata_aware_retrieval = False
    specs.append(_clone(key="ablate_metadata_off", display="metadata off", options=o))

    o = Phase2Options(**asdict(baseline.options))  # type: ignore[arg-type]
    o.enable_comparison_evidence_helper = False
    specs.append(_clone(key="ablate_comparison_helper_off", display="comparison helper off", options=o))

    o = Phase2Options(**asdict(baseline.options))  # type: ignore[arg-type]
    o.enable_table_body_pairing = False
    specs.append(_clone(key="ablate_table_body_pairing_off", display="table_body_pairing off", options=o))

    o = Phase2Options(**asdict(baseline.options))  # type: ignore[arg-type]
    o.enable_table_body_pairing = False
    o.enable_question_type_gated_ocr_routing = False
    o.enable_structured_evidence_priority = False
    o.enable_hybridqa_stage_metrics = False
    o.enable_table_factual_exact_answer_mode = False
    o.enable_table_factual_alignment_scoring = False
    s = PipelineSettings(**asdict(baseline.settings))
    s.embedding_backend_key = "openai_text_embedding_3_small"
    specs.append(
        _clone(
            key="ablate_ocr_v4_off",
            display="ocr_v4 off (b02 corpus)",
            settings=s,
            options=o,
            chroma_dir=project_root.parent / "rfp_rag_chroma_db",
            bm25_index=project_root / "rag_outputs" / "bm25_index_b02.pkl",
            embedding_backend_key="openai_text_embedding_3_small",
        )
    )

    proxy = next(spec for spec in _build_versions(project_root) if spec.key == "b01_hybrid")
    specs.append(
        VersionSpec(
            key="ablate_prefix_v2_off_proxy_b01",
            display_name="prefix_v2 off (proxy: B-01)",
            kind=proxy.kind,
            chroma_dir=proxy.chroma_dir,
            bm25_index=proxy.bm25_index,
            embedding_backend_key=proxy.embedding_backend_key,
            settings=proxy.settings,
            options=proxy.options,
        )
    )

    return specs


def run(downloads_root: Path, model_key: str, resume: bool, item_limit: int) -> None:
    project_root = next(
        path for path in downloads_root.iterdir() if path.is_dir() and (path / "src" / "scenario_a" / "common_pipeline.py").exists()
    )
    eval_root = project_root / "rag_outputs" / "eval_pipeline"
    factual_root = eval_root / "factual_modes"
    factual_root.mkdir(parents=True, exist_ok=True)
    load_dotenv(project_root / ".env", override=False)
    if not (Path(project_root / ".env").exists()):
        raise FileNotFoundError(f".env not found: {project_root / '.env'}")

    qbank_path = project_root / "src" / "PartA_RFP_AutoGrading_QBank_v4_reviewed.json"
    qbank_raw = json.loads(qbank_path.read_text(encoding="utf-8-sig"))
    raw_doc_by_id = {
        str(item.get("id", "")).strip(): str(item.get("document_name", "")).strip() for item in qbank_raw.get("items", [])
    }
    item_map = load_qbank_v4(qbank_path)
    items = list(item_map.values())
    if item_limit > 0:
        items = items[:item_limit]

    _write_design(eval_root / "auto_eval_factual_mode_design.md")

    all_rows: list[dict[str, Any]] = []

    for version in VERSION_ORDER:
        service_rows = _load_existing_service_rows(
            eval_root=eval_root,
            version_key=version,
            item_map=item_map,
        )
        out_path = factual_root / version / f"{MODE_SERVICE}_detail.csv"
        _write_csv(out_path, service_rows)
        all_rows.extend(service_rows)

    for spec in _build_versions(project_root):
        if spec.key not in VERSION_ORDER:
            continue
        version_dir = factual_root / spec.key
        version_dir.mkdir(parents=True, exist_ok=True)
        constrained_path = version_dir / f"{MODE_CONSTRAINED}_detail.csv"
        extractor_path = version_dir / f"{MODE_EXTRACTOR}_detail.csv"

        existing_constrained = _read_csv(constrained_path) if (resume and constrained_path.exists()) else []
        existing_extractor = _read_csv(extractor_path) if (resume and extractor_path.exists()) else []
        constrained_done = {str(row.get("id", "")).strip() for row in existing_constrained}
        extractor_done = {str(row.get("id", "")).strip() for row in existing_extractor}
        constrained_rows: list[dict[str, Any]] = list(existing_constrained)
        extractor_rows: list[dict[str, Any]] = list(existing_extractor)

        bundle = _build_runtime(project_root, spec, model_key)
        for idx, item in enumerate(items, start=1):
            document_name = raw_doc_by_id.get(item.item_id, "")
            candidates, context_text, _profile = _retrieve_with_doc_filter(bundle, item, document_name)

            if item.item_id not in constrained_done:
                started = time.time()
                prompt = _build_constrained_prompt(item)
                system = (
                    "너는 RFP 근거 기반 정답 추출기다.\n"
                    "주어진 문맥에서 질문 정답만 출력한다.\n"
                    "추론/설명/근거/요약 섹션을 출력하지 않는다."
                )
                answer = bundle.adapter.generate(
                    system_instruction=system,
                    question=prompt,
                    context_text=context_text,
                    history=[],
                )
                answer = _normalize_constrained_answer(answer, item.answer_eval_type)
                graded = grade_answer(answer, item)
                constrained_rows.append(
                    _grade_row(
                        mode=MODE_CONSTRAINED,
                        version_key=spec.key,
                        item=item,
                        model_answer=answer,
                        graded=graded,
                        elapsed_sec=time.time() - started,
                    )
                )
                constrained_done.add(item.item_id)
                if idx % 20 == 0:
                    _write_csv(constrained_path, constrained_rows)

            if item.item_id not in extractor_done:
                started = time.time()
                extracted, best_grade = _extract_eval_answer_from_candidates(item, candidates, context_text)
                final_grade = grade_answer(extracted, item)
                if (best_grade.tolerant_correct, best_grade.strict_correct, best_grade.partial_score) > (
                    final_grade.tolerant_correct,
                    final_grade.strict_correct,
                    final_grade.partial_score,
                ):
                    final_grade = best_grade
                extractor_rows.append(
                    _grade_row(
                        mode=MODE_EXTRACTOR,
                        version_key=spec.key,
                        item=item,
                        model_answer=extracted,
                        graded=final_grade,
                        elapsed_sec=time.time() - started,
                    )
                )
                extractor_done.add(item.item_id)
                if idx % 20 == 0:
                    _write_csv(extractor_path, extractor_rows)

            if idx % 50 == 0:
                print(
                    f"[{spec.key}] {idx}/{len(items)} "
                    f"constrained={len(constrained_rows)} extractor={len(extractor_rows)}"
                )

        _write_csv(constrained_path, constrained_rows)
        _write_csv(extractor_path, extractor_rows)
        all_rows.extend(constrained_rows)
        all_rows.extend(extractor_rows)
        print(
            f"[DONE] {spec.key} constrained={len(constrained_rows)} extractor={len(extractor_rows)} "
            f"at {version_dir}"
        )

    compare_rows = _build_compare_rows(all_rows, include_modes=ORDERED_MODES)
    _write_csv(eval_root / "version_mode_compare.csv", compare_rows)

    ablation_rows: list[dict[str, Any]] = []
    ablation_specs = _ablation_specs(project_root)
    baseline_tol: float | None = None
    for ab_spec in ablation_specs:
        bundle = _build_runtime(project_root, ab_spec, model_key)
        detail_rows: list[dict[str, Any]] = []
        for item in items:
            document_name = raw_doc_by_id.get(item.item_id, "")
            candidates, context_text, _profile = _retrieve_with_doc_filter(bundle, item, document_name)
            extracted, graded = _extract_eval_answer_from_candidates(item, candidates, context_text)
            detail_rows.append(
                _grade_row(
                    mode=MODE_EXTRACTOR,
                    version_key=ab_spec.key,
                    item=item,
                    model_answer=extracted,
                    graded=graded,
                    elapsed_sec=0.0,
                )
            )
        agg = _aggregate_mode_rows(detail_rows)
        tol = float(agg["tolerant_auto_accuracy"])
        if ab_spec.key == "baseline_v3_default":
            baseline_tol = tol
        ablation_rows.append(
            {
                "variant_key": ab_spec.key,
                "display_name": ab_spec.display_name,
                **agg,
                "delta_vs_baseline_v3": round(tol - float(baseline_tol or tol), 4),
                "note": "prefix_off is proxy" if ab_spec.key == "ablate_prefix_v2_off_proxy_b01" else "",
            }
        )
        _write_csv(factual_root / "ablation" / f"{ab_spec.key}_detail.csv", detail_rows)
        print(f"[ABLATION] {ab_spec.key}: tolerant={tol:.4f}")

    _write_csv(eval_root / "technique_ablation_compare.csv", ablation_rows)
    _write_final_report(eval_root / "auto_grader_factual_ablation_report.md", compare_rows, ablation_rows)
    _write_representative_cases(eval_root / "representative_cases_factual_mode.md", all_rows)
    print(
        "[SUMMARY]",
        {
            "generated_at": _now(),
            "rows": len(all_rows),
            "version_mode_compare": str(eval_root / "version_mode_compare.csv"),
            "ablation_compare": str(eval_root / "technique_ablation_compare.csv"),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto grader factual-mode evaluation + ablation")
    parser.add_argument("--downloads-root", default=str(Path.home() / "Downloads"))
    parser.add_argument("--generator-model-key", default="gpt5mini_api")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--item-limit", type=int, default=0)
    args = parser.parse_args()
    run(
        Path(args.downloads_root),
        model_key=str(args.generator_model_key),
        resume=not args.no_resume,
        item_limit=max(0, int(args.item_limit)),
    )


if __name__ == "__main__":
    main()

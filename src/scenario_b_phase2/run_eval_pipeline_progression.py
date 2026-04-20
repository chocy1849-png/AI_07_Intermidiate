from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

from scenario_a.common_pipeline import CandidateRow, PipelinePaths, PipelineSettings, ScenarioAAnswer, ScenarioACommonPipeline
from scenario_b_phase2.phase2_pipeline import Phase2Options, ScenarioBPhase2Pipeline


def _now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _normalize_name(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    text = re.sub(r"\s+\.(hwp|pdf|docx?)$", r".\1", text, flags=re.IGNORECASE)
    return text.lower()


def _nospace(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().lower())


def _import_haeun_modules(haeun_dir: Path) -> tuple[Any, Any]:
    import sys

    if str(haeun_dir) not in sys.path:
        sys.path.insert(0, str(haeun_dir))
    import auto_grader  # type: ignore
    import llm_judge  # type: ignore

    return auto_grader, llm_judge


def _find_haeun_dir(downloads_root: Path) -> Path:
    docs_dir = downloads_root / "github" / "docs"
    if not docs_dir.exists():
        raise FileNotFoundError(f"github/docs not found: {docs_dir}")
    candidates = [path for path in docs_dir.iterdir() if path.is_dir() and (path / "auto_grader.py").exists()]
    if not candidates:
        raise FileNotFoundError(f"auto_grader.py not found under: {docs_dir}")
    return candidates[0]


def _resolve_project_root(downloads_root: Path) -> Path:
    candidates = [path for path in downloads_root.iterdir() if path.is_dir() and (path / "src" / "scenario_a" / "common_pipeline.py").exists()]
    if not candidates:
        raise FileNotFoundError("중급프로젝트 root not found under Downloads")
    return candidates[0]


def _load_question_bank(qbank_path: Path) -> list[dict[str, Any]]:
    data = json.loads(qbank_path.read_text(encoding="utf-8-sig"))
    items = data.get("items", [])
    if not items:
        raise RuntimeError(f"question bank empty: {qbank_path}")
    return items


def _load_judge_items(judge_path: Path) -> list[dict[str, Any]]:
    data = json.loads(judge_path.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict):
        items = data.get("items", [])
    else:
        items = data
    if not items:
        raise RuntimeError(f"judge subset empty: {judge_path}")
    return items


def _load_file_name_pool_from_bm25(path: Path) -> list[str]:
    import pickle

    if not path.exists():
        return []
    with path.open("rb") as file:
        payload = pickle.load(file)
    rows = payload.get("chunk_rows", []) or []
    names: list[str] = []
    seen: set[str] = set()
    for row in rows:
        name = str(row.get("source_file_name", "")).strip()
        if not name:
            continue
        key = _nospace(name)
        if key in seen:
            continue
        seen.add(key)
        names.append(name)
    return names


def _resolve_source_file_name(document_name: str, pool: list[str]) -> str:
    if not document_name:
        return ""
    doc = str(document_name).strip()
    if not doc:
        return ""
    doc_norm = _normalize_name(doc)
    doc_ns = _nospace(doc)
    for name in pool:
        if name == doc:
            return name
    for name in pool:
        if _normalize_name(name) == doc_norm:
            return name
    for name in pool:
        name_ns = _nospace(name)
        if doc_ns in name_ns or name_ns in doc_ns:
            return name
    return ""


def _row_matches_filter(row: CandidateRow, target_file_name: str) -> bool:
    if not target_file_name:
        return True
    row_name = str(row.metadata.get("source_file_name", "")).strip()
    if not row_name:
        return False
    a = _nospace(row_name)
    b = _nospace(target_file_name)
    return (a == b) or (a in b) or (b in a)


class FilteredScenarioACommonPipeline(ScenarioACommonPipeline):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._active_file_filter: str = ""

    def set_file_filter(self, file_name: str) -> None:
        self._active_file_filter = str(file_name or "")

    def clear_file_filter(self) -> None:
        self._active_file_filter = ""

    def vector_search(self, query_embedding: list[float]) -> list[CandidateRow]:
        rows = super().vector_search(query_embedding)
        if not self._active_file_filter:
            return rows
        filtered = [row for row in rows if _row_matches_filter(row, self._active_file_filter)]
        return filtered if filtered else rows

    def bm25_search(self, question: str) -> list[CandidateRow]:
        rows = super().bm25_search(question)
        if not self._active_file_filter:
            return rows
        filtered = [row for row in rows if _row_matches_filter(row, self._active_file_filter)]
        return filtered if filtered else rows


class FilteredScenarioBPhase2Pipeline(ScenarioBPhase2Pipeline):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._active_file_filter: str = ""

    def set_file_filter(self, file_name: str) -> None:
        self._active_file_filter = str(file_name or "")

    def clear_file_filter(self) -> None:
        self._active_file_filter = ""

    def vector_search(self, query_embedding: list[float]) -> list[CandidateRow]:
        rows = super().vector_search(query_embedding)
        if not self._active_file_filter:
            return rows
        filtered = [row for row in rows if _row_matches_filter(row, self._active_file_filter)]
        return filtered if filtered else rows

    def bm25_search(self, question: str) -> list[CandidateRow]:
        rows = super().bm25_search(question)
        if not self._active_file_filter:
            return rows
        filtered = [row for row in rows if _row_matches_filter(row, self._active_file_filter)]
        return filtered if filtered else rows


class RawB00Runner:
    def __init__(
        self,
        *,
        project_root: Path,
        chroma_dir: Path,
        collection_name: str,
        model_key: str,
        embedding_model: str = "text-embedding-3-small",
        candidate_k: int = 10,
    ) -> None:
        import chromadb

        self.project_root = project_root
        self.embedding_model = embedding_model
        self.candidate_k = candidate_k

        settings = PipelineSettings(
            embedding_backend_key="openai_text_embedding_3_small",
            factual_or_comparison_route="b02",
            default_route="b02",
            rejection_route="b02",
            follow_up_route="b02",
            candidate_k=candidate_k,
            top_k=5,
        )
        self._helper = ScenarioACommonPipeline(PipelinePaths(project_root=project_root), settings=settings)
        self.adapter = self._helper.create_adapter(model_key)
        client = chromadb.PersistentClient(path=str(chroma_dir))
        self.collection = client.get_collection(collection_name)

    @staticmethod
    def _build_system_instruction() -> str:
        return "\n".join(
            [
                "당신은 한국어 제안요청서(RFP) 기반 요약 도우미다.",
                "반드시 제공된 검색 문맥만 근거로 답한다.",
                "문맥에 없는 사실은 추정하지 않는다.",
                "질문 요구사항만 간결하게 답한다.",
                "문맥에 일정/예산이 없으면 없다고 명시한다.",
            ]
        )

    @staticmethod
    def _build_context(documents: list[str], metadatas: list[dict[str, Any]], distances: list[float]) -> str:
        blocks: list[str] = []
        for idx, (document, metadata, distance) in enumerate(zip(documents, metadatas, distances), start=1):
            blocks.append(
                "\n".join(
                    [
                        f"[검색 결과 {idx}]",
                        f"- 사업명: {metadata.get('사업명', '정보 없음')}",
                        f"- 발주 기관: {metadata.get('발주 기관', '정보 없음')}",
                        f"- 파일명: {metadata.get('source_file_name', '정보 없음')}",
                        f"- 청크 ID: {metadata.get('chunk_id', '정보 없음')}",
                        f"- 거리: {distance}",
                        str(document),
                    ]
                )
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _to_candidates(documents: list[str], metadatas: list[dict[str, Any]], distances: list[float]) -> list[CandidateRow]:
        rows: list[CandidateRow] = []
        for rank, (document, metadata, distance) in enumerate(zip(documents, metadatas, distances), start=1):
            rows.append(
                CandidateRow(
                    chunk_id=str(metadata.get("chunk_id", "")),
                    text=str(document),
                    metadata=dict(metadata),
                    fusion_score=max(0.0, 1.0 - float(distance or 1.0)) / max(1, rank),
                )
            )
        return rows

    def answer(
        self,
        *,
        question: str,
        history: list[dict[str, str]] | None,
        filter_file_name: str,
    ) -> ScenarioAAnswer:
        query_embedding = self._helper.openai_client.embeddings.create(model=self.embedding_model, input=[question]).data[0].embedding
        where = {"source_file_name": {"$eq": filter_file_name}} if filter_file_name else None
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.candidate_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        context_text = self._build_context(documents, metadatas, distances)
        answer_text = self.adapter.generate(
            system_instruction=self._build_system_instruction(),
            question=question,
            context_text=context_text,
            history=history or [],
        )
        return ScenarioAAnswer(
            route="raw_baseline_b00",
            question=question,
            profile={},
            context_text=context_text,
            candidates=self._to_candidates(documents, metadatas, distances),
            answer_text=answer_text,
        )


@dataclass(slots=True)
class VersionSpec:
    key: str
    display_name: str
    mapped_runner: str
    mapped_experiment_key: str
    mapped_collection: str
    mapped_bm25_index: str
    pipeline_kind: str  # raw_b00 | scenario_a | phase2
    optional: bool = False


class VersionEvaluator:
    def __init__(
        self,
        *,
        project_root: Path,
        output_root: Path,
        haeun_dir: Path,
        qbank_items: list[dict[str, Any]],
        judge_items: list[dict[str, Any]],
        generator_model_key: str,
        judge_model: str,
        include_optional: bool,
        auto_limit: int,
        judge_limit: int,
        version_keys: list[str],
        resume: bool,
        aggregate_only: bool,
        skip_auto: bool,
        skip_judge: bool,
        auto_range_start: int,
        auto_range_end: int,
    ) -> None:
        self.project_root = project_root
        self.output_root = output_root
        self.haeun_dir = haeun_dir
        self.qbank_items = qbank_items
        self.judge_items = judge_items
        self.generator_model_key = generator_model_key
        self.judge_model = judge_model
        self.include_optional = include_optional
        self.auto_limit = max(0, int(auto_limit))
        self.judge_limit = max(0, int(judge_limit))
        self.version_keys = [str(value).strip() for value in version_keys if str(value).strip()]
        self.resume = bool(resume)
        self.aggregate_only = bool(aggregate_only)
        self.skip_auto = bool(skip_auto)
        self.skip_judge = bool(skip_judge)
        self.auto_range_start = max(0, int(auto_range_start))
        self.auto_range_end = max(0, int(auto_range_end))
        self.output_root.mkdir(parents=True, exist_ok=True)

        self.auto_grader, self.llm_judge = _import_haeun_modules(haeun_dir)
        load_dotenv(project_root / ".env", override=False)
        if not (os.getenv("OPENAI_API_KEY") or "").strip():
            raise RuntimeError(f"OPENAI_API_KEY missing in {project_root / '.env'}")

        self.versions = self._build_version_specs()
        if self.version_keys:
            wanted = set(self.version_keys)
            self.versions = [spec for spec in self.versions if spec.key in wanted]
            missing = wanted - {spec.key for spec in self.versions}
            if missing:
                raise RuntimeError(f"unknown version_keys: {sorted(missing)}")

    def _build_version_specs(self) -> list[VersionSpec]:
        versions = [
            VersionSpec(
                key="b00_raw_baseline",
                display_name="B-00 raw baseline",
                mapped_runner="experiments/B00/03_나이브_RAG_베이스라인.py + raw_baseline_runner.py",
                mapped_experiment_key="raw_baseline_b00",
                mapped_collection="rfp_contextual_chunks_v1",
                mapped_bm25_index="(not used)",
                pipeline_kind="raw_b00",
            ),
            VersionSpec(
                key="b01_hybrid",
                display_name="B-01",
                mapped_runner="experiments/B01/08_B01_Hybrid_평가.py",
                mapped_experiment_key="b01_hybrid_rrf_0.7_0.3",
                mapped_collection="rfp_contextual_chunks_v1",
                mapped_bm25_index="rag_outputs/bm25_index_b01.pkl",
                pipeline_kind="phase2",
            ),
            VersionSpec(
                key="b02_prefix_v2",
                display_name="B-02",
                mapped_runner="experiments/B02/* + b02_prefix_v2_full_eval",
                mapped_experiment_key="b02_reference_profile",
                mapped_collection="rfp_contextual_chunks_v2_b02",
                mapped_bm25_index="rag_outputs/bm25_index_b02.pkl",
                pipeline_kind="phase2",
            ),
            VersionSpec(
                key="b06_exact_stage1",
                display_name="B-06 exact (stage1 final adopted)",
                mapped_runner="src/scenario_b_phase2/run_phase2_root_cause_ablation.py",
                mapped_experiment_key="baseline_aa (answer-layer parity)",
                mapped_collection="rfp_contextual_chunks_v2_b02",
                mapped_bm25_index="rag_outputs/bm25_index_b02.pkl",
                pipeline_kind="phase2",
            ),
            VersionSpec(
                key="phase2_baseline_v2",
                display_name="phase2 baseline_v2",
                mapped_runner="src/scenario_b_phase2/run_phase2_eval.py",
                mapped_experiment_key="phase2_baseline_v2",
                mapped_collection="rfp_contextual_chunks_v2_b02",
                mapped_bm25_index="rag_outputs/bm25_index_b02.pkl",
                pipeline_kind="phase2",
            ),
            VersionSpec(
                key="baseline_v3_ocr_v4_router_off",
                display_name="baseline_v3 (OCR v4 candidate, router off)",
                mapped_runner="src/scenario_b_phase2/run_phase2_true_table_ocr_v4_exact.py",
                mapped_experiment_key="true_table_ocr_v4_options + router_off",
                mapped_collection="rfp_contextual_chunks_v2_true_table_ocr_v4",
                mapped_bm25_index="rag_outputs/bm25_index_phase2_true_table_ocr_v4.pkl",
                pipeline_kind="phase2",
            ),
            VersionSpec(
                key="phase2_baseline_v1",
                display_name="baseline_v1 (optional)",
                mapped_runner="src/scenario_b_phase2/run_phase2_eval.py",
                mapped_experiment_key="phase2_baseline_v1",
                mapped_collection="rfp_contextual_chunks_v2_b02",
                mapped_bm25_index="rag_outputs/bm25_index_b02.pkl",
                pipeline_kind="phase2",
                optional=True,
            ),
            VersionSpec(
                key="current_serving_router_on",
                display_name="current serving config = baseline_v3 + answer_type_router_v2 (soft_crag_lite off)",
                mapped_runner="streamlit service + phase2 pipeline",
                mapped_experiment_key="baseline_v3 + router_on",
                mapped_collection="rfp_contextual_chunks_v2_true_table_ocr_v4",
                mapped_bm25_index="rag_outputs/bm25_index_phase2_true_table_ocr_v4.pkl",
                pipeline_kind="phase2",
                optional=True,
            ),
        ]
        if self.include_optional:
            return versions
        return [row for row in versions if not row.optional]

    def _build_pipeline(self, spec: VersionSpec) -> tuple[Any, Any, list[str], Callable[[str, list[dict[str, str]] | None, str], ScenarioAAnswer]]:
        if spec.pipeline_kind == "raw_b00":
            chroma_dir = self.project_root.parent / "rfp_rag_chroma_db"
            runner = RawB00Runner(
                project_root=self.project_root,
                chroma_dir=chroma_dir,
                collection_name="rfp_contextual_chunks_v1",
                model_key=self.generator_model_key,
                candidate_k=10,
            )
            pool = _load_file_name_pool_from_bm25(self.project_root / "rag_outputs" / "bm25_index_b01.pkl")

            def _answer(question: str, history: list[dict[str, str]] | None, filter_name: str) -> ScenarioAAnswer:
                return runner.answer(question=question, history=history, filter_file_name=filter_name)

            return runner, runner.adapter, pool, _answer

        if spec.key == "b01_hybrid":
            options = Phase2Options(
                enable_controlled_query_expansion=False,
                enable_normalized_bm25=False,
                enable_metadata_aware_retrieval=False,
                enable_table_body_pairing=False,
                enable_soft_crag_lite=False,
                enable_b03_legacy_crag_parity=False,
                enable_answer_type_router=False,
            )
            settings = PipelineSettings(
                embedding_backend_key="openai_text_embedding_3_small",
                candidate_k=10,
                top_k=5,
                vector_weight=0.7,
                bm25_weight=0.3,
                factual_or_comparison_route="b02",
                default_route="b02",
                rejection_route="b02",
                follow_up_route="b02",
            )
            pipeline = FilteredScenarioBPhase2Pipeline(
                PipelinePaths(
                    project_root=self.project_root,
                    chroma_dir=self.project_root.parent / "rfp_rag_chroma_db",
                    bm25_index_path=self.project_root / "rag_outputs" / "bm25_index_b01.pkl",
                ),
                settings=settings,
                options=options,
            )
            adapter = pipeline.create_adapter(self.generator_model_key)
            pool = _load_file_name_pool_from_bm25(self.project_root / "rag_outputs" / "bm25_index_b01.pkl")

            def _answer(question: str, history: list[dict[str, str]] | None, filter_name: str) -> ScenarioAAnswer:
                pipeline.set_file_filter(filter_name)
                try:
                    row = {"question": question, "answer_type": "factual", "depends_on_list": []}
                    return pipeline.answer(row, adapter, question=question, history=history)
                finally:
                    pipeline.clear_file_filter()

            return pipeline, adapter, pool, _answer

        if spec.key == "b02_prefix_v2":
            options = Phase2Options(
                enable_controlled_query_expansion=False,
                enable_normalized_bm25=False,
                enable_metadata_aware_retrieval=False,
                enable_table_body_pairing=False,
                enable_soft_crag_lite=False,
                enable_b03_legacy_crag_parity=False,
                enable_answer_type_router=False,
            )
            settings = PipelineSettings(
                embedding_backend_key="openai_text_embedding_3_small",
                candidate_k=10,
                top_k=5,
                vector_weight=0.7,
                bm25_weight=0.3,
                factual_or_comparison_route="b02",
                default_route="b02",
                rejection_route="b02",
                follow_up_route="b02",
            )
            pipeline = FilteredScenarioBPhase2Pipeline(
                PipelinePaths(
                    project_root=self.project_root,
                    chroma_dir=self.project_root.parent / "rfp_rag_chroma_db",
                    bm25_index_path=self.project_root / "rag_outputs" / "bm25_index_b02.pkl",
                ),
                settings=settings,
                options=options,
            )
            adapter = pipeline.create_adapter(self.generator_model_key)
            pool = _load_file_name_pool_from_bm25(self.project_root / "rag_outputs" / "bm25_index_b02.pkl")

            def _answer(question: str, history: list[dict[str, str]] | None, filter_name: str) -> ScenarioAAnswer:
                pipeline.set_file_filter(filter_name)
                try:
                    row = {"question": question, "answer_type": "factual", "depends_on_list": []}
                    return pipeline.answer(row, adapter, question=question, history=history)
                finally:
                    pipeline.clear_file_filter()

            return pipeline, adapter, pool, _answer

        if spec.key == "b06_exact_stage1":
            options = Phase2Options(
                enable_controlled_query_expansion=False,
                enable_normalized_bm25=False,
                enable_metadata_aware_retrieval=False,
                enable_table_body_pairing=False,
                enable_soft_crag_lite=False,
                enable_b03_legacy_crag_parity=True,
                enable_answer_type_router=False,
            )
            settings = PipelineSettings(
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
            )
            pipeline = FilteredScenarioBPhase2Pipeline(
                PipelinePaths(
                    project_root=self.project_root,
                    chroma_dir=self.project_root.parent / "rfp_rag_chroma_db",
                    bm25_index_path=self.project_root / "rag_outputs" / "bm25_index_b02.pkl",
                ),
                settings=settings,
                options=options,
            )
            adapter = pipeline.create_adapter(self.generator_model_key)
            pool = _load_file_name_pool_from_bm25(self.project_root / "rag_outputs" / "bm25_index_b02.pkl")

            def _answer(question: str, history: list[dict[str, str]] | None, filter_name: str) -> ScenarioAAnswer:
                pipeline.set_file_filter(filter_name)
                try:
                    row = {"question": question, "answer_type": "factual", "depends_on_list": []}
                    return pipeline.answer(row, adapter, question=question, history=history)
                finally:
                    pipeline.clear_file_filter()

            return pipeline, adapter, pool, _answer

        if spec.key == "phase2_baseline_v1":
            options = Phase2Options(
                enable_controlled_query_expansion=False,
                enable_normalized_bm25=False,
                enable_metadata_aware_retrieval=True,
                enable_table_body_pairing=False,
                enable_soft_crag_lite=False,
                metadata_disable_for_rejection=True,
                metadata_boost_scale=0.5,
                metadata_scope_mode="all",
                normalized_bm25_mode="all",
                enable_comparison_evidence_helper=False,
                enable_b03_legacy_crag_parity=True,
                enable_answer_type_router=False,
            )
            settings = PipelineSettings(
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
            )
            pipeline = FilteredScenarioBPhase2Pipeline(
                PipelinePaths(
                    project_root=self.project_root,
                    chroma_dir=self.project_root.parent / "rfp_rag_chroma_db",
                    bm25_index_path=self.project_root / "rag_outputs" / "bm25_index_b02.pkl",
                ),
                settings=settings,
                options=options,
            )
            adapter = pipeline.create_adapter(self.generator_model_key)
            pool = _load_file_name_pool_from_bm25(self.project_root / "rag_outputs" / "bm25_index_b02.pkl")

            def _answer(question: str, history: list[dict[str, str]] | None, filter_name: str) -> ScenarioAAnswer:
                pipeline.set_file_filter(filter_name)
                try:
                    row = {"question": question, "answer_type": "factual", "depends_on_list": []}
                    return pipeline.answer(row, adapter, question=question, history=history)
                finally:
                    pipeline.clear_file_filter()

            return pipeline, adapter, pool, _answer

        if spec.key in {"phase2_baseline_v2", "baseline_v3_ocr_v4_router_off", "current_serving_router_on"}:
            is_v3 = spec.key in {"baseline_v3_ocr_v4_router_off", "current_serving_router_on"}
            chroma_dir = self.project_root.parent / ("rfp_rag_chroma_db_phase2_true_table_ocr_v4" if is_v3 else "rfp_rag_chroma_db")
            bm25_path = self.project_root / "rag_outputs" / ("bm25_index_phase2_true_table_ocr_v4.pkl" if is_v3 else "bm25_index_b02.pkl")
            backend_key = "openai_text_embedding_3_small_true_table_ocr_v4" if is_v3 else "openai_text_embedding_3_small"

            options = Phase2Options(
                enable_controlled_query_expansion=False,
                enable_normalized_bm25=False,
                enable_metadata_aware_retrieval=True,
                enable_table_body_pairing=True if is_v3 else False,
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
                enable_groupc_table_plus_text_guard=True if is_v3 else False,
                groupc_pair_bonus=0.008 if is_v3 else 0.006,
                groupc_parent_bonus=0.004 if is_v3 else 0.003,
                groupc_table_penalty_without_body=0.015 if is_v3 else 0.012,
                enable_question_type_gated_ocr_routing=True if is_v3 else False,
                enable_structured_evidence_priority=True if is_v3 else False,
                enable_hybridqa_stage_metrics=True if is_v3 else False,
                enable_table_factual_exact_answer_mode=True if is_v3 else False,
                enable_table_factual_alignment_scoring=True if is_v3 else False,
                table_factual_generic_penalty=0.012,
                enable_answer_type_router=True if spec.key == "current_serving_router_on" else False,
                answer_type_router_force=False,
                answer_type_router_confidence_threshold=0.58,
            )
            settings = PipelineSettings(
                embedding_backend_key=backend_key,
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
            )
            pipeline = FilteredScenarioBPhase2Pipeline(
                PipelinePaths(
                    project_root=self.project_root,
                    chroma_dir=chroma_dir,
                    bm25_index_path=bm25_path,
                ),
                settings=settings,
                options=options,
            )
            adapter = pipeline.create_adapter(self.generator_model_key)
            pool = _load_file_name_pool_from_bm25(bm25_path)

            def _answer(question: str, history: list[dict[str, str]] | None, filter_name: str) -> ScenarioAAnswer:
                pipeline.set_file_filter(filter_name)
                try:
                    row = {"question": question, "answer_type": "factual", "depends_on_list": []}
                    return pipeline.answer(row, adapter, question=question, history=history)
                finally:
                    pipeline.clear_file_filter()

            return pipeline, adapter, pool, _answer

        raise RuntimeError(f"unsupported spec: {spec.key}")

    @staticmethod
    def _infer_answer_type_from_judge_type(item: dict[str, Any]) -> str:
        judge_type = str(item.get("judge_type", "")).strip().lower()
        return {
            "single_doc": "factual",
            "comparison": "comparison",
            "follow_up": "follow_up",
            "rejection": "rejection",
        }.get(judge_type, "factual")

    @staticmethod
    def _convert_candidates_for_judge(candidates: list[CandidateRow]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for candidate in candidates[:10]:
            score = float(candidate.fusion_score or 0.0)
            distance = max(0.0, 1.0 - min(1.0, score))
            rows.append(
                {
                    "chunk_id": candidate.chunk_id,
                    "text": candidate.text,
                    "metadata": dict(candidate.metadata),
                    "distance": distance,
                }
            )
        return rows

    def _build_mapping_md(self) -> str:
        lines = [
            "# Version Mapping",
            "",
            f"- generated_at: `{_now_iso()}`",
            f"- project_root: `{self.project_root}`",
            f"- output_root: `{self.output_root}`",
            f"- generator_model_fixed: `{self.generator_model_key}`",
            f"- judge_model_fixed: `{self.judge_model}`",
            "- router policy: historical baselines `router_off`, current serving row only `router_on`",
            "- soft_crag_lite policy: all rows `off`",
            "",
            "| version_key | display_name | mapped_runner | mapped_experiment_key | collection | bm25_index | status |",
            "|---|---|---|---|---|---|---|",
        ]
        for spec in self.versions:
            lines.append(
                "| "
                + " | ".join(
                    [
                        spec.key,
                        spec.display_name,
                        spec.mapped_runner,
                        spec.mapped_experiment_key,
                        spec.mapped_collection,
                        spec.mapped_bm25_index,
                        "mapped",
                    ]
                )
                + " |"
            )
        lines.append("")
        lines.append("## Mapping Gap Check")
        lines.append("")
        lines.append("- unmapped version: none")
        lines.append("- approximation used: none")
        return "\n".join(lines)

    def _run_auto_eval(
        self,
        *,
        spec: VersionSpec,
        answer_fn: Callable[[str, list[dict[str, str]] | None, str], ScenarioAAnswer],
        file_pool: list[str],
        existing_rows: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        results: list[dict[str, Any]] = list(existing_rows or [])
        done_ids = {str(row.get("id", "")).strip() for row in results}

        auto_items = self.qbank_items[: self.auto_limit] if self.auto_limit > 0 else self.qbank_items
        if self.auto_range_start > 0 or self.auto_range_end > 0:
            start_idx = max(1, self.auto_range_start or 1)
            end_idx = self.auto_range_end or len(auto_items)
            auto_items = auto_items[start_idx - 1 : end_idx]
        total = len(auto_items)
        for idx, item in enumerate(auto_items, start=1):
            item_id = str(item.get("id", "")).strip()
            if item_id in done_ids:
                continue
            question = str(item.get("question", "")).strip()
            document_name = str(item.get("document_name", "")).strip()
            choices = item.get("choices", []) or []
            if document_name:
                query = f"[{document_name}] {question}"
            else:
                query = question
            if choices:
                choices_text = "\n".join(f"  {choice}" for choice in choices)
                query = f"{query}\n\n[선택지]\n{choices_text}\n\n반드시 위 선택지 중 하나만 골라 번호와 텍스트를 함께 답하세요."
            filter_name = _resolve_source_file_name(document_name, file_pool)

            started = time.time()
            try:
                answered = answer_fn(query, None, filter_name)
                model_answer = answered.answer_text
            except Exception as exc:  # noqa: BLE001
                model_answer = f"[오류: {exc}]"
            elapsed = round(time.time() - started, 2)

            is_correct = bool(self.auto_grader.grade(model_answer, item))
            print(f"    [AUTO {idx:03d}/{total}] {spec.key} {item.get('id','')} => {'OK' if is_correct else 'FAIL'}")
            results.append(
                {
                    "id": item_id,
                    "document_id": item.get("document_id", ""),
                    "document_name": item.get("document_name", ""),
                    "question_type": item.get("question_type", ""),
                    "difficulty": item.get("difficulty", ""),
                    "category": item.get("category", ""),
                    "question": question,
                    "ground_truth": item.get("answer", ""),
                    "model_answer": model_answer,
                    "is_correct": is_correct,
                    "elapsed_sec": elapsed,
                    "version_key": spec.key,
                }
            )
            done_ids.add(item_id)
        normalized_results = self._normalize_is_correct_rows(results)
        summary = self.auto_grader.aggregate(normalized_results)
        return normalized_results, summary

    def _run_judge_eval(
        self,
        *,
        spec: VersionSpec,
        answer_fn: Callable[[str, list[dict[str, str]] | None, str], ScenarioAAnswer],
        file_pool: list[str],
        existing_rows: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        judge_items = self.judge_items[: self.judge_limit] if self.judge_limit > 0 else self.judge_items
        items_by_id = {str(item["id"]): item for item in judge_items}
        results: list[dict[str, Any]] = list(existing_rows or [])
        answers_by_id: dict[str, dict[str, Any]] = {
            str(row.get("id", "")).strip(): {"rag_answer": str(row.get("rag_answer", ""))}
            for row in results
            if str(row.get("id", "")).strip()
        }
        done_ids = {str(row.get("id", "")).strip() for row in results}
        total = len(judge_items)

        for idx, item in enumerate(judge_items, start=1):
            item_id = str(item.get("id", ""))
            if item_id in done_ids:
                continue
            started = time.time()
            depends_on_ids = self.llm_judge._normalize_depends_on(item.get("depends_on"))
            chat_history = self.llm_judge._build_chat_history(items_by_id, answers_by_id, depends_on_ids)

            question = str(item.get("question", "")).strip()
            document_name = str(item.get("document_name", "") or "").strip()
            filter_name = _resolve_source_file_name(document_name, file_pool)
            if document_name:
                query = f"[{document_name}] {question}"
            else:
                query = question

            answer_type = self._infer_answer_type_from_judge_type(item)
            try:
                answered = answer_fn(query, chat_history, filter_name)
                rag_answer = answered.answer_text
                retrieved_docs = self._convert_candidates_for_judge(answered.candidates)
            except Exception as exc:  # noqa: BLE001
                rag_answer = f"[RAG 오류: {exc}]"
                retrieved_docs = []

            retrieval_context = self.llm_judge._build_retrieval_context(retrieved_docs)
            judged = self.llm_judge._call_judge_model(
                item=item,
                rag_answer=rag_answer,
                retrieval_context=retrieval_context,
                model=self.judge_model,
            )
            avg_score, verdict = self.llm_judge._compute_verdict(item, judged)
            elapsed = round(time.time() - started, 2)
            answers_by_id[item_id] = {"rag_answer": rag_answer}
            done_ids.add(item_id)

            print(f"    [JUDGE {idx:02d}/{total}] {spec.key} {item_id} => {verdict.upper()} ({elapsed}s)")
            results.append(
                {
                    "id": item_id,
                    "judge_type": item.get("judge_type", ""),
                    "category": item.get("category", ""),
                    "difficulty": item.get("difficulty", ""),
                    "document_name": document_name,
                    "document_names": self.llm_judge._serialize_document_names(item),
                    "depends_on": ", ".join(depends_on_ids),
                    "question": question,
                    "answer_type": answer_type,
                    "ground_truth_hint": item.get("ground_truth_hint", "") or item.get("answer", ""),
                    "expected": item.get("expected", ""),
                    "eval_focus": item.get("eval_focus", ""),
                    "rag_answer": rag_answer,
                    "retrieval_context": retrieval_context[:6000],
                    "faithfulness_score": judged["faithfulness_score"],
                    "completeness_score": judged["completeness_score"],
                    "groundedness_score": judged["groundedness_score"],
                    "relevancy_score": judged["relevancy_score"],
                    "avg_score": avg_score,
                    "verdict": verdict,
                    "evaluator_note": judged["evaluator_note"],
                    "elapsed_sec": elapsed,
                    "version_key": spec.key,
                }
            )
        summary = self.llm_judge.aggregate(results)
        return results, summary

    @staticmethod
    def _pick_group_metric(rows: list[dict[str, Any]], group: str, key: str) -> float | None:
        row = next((item for item in rows if str(item.get("group", "")).strip() == group), None)
        if row is None:
            return None
        try:
            return float(row.get(key))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_is_correct_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        def _to_bool(value: Any) -> bool:
            text = str(value).strip().lower()
            return text in {"1", "true", "t", "yes", "y"}

        normalized: list[dict[str, Any]] = []
        for row in rows:
            copied = dict(row)
            copied["is_correct"] = _to_bool(copied.get("is_correct"))
            normalized.append(copied)
        return normalized

    def _build_compare_rows(
        self,
        version_rows: list[dict[str, Any]],
        *,
        metric_type: str,
    ) -> list[dict[str, Any]]:
        compare_rows: list[dict[str, Any]] = []
        for idx, row in enumerate(version_rows):
            out: dict[str, Any] = {
                "version_key": row["version_key"],
                "display_name": row["display_name"],
            }
            if metric_type == "auto":
                summary = row["auto_summary"]
                out.update(
                    {
                        "auto_overall_accuracy": self._pick_group_metric(summary, "전체", "accuracy"),
                        "multiple_choice_accuracy": self._pick_group_metric(summary, "type:multiple_choice", "accuracy"),
                        "short_answer_accuracy": self._pick_group_metric(summary, "type:short_answer", "accuracy"),
                        "basic_accuracy": self._pick_group_metric(summary, "difficulty:basic", "accuracy"),
                        "intermediate_accuracy": self._pick_group_metric(summary, "difficulty:intermediate", "accuracy"),
                        "advanced_accuracy": self._pick_group_metric(summary, "difficulty:advanced", "accuracy"),
                    }
                )
            else:
                summary = row["judge_summary"]
                out.update(
                    {
                        "llm_judge_average": self._pick_group_metric(summary, "overall", "avg_score"),
                        "faithfulness": self._pick_group_metric(summary, "overall", "avg_faithfulness_score"),
                        "completeness": self._pick_group_metric(summary, "overall", "avg_completeness_score"),
                        "groundedness": self._pick_group_metric(summary, "overall", "avg_groundedness_score"),
                        "relevancy": self._pick_group_metric(summary, "overall", "avg_relevancy_score"),
                    }
                )

            prev = compare_rows[idx - 1] if idx > 0 else None
            if metric_type == "auto":
                current = out.get("auto_overall_accuracy")
                base_value = compare_rows[0].get("auto_overall_accuracy") if compare_rows else current
                prev_value = prev.get("auto_overall_accuracy") if prev else None
            else:
                current = out.get("llm_judge_average")
                base_value = compare_rows[0].get("llm_judge_average") if compare_rows else current
                prev_value = prev.get("llm_judge_average") if prev else None

            if current is not None and base_value is not None:
                out["delta_vs_b00"] = round(float(current) - float(base_value), 4)
            else:
                out["delta_vs_b00"] = None
            if current is not None and prev_value is not None:
                out["delta_vs_prev"] = round(float(current) - float(prev_value), 4)
            else:
                out["delta_vs_prev"] = None
            compare_rows.append(out)
        return compare_rows

    def _write_csv(self, path: Path, rows: list[dict[str, Any]]) -> None:
        import csv

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

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, Any]]:
        import csv

        if not path.exists():
            return []
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            return list(csv.DictReader(file))

    def _write_summary_md(self, auto_compare: list[dict[str, Any]], judge_compare: list[dict[str, Any]]) -> None:
        lines = [
            "# Final Stage Progression Summary",
            "",
            f"- generated_at: `{_now_iso()}`",
            f"- generator_model_fixed: `{self.generator_model_key}`",
            f"- judge_model_fixed: `{self.judge_model}`",
            f"- auto_grader_count: `{len(self.qbank_items)}`",
            f"- llm_judge_subset_count: `{len(self.judge_items)}`",
            "- soft_crag_lite: `off` (all rows)",
            "",
            "## Auto Grader (340)",
            "",
            "| version | overall | MC | short | basic | intermediate | advanced | delta_vs_b00 | delta_vs_prev |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in auto_compare:
            lines.append(
                f"| {row['version_key']} | {row.get('auto_overall_accuracy')} | {row.get('multiple_choice_accuracy')} | "
                f"{row.get('short_answer_accuracy')} | {row.get('basic_accuracy')} | {row.get('intermediate_accuracy')} | "
                f"{row.get('advanced_accuracy')} | {row.get('delta_vs_b00')} | {row.get('delta_vs_prev')} |"
            )
        lines.extend(
            [
                "",
                "## LLM Judge (20)",
                "",
                "| version | avg | faithfulness | completeness | groundedness | relevancy | delta_vs_b00 | delta_vs_prev |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in judge_compare:
            lines.append(
                f"| {row['version_key']} | {row.get('llm_judge_average')} | {row.get('faithfulness')} | "
                f"{row.get('completeness')} | {row.get('groundedness')} | {row.get('relevancy')} | "
                f"{row.get('delta_vs_b00')} | {row.get('delta_vs_prev')} |"
            )
        lines.append("")
        lines.append("## Interpretation Rule")
        lines.append("")
        lines.append("- historical baseline rows: router off")
        lines.append("- current serving row only: router on")
        lines.append("- serving row is reference only (not mixed into historical baseline interpretation)")
        (self.output_root / "final_stage_progression_summary.md").write_text("\n".join(lines), encoding="utf-8")

    def run(self) -> None:
        (self.output_root / "version_mapping.md").write_text(self._build_mapping_md(), encoding="utf-8")

        version_rows: list[dict[str, Any]] = []
        for spec in self.versions:
            version_dir = self.output_root / spec.key
            version_dir.mkdir(parents=True, exist_ok=True)
            auto_summary_path = version_dir / "auto_eval_summary.csv"
            judge_summary_path = version_dir / "judge_summary.csv"
            auto_detail_path = version_dir / "auto_eval_detail.csv"
            judge_detail_path = version_dir / "judge_detail.csv"
            existing_auto = self._read_csv(auto_summary_path)
            existing_judge = self._read_csv(judge_summary_path)
            existing_auto_detail = self._read_csv(auto_detail_path)
            existing_judge_detail = self._read_csv(judge_detail_path)

            if self.aggregate_only:
                if not existing_auto or not existing_judge:
                    raise RuntimeError(f"aggregate_only mode requires existing files: {version_dir}")
                version_rows.append(
                    {
                        "version_key": spec.key,
                        "display_name": spec.display_name,
                        "auto_summary": existing_auto,
                        "judge_summary": existing_judge,
                    }
                )
                continue

            if self.resume and existing_auto and existing_judge and not self.skip_auto and not self.skip_judge and self.auto_range_start == 0 and self.auto_range_end == 0:
                print(f"\n[SKIP-RESUME] {spec.key} ({spec.display_name})")
                version_rows.append(
                    {
                        "version_key": spec.key,
                        "display_name": spec.display_name,
                        "auto_summary": existing_auto,
                        "judge_summary": existing_judge,
                    }
                )
                continue

            print(f"\n[RUN] {spec.key} ({spec.display_name})")
            _, _, file_pool, answer_fn = self._build_pipeline(spec)

            if self.skip_auto:
                if not existing_auto_detail:
                    raise RuntimeError(f"--skip-auto set but auto detail missing: {auto_detail_path}")
                auto_detail = self._normalize_is_correct_rows(existing_auto_detail)
                auto_summary = self.auto_grader.aggregate(auto_detail)
            else:
                auto_detail, auto_summary = self._run_auto_eval(
                    spec=spec,
                    answer_fn=answer_fn,
                    file_pool=file_pool,
                    existing_rows=existing_auto_detail if self.resume else None,
                )
            self._write_csv(version_dir / "auto_eval_detail.csv", auto_detail)
            self._write_csv(version_dir / "auto_eval_summary.csv", auto_summary)

            if self.skip_judge:
                judge_detail = existing_judge_detail
                judge_summary = self.llm_judge.aggregate(judge_detail) if judge_detail else []
            else:
                judge_detail, judge_summary = self._run_judge_eval(
                    spec=spec,
                    answer_fn=answer_fn,
                    file_pool=file_pool,
                    existing_rows=existing_judge_detail if self.resume else None,
                )
            if judge_detail:
                self.llm_judge.save_csv(version_dir / "judge_detail.csv", judge_detail)
                self.llm_judge.save_csv(version_dir / "judge_summary.csv", judge_summary)
                (version_dir / "judge_report.md").write_text(self.llm_judge.build_report(judge_detail, judge_summary, model=self.judge_model), encoding="utf-8")

            version_rows.append(
                {
                    "version_key": spec.key,
                    "display_name": spec.display_name,
                    "auto_summary": auto_summary,
                    "judge_summary": judge_summary,
                }
            )

        auto_compare = self._build_compare_rows(version_rows, metric_type="auto")
        judge_compare = self._build_compare_rows(version_rows, metric_type="judge")
        self._write_csv(self.output_root / "auto_grader_compare.csv", auto_compare)
        self._write_csv(self.output_root / "llm_judge_compare.csv", judge_compare)
        self._write_summary_md(auto_compare, judge_compare)

        manifest = {
            "generated_at": _now_iso(),
            "project_root": str(self.project_root),
            "output_root": str(self.output_root),
            "generator_model_key": self.generator_model_key,
            "judge_model": self.judge_model,
            "version_keys": [spec.key for spec in self.versions],
        }
        (self.output_root / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[DONE] outputs: {self.output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified auto-grader(340) + llm-judge(20) progression eval.")
    parser.add_argument("--downloads-root", default=str(Path.home() / "Downloads"))
    parser.add_argument("--generator-model-key", default="gpt5mini_api")
    parser.add_argument("--judge-model", default="gpt-5")
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--auto-limit", type=int, default=0)
    parser.add_argument("--judge-limit", type=int, default=0)
    parser.add_argument("--version-keys", nargs="*", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--skip-auto", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--auto-range-start", type=int, default=0)
    parser.add_argument("--auto-range-end", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    downloads_root = Path(args.downloads_root).resolve()
    project_root = _resolve_project_root(downloads_root)
    haeun_dir = _find_haeun_dir(downloads_root)

    qbank_path = next(path for path in haeun_dir.iterdir() if "QBank" in path.name and path.suffix.lower() == ".json")
    judge_path = haeun_dir / "judge_subset_20.json"
    qbank_items = _load_question_bank(qbank_path)
    judge_items = _load_judge_items(judge_path)

    output_root = project_root / "rag_outputs" / "eval_pipeline"
    evaluator = VersionEvaluator(
        project_root=project_root,
        output_root=output_root,
        haeun_dir=haeun_dir,
        qbank_items=qbank_items,
        judge_items=judge_items,
        generator_model_key=args.generator_model_key,
        judge_model=args.judge_model,
        include_optional=bool(args.include_optional),
        auto_limit=args.auto_limit,
        judge_limit=args.judge_limit,
        version_keys=args.version_keys,
        resume=bool(args.resume),
        aggregate_only=bool(args.aggregate_only),
        skip_auto=bool(args.skip_auto),
        skip_judge=bool(args.skip_judge),
        auto_range_start=args.auto_range_start,
        auto_range_end=args.auto_range_end,
    )
    evaluator.run()


if __name__ == "__main__":
    main()

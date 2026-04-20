from __future__ import annotations

import json
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv

from scenario_a.embedding_backends import create_embedding_backend
from scenario_a.embedding_backends.base import EmbeddingBackendConfig, EmbeddingBackendProtocol
from scenario_a.model_adapters import create_adapter
from scenario_a.model_adapters.base import AdapterConfig, BaseModelAdapter


def _load_yaml() -> Any:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required for Scenario A config loading.") from exc
    return yaml


def _require_openai() -> Any:
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError("openai package is required for Scenario A routing.") from exc
    return OpenAI


def _require_chromadb() -> Any:
    try:
        import chromadb
    except ModuleNotFoundError as exc:
        raise RuntimeError("chromadb package is required for Scenario A retrieval.") from exc
    return chromadb


def _is_ascii_only_path(path: Path) -> bool:
    try:
        str(path).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


@dataclass(slots=True)
class PipelinePaths:
    project_root: Path
    chroma_dir: Path | None = None
    bm25_index_path: Path | None = None
    model_config_path: Path | None = None
    env_path: Path | None = None

    def resolved(self) -> "PipelinePaths":
        root = self.project_root.resolve()
        chroma_override = (os.getenv("RFP_CHROMA_DIR") or "").strip()
        if self.chroma_dir is not None:
            chroma_dir = self.chroma_dir
        elif chroma_override:
            chroma_dir = Path(chroma_override)
        else:
            repo_local_chroma = root / "rag_outputs" / "chroma_db"
            legacy_external_chroma = root.parent / "rfp_rag_chroma_db"
            if _is_ascii_only_path(repo_local_chroma):
                chroma_dir = repo_local_chroma
            else:
                chroma_dir = legacy_external_chroma

        return PipelinePaths(
            project_root=root,
            chroma_dir=Path(chroma_dir),
            bm25_index_path=self.bm25_index_path,
            model_config_path=self.model_config_path or (root / "config" / "scenario_a_models.yaml"),
            env_path=self.env_path or (root / ".env"),
        )


@dataclass(slots=True)
class PipelineSettings:
    embedding_backend_key: str = "openai_text_embedding_3_small"
    routing_model: str = "gpt-5-mini"
    candidate_k: int = 10
    top_k: int = 5
    crag_top_n: int = 5
    vector_weight: float = 0.7
    bm25_weight: float = 0.3
    rrf_k: int = 60
    factual_or_comparison_route: str = "b03a"
    default_route: str = "b02"
    rejection_route: str = "b02"
    follow_up_route: str = "b02"


@dataclass(slots=True)
class CandidateRow:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    fusion_score: float
    adjusted_score: float | None = None
    crag_label: str = ""
    crag_reason: str = ""


@dataclass(slots=True)
class RetrievalResult:
    route: str
    profile: dict[str, Any]
    candidates: list[CandidateRow]
    context_text: str


@dataclass(slots=True)
class ScenarioAAnswer:
    route: str
    question: str
    profile: dict[str, Any]
    context_text: str
    candidates: list[CandidateRow]
    answer_text: str


class ScenarioACommonPipeline:
    def __init__(self, paths: PipelinePaths, settings: PipelineSettings | None = None) -> None:
        self.paths = paths.resolved()
        self.settings = settings or PipelineSettings()
        self._openai_client: Any | None = None
        self._chroma_collection: Any | None = None
        self._bm25_index: dict[str, Any] | None = None
        self._embedding_backend: EmbeddingBackendProtocol | None = None
        self._model_config_payload: dict[str, Any] | None = None

    def candidate_chroma_dirs(self) -> list[Path]:
        seen: set[str] = set()
        candidates: list[Path] = []
        for candidate in [
            self.paths.chroma_dir,
            self.paths.project_root / "rag_outputs" / "chroma_db",
            self.paths.project_root.parent / "rfp_rag_chroma_db",
        ]:
            if candidate is None:
                continue
            key = str(Path(candidate).resolve())
            if key in seen:
                continue
            seen.add(key)
            candidates.append(Path(candidate))
        return candidates

    def validate(self) -> None:
        if not self.paths.model_config_path.exists():
            raise FileNotFoundError(f"Scenario A config not found: {self.paths.model_config_path}")
        if not self.paths.chroma_dir.exists():
            raise FileNotFoundError(f"Chroma directory not found: {self.paths.chroma_dir}")

    @property
    def model_config_payload(self) -> dict[str, Any]:
        if self._model_config_payload is None:
            yaml = _load_yaml()
            self._model_config_payload = yaml.safe_load(
                self.paths.model_config_path.read_text(encoding="utf-8")
            )
        return self._model_config_payload

    @property
    def openai_client(self) -> Any:
        if self._openai_client is None:
            self.validate()
            load_dotenv(self.paths.env_path, override=False)
            api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is required for Scenario A routing.")
            base_url = (os.getenv("OPENAI_BASE_URL") or "").strip() or "https://api.openai.com/v1"
            OpenAI = _require_openai()
            self._openai_client = OpenAI(api_key=api_key, base_url=base_url)
        return self._openai_client

    def load_embedding_config(self, backend_key: str | None = None) -> EmbeddingBackendConfig:
        key = backend_key or self.settings.embedding_backend_key
        payload = self.model_config_payload
        defaults = payload.get("embedding_defaults", {}) or {}
        backends = payload.get("embedding_backends", {}) or {}
        if key not in backends:
            raise KeyError(f"Unknown embedding backend key: {key}")

        merged = dict(defaults)
        merged.update(backends[key] or {})
        return EmbeddingBackendConfig(
            backend_key=key,
            backend_name=merged["backend"],
            model_name=merged["model_name"],
            collection_name=merged["collection_name"],
            bm25_index_name=merged["bm25_index_name"],
            query_prefix=merged.get("query_prefix", ""),
            document_prefix=merged.get("document_prefix", ""),
            normalize_embeddings=bool(merged.get("normalize_embeddings", True)),
            encode_kwargs=dict(merged.get("encode_kwargs", {}) or {}),
            screening_only=bool(merged.get("screening_only", False)),
            note=str(merged.get("note", "")),
        )

    @property
    def embedding_backend(self) -> EmbeddingBackendProtocol:
        if self._embedding_backend is None:
            backend = create_embedding_backend(self.load_embedding_config())
            if hasattr(backend, "bind_client"):
                backend.bind_client(self.openai_client)
            self._embedding_backend = backend
        return self._embedding_backend

    def resolve_bm25_index_path(self, backend_key: str | None = None) -> Path:
        if self.paths.bm25_index_path is not None:
            return self.paths.bm25_index_path
        config = self.load_embedding_config(backend_key)
        return self.paths.project_root / "rag_outputs" / config.bm25_index_name

    @property
    def chroma_collection(self) -> Any:
        if self._chroma_collection is None:
            self.validate()
            chromadb = _require_chromadb()
            target_name = self.embedding_backend.config.collection_name
            last_error: Exception | None = None
            for chroma_dir in self.candidate_chroma_dirs():
                if not chroma_dir.exists():
                    continue
                try:
                    client = chromadb.PersistentClient(path=str(chroma_dir))
                    collection = client.get_collection(target_name)
                    self.paths.chroma_dir = chroma_dir
                    self._chroma_collection = collection
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = exc

            if self._chroma_collection is None:
                if last_error is not None:
                    raise last_error
                raise FileNotFoundError(
                    f"Could not resolve Chroma collection {target_name} from candidate paths: "
                    + ", ".join(str(path) for path in self.candidate_chroma_dirs())
                )
        return self._chroma_collection

    @property
    def bm25_index(self) -> dict[str, Any]:
        if self._bm25_index is None:
            bm25_path = self.resolve_bm25_index_path()
            if not bm25_path.exists():
                raise FileNotFoundError(f"BM25 index not found: {bm25_path}")
            with bm25_path.open("rb") as file:
                self._bm25_index = pickle.load(file)
        return self._bm25_index

    def load_model_config(self, model_key: str) -> AdapterConfig:
        payload = self.model_config_payload
        defaults = payload.get("defaults", {}) or {}
        models = payload.get("models", {}) or {}
        if model_key not in models:
            raise KeyError(f"Unknown model key: {model_key}")
        merged = dict(defaults)
        merged.update(models[model_key] or {})
        return AdapterConfig(
            model_key=model_key,
            adapter_name=merged["adapter"],
            model_id=merged["model_id"],
            tokenizer_id=merged.get("tokenizer_id"),
            runtime=merged.get("runtime", "transformers"),
            ollama_model_name=merged.get("ollama_model_name"),
            device_map=merged.get("device_map", "auto"),
            torch_dtype=merged.get("torch_dtype", "auto"),
            trust_remote_code=bool(merged.get("trust_remote_code", False)),
            max_new_tokens=int(merged.get("max_new_tokens", 768)),
            temperature=float(merged.get("temperature", 0.0)),
            top_p=float(merged.get("top_p", 1.0)),
            repetition_penalty=float(merged.get("repetition_penalty", 1.0)),
            stop_sequences=tuple(merged.get("stop_sequences", []) or []),
            use_chat_template=bool(merged.get("use_chat_template", True)),
            extra_generate_kwargs=dict(merged.get("extra_generate_kwargs", {}) or {}),
        )

    def create_adapter(self, model_key: str) -> BaseModelAdapter:
        return create_adapter(self.load_model_config(model_key))

    def decide_route(self, question_row: dict[str, Any]) -> str:
        answer_type = str(question_row.get("answer_type", "")).strip()
        has_dependency = bool(question_row.get("depends_on_list"))
        if has_dependency or answer_type == "follow_up":
            return self.settings.follow_up_route
        if answer_type == "rejection":
            return self.settings.rejection_route
        if answer_type in {"factual", "comparison"}:
            return self.settings.factual_or_comparison_route
        return self.settings.default_route

    def build_question_profile(self, question_row: dict[str, Any], question: str) -> dict[str, Any]:
        q = str(question or "")
        answer_type = str(question_row.get("answer_type", ""))
        return {
            "answer_type": answer_type,
            "budget": bool(re.search(r"(예산|금액|사업비|기초금액|추정금액|얼마)", q)),
            "schedule": bool(re.search(r"(기간|일정|마감|언제|몇\s*일|개월|사업기간|수행기간|완료일)", q)),
            "contract": bool(re.search(r"(계약방식|입찰방식|계약|입찰|협상)", q)),
            "purpose": bool(re.search(r"(목적|배경|필요성|목표)", q)),
            "comparison": answer_type == "comparison" or bool(re.search(r"(비교|차이|각각|모두|공통|서로|이상|이하)", q)),
            "follow_up": answer_type == "follow_up" or bool(question_row.get("depends_on_list")),
            "rejection": answer_type == "rejection",
        }

    def embed_question(self, question: str) -> list[float]:
        return self.embedding_backend.embed_queries([question])[0]

    def vector_search(self, query_embedding: list[float]) -> list[CandidateRow]:
        result = self.chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=self.settings.candidate_k,
            include=["documents", "metadatas", "distances"],
        )
        rows: list[CandidateRow] = []
        for rank, (document, metadata, _) in enumerate(
            zip(result["documents"][0], result["metadatas"][0], result["distances"][0]),
            start=1,
        ):
            rows.append(
                CandidateRow(
                    chunk_id=str(metadata.get("chunk_id", "")),
                    text=document,
                    metadata=dict(metadata),
                    fusion_score=(1.0 / (self.settings.rrf_k + rank)) * self.settings.vector_weight,
                )
            )
        return rows

    @staticmethod
    def _bm25_tokenize(text: str) -> list[str]:
        return re.findall(r"[0-9A-Za-z가-힣]+", str(text or "").lower())

    def bm25_search(self, question: str) -> list[CandidateRow]:
        model = self.bm25_index["model"]
        chunk_rows = self.bm25_index["chunk_rows"]
        scores = model.get_scores(self._bm25_tokenize(question))
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[: self.settings.candidate_k]
        rows: list[CandidateRow] = []
        for rank, (row_index, _) in enumerate(ranked, start=1):
            source = chunk_rows[row_index]
            rows.append(
                CandidateRow(
                    chunk_id=str(source.get("chunk_id", "")),
                    text=str(source.get("contextual_chunk_text", "")),
                    metadata=dict(source),
                    fusion_score=(1.0 / (self.settings.rrf_k + rank)) * self.settings.bm25_weight,
                )
            )
        return rows

    def fuse_candidates(self, vector_rows: Iterable[CandidateRow], bm25_rows: Iterable[CandidateRow]) -> list[CandidateRow]:
        fused: dict[str, CandidateRow] = {}
        for row in [*vector_rows, *bm25_rows]:
            if row.chunk_id not in fused:
                fused[row.chunk_id] = CandidateRow(
                    chunk_id=row.chunk_id,
                    text=row.text,
                    metadata=dict(row.metadata),
                    fusion_score=0.0,
                )
            fused[row.chunk_id].fusion_score += row.fusion_score
        return sorted(fused.values(), key=lambda item: item.fusion_score, reverse=True)

    def field_bonus(self, row: CandidateRow, profile: dict[str, Any], question: str) -> float:
        metadata = row.metadata
        metadata_text = " ".join(
            str(metadata.get(key, ""))
            for key in [
                "source_file_name",
                "section_title",
                "chunk_role",
                "chunk_role_tags",
                "purpose_summary",
                "period_raw",
                "contract_method",
                "bid_method",
                "budget_text",
            ]
        )
        bonus = 0.0
        if profile["budget"] and int(metadata.get("has_budget_signal", 0) or 0):
            bonus += 0.0045
        if profile["schedule"] and int(metadata.get("has_schedule_signal", 0) or 0):
            bonus += 0.0045
        if profile["contract"] and int(metadata.get("has_contract_signal", 0) or 0):
            bonus += 0.0045
        if profile["purpose"] and any(token in metadata_text for token in ["목적", "배경", "필요"]):
            bonus += 0.0035
        if int(metadata.get("has_table", 0) or 0) and (profile["budget"] or profile["schedule"]):
            bonus += 0.0015

        overlap = set(self._bm25_tokenize(question)) & set(self._bm25_tokenize(metadata_text))
        bonus += min(0.0030, len(overlap) * 0.0005)
        return bonus

    def rerank_with_profile(self, candidates: list[CandidateRow], profile: dict[str, Any], question: str) -> list[CandidateRow]:
        reranked: list[CandidateRow] = []
        for row in candidates:
            reranked.append(
                CandidateRow(
                    chunk_id=row.chunk_id,
                    text=row.text,
                    metadata=dict(row.metadata),
                    fusion_score=row.fusion_score,
                    adjusted_score=row.fusion_score + self.field_bonus(row, profile, question),
                )
            )
        return sorted(reranked, key=lambda item: item.adjusted_score or item.fusion_score, reverse=True)

    @staticmethod
    def _extract_json_block(text: str) -> Any:
        match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if not match:
            raise ValueError("JSON block not found in model response.")
        return json.loads(match.group(1))

    def _call_router_llm(self, prompt: str) -> str:
        response = self.openai_client.responses.create(
            model=self.settings.routing_model,
            input=prompt,
        )
        return (response.output_text or "").strip()

    def apply_crag(self, question_row: dict[str, Any], question: str, candidates: list[CandidateRow]) -> list[CandidateRow]:
        if not candidates:
            return []
        target = candidates[: self.settings.crag_top_n]
        candidate_blocks = []
        for index, row in enumerate(target, start=1):
            candidate_blocks.append(
                "\n".join(
                    [
                        f"[candidate_{index}]",
                        f"chunk_id: {row.chunk_id}",
                        f"source_file_name: {row.metadata.get('source_file_name', '')}",
                        f"section_title: {row.metadata.get('section_title', '')}",
                        f"chunk_role: {row.metadata.get('chunk_role', '')}",
                        "snippet:",
                        row.text[:900],
                    ]
                )
            )

        prompt = "\n\n".join(
            [
                "You are the fixed CRAG routing layer for Scenario A.",
                "Classify each candidate as CORRECT, AMBIGUOUS, or INCORRECT for answering the question.",
                "Use only the question and candidate evidence. Do not guess missing facts.",
                "Return JSON only in this format:",
                '[{"chunk_id":"...","label":"CORRECT|AMBIGUOUS|INCORRECT","reason":"..."}]',
                f"question_id: {question_row.get('question_id', '')}",
                f"answer_type: {question_row.get('answer_type', '')}",
                f"question: {question}",
                "\n\n".join(candidate_blocks),
            ]
        )

        try:
            raw_text = self._call_router_llm(prompt)
            judgments = self._extract_json_block(raw_text)
        except Exception:  # noqa: BLE001
            return candidates[: self.settings.top_k]
        label_by_chunk = {
            str(item.get("chunk_id", "")): (
                str(item.get("label", "")).upper(),
                str(item.get("reason", "")),
            )
            for item in judgments
            if item.get("chunk_id")
        }

        filtered: list[CandidateRow] = []
        for row in candidates:
            label, reason = label_by_chunk.get(row.chunk_id, ("", ""))
            enriched = CandidateRow(
                chunk_id=row.chunk_id,
                text=row.text,
                metadata=dict(row.metadata),
                fusion_score=row.fusion_score,
                adjusted_score=row.adjusted_score,
                crag_label=label,
                crag_reason=reason,
            )
            if label == "CORRECT":
                filtered.append(enriched)
            elif label == "AMBIGUOUS" and len(filtered) < self.settings.top_k:
                filtered.append(enriched)

        return filtered[: self.settings.top_k] if filtered else candidates[: self.settings.top_k]

    def build_context_text(self, candidates: list[CandidateRow]) -> str:
        blocks = []
        for index, row in enumerate(candidates[: self.settings.top_k], start=1):
            blocks.append(
                "\n".join(
                    [
                        f"[evidence_{index}]",
                        f"source_file_name: {row.metadata.get('source_file_name', '')}",
                        f"section_title: {row.metadata.get('section_title', '')}",
                        f"chunk_role: {row.metadata.get('chunk_role', '')}",
                        row.text,
                    ]
                )
            )
        return "\n\n".join(blocks)

    def retrieve(self, question_row: dict[str, Any], question: str) -> RetrievalResult:
        route = self.decide_route(question_row)
        profile = self.build_question_profile(question_row, question)
        query_embedding = self.embed_question(question)
        vector_rows = self.vector_search(query_embedding)
        bm25_rows = self.bm25_search(question)
        fused = self.fuse_candidates(vector_rows, bm25_rows)
        reranked = self.rerank_with_profile(fused, profile, question)
        selected = reranked[: self.settings.top_k]
        if route == "b03a":
            selected = self.apply_crag(question_row, question, reranked)
        return RetrievalResult(
            route=route,
            profile=profile,
            candidates=selected,
            context_text=self.build_context_text(selected),
        )

    @staticmethod
    def build_system_instruction() -> str:
        return "\n".join(
            [
                "You are the fixed answer layer for Scenario A.",
                "Use only the provided document evidence.",
                "Do not invent facts that are not supported by the evidence.",
                "If the evidence is insufficient, say it is insufficient.",
                "For comparison questions, state the comparison axis and differences clearly.",
                "For table-plus-text questions, reflect both the table evidence and the body evidence when available.",
            ]
        )

    def answer(
        self,
        question_row: dict[str, Any],
        adapter: BaseModelAdapter,
        *,
        question: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> ScenarioAAnswer:
        resolved_question = question or str(question_row.get("question", "")).strip()
        retrieval = self.retrieve(question_row, resolved_question)
        answer_text = adapter.generate(
            system_instruction=self.build_system_instruction(),
            question=resolved_question,
            context_text=retrieval.context_text,
            history=history or [],
        )
        return ScenarioAAnswer(
            route=retrieval.route,
            question=resolved_question,
            profile=retrieval.profile,
            context_text=retrieval.context_text,
            candidates=retrieval.candidates,
            answer_text=answer_text,
        )

    @staticmethod
    def allowed_adapter_differences() -> list[str]:
        return [
            "tokenizer",
            "chat template",
            "max_new_tokens",
            "stop sequence",
            "local inference runtime",
        ]

    @staticmethod
    def forbidden_adapter_differences() -> list[str]:
        return [
            "retrieval 결과 변경",
            "질문 포맷 의미 변경",
            "답변 지시문 의미 변경",
            "평가셋/평가 로직 분기",
        ]

    def describe_retrieval_assets(self) -> dict[str, str]:
        config = self.load_embedding_config()
        return {
            "embedding_backend": config.backend_key,
            "embedding_model": config.model_name,
            "chroma_dir": str(self.paths.chroma_dir),
            "collection_name": config.collection_name,
            "bm25_index_path": str(self.resolve_bm25_index_path(config.backend_key)),
        }

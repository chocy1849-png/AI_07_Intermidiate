from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from scenario_a.common_pipeline import PipelinePaths, ScenarioAAnswer
from scenario_b_phase2.phase2_pipeline import ScenarioBPhase2Pipeline

from streamlit_qa.config import build_phase2_options, build_pipeline_settings


def guess_answer_type(question: str) -> str:
    lowered = str(question or "").lower()
    if re.search(r"(비교|차이|공통|각각|대비|comparison)", lowered):
        return "comparison"
    if re.search(r"(불가|없나요|없는가|불가능|해당 없음|없습니까)", lowered):
        return "rejection"
    return "factual"


def build_chat_question_row(
    question: str,
    *,
    answer_type: str | None = None,
    router_result: dict[str, Any] | None = None,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    resolved_answer_type = str(answer_type or "").strip() or guess_answer_type(question)
    depends_on_list = ["__history__"] if resolved_answer_type == "follow_up" and history else []
    return {
        "question_id": f"chat_{uuid4().hex[:12]}",
        "question_index": 0,
        "type_group": "CHAT",
        "answer_type": resolved_answer_type,
        "router_route": str((router_result or {}).get("route", "")),
        "router_confidence": float((router_result or {}).get("confidence", 0.0) or 0.0),
        "router_signals": "|".join(str(item) for item in (router_result or {}).get("signals", []) or []),
        "router_reason": str((router_result or {}).get("reason", "")),
        "question": question,
        "ground_truth_doc": "",
        "ground_truth_docs": "",
        "depends_on": "",
        "depends_on_list": depends_on_list,
    }


def build_history_for_adapter(messages: list[dict[str, Any]], max_turns: int = 12) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    for row in messages:
        role = str(row.get("role", "")).strip()
        if role not in {"user", "assistant"}:
            continue
        content = str(row.get("content", "")).strip()
        if not content:
            continue
        turns.append({"role": role, "content": content})
    if max_turns <= 0:
        return turns
    return turns[-max_turns:]


def build_attachment_context(attachments: list[dict[str, Any]], max_chars: int = 4000) -> str:
    if not attachments:
        return ""
    blocks: list[str] = []
    used = 0
    for item in attachments:
        if not bool(item.get("included", True)):
            continue
        name = str(item.get("name", "")).strip() or "unnamed"
        preview = str(item.get("preview_text", "")).strip()
        if not preview:
            continue
        budget = max(0, max_chars - used)
        if budget <= 0:
            break
        clipped = preview[:budget]
        blocks.append(f"[file] {name}\n{clipped}")
        used += len(clipped)
    return "\n\n".join(blocks)


@dataclass(slots=True)
class RunResult:
    answer_text: str
    latency_sec: float
    mode: str
    model_label: str
    route: str
    profile: dict[str, Any]
    top_k: int
    chunks: list[dict[str, Any]]
    retrieval_context: str


def extract_chunks(answered: ScenarioAAnswer) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(answered.candidates, start=1):
        rows.append(
            {
                "rank": index,
                "chunk_id": row.chunk_id,
                "source_file_name": str(row.metadata.get("source_file_name", "")),
                "section_title": str(row.metadata.get("section_title", "")),
                "chunk_role": str(row.metadata.get("chunk_role", "")),
                "fusion_score": float(row.fusion_score),
                "adjusted_score": float(row.adjusted_score) if row.adjusted_score is not None else None,
                "crag_label": row.crag_label,
                "crag_reason": row.crag_reason,
                "text": row.text,
            }
        )
    return rows


def run_chat_turn(
    *,
    adapter: Any,
    question: str,
    history: list[dict[str, str]],
    attachment_context: str,
    model_label: str,
    preset_name: str,
) -> RunResult:
    started = time.time()
    context_text = attachment_context or "(no evidence)"
    answer_text = adapter.generate(
        system_instruction=(
            "You are an internal QA assistant for a Korean RFP RAG project.\n"
            "Answer directly and keep claims explicit.\n"
            "If the prompt asks for comparison, provide a compact side-by-side summary."
        ),
        question=question,
        context_text=context_text,
        history=history,
    )
    elapsed = time.time() - started
    profile = {
        "answer_layer_policy": "chat_only",
        "prompt_preset": preset_name,
        "attachment_context_chars": len(attachment_context or ""),
    }
    return RunResult(
        answer_text=answer_text,
        latency_sec=round(elapsed, 3),
        mode="chat",
        model_label=model_label,
        route="chat_only",
        profile=profile,
        top_k=0,
        chunks=[],
        retrieval_context="",
    )


def run_rag_turn(
    *,
    project_root: Path,
    adapter: Any,
    question: str,
    history: list[dict[str, str]],
    attachment_context: str,
    model_label: str,
    rag_profile_key: str,
    rag_profile_options: dict[str, Any],
    embedding_backend_key: str,
    routing_model: str,
    candidate_k: int,
    top_k: int,
    crag_top_n: int,
    vector_weight: float,
    bm25_weight: float,
    chroma_dir: str = "",
    answer_type: str | None = None,
    router_result: dict[str, Any] | None = None,
) -> RunResult:
    settings = build_pipeline_settings(
        embedding_backend_key=embedding_backend_key,
        routing_model=routing_model,
        candidate_k=candidate_k,
        top_k=top_k,
        crag_top_n=crag_top_n,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
    )
    options = build_phase2_options(rag_profile_options)
    pipeline = ScenarioBPhase2Pipeline(
        PipelinePaths(
            project_root=project_root,
            chroma_dir=Path(chroma_dir).resolve() if str(chroma_dir).strip() else None,
        ),
        settings=settings,
        options=options,
    )
    return run_rag_turn_with_pipeline(
        pipeline=pipeline,
        adapter=adapter,
        question=question,
        history=history,
        attachment_context=attachment_context,
        model_label=model_label,
        rag_profile_key=rag_profile_key,
        answer_type=answer_type,
        router_result=router_result,
    )


def run_rag_turn_with_pipeline(
    *,
    pipeline: ScenarioBPhase2Pipeline,
    adapter: Any,
    question: str,
    history: list[dict[str, str]],
    attachment_context: str,
    model_label: str,
    rag_profile_key: str,
    answer_type: str | None = None,
    router_result: dict[str, Any] | None = None,
) -> RunResult:
    rag_history = list(history)
    question_row = build_chat_question_row(
        question,
        answer_type=answer_type,
        router_result=router_result,
        history=rag_history,
    )
    if attachment_context:
        rag_history.append(
            {
                "role": "user",
                "content": f"[SESSION FILE CONTEXT]\n{attachment_context}",
            }
        )
    started = time.time()
    answered = pipeline.answer(
        question_row,
        adapter,
        question=question,
        history=rag_history,
    )
    elapsed = time.time() - started
    profile = dict(answered.profile or {})
    profile.update(
        {
            "rag_profile_key": rag_profile_key,
            "attachment_context_chars": len(attachment_context),
        }
    )
    return RunResult(
        answer_text=answered.answer_text,
        latency_sec=round(elapsed, 3),
        mode="rag",
        model_label=model_label,
        route=answered.route,
        profile=profile,
        top_k=len(answered.candidates),
        chunks=extract_chunks(answered),
        retrieval_context=answered.context_text,
    )

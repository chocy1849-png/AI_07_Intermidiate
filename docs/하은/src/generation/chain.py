"""LangChain 기반 RAG 응답 생성 체인.

generate_answer()  : 논스트리밍 (완성된 dict 반환)
stream_answer()    : 스트리밍 제너레이터 (text chunk yield)
generate_dummy_answer() : 임베딩 연결 전 흐름 검증용 더미 응답
"""
from __future__ import annotations

from typing import Any, Generator

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .prompts import SYSTEM_PROMPT, EVAL_SYSTEM_PROMPT

_LLM_MODEL = "gpt-5"
_chain = None
_eval_chain = None


def _build_llm() -> ChatOpenAI:
    kwargs = {
        "model": _LLM_MODEL,
        "streaming": True,
    }
    # gpt-5 계열은 temperature=0 명시를 지원하지 않아 기본값으로 호출한다.
    if not _LLM_MODEL.startswith("gpt-5"):
        kwargs["temperature"] = 0
    return ChatOpenAI(**kwargs)


def _get_chain(eval_mode: bool = False):
    global _chain, _eval_chain
    if eval_mode:
        if _eval_chain is None:
            llm = _build_llm()
            prompt = ChatPromptTemplate.from_messages(
                [("system", EVAL_SYSTEM_PROMPT), ("human", "{question}")]
            )
            _eval_chain = prompt | llm | StrOutputParser()
        return _eval_chain
    if _chain is None:
        llm = _build_llm()
        prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", "{question}")]
        )
        _chain = prompt | llm | StrOutputParser()
    return _chain


def _format_context(retrieved_docs: list[dict]) -> str:
    if not retrieved_docs:
        return "관련 문서를 찾지 못했습니다."
    parts = []
    for i, doc in enumerate(retrieved_docs[:10], 1):
        meta = doc.get("metadata", {})
        header = f"[문서 {i}] {meta.get('file_name', '')} ({meta.get('agency', '')})"
        # 메타데이터에 날짜/예산 등 핵심 정보가 있으면 컨텍스트에 포함
        meta_lines = []
        if meta.get("published_at"):
            meta_lines.append(f"공고일: {meta['published_at']}")
        if meta.get("bid_end_at"):
            meta_lines.append(f"입찰마감: {meta['bid_end_at']}")
        if meta.get("project_name"):
            meta_lines.append(f"사업명: {meta['project_name']}")
        meta_str = " | ".join(meta_lines)
        if meta_str:
            header += f"\n{meta_str}"
        parts.append(f"{header}\n{doc['text']}")
    return "\n\n---\n\n".join(parts)


def _format_history(chat_history: list[dict] | None) -> str:
    if not chat_history:
        return "없음"
    lines = []
    for msg in chat_history[-6:]:
        role = "사용자" if msg["role"] == "user" else "어시스턴트"
        lines.append(f"{role}: {msg['content'][:300]}")
    return "\n".join(lines)


def _build_citations(retrieved_docs: list[dict]) -> list[dict]:
    return [
        {
            "file_name": doc.get("metadata", {}).get("file_name", ""),
            "chunk_id": doc.get("chunk_id", ""),
            "score": round(max(0.0, 1 - doc.get("distance", 1.0)), 4),
            "text": doc.get("text", "")[:500],
        }
        for doc in retrieved_docs[:5]
    ]


def generate_answer(
    query: str,
    retrieved_docs: list[dict],
    chat_history: list[dict] | None = None,
    eval_mode: bool = False,
) -> dict[str, Any]:
    """논스트리밍 RAG 응답. 완성된 dict를 반환한다.
    eval_mode=True이면 평가 전용 프롬프트(거부 없음, 선택지 강제) 사용."""
    context = _format_context(retrieved_docs)
    history = _format_history(chat_history)
    answer = _get_chain(eval_mode=eval_mode).invoke(
        {"context": context, "chat_history": history, "question": query}
    )
    return {
        "answer": answer,
        "citations": _build_citations(retrieved_docs),
        "debug": {
            "retrieved_count": len(retrieved_docs),
            "model": _LLM_MODEL,
        },
    }


def stream_answer(
    query: str,
    retrieved_docs: list[dict],
    chat_history: list[dict] | None = None,
) -> Generator[str, None, None]:
    """스트리밍 제너레이터 — st.write_stream()에 직접 전달 가능."""
    context = _format_context(retrieved_docs)
    history = _format_history(chat_history)
    yield from _get_chain().stream(
        {"context": context, "chat_history": history, "question": query}
    )


def generate_dummy_answer(
    query: str,
    documents: list[dict[str, Any]],
    filters: dict[str, str] | None = None,
) -> dict[str, Any]:
    """임베딩 연결 전 UI 흐름 검증용 더미 응답."""
    filters = filters or {}
    matched_docs = documents[:3]

    if matched_docs:
        references = "\n".join(
            f"- {doc['project_name']} / {doc['agency']} / {doc['file_name']}"
            for doc in matched_docs
        )
        answer = (
            f"질문: {query}\n\n"
            f"아래 문서를 후보로 찾았습니다.\n{references}\n\n"
            "실제 RAG 응답은 임베딩 연결 후 제공됩니다."
        )
    else:
        answer = (
            f"질문: {query}\n\n"
            "필터 조건에서 관련 문서를 찾지 못했습니다. "
            "발주기관 필터를 넓히거나 키워드를 줄여보세요."
        )

    return {
        "answer": answer,
        "citations": [
            {"file_name": doc["file_name"], "chunk_id": f"{doc['file_name']}::0", "text": ""}
            for doc in matched_docs
        ],
        "debug": {
            "retrieved_count": len(documents),
            "model": "dummy-v1",
            "filters": filters,
        },
    }

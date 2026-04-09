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

from .prompts import SYSTEM_PROMPT

_LLM_MODEL = "gpt-5-mini"
_chain = None


def _get_chain():
    global _chain
    if _chain is None:
        llm = ChatOpenAI(model=_LLM_MODEL, temperature=0, streaming=True)
        prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", "{question}")]
        )
        _chain = prompt | llm | StrOutputParser()
    return _chain


def _format_context(retrieved_docs: list[dict]) -> str:
    if not retrieved_docs:
        return "관련 문서를 찾지 못했습니다."
    parts = []
    for i, doc in enumerate(retrieved_docs[:5], 1):
        meta = doc.get("metadata", {})
        header = f"[문서 {i}] {meta.get('file_name', '')} ({meta.get('agency', '')})"
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
) -> dict[str, Any]:
    """논스트리밍 RAG 응답. 완성된 dict를 반환한다."""
    context = _format_context(retrieved_docs)
    history = _format_history(chat_history)
    answer = _get_chain().invoke(
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

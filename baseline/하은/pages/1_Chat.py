"""Chat 페이지 — RAG 기반 대화 인터페이스.

구현 내용 (4/8~4/9):
  B-4.3  End-to-End RAG Chain 연동
  B-4.4  출처 문서(source documents) expander 표시
  B-5.1  세션 관리: 새 대화 시작, 세션 목록, 대화 초기화
  B-5.2  검색 모드 선택: 기본 검색 / MMR / 하이브리드
  B-5.3  응답 스트리밍 (st.write_stream)
  B-5.4  검색 결과 디버그 패널 (expander로 청크 표시)
"""
from __future__ import annotations

import uuid

import streamlit as st

from rag_pipeline import answer_query_stream
from src.db.metadata_store import list_agencies


# ── 상수 ──────────────────────────────────────────────────────────────────────
_INITIAL_MSG = {
    "role": "assistant",
    "content": "안녕하세요! RFP 문서에 대해 질문해 주세요. 발주기관 필터와 검색 모드를 설정하면 더 정확한 답변을 드릴 수 있습니다.",
}


# ── 헬퍼 함수 ─────────────────────────────────────────────────────────────────
def _new_session() -> dict:
    return {"id": str(uuid.uuid4())[:8], "title": "새 대화", "messages": [dict(_INITIAL_MSG)]}


def _active_session() -> dict:
    for s in st.session_state.sessions:
        if s["id"] == st.session_state.active_sid:
            return s
    st.session_state.active_sid = st.session_state.sessions[0]["id"]
    return st.session_state.sessions[0]


def _render_citations(citations: list[dict], key_prefix: str = "") -> None:
    if not citations:
        return
    with st.expander(f"📄 출처 문서 {len(citations)}건"):
        for i, c in enumerate(citations, 1):
            st.markdown(f"**[{i}] {c.get('file_name', '')}**")
            score = c.get("score")
            label = f"chunk: `{c.get('chunk_id', '')}`"
            if score is not None:
                label += f" | 관련도: {score:.3f}"
            st.caption(label)
            if c.get("text"):
                st.text_area(
                    label="preview",
                    value=c["text"][:600],
                    height=100,
                    key=f"{key_prefix}_cite_{i}_{c.get('chunk_id', '')}",
                    label_visibility="collapsed",
                )
            if i < len(citations):
                st.divider()


def _render_debug(
    retrieved_docs: list[dict],
    search_mode: str,
    filters: dict,
    key_prefix: str = "",
) -> None:
    with st.expander("🔍 검색 결과 디버그"):
        st.caption(
            f"검색 모드: `{search_mode}` | 청크 수: {len(retrieved_docs)} | 필터: {filters}"
        )
        for i, doc in enumerate(retrieved_docs, 1):
            meta = doc.get("metadata", {})
            dist = doc.get("distance")
            sim_str = f"{1 - dist:.3f}" if dist is not None else "-"
            st.markdown(f"**[{i}] {meta.get('file_name', '')}** — 유사도 {sim_str}")
            st.text_area(
                label="chunk",
                value=doc.get("text", "")[:400],
                height=90,
                key=f"{key_prefix}_dbg_{i}_{doc.get('chunk_id', '')}",
                label_visibility="collapsed",
            )


# ── 세션 상태 초기화 ──────────────────────────────────────────────────────────
if "sessions" not in st.session_state:
    first = _new_session()
    st.session_state.sessions = [first]
    st.session_state.active_sid = first["id"]


# ── 사이드바 ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💬 대화 세션")

    if st.button("➕ 새 대화 시작", use_container_width=True):
        session = _new_session()
        st.session_state.sessions.append(session)
        st.session_state.active_sid = session["id"]
        st.rerun()

    for s in reversed(st.session_state.sessions):
        is_active = s["id"] == st.session_state.active_sid
        label = f"{'▶ ' if is_active else ''}{s['title']}"
        if st.button(label, key=f"ses_{s['id']}", use_container_width=True):
            st.session_state.active_sid = s["id"]
            st.rerun()

    if st.button("🗑 현재 대화 초기화", use_container_width=True):
        cur = _active_session()
        cur["messages"] = [dict(_INITIAL_MSG)]
        cur["title"] = "새 대화"
        st.rerun()

    st.divider()
    st.markdown("### 🔎 검색 설정")

    agencies = ["전체"] + list_agencies()
    selected_agency = st.selectbox("발주기관", agencies, index=0)
    keyword = st.text_input("사업명/요약 검색", placeholder="예: ERP, 학사, LMS")
    search_mode = st.radio(
        "검색 모드",
        ["기본 검색", "MMR", "하이브리드"],
        index=0,
        help="기본: 벡터 유사도 | MMR: 다양성 보장 | 하이브리드: BM25+벡터 결합",
    )
    top_k = st.slider("검색 청크 수", min_value=3, max_value=10, value=5)


# ── 메인 채팅 ─────────────────────────────────────────────────────────────────
session = _active_session()
st.title(f"Chat — {session['title']}")

# 기존 메시지 렌더
for idx, msg in enumerate(session["messages"]):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            _render_citations(msg["citations"], key_prefix=f"hist_{idx}")


# ── 사용자 입력 처리 ──────────────────────────────────────────────────────────
prompt = st.chat_input("RFP 문서에 대해 질문해 주세요")

if prompt:
    # 세션 타이틀 (첫 질문 기준)
    if session["title"] == "새 대화":
        session["title"] = prompt[:24] + ("…" if len(prompt) > 24 else "")

    session["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    filters = {
        "agency": "" if selected_agency == "전체" else selected_agency,
        "keyword": keyword,
    }

    with st.chat_message("assistant"):
        # 스트리밍 응답
        stream_gen, retrieved_docs = answer_query_stream(
            query=prompt,
            search_mode=search_mode,
            filters=filters,
            top_k=top_k,
            chat_history=session["messages"],
        )
        answer_text = st.write_stream(stream_gen)

        # 출처 문서 표시
        citations = [
            {
                "file_name": doc.get("metadata", {}).get("file_name", ""),
                "chunk_id": doc.get("chunk_id", ""),
                "score": round(max(0.0, 1 - doc.get("distance", 1.0)), 4),
                "text": doc.get("text", "")[:600],
            }
            for doc in retrieved_docs[:5]
        ]
        key_pfx = f"new_{len(session['messages'])}"
        _render_citations(citations, key_prefix=key_pfx)
        _render_debug(retrieved_docs, search_mode, filters, key_prefix=key_pfx)

    session["messages"].append(
        {"role": "assistant", "content": answer_text, "citations": citations}
    )

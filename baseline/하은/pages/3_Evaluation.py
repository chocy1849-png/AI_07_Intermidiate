from __future__ import annotations

from pathlib import Path

import streamlit as st

from config import CHUNK_PARAMETER_GRID, PARSED_DATA_DIR, REPORTS_DIR
from src.chunking.chunker import chunk_document
from src.db.parsed_store import get_parsed_document
from src.db.metadata_store import query_documents, list_agencies
from src.db.vector_store import count as chroma_count
from src.evaluation.evaluator import build_dataset_report, load_latest_report
from rag_pipeline import retrieve, generate_answer


st.title("Evaluation")
st.caption("데이터셋 현황 · 청킹 실험 · RAG 파이프라인 품질 테스트")

# ── 1. 데이터셋 현황 ──────────────────────────────────────────────────────────
st.subheader("1. 데이터셋 현황")
report = build_dataset_report()
col1, col2, col3 = st.columns(3)
col1.metric("저장된 parsed JSON", len(list(PARSED_DATA_DIR.glob("*.json"))))
col2.metric("저장된 리포트", len(list(REPORTS_DIR.glob("parse_report*.json"))))
col3.metric("ChromaDB 벡터 수", chroma_count())
st.json(report["summary"])

# ── 2. 청킹 실험 ──────────────────────────────────────────────────────────────
st.divider()
st.subheader("2. 청킹 실험 파라미터")
st.dataframe(report["chunk_grid"], use_container_width=True)

chunk_report = load_latest_report(REPORTS_DIR, "chunking_experiment_report")
quality_report = load_latest_report(REPORTS_DIR, "parsing_quality_report")
parser_report = load_latest_report(REPORTS_DIR, "parser_comparison_report")

if chunk_report:
    st.markdown("#### 최신 청킹 실험")
    st.dataframe(chunk_report["rows"], use_container_width=True)

if quality_report:
    st.markdown("#### 최신 파싱 품질 점검")
    st.dataframe(quality_report["samples"], use_container_width=True)

if parser_report:
    st.markdown("#### 최신 파서 비교")
    st.json(parser_report["summary"])

# ── 3. 문서 청킹 미리보기 ─────────────────────────────────────────────────────
st.divider()
st.subheader("3. 문서 청킹 미리보기")

sample_docs = query_documents(limit=20)
file_names = [doc["file_name"] for doc in sample_docs]

if not file_names:
    st.warning("평가용 문서가 없습니다.")
else:
    selected_file = st.selectbox("샘플 문서", file_names)
    selected_doc = next(doc for doc in sample_docs if doc["file_name"] == selected_file)
    selected_params = st.selectbox(
        "파라미터 조합",
        CHUNK_PARAMETER_GRID,
        format_func=lambda item: f"chunk_size={item['chunk_size']}, overlap={item['overlap']}",
    )
    selected_strategy = st.selectbox(
        "청킹 전략",
        ["langchain_recursive", "recursive", "section_aware"],
        index=0,
    )
    chunks = chunk_document(
        (get_parsed_document(selected_doc["file_name"]) or {}).get("text", selected_doc["csv_text"]),
        metadata={"file_name": selected_doc["file_name"]},
        chunk_size=selected_params["chunk_size"],
        overlap=selected_params["overlap"],
        strategy=selected_strategy,
    )

    st.metric("생성 청크 수", len(chunks))
    if chunks:
        st.markdown("#### 첫 번째 청크 미리보기")
        st.text_area(
            "chunk_preview",
            value=chunks[0]["text"],
            height=300,
            label_visibility="collapsed",
        )

markdown_reports = sorted(Path(REPORTS_DIR).glob("*.md"))
if markdown_reports:
    selected_report = st.selectbox("마크다운 리포트 보기", [path.name for path in markdown_reports])
    report_path = next(path for path in markdown_reports if path.name == selected_report)
    st.markdown("#### 리포트 본문")
    st.markdown(report_path.read_text(encoding="utf-8"))

# ── 4. RAG 파이프라인 품질 테스트 ─────────────────────────────────────────────
st.divider()
st.subheader("4. RAG 파이프라인 품질 테스트")
st.caption("검색 모드별로 동일한 질문에 대한 검색 결과와 LLM 응답을 비교합니다.")

if chroma_count() == 0:
    st.warning("ChromaDB가 비어있습니다. `python scripts/embed_and_index.py` 먼저 실행하세요.")
else:
    agencies_eval = ["전체"] + list_agencies()
    eval_agency = st.selectbox("발주기관 필터", agencies_eval, index=0, key="eval_agency")
    eval_top_k = st.slider("검색 청크 수 (top_k)", min_value=3, max_value=10, value=5, key="eval_top_k")

    test_modes = st.multiselect(
        "비교할 검색 모드",
        ["기본 검색", "MMR", "하이브리드"],
        default=["기본 검색", "MMR"],
    )

    eval_query = st.text_area(
        "테스트 질문",
        placeholder="예: 사업 규모가 가장 큰 프로젝트는 무엇인가요?",
        height=80,
        key="eval_query",
    )

    if st.button("검색 실행", type="primary") and eval_query.strip():
        filters = {"agency": "" if eval_agency == "전체" else eval_agency}

        for mode in test_modes:
            st.markdown(f"### 검색 모드: `{mode}`")
            with st.spinner(f"{mode} 검색 중..."):
                docs = retrieve(
                    query=eval_query,
                    search_mode=mode,
                    filters=filters,
                    top_k=eval_top_k,
                )

            if not docs:
                st.info("검색 결과 없음.")
                continue

            # 검색 결과 표
            rows = [
                {
                    "순위": i,
                    "파일명": d.get("metadata", {}).get("file_name", ""),
                    "발주기관": d.get("metadata", {}).get("agency", ""),
                    "유사도": round(max(0.0, 1 - d.get("distance", 1.0)), 4),
                    "청크ID": d.get("chunk_id", ""),
                }
                for i, d in enumerate(docs, 1)
            ]
            st.dataframe(rows, use_container_width=True)

            # 청크 내용 expander
            with st.expander(f"검색된 청크 내용 ({len(docs)}건)"):
                for i, d in enumerate(docs, 1):
                    meta = d.get("metadata", {})
                    sim = max(0.0, 1 - d.get("distance", 1.0))
                    st.markdown(f"**[{i}] {meta.get('file_name', '')}** — 유사도 {sim:.4f}")
                    st.text_area(
                        label=f"chunk_{mode}_{i}",
                        value=d.get("text", "")[:500],
                        height=100,
                        label_visibility="collapsed",
                        key=f"eval_chunk_{mode}_{i}",
                    )
                    if i < len(docs):
                        st.divider()

        # LLM 응답 생성 (마지막 모드 기준)
        if test_modes:
            st.markdown("---")
            st.markdown(f"### LLM 응답 (`{test_modes[-1]}` 기준)")
            last_docs = retrieve(
                query=eval_query,
                search_mode=test_modes[-1],
                filters=filters,
                top_k=eval_top_k,
            )
            with st.spinner("LLM 응답 생성 중..."):
                result = generate_answer(eval_query, last_docs)

            st.markdown(result["answer"])

            with st.expander("디버그 정보"):
                st.json(result["debug"])
                st.json(result["citations"])

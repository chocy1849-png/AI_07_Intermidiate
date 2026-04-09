from __future__ import annotations

import streamlit as st

from src.db.parsed_store import get_parsed_document
from src.db.metadata_store import get_dataset_summary, list_agencies, query_documents


st.title("Documents")
st.caption("CSV 메타데이터를 SQLite에 적재한 뒤 필터 흐름을 확인하는 화면입니다.")

summary = get_dataset_summary()
col1, col2, col3 = st.columns(3)
col1.metric("총 문서", summary["document_count"])
col2.metric("기관 수", summary["agency_count"])
col3.metric("형식", ", ".join(f"{k}:{v}" for k, v in summary["file_format_counts"].items()))

agencies = ["전체"] + list_agencies()
selected_agency = st.selectbox("발주기관", agencies, index=0)
keyword = st.text_input("사업명/요약 검색")
limit = st.slider("표시 개수", min_value=10, max_value=100, value=30, step=10)

documents = query_documents(
    agency=None if selected_agency == "전체" else selected_agency,
    keyword=keyword,
    limit=limit,
)

st.subheader(f"조회 결과 {len(documents)}건")
table_rows = [
    {
        "project_name": doc["project_name"],
        "agency": doc["agency"],
        "file_format": doc["file_format"],
        "published_at": doc["published_at"],
        "file_name": doc["file_name"],
    }
    for doc in documents
]
st.dataframe(table_rows, use_container_width=True)

if documents:
    selected_file = st.selectbox("문서 상세", [doc["file_name"] for doc in documents])
    selected_document = next(doc for doc in documents if doc["file_name"] == selected_file)

    st.markdown("### 문서 요약")
    st.write(selected_document["project_summary"] or "요약 없음")
    st.markdown("### 원본 경로")
    st.code(selected_document["source_path"] or "원본 경로 없음")
    parsed_document = get_parsed_document(selected_document["file_name"])
    if parsed_document:
        st.markdown("### 파싱 메타데이터")
        st.json(parsed_document["metadata"])
        st.markdown("### 파싱 텍스트 미리보기")
        st.text_area(
            "parsed_preview",
            value=parsed_document["text"][:4000],
            height=260,
            label_visibility="collapsed",
        )
    st.markdown("### CSV 텍스트 미리보기")
    st.text_area(
        "preview",
        value=(selected_document["csv_text"] or "")[:4000],
        height=240,
        label_visibility="collapsed",
    )

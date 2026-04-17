from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from config import PARSED_DATA_DIR, REPORTS_DIR, ensure_directories
from src.db.metadata_store import bootstrap_metadata_db, get_dataset_summary
from src.db.vector_store import count as chroma_count


st.set_page_config(
    page_title="RFP RAG Chatbot",
    page_icon="📄",
    layout="wide",
)

ensure_directories()
bootstrap_metadata_db()
summary = get_dataset_summary()
parsed_count = len(list(PARSED_DATA_DIR.glob("*.json")))
report_count = len(list(REPORTS_DIR.glob("*.json")))
vector_count = chroma_count()

st.title("RFP RAG Chatbot — 입찰메이트")
st.caption("RFP(제안요청서) 문서 기반 RAG 챗봇. 발주기관 필터·검색 모드 선택·스트리밍 응답 지원.")

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("문서 수", summary["document_count"])
col2.metric("발주 기관 수", summary["agency_count"])
col3.metric("파일 형식 수", len(summary["file_format_counts"]))
col4.metric("파싱 결과", parsed_count)
col5.metric("ChromaDB 벡터", vector_count)
col6.metric("리포트", report_count)

if vector_count == 0:
    st.warning(
        "ChromaDB가 비어있습니다. 아래 명령으로 임베딩을 먼저 적재하세요.\n\n"
        "```bash\npython scripts/embed_and_index.py\n```"
    )

st.divider()
st.subheader("구현 완료 기능")
st.markdown(
    """
    | 페이지 | 기능 |
    |---|---|
    | **Chat** | 세션 관리 · 발주기관 필터 · 검색 모드(기본/MMR/하이브리드) · 스트리밍 응답 · 출처 문서 expander · 디버그 패널 |
    | **Documents** | SQLite 메타데이터 조회 · 발주기관/키워드 필터 · 파싱 텍스트 미리보기 |
    | **Evaluation** | 데이터셋 현황 · 청킹 실험 결과 · RAG 파이프라인 품질 테스트 |

    **검색 모드**
    - 기본 검색: ChromaDB 코사인 유사도
    - MMR: 다양성 보장 (Maximum Marginal Relevance)
    - 하이브리드: BM25 + 벡터 앙상블 (α=0.5)

    **임베딩**: `text-embedding-3-small` (1536차원) — ChromaDB PersistentClient (cosine space)

    **생성**: `gpt-5-mini` via LangChain LCEL, 스트리밍 지원
    """
)

st.divider()
st.subheader("빠른 시작")
st.code(
    """# 1. 임베딩 적재 (최초 1회 또는 신규 문서 추가 시)
export OPENAI_API_KEY=...
python scripts/embed_and_index.py

# 2. 앱 실행
streamlit run app.py""",
    language="bash",
)

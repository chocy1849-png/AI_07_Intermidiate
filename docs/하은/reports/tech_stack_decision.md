# Tech Stack Decision

## 선택 결과

- UI 프레임워크: `Streamlit`
- 메타데이터 DB: `SQLite`
- 문서 파싱: `olefile` + `PyMuPDF`/`pdfplumber`
- 청킹 기준선: `LangChain RecursiveCharacterTextSplitter`
- 1차 검색: `BM25(rank-bm25)` 기반 하이브리드 검색
- 중간 저장 포맷: `JSON`

## Streamlit vs Gradio

- `Streamlit`을 선택한 이유:
  - 멀티페이지 구조가 자연스럽고 `pages/` 기반 구성이 빠르다.
  - `st.chat_input`, `st.chat_message`, `st.session_state`로 챗 UI 프로토타입 구현이 빠르다.
  - 데이터프레임, 사이드바 필터, 실험 결과 화면을 한 앱에서 처리하기 쉽다.
- `Gradio`를 이번 단계에서 제외한 이유:
  - 데모형 챗 인터페이스는 강하지만, 문서 탐색/평가 대시보드 구성이 상대적으로 덜 직관적이다.

## SQLite 선택 이유

- `data_list.csv` 기반 메타데이터 필터 요구사항에 충분하다.
- 배포 초기 단계에서 별도 DB 서버 없이 바로 사용할 수 있다.
- 문서 수 100개 규모에서는 조회 성능이 충분하다.

## 파서 선택 이유

- HWP는 현재 환경에서 `olefile` 기반 커스텀 파서가 가장 안정적으로 동작했다.
- PDF는 `PyMuPDF`가 기본, 예외 시 `pdfplumber` 폴백이 적절하다.

## 결론

4/2~4/3 단계 목표에는 `Streamlit + SQLite + JSON + 경량 파서 + LangChain 기준 청킹 + BM25 검색` 조합이 가장 빠르고 리스크가 낮다.

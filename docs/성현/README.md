# Handoff Table Pipeline

이 폴더는 **표 분석 + 적재 경로**만 전달하기 위한 최소 파일 묶음이다.  
retrieval/QA 평가는 제외했고, HWP 입력, 표 분석, OCR 보강, semantic text 조립, 벡터 DB 적재에 필요한 파일만 포함했다.

## 폴더 구성

### `rag_system/`

- `ingest.py`
  - 실행 진입점
  - `files/` 아래 문서를 읽고 최종적으로 벡터 DB에 적재

- `config.py`
  - 입력 경로, 벡터 DB 경로, chunk 크기, 모델 설정 등 실행 설정 관리

### `table_pipeline/`

- `rag_utils.py`
  - 문서 읽기
  - HWP/PDF 텍스트 추출
  - semantic text 조립
  - chunk 생성
  - 최종 `Document` 리스트 생성

- `table_enrichment.py`
  - 표 분석 본체
  - 키-값 추출, 행 요약, 표 타입 분류, 표 제목 추정, 비교 요약 생성

### `table_pipeline/ocr_support/`

- `extract_hwp_artifacts.py`
  - HWP 입력의 시작점
  - HWP 내부 구조를 읽어 표, 설명문, 섹션 헤더, 이미지 후보를 분리

- `run_hwp_ocr_pipeline.py`
  - 이미지 OCR 실행
  - RapidOCR 우선, 실패 시 Windows OCR fallback 사용

- `windows_ocr_fallback.ps1`
  - Windows 기본 OCR 호출 스크립트

## 실행 흐름

1. `rag_system/ingest.py`
2. `table_pipeline/rag_utils.py`
3. `table_pipeline/ocr_support/extract_hwp_artifacts.py`
4. `table_pipeline/table_enrichment.py`
5. `table_pipeline/ocr_support/run_hwp_ocr_pipeline.py`
6. `table_pipeline/rag_utils.py`에서 chunk 생성
7. `rag_system/ingest.py`에서 벡터 DB 적재

즉 사용자는 보통 `extract_hwp_artifacts.py`나 `table_enrichment.py`를 직접 실행하지 않고, 최종적으로 `ingest.py`를 실행하면 내부에서 모두 호출되는 구조다.

## 역할 요약

- **입력/추출**
  - `extract_hwp_artifacts.py`

- **표 분석**
  - `table_enrichment.py`

- **조립/청킹**
  - `rag_utils.py`

- **적재**
  - `ingest.py`

- **설정**
  - `config.py`

- **OCR 보강**
  - `run_hwp_ocr_pipeline.py`
  - `windows_ocr_fallback.ps1`

## 실행 의존성

### 권장 환경

- Python `3.12` 이상
- Windows 환경 사용 시 `windows_ocr_fallback.ps1` 사용 가능
- `.env`에 필요한 API 키 설정 필요

### 주요 Python 패키지

현재 프로젝트 기준 핵심 의존성과 버전은 아래와 같다.

- `langchain-chroma>=1.1.0`
- `langchain-core>=1.2.30`
- `langchain-huggingface>=1.2.1`
- `langchain-openai>=1.1.13`
- `langchain-text-splitters>=1.1.1`
- `olefile>=0.47`
- `pillow>=12.2.0`
- `pydantic>=2.13.1`
- `pymupdf>=1.27.2.2`
- `pymupdf4llm>=1.27.2.2`
- `python-dotenv>=1.2.2`
- `rapidocr-onnxruntime>=1.4.4`

### 참고

- `RapidOCR`를 쓰려면 `rapidocr-onnxruntime`가 설치되어 있어야 한다.
- 환경에 따라 `RapidOCR`가 동작하지 않으면 `windows_ocr_fallback.ps1` 경로를 사용할 수 있다.
- 원본 프로젝트에는 `pyproject.toml`, `uv.lock`가 있으므로 의존성 재현은 그 파일 기준으로 맞추면 된다.

## 포함 문서

- `table_pipeline_work_summary.md`
  - 작업 정리 문서
  - 작업 범위, 구현 구조, 표 분석 방식, 검증 결과 요약

- `table_pipeline_final_report.md`
  - 최종 보고서 문서
  - 카테고리 설정 이유, 점수 변화, 성능 해석, 한계 정리

## 참고

- 이 폴더에는 retrieval/QA 코드는 포함하지 않았다.
- 목적은 **HWP 표 분석과 벡터 DB 적재 경로만 전달하는 것**이다.

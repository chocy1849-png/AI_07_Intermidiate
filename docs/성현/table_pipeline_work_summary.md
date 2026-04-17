## 1. 담당 범위 요약

| 구분 | 내용 |
|------|------|
| HWP 입력 파이프라인 | HWP 파일 직접 읽기 및 구조 추출 |
| 표 분석 | HWP 표 구조 분석 및 QA 친화적 재구성 |
| OCR 보강 | HWP 내부 이미지 OCR 경로 연결 |
| semantic text / 적재 | 표/본문/OCR 결과 조립 및 벡터 DB 적재 경로 연결 |
| 검증 | 표 중심 retrieval 및 LLM 응답 테스트 수행 |

---

## 2. 작업 배경

### 문제 상황
- 실제 입력 문서는 PDF와 HWP가 혼재되어 있었음
- 기존 OCR 중심 방식은 PDF/이미지에는 적용 가능했지만, HWP 파일 자체를 직접 읽고 내부 표 구조를 안정적으로 복원하는 데 한계가 있었음
- 그 결과 HWP 문서에서는 표 구조 유실, 설명문/섹션 헤더 분리 실패, 이미지형 정보 누락이 발생할 가능성이 높았음

### 작업 목적
- HWP 파일을 직접 읽는 입력 경로를 별도로 구성
- 표는 OCR에만 의존하지 않고 parser-first 방식으로 구조를 보존
- 이미지는 OCR로 보강
- 최종적으로 HWP/PDF 문서를 모두 RAG 입력으로 사용할 수 있는 통합 경로를 만드는 것이 목적이었음

---

## 3. 구현한 파이프라인 구조

```text
files/ 아래 HWP/PDF 문서
  ↓
rag_system/ingest.py
  ↓
table_pipeline/ocr_support/extract_hwp_artifacts.py
  - HWP에서 표, 설명문, 섹션 헤더, 이미지 후보 추출
  ↓
table_pipeline/table_enrichment.py
  - 표 구조 분석
  - 키-값 추출 / 행 요약 / 표 타입 분류 / 표 제목 추정
  ↓
table_pipeline/ocr_support/run_hwp_ocr_pipeline.py
  - 이미지 OCR 실행
  - RapidOCR 또는 Windows OCR fallback
  ↓
table_pipeline/rag_utils.py
  - semantic text 조립
  - chunk 생성
  ↓
Chroma 벡터 DB 적재
```

---

## 4. 주요 코드 및 역할

| 파일 | 역할 |
|------|------|
| `rag_system/ingest.py` | 적재 실행 진입점, 최종적으로 벡터 DB에 적재 |
| `rag_system/config.py` | 입력 경로, DB 경로, chunk 설정 관리 |
| `table_pipeline/ocr_support/extract_hwp_artifacts.py` | HWP 입력 시작점, 표/설명문/섹션 헤더/이미지 후보 추출 |
| `table_pipeline/table_enrichment.py` | 표 분석 본체, 키-값 추출/행 요약/표 타입 분류/표 제목 추정 |
| `table_pipeline/ocr_support/run_hwp_ocr_pipeline.py` | 이미지 OCR 실행 |
| `table_pipeline/ocr_support/windows_ocr_fallback.ps1` | Windows OCR fallback |
| `table_pipeline/rag_utils.py` | semantic text 조립, chunk 생성, `Document` 리스트 생성 |

---

## 5. 표 분석 방식

### 처음 설정한 카테고리

#### HWP 입력 단계

| 카테고리 | 의미 | 설정 이유 |
|------|------|------|
| `structural_table` | 구조적으로 읽을 수 있는 표 | 표는 OCR보다 구조 추출이 우선이기 때문 |
| `explanatory_block` | 표 주변 설명문 | 표만 있으면 질문 의도를 놓칠 수 있어 문맥 연결이 필요했기 때문 |
| `section_header_block` | 섹션 헤더 | 표 제목이나 상위 섹션명을 붙여야 검색이 잘 되기 때문 |
| `image_ocr_candidate` | 이미지 OCR 후보 | 표가 아닌 이미지형 정보는 OCR 경로가 필요했기 때문 |

#### 표 분석 단계

| 카테고리 | 의미 | 설정 이유 |
|------|------|------|
| `requirement_table` | 요구사항 표 | ID, 명칭, 세부내용 구조가 반복되어 별도 요약이 필요했기 때문 |
| `score_table` | 평가/배점 표 | 배점, 점수, 구간 정보는 질문과 직접 연결되기 때문 |
| `schedule_table` | 일정 표 | 날짜, 기간, 단계 정보가 중심이라 별도 처리 필요 |
| `equipment_region` | 장비/지역 표 | 장비명과 대상 지역을 같이 묻는 질문이 많았기 때문 |
| `service_status` | 기능 현황/앱 목록 표 | 서비스 구성, 기존 기능 현황 질문에 대응하기 위해 |
| `ai_requirement` | AI 기능 요구사항 표 | 기능 요구사항과 설명을 묶어 읽어야 했기 때문 |

### 적용한 핵심 함수

| 함수 | 역할 |
|------|------|
| `extract_key_value_pairs()` | 표에서 `라벨 -> 값` 구조 추출 |
| `build_table_row_summary_lines()` | 표를 행 단위 의미 요약으로 재구성 |
| `detect_table_type()` | 표 유형 분류 |
| `infer_table_title()` | 표 제목 추정 |
| `build_type_template_summary()` | 표 유형별 핵심 요약 생성 |
| `build_comparison_summary()` | 비교형 질문 대응 요약 생성 |
| `extract_doc_context_hints()` | 설명문/섹션 헤더에서 문맥 힌트 추출 |
| `build_table_block()` | 최종 표 블록 생성 |

### 최종 표 블록 구성
- `DOC_TITLE`
- `TABLE_TITLE`
- `TABLE_TYPE`
- `TABLE_CONTEXT`
- `TYPE_TEMPLATE_SUMMARY`
- `COMPARISON_SUMMARY`
- `DOC_HINT_SUMMARY`
- `DOC_FOCUS_SUMMARY`
- `KEY_VALUE_SUMMARY`
- `ROW_SUMMARY`
- `RAW_TABLE_TEXT`

즉 표를 원문 그대로 적재한 것이 아니라, 검색과 QA에 유리한 구조화 블록으로 변환해서 적재하도록 구성했음

---

## 6. OCR 처리 방식

### 목적
- 이미지형 정보도 후속 검색에 활용할 수 있도록 OCR 경로를 연결

### 방식
- 1차: `RapidOCR`
- 2차: `Windows OCR fallback`

### 현재 반영 수준
- 이미지는 `[IMAGE_OCR]` 블록으로 semantic text에 포함
- 현재는 구조도나 양식의 의미를 완전히 재구성한 수준은 아니고, OCR 텍스트 블록 수준까지 연결함

---

## 7. semantic text 조립 및 chunking

### semantic text 조립 순서
1. 문서 context 힌트 블록
2. 구조 표 블록
3. 설명문 블록
4. 섹션 헤더 블록
5. 이미지 OCR 블록

### chunking 방식
- section 기준으로 분할
- 최종적으로 `Document` 리스트 생성

### 적재 대상
- 원문 그대로가 아니라 가공된 semantic text chunk를 적재

---

## 8. 검증 결과

### retrieval 검증

| 평가셋 | 결과 |
|------|------|
| `eval_questions_table_v1` Group B/C | `Top1 10/10`, `Top3 10/10` |
| 전체 HWP 문서명 자동 평가 | `Top1 479/480`, `Top3 480/480` |

### LLM 응답 테스트

#### 잘 나온 질문
- 기술평가 / 가격평가 배점
- 요구사항 ID 기반 세부내용
- 평가항목 및 배점 기준
- 수행실적 평가등급별 평점

#### 흔들린 질문
- 사업기간 / 예산
- 제출서류 목록

### 해석
- 문서 추출, 표 구조화, semantic text 조립, chunk 생성, 벡터 DB 적재, retrieval 단계까지는 안정적으로 연결됨
- retrieval은 성공했지만, 최종 답변 단계에서는 LLM의 근거 조합과 답변 구성 과정에서 품질 저하가 발생할 수 있음을 확인함

---

## 9. 구조 정리 및 코드 정비

### 폴더 구조 정리
- 기존 `rag_system/table_pipeline/` 구조를 루트 1계층 `table_pipeline/`으로 이동
- `evaluation/`, `ocr_support/` 하위 폴더로 역할 분리

### 추가 정리
- `launch_table_pipeline.bat`로 메뉴 기반 실행 지원
- `pyproject.toml`, `uv.lock` 추가
- `README.md`, 작업 정리 문서, 보고서 문서 정리
- `table_enrichment.py`, `rag_utils.py` 중복 계산 및 import 정리

---

## 10. 해결한 사항

- HWP 파일 직접 읽기 경로 구축
- HWP 내부 표 구조 추출
- 표 enrichment 구현
- 이미지 OCR 보강 경로 연결
- semantic text 조립 및 chunk 생성
- 벡터 DB 적재 경로 연결
- retrieval 평가 스크립트 구축
- LLM 응답 테스트 수행
- `table_pipeline/` 중심으로 코드 구조 정리

---

## 11. 현재 한계 및 최종 판단

### 현재 한계
- 이미지 OCR은 텍스트 블록 수준까지만 반영됨
- 병합 셀이 심하거나 구조가 특이한 표는 추가 보강 필요
- retrieval은 안정적이지만, LLM 답변 단계에서 표 + 본문 + OCR 근거 조합이 흔들릴 수 있음

### 최종 판단
- 이번 작업의 핵심 성과는 기존 OCR의 HWP 처리 한계를 보완하고, HWP/PDF 문서를 모두 RAG 입력으로 사용할 수 있는 통합 입력 파이프라인을 만든 것
- 문서 추출, HWP 표 구조 분석, semantic text 조립, chunk 생성, 벡터 DB 적재, retrieval 검증까지는 안정적으로 연결됨
- 현재 남은 과제는 retrieval보다는 LLM의 근거 조합 및 답변 구성 단계에 가까움

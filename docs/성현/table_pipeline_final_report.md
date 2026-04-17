
## 1. 작업 목적

이번 작업의 목적은 기존 OCR 중심 처리 방식의 한계를 보완하여, PDF뿐 아니라 HWP 문서도 안정적으로 RAG 입력으로 사용할 수 있도록 만드는 것이었다.

기존 방식의 문제는 다음과 같았다.

- HWP 파일 자체를 직접 읽는 경로가 약했음
- HWP 내부 표 구조가 본문 텍스트로만 평탄화되면서 의미가 깨졌음
- 이미지형 정보는 OCR 대상이 될 수 있어도, 표/본문과 같은 문맥으로 연결되지 않았음

따라서 이번 작업에서는 다음 세 가지를 중심으로 개선했다.

1. HWP를 직접 읽는 전처리 경로 구성
2. 표를 parser-first 방식으로 구조 분석 후 semantic text로 재구성
3. 이미지형 정보에 대해 OCR 보강 경로 연결

---

## 2. 처음 설정한 카테고리와 설정 이유

이번 작업에서는 처음부터 문서를 한 번에 처리하지 않고, 입력 단계와 표 분석 단계에서 카테고리를 나누어 처리했다. 이유는 HWP 문서 안에 있는 정보가 모두 같은 성격이 아니었고, 같은 방식으로 적재하면 검색 품질이 떨어질 가능성이 높았기 때문이다.

### 2.1 HWP 입력 단계 카테고리

`extract_hwp_artifacts.py`에서 HWP 내부 정보를 아래처럼 분리했다.

| 카테고리 | 의미 | 설정 이유 |
|------|------|------|
| `structural_table` | 구조적으로 읽을 수 있는 표 | 표는 OCR보다 구조 추출이 우선이기 때문 |
| `explanatory_block` | 표 주변 설명문 | 표만 있으면 질문 의도를 놓칠 수 있어 문맥 연결이 필요했기 때문 |
| `section_header_block` | 섹션 헤더 | 표 제목이나 상위 섹션명을 붙여야 검색이 잘 되기 때문 |
| `image_ocr_candidate` | 이미지 OCR 후보 | 표가 아닌 이미지형 정보는 OCR 경로가 필요했기 때문 |

이 카테고리를 먼저 나눈 이유는, 표/본문/이미지가 섞여 있는 HWP를 그대로 chunking하면 중요한 구조 정보가 사라질 수 있었기 때문이다.

### 2.2 표 분석 단계 카테고리

`table_enrichment.py`에서는 표를 다시 표 유형별로 분류했다.

| 카테고리 | 의미 | 설정 이유 |
|------|------|------|
| `requirement_table` | 요구사항 표 | ID, 명칭, 세부내용 구조가 자주 반복되어 별도 요약이 필요했기 때문 |
| `score_table` | 평가/배점 표 | 배점, 점수, 구간 정보는 검색 질문과 직접 연결되기 때문 |
| `schedule_table` | 일정 표 | 날짜, 기간, 단계 정보가 중심이라 별도 처리 필요 |
| `equipment_region` | 장비/지역 표 | 장비명과 대상 지역을 같이 묻는 질문이 많았기 때문 |
| `service_status` | 기능 현황/앱 목록 표 | 서비스 구성, 기존 기능 현황 질문에 대응하기 위해 |
| `ai_requirement` | AI 기능 요구사항 표 | 기능 요구사항과 설명을 묶어 읽어야 했기 때문 |

이렇게 나눈 이유는 모든 표를 같은 형식으로 적재하면 질문과 직접 맞닿는 표현이 부족해지기 때문이다.  
즉, 표 유형별로 다른 요약 방식을 써야 retrieval 성능이 올라간다고 판단했다.

### 2.3 평가 단계 카테고리

성능 검증도 하나로 보지 않고 세 가지 경우로 나누었다.

| 카테고리 | 의미 | 설정 이유 |
|------|------|------|
| 표 질문 샘플셋 | 직접 만든 표 중심 질문 | 실제로 약한 케이스를 빠르게 반복 검증하기 위해 |
| 공식 Group B/C | `eval_questions_table_v1` 기준 | HWP enrichment 효과를 공식 질문셋 기준으로 확인하기 위해 |
| 전체 HWP 자동 평가 | 전체 HWP 문서명 기반 | 특정 문서만이 아니라 전체 일반화 성능을 보기 위해 |

즉, 처음부터 카테고리를 나눠서 처리한 이유는 입력 구조가 다르고, 질문 유형이 다르고, 평가 목적도 다르기 때문이다.

---

## 3. 전체 파이프라인 구조

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

핵심은 HWP를 단순 OCR 대상으로 취급하지 않고, 먼저 내부 구조를 읽은 뒤 표와 설명문, 이미지 후보를 분리해서 처리했다는 점이다.

---

## 4. 주요 코드 및 역할

| 파일 | 역할 |
|------|------|
| `rag_system/ingest.py` | 적재 실행 진입점, 최종적으로 벡터 DB 적재 |
| `rag_system/config.py` | 입력 경로, DB 경로, chunk 설정 관리 |
| `table_pipeline/ocr_support/extract_hwp_artifacts.py` | HWP 입력 시작점, 표/설명문/섹션 헤더/이미지 후보 추출 |
| `table_pipeline/table_enrichment.py` | 표 분석 본체, 키-값 추출/행 요약/표 타입 분류/표 제목 추정 |
| `table_pipeline/ocr_support/run_hwp_ocr_pipeline.py` | 이미지 OCR 실행 |
| `table_pipeline/ocr_support/windows_ocr_fallback.ps1` | Windows OCR fallback |
| `table_pipeline/rag_utils.py` | semantic text 조립, chunk 생성, `Document` 리스트 생성 |

---

## 5. 표 분석 방식

표 분석은 `table_pipeline/table_enrichment.py`에서 처리했다. 단순히 표 텍스트를 그대로 적재하지 않고, 검색과 QA에 유리한 구조화 블록으로 재구성했다.

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

즉 표를 원문 그대로 넣는 것이 아니라, 구조와 문맥을 유지한 검색 친화적 형태로 바꿔서 적재했다.

---

## 6. OCR 처리 방식

OCR은 이번 작업의 메인이 아니라 보조 경로로 연결했다.

### 목적

- 이미지형 정보도 후속 검색에 사용할 수 있도록 최소한의 OCR 경로 확보

### 방식

- 1차: `RapidOCR`
- 2차: `Windows OCR fallback`

### 현재 반영 수준

- 이미지는 `[IMAGE_OCR]` 블록으로 semantic text에 포함
- 현재는 구조도나 양식의 의미를 완전히 재구성한 수준은 아니고, OCR 텍스트 블록 수준까지 반영

즉 표는 parser-first, 이미지는 OCR 보강이라는 역할 분리를 유지했다.

---

## 7. semantic text 조립 및 적재 방식

`table_pipeline/rag_utils.py`에서 아래 순서로 semantic text를 조립했다.

1. 문서 context 힌트 블록
2. 구조 표 블록
3. 설명문 블록
4. 섹션 헤더 블록
5. 이미지 OCR 블록

그 다음 section 기준으로 chunk를 생성하고, 최종적으로 `Document` 리스트를 만든 뒤 Chroma 벡터 DB(`my_rfp_vectordb`)에 적재했다.

즉 적재 대상은 원문 그대로가 아니라, 전처리와 표 분석을 거친 semantic text chunk이다.

---

## 8. 점수 및 성능 변화

이번 작업은 세 가지 경우로 나누어 성능을 확인했다.

### 경우 1. 표 중심 질문 샘플셋 개선

표 질문 전용 샘플셋으로 초기 성능과 보강 후 성능을 비교했다.

| 단계 | 설정 | Top1 | Top3 |
|------|------|------|------|
| 초기 | baseline | 7/10 | 9/10 |
| 초기 | mmr | 8/10 | 10/10 |
| 표 enrichment 및 문맥 보강 후 | mmr | 9/10 | 10/10 |

### 해석

- 단순 표 텍스트 적재만으로는 일부 문서에서 표 제목/문맥이 약해서 Top1이 흔들렸다.
- `KEY_VALUE_SUMMARY`, `ROW_SUMMARY`, `TABLE_TITLE`, `TABLE_TYPE`, `COMPARISON_SUMMARY`, `DOC_HINT_SUMMARY`를 추가하면서 검색어와 직접 맞닿는 표현이 늘어났다.
- 그 결과 샘플셋 기준 Top1은 7/10 또는 8/10 수준에서 9/10까지 개선되었고, Top3는 10/10 수준으로 유지되었다.

---

### 경우 2. 공식 평가셋 Group B/C

`eval_questions_table_v1`의 Group B/C를 기준으로 표 enrichment 효과를 검증했다.

| 평가셋 | Top1 | Top3 |
|------|------|------|
| `eval_questions_table_v1` Group B/C | 10/10 | 10/10 |

### 해석

- Group B는 HWP 표 enrichment 효과를 보기 위한 질문이 많았고, Group C는 표+본문 결합형 질문이었다.
- 이 구간에서 Top1/Top3가 모두 10/10이 나온 것은, HWP 표를 구조화해서 semantic text로 넣는 방식이 retrieval 단계에서는 충분히 효과적이었다는 뜻이다.

---

### 경우 3. 전체 HWP 문서명 자동 평가

고정 질문 템플릿에 전체 HWP 문서명을 넣어 자동 평가를 수행했다.

| 평가셋 | Top1 | Top3 |
|------|------|------|
| 전체 HWP 문서명 자동 평가 | 479/480 | 480/480 |

### 해석

- 전체 문서 기준으로도 거의 모든 경우에서 목표 문서를 상위로 가져왔다.
- 유일하게 흔들린 케이스는 유사 사업 문서 간 용어가 겹치는 경우였다.
- 이는 표 구조가 DB에 들어가지 못한 문제가 아니라, 같은 기관 내 유사 문서끼리 제목/문맥 구분력이 약할 때 발생한 문제로 해석했다.

---

## 9. LLM 응답 테스트 결과

retrieval만 보는 것이 아니라, 실제 LLM 답변 단계까지 확인하기 위해 표 중심 질문 6개를 대상으로 응답 테스트를 수행했다.

### 결과 요약

| 구분 | 결과 |
|------|------|
| 안정적으로 답한 질문 | 4개 |
| 부분적으로 흔들린 질문 | 2개 |

### 잘 나온 질문

- 기술평가 / 가격평가 배점
- 요구사항 ID 기반 세부내용
- 평가항목 및 배점 기준
- 수행실적 평가등급별 평점

### 흔들린 질문

- 사업기간 / 예산
  - 본문 값과 OCR 이미지 값이 함께 들어오면서 혼선 발생
- 제출서류 목록
  - retrieval은 성공했지만, LLM이 완전한 목록 구성에는 실패

### 해석

- 문서 추출, 청킹, 적재, retrieval 자체는 성공했다.
- 하지만 최종 답변 단계에서는 LLM이 표 + 본문 + OCR 근거를 조합하는 과정에서 일부 혼선이 발생할 수 있음을 확인했다.
- 따라서 현재 남은 문제는 retrieval 이전 단계가 아니라, answer synthesis 단계에 더 가깝다.

---

## 10. 구조 정리 및 코드 정비

이번 작업에서는 성능 개선뿐 아니라 코드 구조도 함께 정리했다.

### 정리한 내용

- 기존 `rag_system/table_pipeline/` 구조를 루트 1계층 `table_pipeline/`으로 이동
- `evaluation/`, `ocr_support/` 하위 폴더로 역할 분리
- `launch_table_pipeline.bat` 추가
- `pyproject.toml`, `uv.lock` 추가
- `table_enrichment.py`, `rag_utils.py` 중복 계산 및 import 정리

### 코드 정비 효과

- 표 분석 본체, HWP 입력, OCR 보강, 평가 코드가 역할별로 분리됨
- 인수인계 시 필요한 파일 범위를 명확히 나눌 수 있게 됨
- 문서화(`README.md`, 작업 정리, 보고서)까지 함께 정리됨

---

## 11. 해결한 사항

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

## 12. 현재 한계

- 이미지 OCR은 현재 텍스트 블록 수준까지만 반영됨
- 병합 셀이 심하거나 구조가 특이한 표는 추가 보강이 필요함
- retrieval은 안정적이지만, LLM 답변 단계에서 표 + 본문 + OCR 근거 조합이 흔들릴 수 있음

---

## 13. 최종 결론

이번 작업의 핵심 성과는 다음과 같다.

1. 기존 OCR 방식으로 처리하기 어려웠던 HWP를 직접 읽는 입력 경로를 구축했다.
2. HWP 표를 parser-first 방식으로 분석하여 구조를 보존한 상태로 semantic text에 반영했다.
3. 표 중심 retrieval 성능을 실제 수치로 개선했다.
   - 샘플셋 기준 Top1 최대 7/10 -> 9/10 개선
   - 공식 Group B/C 기준 Top1, Top3 모두 10/10
   - 전체 HWP 문서 자동 평가 기준 Top1 479/480, Top3 480/480

즉 이번 작업은 단순 OCR 보강이 아니라, HWP/PDF 문서를 모두 RAG 입력으로 사용할 수 있는 통합 입력 파이프라인을 만든 작업이었다.

현재 남은 과제는 retrieval보다는 LLM의 근거 조합 및 답변 구성 단계에 가깝다.

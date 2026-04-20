# 01. 시스템 아키텍처 및 핵심 코드 맵

## 참조한 원본 파일 목록

| 구분 | 원본 경로 |
|---|---|
| 공통 RAG 파이프라인 | `C:\Users\UserK\Downloads\중급프로젝트\src\scenario_a\common_pipeline.py` |
| 모델 설정 | `C:\Users\UserK\Downloads\중급프로젝트\config\scenario_a_models.yaml` |
| OpenAI adapter | `C:\Users\UserK\Downloads\중급프로젝트\src\scenario_a\model_adapters\openai_chat_adapter.py` |
| Qwen adapter | `C:\Users\UserK\Downloads\중급프로젝트\src\scenario_a\model_adapters\qwen_adapter.py` |
| Gemma adapter | `C:\Users\UserK\Downloads\중급프로젝트\src\scenario_a\model_adapters\gemma4_adapter.py` |
| Phase2 파이프라인 | `C:\Users\UserK\Downloads\중급프로젝트\src\scenario_b_phase2\phase2_pipeline.py` |
| Metadata retrieval | `C:\Users\UserK\Downloads\중급프로젝트\src\scenario_b_phase2\metadata_aware_retrieval.py` |
| Normalized BM25 | `C:\Users\UserK\Downloads\중급프로젝트\src\scenario_b_phase2\normalized_bm25.py` |
| Soft CRAG-lite | `C:\Users\UserK\Downloads\중급프로젝트\src\scenario_b_phase2\soft_crag_lite.py` |
| Answer type router | `C:\Users\UserK\Downloads\중급프로젝트\src\scenario_b_phase2\answer_type_router.py` |
| Streamlit 앱 | `C:\Users\UserK\Downloads\중급프로젝트\app.py` |
| Streamlit 서비스 | `C:\Users\UserK\Downloads\중급프로젝트\src\streamlit_qa\rag_service.py` |
| 평가 파이프라인 | `C:\Users\UserK\Downloads\중급프로젝트\src\scenario_b_phase2\run_improved_auto_grader_v41.py` |

## 시스템 전체 구조

```text
중급프로젝트/
  app.py
  config/
    scenario_a_models.yaml
  src/
    scenario_a/
      common_pipeline.py
      model_adapters/
      embedding_backends/
    scenario_a_qwen_ft/
      finetune_qwen_v3.py
      sft_targeted_v3_pipeline.py
    scenario_b_phase2/
      phase2_pipeline.py
      answer_type_router.py
      soft_crag_lite.py
      true_hwp_table_ocr_*.py
      improved_auto_grader*.py
    streamlit_qa/
      config.py
      providers.py
      rag_service.py
      storage.py
  rag_outputs/
    compare/report/eval 결과
```

## 핵심 컴포넌트 역할

| 영역 | 역할 | 대표 파일 |
|---|---|---|
| 공통 RAG 실행 | Chroma/BM25 검색, context assembly, generator 호출, 결과 패키징 | `src\scenario_a\common_pipeline.py` |
| 모델 adapter | OpenAI/Qwen/Gemma/EXAONE을 동일 인터페이스로 호출 | `src\scenario_a\model_adapters\*.py` |
| embedding backend | OpenAI embedding과 HF embedding을 교체 가능하게 관리 | `src\scenario_a\embedding_backends\*.py` |
| Scenario A | 로컬/HF 모델 비교, embedding ablation, FT 평가 | `src\scenario_a\*.py`, `src\scenario_a_qwen_ft\*.py` |
| Phase2 | metadata boost, comparison helper, OCR routing, table/body pairing | `src\scenario_b_phase2\phase2_pipeline.py` |
| OCR augment | HWP table OCR 결과를 replace가 아닌 augment chunk로 추가 | `src\scenario_b_phase2\true_hwp_table_ocr_*.py` |
| Router | 실서비스 질문에 answer_type과 route를 부여 | `src\scenario_b_phase2\answer_type_router.py` |
| Soft CRAG-lite | retrieval evidence quality를 가볍게 점검하고 down-rank/flag 처리 | `src\scenario_b_phase2\soft_crag_lite.py` |
| Streamlit | 웹 데모 UI와 RAG 서비스 연결 | `app.py`, `src\streamlit_qa\*.py` |
| 평가 | auto grader, improved grader, LLM Judge 결과 비교 | `src\scenario_b_phase2\improved_auto_grader*.py` |

## Scenario A 구조

Scenario A는 generator/model 교체 실험을 위해 만들어진 구조다. 핵심은 retrieval pipeline을 최대한 동일하게 유지한 상태에서 Qwen, Gemma4, OpenAI 모델을 비교하는 것이다.

| 파일 | 역할 |
|---|---|
| `src\scenario_a\common_pipeline.py` | 공통 retrieval/answer pipeline |
| `src\scenario_a\baseline_runner.py` | 모델별 baseline 평가 |
| `src\scenario_a\screening_runner.py` | 후보 모델 screening |
| `src\scenario_a\model_adapters\qwen_adapter.py` | Qwen 로컬 추론 adapter |
| `src\scenario_a\model_adapters\gemma4_adapter.py` | Gemma4 로컬 추론 adapter |
| `config\scenario_a_models.yaml` | 모델명, embedding backend, 출력 경로 설정 |

## Phase2 구조

Phase2는 Stage1에서 남은 retrieval 병목을 더 세밀하게 분해한 실험 구조다.

| 기능 | 구현 방식 |
|---|---|
| metadata-aware retrieval | hard filter가 아닌 soft boost |
| comparison helper | 비교형 질문에서 dual-doc evidence coverage 강화 |
| table/body pairing | table_plus_text에서 table chunk와 nearby/parent body를 함께 pack |
| OCR augment | 기존 corpus를 대체하지 않고 OCR evidence chunk만 추가 |
| answer_type router | 사용자 질문을 factual/comparison/table_factual/table_plus_text/rejection/follow_up으로 분류 |
| Soft CRAG-lite | low-quality evidence를 keep/down-rank/flag로 조정 |

## Streamlit 구조

Streamlit 앱은 `app.py`를 진입점으로 사용한다. UI와 pipeline 실행 로직은 `src\streamlit_qa` 아래로 분리되어 있다.

| 파일 | 역할 |
|---|---|
| `app.py` | Streamlit UI, 대화 화면, 설정 선택 |
| `src\streamlit_qa\config.py` | 실행 설정 및 환경 변수 로딩 |
| `src\streamlit_qa\providers.py` | OpenAI/HF provider 연결 |
| `src\streamlit_qa\rag_service.py` | RAG pipeline 호출 및 응답 반환 |
| `src\streamlit_qa\storage.py` | 세션/대화 기록 저장 보조 |

## 평가 파이프라인 구조

평가는 두 축으로 분리했다.

| 평가 축 | 목적 | 대표 산출물 |
|---|---|---|
| LLM Judge | 서비스형 장문 답변의 faithfulness/completeness/groundedness/relevancy 평가 | `rag_outputs\eval_pipeline\llm_judge_compare.csv` |
| Auto grader | 정답형 factual extraction 성능 보조 측정 | `rag_outputs\eval_pipeline\auto_grader_v41_compare.csv` |
| Factual mode | 장문 답변 대신 constrained/extractor path로 factual 측정 재정의 | `rag_outputs\eval_pipeline\auto_eval_factual_mode_design.md` |


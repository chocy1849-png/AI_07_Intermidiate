# 12. Streamlit 및 Answer Type Router 요약

## 참조한 원본 파일 목록

| 구분 | 원본 경로 |
|---|---|
| Streamlit 앱 | `C:\Users\UserK\Downloads\중급프로젝트\app.py` |
| Streamlit config | `C:\Users\UserK\Downloads\중급프로젝트\src\streamlit_qa\config.py` |
| Provider 연결 | `C:\Users\UserK\Downloads\중급프로젝트\src\streamlit_qa\providers.py` |
| RAG service | `C:\Users\UserK\Downloads\중급프로젝트\src\streamlit_qa\rag_service.py` |
| Storage | `C:\Users\UserK\Downloads\중급프로젝트\src\streamlit_qa\storage.py` |
| Router 구현 | `C:\Users\UserK\Downloads\중급프로젝트\src\scenario_b_phase2\answer_type_router.py` |
| Router smoke 실행 | `C:\Users\UserK\Downloads\중급프로젝트\src\scenario_b_phase2\run_answer_type_router_smoke.py` |
| Router smoke 결과 | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\phase2_runs\p2_answer_type_router_smoke_v2_thr056\answer_type_router_smoke_summary.json` |
| Router 분석 결과 | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\phase2_runs\p2_answer_type_router_analysis_v2_thr056\answer_type_router_analysis_summary.json` |

## Streamlit 앱 구조

Streamlit 앱은 RAG 데모를 위한 사용자 인터페이스다. UI 자체는 `app.py`에서 시작하고, 설정/모델 provider/RAG 호출/대화 저장은 `src\streamlit_qa` 아래로 분리했다.

| 파일 | 역할 |
|---|---|
| `app.py` | Streamlit 실행 진입점, 화면 구성, 사용자 입력 처리 |
| `config.py` | 환경변수, 경로, 모델 설정 로딩 |
| `providers.py` | OpenAI/HF provider 생성 및 adapter 연결 |
| `rag_service.py` | RAG pipeline 호출, router 결과 반영, 답변 반환 |
| `storage.py` | 대화 세션 및 기록 저장 보조 |

## 웹 데모 흐름

```text
사용자 질문
  -> answer_type_router
  -> route/profile 결정
  -> RAG retrieval
  -> context assembly
  -> generator 호출
  -> 답변 + 근거 표시
```

현재 웹 데모 기준으로 Soft CRAG-lite는 보류 상태이므로 기본적으로 off로 유지하는 것이 맞다. Router는 질문 유형을 부여하는 보조 모듈이며, retrieval/generation 전체를 대체하지 않는다.

## Answer Type Router 목적

평가셋에는 `answer_type`이 이미 포함되어 있지만, 실제 웹 사용자의 질문에는 이 정보가 없다. Router의 목적은 런타임에서 사용자 질문에 answer type과 route를 부여하는 것이다.

출력 형식은 다음 구조를 따른다.

```json
{
  "answer_type": "table_plus_text",
  "route": "baseline_v3",
  "confidence": 0.82,
  "signals": ["needs_table_and_body"]
}
```

## 분류 대상

| answer_type | 의미 |
|---|---|
| factual | 일반 단일 문서 사실 질문 |
| comparison | 두 문서 이상 비교/정렬/추천 질문 |
| follow_up | 이전 턴 맥락이 필요한 후속 질문 |
| rejection | 문서 밖 정보라 답변 거부가 필요한 질문 |
| table_factual | 표 안의 값/목록/현황/배점 등 직접 추출 |
| table_plus_text | 표와 본문을 함께 봐야 하는 질문 |

## 규칙 기반 신호

| 유형 | 대표 신호 |
|---|---|
| comparison | 비교, 차이, 각각, 모두, 공통, 이상/이하, 정렬, 추천 |
| follow_up | 그럼, 그중, 그 사업, 앞에서 말한, 이전, 이어서 |
| rejection | 실제 낙찰업체, 실제 완료일, 실제 URL, 실제 성과 |
| table_factual | 목록, 내역, 현황, 기능, 앱, 장비, 정의, 배점, 평가항목 |
| table_plus_text | 왜, 어떻게, 연결해서, 목적, 배경, 역할, 해결 |
| factual | 위 분류에 걸리지 않는 일반 사실 질문 |

## Router 결과

| 항목 | 값 |
|---|---:|
| smoke sample size | 20 |
| exact match | 19 |
| exact match rate | 0.95 |
| confidence threshold | 0.56 |
| fallback route | `comparison_safe` |

Router는 최종 성능 모델이라기보다 웹서비스 연결 가능한 최소 동작 버전이다. 규칙 기반이므로 해석 가능하고, 낮은 confidence에서는 fallback route를 사용한다.

## 현재 웹 데모 기준 구성

| 항목 | 상태 |
|---|---|
| 기본 pipeline | baseline_v3 후보 또는 운영 기준선 |
| router | on 가능, smoke 19/20 |
| soft_crag_lite | 보류, 기본 off |
| follow-up/source-state | 완성형 구현 전, TODO 상태 |
| post-verification | 완성형 구현 전, TODO 상태 |


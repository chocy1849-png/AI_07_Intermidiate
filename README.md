# RFP 기반 한국어 RAG 챗봇 고도화 프로젝트

정부·공공 RFP/HWP/PDF 문서를 대상으로 한국어 RAG 챗봇을 구축하고, 검색·문맥·표 OCR·평가 체계를 단계적으로 고도화한 중급 프로젝트입니다.
본 리포지토리는 LMS 제출용 GitHub Repository이며, **원본 RFP 문서와 API key, Chroma DB, 대용량 산출물은 포함하지 않습니다.**

협업일지 링크 : https://www.notion.so/3367f345c16a803f9f8ed3765b0e42d9

## 제출물 바로가기

| 항목 | 경로 | 설명 |
|---|---|---|
| 최종 보고서 PDF | `docs/report/final_report.pdf` | 제출 전 PDF 파일 추가 필요 |
| 보고서 작성용 통합 번들 | [`docs/final_bundle/`](docs/final_bundle/) | 최종 요약 문서 15개 + manifest/missing items |
| 프로젝트 개요/타임라인 | [`docs/final_bundle/00_project_overview_and_timeline.md`](docs/final_bundle/00_project_overview_and_timeline.md) | 전체 흐름 요약 |
| 시스템 아키텍처 | [`docs/final_bundle/01_system_architecture_and_core_code_map.md`](docs/final_bundle/01_system_architecture_and_core_code_map.md) | 코드 구조와 핵심 파일 역할 |
| Stage1 결과 비교 | [`docs/final_bundle/03_stage1_b00_to_b06_compare.csv`](docs/final_bundle/03_stage1_b00_to_b06_compare.csv) | B-00~B-06 핵심 수치 |
| Phase2 결과 비교 | [`docs/final_bundle/05_phase2_key_compare_tables.csv`](docs/final_bundle/05_phase2_key_compare_tables.csv) | baseline_v2/v3, OCR, Soft CRAG, router |
| 평가 결과 요약 | [`docs/final_bundle/08_evaluation_key_results.csv`](docs/final_bundle/08_evaluation_key_results.csv) | auto/Judge/improved auto 결과 |
| 협업일지 링크 | [`docs/collaboration_logs.md`](docs/collaboration_logs.md) | 개인별 링크 또는 PDF 추가 필요 |

## 프로젝트 목표

RFP 문서는 긴 본문, 표, 그림, 메타데이터, 평가 기준이 섞여 있어 단순 similarity search만으로 안정적인 답변을 만들기 어렵습니다. 본 프로젝트는 다음 문제를 중심으로 개선했습니다.

| 문제 | 대응 |
|---|---|
| 비교형 질문에서 두 문서 근거를 동시에 찾기 어려움 | Hybrid retrieval, prefix v2, comparison helper |
| 표/그림 정보가 파싱 중 손실됨 | OCR augment, table evidence, table/body pairing |
| 후속 질문에서 이전 문서 맥락이 흔들림 | history-aware/source-state 설계 및 TODO 정리 |
| 장문 답변과 exact auto grader가 충돌함 | LLM Judge와 factual auto grader 역할 분리 |
| 로컬 모델 적용 가능성 확인 필요 | Qwen/Gemma/embedding screening, Qwen SFT 실험 |

## 전체 아키텍처

```text
RFP 원문(HWP/PDF)
  -> 파싱 및 processed_data 생성
  -> contextual chunk + metadata prefix
  -> embedding + ChromaDB / BM25 index
  -> hybrid retrieval / helper / OCR augment
  -> answer type routing
  -> answer generation
  -> LLM Judge / auto grader / manual evaluation
```

## 현재 최종 기준선

| 기준 | 구성 | 상태 |
|---|---|---|
| Stage1 운영 기준 | B-06 exact + `gpt-5-mini` | 운영 기준 |
| Stage1 ceiling | 동일 retrieval + `gpt-5` | 최고 성능 확인용 |
| Phase2 기준 | `phase2_baseline_v2` | metadata half + comparison helper |
| Phase2 후보 | `baseline_v3` | OCR v4 후보, 추가 운영 검증 필요 |
| Soft CRAG-lite | baseline_v3 + targeted soft check | 구현 완료, 채택 보류 |
| Router | rule-based answer type router v2 | smoke 19/20, 웹 연결 가능 |

## 핵심 결과 요약

| 구간 | 핵심 결론 |
|---|---|
| Stage1 B-02 | structured prefix v2가 가장 안정적인 기본선으로 채택됨 |
| Stage1 B-03a | CRAG 계열은 Type2에는 유리하지만 전역 적용 시 Type3/Type4 하락으로 조건부만 유지 |
| Stage1 B-04 | 1200/160 chunk는 full45에서 500/80보다 낮아 미채택 |
| Stage1 B-05 | 초기 table enrichment는 full45/Type4 하락으로 Phase2 OCR 트랙으로 이관 |
| Stage1 B-06 | `gpt-5-mini`는 운영 기준, `gpt-5`는 ceiling |
| Phase2 baseline_v2 | comparison helper로 Type2와 overall 개선 |
| OCR v4 | table_15와 full45를 개선해 `baseline_v3` 후보로 유지 |
| Soft CRAG-lite | Group C/Type4/latency는 개선했지만 overall/Type2 하락으로 보류 |
| Qwen FT | 일부 조건에서는 소폭 개선이나 전체적으로 RAG 구조 개선이 더 유의미 |
| Auto grader | 최종 품질 지표가 아니라 factual retrieval/extraction 측정기로 재정의 |

상세 수치와 근거는 [`docs/final_bundle/`](docs/final_bundle/)을 확인하면 됩니다.

## 리포지토리 구조

```text
.
├─ app.py
├─ requirements.txt
├─ .env.example
├─ config/
│  ├─ scenario_a_models.yaml
│  ├─ phase2_experiments.yaml
│  └─ *_main_experiment*.yaml
├─ src/
│  ├─ scenario_a/
│  │  ├─ common_pipeline.py
│  │  ├─ baseline_runner.py
│  │  ├─ screening_runner.py
│  │  └─ model_adapters/
│  ├─ scenario_a_qwen_ft/
│  │  ├─ finetune_qwen_v3.py
│  │  ├─ sft_targeted_v3_pipeline.py
│  │  └─ run_*.py
│  ├─ scenario_b_phase2/
│  │  ├─ phase2_pipeline.py
│  │  ├─ answer_type_router.py
│  │  ├─ soft_crag_lite.py
│  │  ├─ true_hwp_table_ocr_*.py
│  │  ├─ improved_auto_grader*.py
│  │  └─ run_*.py
│  ├─ streamlit_qa/
│  │  ├─ config.py
│  │  ├─ providers.py
│  │  ├─ rag_service.py
│  │  └─ storage.py
│  └─ ocr_tools/
├─ experiments/
│  ├─ B00/ ... B06/
│  └─ shared/
├─ evaluation/
│  ├─ day3_partA_eval_questions_v1.txt
│  ├─ eval_questions_table_v1.txt
│  └─ auto_grader.py
├─ docs/
│  ├─ final_bundle/
│  ├─ report/
│  ├─ collaboration_logs.md
│  ├─ 건호/
│  ├─ 성현/
│  └─ 하은,윤민/
└─ baseline/
```

## 핵심 코드 맵

| 파일/폴더 | 역할 |
|---|---|
| `app.py` | Streamlit 웹 데모 진입점 |
| `src/scenario_a/common_pipeline.py` | 공통 RAG pipeline, Chroma/BM25 검색, context assembly |
| `src/scenario_a/model_adapters/` | OpenAI/Qwen/Gemma/EXAONE adapter |
| `src/scenario_a_qwen_ft/` | Qwen SFT 데이터셋 생성, refinement, LoRA FT 실행 코드 |
| `src/scenario_b_phase2/phase2_pipeline.py` | Phase2 helper, metadata boost, comparison helper, OCR/table routing |
| `src/scenario_b_phase2/answer_type_router.py` | 실사용 질문의 answer type 분류 |
| `src/scenario_b_phase2/soft_crag_lite.py` | retrieval evidence quality check |
| `src/scenario_b_phase2/true_hwp_table_ocr_*.py` | true HWP table OCR augment |
| `src/scenario_b_phase2/improved_auto_grader*.py` | qbank v4 기반 improved auto grader |
| `experiments/B00`~`B06` | Stage1 재현 코드 |
| `docs/final_bundle` | 보고서 작성용 통합 요약 문서 |

## 실행 준비

### 1. Python 환경

권장 환경은 Python 3.11입니다. 일부 로컬 모델/FT 코드는 GPU 환경과 별도 패키지가 필요합니다.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. API key

`.env.example`을 복사해 `.env`를 만들고 API key를 입력합니다. `.env`는 `.gitignore`에 포함되어 있어 커밋하지 않습니다.

```text
OPENAI_API_KEY=...
```

### 3. 데이터 준비

보안 정책상 원본 RFP 문서는 리포지토리에 포함하지 않습니다. 재현하려면 로컬에 다음 파일이 필요합니다.

```text
processed_data/
rag_outputs/
Chroma DB directory
data_list.csv
files/files/*.hwp 또는 *.pdf
```

원본 문서와 전처리 산출물은 LMS/GitHub 제출 대상이 아니며, 로컬에서만 사용해야 합니다.

## 대표 실행 명령

### Streamlit 데모

```bash
streamlit run app.py
```

### Stage1 재현

```bash
python experiments/B00/01_컨텍스트_청킹.py
python experiments/B00/02_임베딩_생성_크로마적재.py --기존컬렉션초기화
python experiments/B00/03_나이브_RAG_베이스라인.py --질문 "사업 예산은 얼마야?" --상위개수 5
```

Stage1 전체 흐름은 다음 노트북도 참고할 수 있습니다.

| 노트북 | 설명 |
|---|---|
| `experiments/B00/RAG_베이스라인_재현.ipynb` | baseline RAG 재현 |
| `experiments/B06/RAG_고도화1단계_재현.ipynb` | Stage1 adopted pipeline 재현 |

### Phase2 예시

```bash
set PYTHONPATH=%CD%\src
python src/scenario_b_phase2/run_answer_type_router_smoke.py
python src/scenario_b_phase2/run_phase2_baseline_v2_confirm.py
python src/scenario_b_phase2/run_phase2_true_table_ocr_v4_exact.py
```

### 평가 예시

```bash
set PYTHONPATH=%CD%\src
python src/scenario_b_phase2/run_improved_auto_grader_v41.py
```

LLM Judge는 API 비용과 시간이 발생하므로 제출용 리포지토리에는 결과 요약만 포함합니다.

## 보안 및 제출 정책

이 리포지토리에 포함하지 않는 항목은 다음과 같습니다.

| 제외 항목 | 이유 |
|---|---|
| `.env`, API key | 비밀정보 |
| 원본 `.hwp`, `.pdf`, `.hwpx` | NDA 및 원본 데이터 외부 공유 금지 |
| `data_list.csv` | 원본 데이터 목록 |
| `processed_data/` | 원문 파싱 결과 포함 가능 |
| `rag_outputs/` | 대용량 실험 산출물, Chroma/BM25/중간 결과 |
| Chroma DB / pickle / cache | 대용량 로컬 인덱스 |
| 모델 checkpoint / adapter | 대용량 모델 산출물 |

GitHub에는 코드, 설정 템플릿, 평가 로직, 2차 가공 요약 문서, 핵심 비교표만 포함합니다.

## 팀원 기여

| 담당 | 주요 기여 |
|---|---|
| 조찬영 | baseline/Stage1/Phase2/Scenario A/FT/Streamlit 통합 |
| 박윤민 | 문제은행 설계, 자동평가 문항 구조화, 검수 결과 통합 |
| 강하은 | 평가 파이프라인, LLM Judge, qbank/auto grader 정리 |
| 윤성현 | OCR/table extraction, structural table 후보 및 table pipeline |
| 이건호 | Dense/Sparse/Hybrid retrieval 분석, Type별 retrieval 병목 분석 |

상세 내용은 [`docs/final_bundle/11_team_contributions_summary.md`](docs/final_bundle/11_team_contributions_summary.md)를 참고하세요.

## 보고서 작성 참고

보고서/PPT 작성 시에는 아래 파일을 우선 확인하면 됩니다.

| 목적 | 파일 |
|---|---|
| 전체 요약 | `docs/final_bundle/00_project_overview_and_timeline.md` |
| 아키텍처 | `docs/final_bundle/01_system_architecture_and_core_code_map.md` |
| Stage1 결과 | `docs/final_bundle/02_stage1_b00_to_b06_summary.md` |
| Phase2 결과 | `docs/final_bundle/04_phase2_progression_summary.md` |
| OCR 설명 | `docs/final_bundle/06_ocr_pipeline_and_evidence_summary.md` |
| 평가 체계 | `docs/final_bundle/07_evaluation_framework_summary.md` |
| FT/데이터셋 | `docs/final_bundle/10_finetuning_and_dataset_summary.md` |
| 한계/향후 과제 | `docs/final_bundle/13_limitations_and_future_work.md` |

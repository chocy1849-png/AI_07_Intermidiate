# 00. 프로젝트 개요 및 타임라인

## 참조한 원본 파일 목록

| 구분 | 원본 경로 |
|---|---|
| Stage1 비교 | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\b02_b01_compare.csv` |
| Stage1 비교 | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\b03_b02_compare.csv` |
| Stage1 비교 | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\b04_full_compare.csv` |
| Stage1 비교 | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\b05_full_compare.csv` |
| Stage1 모델 비교 | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\b06_compare.csv` |
| Scenario A 요약 | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\scenario_a_ablation_summary.md` |
| Phase2 OCR v4 | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\phase2_runs\p2_true_table_ocr_v4_exact_v1\true_table_ocr_v4_exact_report.json` |
| Router 결과 | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\phase2_runs\p2_answer_type_router_smoke_v2_thr056\answer_type_router_smoke_summary.json` |
| 평가 파이프라인 | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\eval_pipeline\final_stage_progression_summary.md` |
| 팀 평가 문서 | `C:\Users\UserK\Downloads\github\docs\하은,윤민\최종보고서_평가파이프라인_강하은&박윤민.md` |

## 프로젝트 목표

본 프로젝트의 목표는 정부·공공 RFP/HWP/PDF 문서를 대상으로 한국어 RAG 챗봇을 구축하고, 문서 근거 기반 답변 품질을 단계적으로 개선하는 것이다. 단순히 답변을 생성하는 데 그치지 않고, 어떤 고도화가 실제 성능을 올렸는지 분리 검증하는 것을 핵심 원칙으로 두었다.

주요 개선 대상은 다음 네 가지였다.

| 병목 | 문제 | 대응 방향 |
|---|---|---|
| 비교형 질문 | 두 개 이상 문서의 근거를 동시에 회수해야 하며 completeness가 낮아짐 | Hybrid retrieval, structured prefix, comparison helper |
| 표/그림 정보 손실 | HWP/PDF 표와 그림이 본문 파싱에서 누락되거나 분리됨 | OCR augment, table evidence, table/body pairing |
| 후속 질문 | 이전 턴의 문서 상태와 비교쌍을 유지하지 못함 | history-aware rewriting과 source-state 설계 |
| 평가 불일치 | 장문 서비스 답변이 exact auto grader와 맞지 않음 | LLM Judge와 factual auto grader 역할 분리 |

## 전체 시스템 흐름

```text
RFP 원문(HWP/PDF)
  -> 파싱 및 processed_data 생성
  -> contextual chunk + metadata prefix
  -> embedding + ChromaDB / BM25 index
  -> retrieval / reranking / conditional helper
  -> answer generation
  -> LLM Judge / auto grader / manual review
  -> 채택, 보류, 기각 판단
```

## Stage1에서 Stage2로 이어진 흐름

Stage1은 B-00부터 B-06까지의 기본 RAG 고도화 구간이다. 이 단계에서 `B-02 structured prefix`가 안정적인 기본선으로 채택되었고, `B-03a CRAG 계열 rescue`는 전역 적용이 아니라 조건부 적용 후보로 정리되었다. `B-04 chunking`, `B-05 초기 표 enrichment`는 전체 성능 기준에서 미채택되었다. `B-06`에서는 동일 retrieval/routing 위에서 generator를 비교하여 `gpt-5-mini`를 운영 기준, `gpt-5`를 ceiling으로 정리했다.

Stage2는 Phase2 고도화 구간이다. Stage1에서 남은 병목인 비교형 evidence coverage, metadata-aware retrieval, 표 OCR, answer type routing, Soft CRAG-lite를 분리 실험했다. 최종적으로 `phase2_baseline_v2`는 comparison helper를 포함한 기준선으로, `baseline_v3`는 OCR v4 후보를 포함한 승격 후보로 정리되었다. Soft CRAG-lite는 구현되었지만 judge-on 결과상 보류로 정리했다.

## 주요 의사결정 타임라인

| 시점 | 주요 작업 | 의사결정 |
|---|---|---|
| 2026-04-06 전후 | Naive RAG, 500/80 chunk, ChromaDB 적재, 45문항 평가 시작 | RAG baseline 구축 완료 |
| 2026-04-07 전후 | B-01 Hybrid, B-02 Prefix v2, B-03 CRAG, B-04 chunking, B-05 table enrichment | B-02 기본선 채택, B-03 조건부, B-04/B-05 미채택 |
| 2026-04-08 전후 | B-06 generator 비교, Scenario A 구조 설계 | gpt-5-mini 운영 기준, gpt-5 ceiling |
| 2026-04-09 전후 | Qwen/Gemma/embedding screening, Gemma raw baseline vs Stage1 | RAG 구조 효과와 모델 자체 성능을 분리 해석 |
| 2026-04-10 전후 | Qwen SFT dataset v3 구축 | accepted 2,952건, FT 본실험 가능 판정 |
| 2026-04-13 전후 | Qwen FT 본실험 및 평가 | FT 단독 효과는 제한적, RAG 구조 개선 우선 |
| 2026-04-15 전후 | Phase2 P0, answer layer parity, metadata/comparison helper | phase2_baseline_v2 승격 |
| 2026-04-16~17 전후 | true HWP table OCR v1~v4, router, Soft CRAG-lite | OCR v4는 baseline_v3 후보, router 19/20, Soft CRAG-lite 보류 |
| 2026-04-18 이후 | 팀 평가 파이프라인 통합, qbank v4, improved auto grader | auto grader는 LLM Judge 대체가 아닌 factual 측정기로 재정의 |

## 최종 기준선 요약

| 기준 | 구성 | 용도 |
|---|---|---|
| 운영 기준선 | B-06 exact + gpt-5-mini | 안정적 서비스형 답변 기준 |
| Stage2 기준선 | phase2_baseline_v2 | metadata half + comparison helper 포함 |
| Stage2 승격 후보 | baseline_v3 / OCR v4 후보 | table factual과 full45에서 개선된 후보 |
| 최고 성능 확인 | 동일 retrieval 위 gpt-5 | 운영 기준이 아닌 ceiling check |


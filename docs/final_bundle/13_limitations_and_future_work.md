# 13. 한계 및 향후 과제

## 참조한 원본 파일 목록

| 구분 | 원본 경로 |
|---|---|
| Stage1 비교 | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\b03_b02_compare.csv` |
| Stage1 chunking 비교 | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\b04_full_compare.csv` |
| Stage1 table 비교 | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\b05_full_compare.csv` |
| OCR failure analysis | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\phase2_runs\p1_ocr_single_upgrade_v1\ocr_failure_analysis.md` |
| OCR v4 결과 | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\phase2_runs\p2_true_table_ocr_v4_exact_v1\true_table_ocr_v4_exact_report.json` |
| Soft CRAG 결과 | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\phase2_runs\p2_soft_crag_lite_compare_v1\soft_crag_lite_compare.csv` |
| Auto grader report | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\eval_pipeline\auto_trend_vs_reference_judge_report_v41.md` |
| FT root cause | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\scenario_a_ft_rootcause_and_gemma4_assessment.md` |

## 현재까지 드러난 한계

### 1. Retrieval miss와 multi-document coverage

비교형 질문은 단일 문서 hit가 아니라 두 문서 이상의 evidence coverage가 중요하다. 기존 top1/topk hit만으로는 Type2 성능을 충분히 설명하지 못했다. Phase2에서 comparison helper를 도입해 개선했지만, 비교축이 복잡하거나 한쪽 문서의 정보가 표에만 있는 경우는 여전히 취약하다.

향후 과제는 `multi-document evidence coverage`를 별도 지표로 고정하고, 질문이 요구하는 비교축별 evidence가 모두 확보됐는지 보는 것이다.

### 2. Long answer bias와 auto grader mismatch

서비스형 답변은 읽기 좋고 근거를 포함하지만, exact auto grader에는 불리하다. 후반 pipeline일수록 답변이 길고 구조화되면서 정답이 포함되어도 자동채점에서 실패하는 경우가 많았다.

따라서 auto grader는 최종 서비스 품질 지표가 아니라 factual extraction 보조 지표로 써야 한다. Constrained Factual Answer Mode와 Retrieval+Extractor Mode의 full run이 필요하다.

### 3. OCR/table evidence의 불균형

OCR v4는 table_15와 full45를 개선했지만, Group B와 Group C의 요구가 다르다.

| 유형 | 요구 |
|---|---|
| table_factual | 표 안의 값을 정확하고 짧게 추출 |
| table_plus_text | 표와 nearby/parent body를 함께 이해 |

OCR v4는 승격 후보지만, Group B가 baseline_v2보다 낮은 부분은 향후 보완 대상이다. 특히 header/row matching, generic row pollution 방지, table AST 기반 extraction이 더 필요하다.

### 4. Soft CRAG-lite의 제한

Soft CRAG-lite는 Group C, Type4, latency를 개선했지만 overall과 Type2가 하락했다. 즉 evidence quality check는 유효한 신호를 주지만, 전역 또는 넓은 적용은 위험하다. 향후에는 comparison/table_plus_text/follow_up 등 좁은 라우트에서만 gate하는 방식으로 재검토해야 한다.

### 5. Follow-up/source-state 미완성

history-aware query rewriting과 source-state 유지는 설계와 일부 실험은 있었지만, 최종 baseline에 완성형으로 포함하지 않았다. 실제 챗봇에서는 후속 질문이 중요하므로 다음 단계에서 반드시 별도 트랙으로 분리해야 한다.

필요 기능은 다음과 같다.

| 기능 | 설명 |
|---|---|
| source-state memory | 이전 턴 문서/기관/사업명/비교쌍 유지 |
| bounded follow-up | 2~3턴 제한 follow-up만 안정 처리 |
| follow-up rewriting | 이전 문맥을 반영한 query rewrite |
| answer audit | 후속 질문에서 문서가 튀었는지 검증 |

### 6. FT의 제한

Qwen FT는 B02 고정 조건에서는 소폭 개선이 있었지만, Stage1 전체 기준에서는 하락했다. 이는 FT가 retrieval miss를 해결하지 못하기 때문이다. SFT 데이터셋 품질은 충분히 확보했지만, FT는 RAG 구조 개선 이후에 보조적으로 적용해야 한다.

## 향후 확장 과제

| 우선순위 | 과제 | 기대 효과 |
|---|---|---|
| P1 | factual mode full run 완료 | auto grader를 retrieval/fact 측정기로 안정화 |
| P1 | baseline_v3 full serving 검증 | OCR v4 후보의 실제 서비스 안정성 확인 |
| P1 | follow-up/source-state 완성 | 챗봇형 사용성 개선 |
| P2 | OCR v4 Group B 보강 | table_factual 정확도 개선 |
| P2 | multi-doc coverage metric 정식화 | Type2 비교형 평가 신뢰도 향상 |
| P2 | post-generation verification | hallucination과 wrong-doc 답변 감지 |
| P3 | Gemma/Qwen FT 재검토 | RAG pipeline 고정 후 모델별 최적화 |

## 보고서용 결론

현재 프로젝트의 가장 중요한 한계는 모델 성능이 아니라, retrieval evidence 구성과 평가 방식의 불일치다. 따라서 향후 개선은 “더 큰 모델”보다 “더 정확한 evidence coverage, OCR/table 구조화, answer type별 routing, factual 평가 path 분리”가 우선이다.


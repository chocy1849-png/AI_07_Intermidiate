# 04. Phase2 진행 요약

## 참조한 원본 파일 목록

| 구분 | 원본 경로 |
|---|---|
| Hybrid sweep | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\phase2_runs\hybrid_sweep_compare.csv` |
| Baseline v2 확인 | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\phase2_runs\p2_baseline_v2_confirm_v1\baseline_v2_confirm_compare.csv` |
| OCR single upgrade | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\phase2_runs\p1_ocr_single_upgrade_v1\ocr_single_upgrade_summary.csv` |
| OCR failure analysis | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\phase2_runs\p1_ocr_single_upgrade_v1\ocr_failure_analysis.md` |
| OCR v4 결과 | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\phase2_runs\p2_true_table_ocr_v4_exact_v1\true_table_ocr_v4_exact_compare.csv` |
| OCR v4 리포트 | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\phase2_runs\p2_true_table_ocr_v4_exact_v1\true_table_ocr_v4_exact_report.json` |
| Soft CRAG-lite | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\phase2_runs\p2_soft_crag_lite_compare_v1\soft_crag_lite_compare.csv` |
| Router smoke | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\phase2_runs\p2_answer_type_router_smoke_v2_thr056\answer_type_router_smoke_summary.json` |

## Phase2 목표

Phase2의 목표는 Stage1에서 남은 병목을 단일 요소별로 검증하는 것이었다. 원칙은 “현재 최고 manual mean 조합을 잠정 baseline으로 승격하고, 그 위에서 single upgrade만 수행”이었다. 즉 여러 기능을 동시에 섞지 않고, 각 기능이 실제로 성능에 기여하는지 분리 판단했다.

## 기준선 변화

| 기준선 | 구성 | 판단 |
|---|---|---|
| baseline_v1 | B-06 exact + metadata_t4off_half | metadata helper를 약하게 적용한 기준 |
| baseline_v2 | baseline_v1 + comparison_helper_only | full45와 Type2 개선으로 Phase2 공식 기준선 |
| baseline_v3 후보 | baseline_v2 + true_table_ocr_v4 | OCR v4가 full45/table_15에서 개선되어 승격 후보 |

## 주요 실험과 판단

| 실험 | 목적 | 결과 | 판단 |
|---|---|---|---|
| Hybrid weight sweep | dense/sparse 비율 조정 | 0.6/0.4는 latency만 유리하고 manual mean/Type2/Type4 불리 | 종료/제외 |
| Metadata boost | 문서 메타 기반 soft boost | 강한 boost는 rejection 훼손 가능, half+t4off가 더 안정 | baseline_v1 구성 |
| Comparison helper | 비교형 질문의 dual-doc coverage 개선 | baseline_v1 대비 overall 4.4167 -> 4.4778, Type2 3.875 -> 4.025 | baseline_v2 승격 |
| OCR single upgrade v1 | OCR pilot corpus 적용 | table_15/group_bc 모두 하락 | 미채택 |
| Structural/Image OCR A/B/C | 팀 OCR 결과 augment | 일부 table 개선 외 group_bc/Type4/coverage 불안정 | 미채택 |
| True HWP table OCR v1~v3 | 실제 table crop/OCR 및 pairing 실험 | OCR 경로는 안정화됐으나 answer layer와 table factual 품질 병목 | shadow |
| OCR v4 | table_factual exact answer mode + structured evidence | full45 4.4389, table_15 3.7833, Type4 1.0 | baseline_v3 후보 |
| Soft CRAG-lite | retrieval evidence quality check | GroupC/Type4/latency 개선, overall/Type2 하락 | 보류 |
| Answer type router | 웹 사용자 질문의 runtime routing | 20문항 smoke 19/20, 0.95 | 최소 구현 성공 |

## baseline_v2의 의미

baseline_v2는 metadata helper와 comparison helper를 포함한다. 핵심은 비교형 질문에서 두 문서 evidence를 더 안정적으로 확보하는 것이다.

| 항목 | baseline_v1 | baseline_v2 |
|---|---:|---:|
| overall manual mean | 4.4167 | 4.4778 |
| Type2 manual mean | 3.8750 | 4.0250 |
| Type4 rejection success | 1.0000 | 0.9000 |
| dual_doc_coverage | 0.7500 | 0.8333 |
| comparison_evidence_coverage | 0.6923 | 0.6923 |
| latency | 29.12초 | 30.72초 |

baseline_v2는 Type4가 소폭 낮아졌지만, overall과 Type2 개선이 더 중요하다고 판단되어 Phase2 공식 기준으로 승격했다.

## baseline_v3 후보의 의미

baseline_v3 후보는 OCR v4를 포함한다. OCR v4는 replace가 아니라 augment 방식이며, 기존 body chunk를 유지한 채 table structured evidence를 추가한다. OCR v4는 table_15와 full45를 모두 개선했으며 Type4도 유지했다.

| 항목 | baseline_v2 | OCR v4 |
|---|---:|---:|
| table_15 mean | 3.7500 | 3.7833 |
| group_b mean | 3.8000 | 3.2500 |
| group_c mean | 4.0500 | 4.1000 |
| full45 overall | 4.3611 | 4.4389 |
| Type4 rejection success | 1.0000 | 1.0000 |
| latency | 32.89초 | 30.15초 |

Group B는 baseline보다 낮지만 v3 대비 개선되었고, table_15/full45/Type4 기준을 충족해 baseline_v3 후보로 유지했다.

## Soft CRAG-lite 판정

Soft CRAG-lite는 구현과 judge-on 비교까지 완료되었다. 결과적으로 Group C와 Type4는 개선됐지만, full45 overall과 Type2가 하락했다. 따라서 “기능 구현 완료, 채택은 보류”로 정리했다.

| 항목 | baseline_v3 soft off | soft_crag_targeted |
|---|---:|---:|
| full45 overall | 4.4111 | 4.3889 |
| Type2 | 4.3000 | 4.1750 |
| Group C | 3.9500 | 4.0500 |
| Type4 success | 0.9000 | 1.0000 |
| latency | 27.53초 | 23.55초 |

## Answer type router 판정

Answer type router는 웹 사용자의 자유 질문에 대해 `factual`, `comparison`, `follow_up`, `rejection`, `table_factual`, `table_plus_text` 중 하나를 부여하는 규칙 기반 모듈이다. 20개 smoke set 기준 19/20, exact match 0.95를 기록했으며, confidence threshold와 fallback route를 포함한다. 현재 웹 데모 연결 가능한 최소 구현으로 판단했다.


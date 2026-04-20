# 06. OCR 파이프라인 및 Evidence 요약

## 참조한 원본 파일 목록

| 구분 | 원본 경로 |
|---|---|
| 팀 OCR 최종 리포트 | `C:\Users\UserK\Downloads\github\docs\성현\table_pipeline_final_report.md` |
| 팀 OCR 작업 요약 | `C:\Users\UserK\Downloads\github\docs\성현\table_pipeline_work_summary.md` |
| 팀 OCR 코드 | `C:\Users\UserK\Downloads\중급프로젝트\ocr\build_hwp_ocr_payload.py` |
| 팀 OCR 코드 | `C:\Users\UserK\Downloads\중급프로젝트\ocr\format_hwp_ocr_for_rag.py` |
| 팀 OCR 코드 | `C:\Users\UserK\Downloads\중급프로젝트\ocr\run_hwp_ocr_pipeline.py` |
| Phase2 OCR v2 구현 | `C:\Users\UserK\Downloads\중급프로젝트\src\scenario_b_phase2\true_hwp_table_ocr_v2_augment.py` |
| Phase2 OCR v4 실행 | `C:\Users\UserK\Downloads\중급프로젝트\src\scenario_b_phase2\run_phase2_true_table_ocr_v4_exact.py` |
| OCR v4 결과 | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\phase2_runs\p2_true_table_ocr_v4_exact_v1\true_table_ocr_v4_exact_compare.csv` |
| OCR v4 리포트 | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\phase2_runs\p2_true_table_ocr_v4_exact_v1\true_table_ocr_v4_exact_report.json` |
| OCR asset | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\phase2_true_table_ocr_v4_assets` |
| OCR BM25 | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\bm25_index_phase2_true_table_ocr_v4.pkl` |

## OCR 적용 배경

Stage1의 B-05는 표 정보를 단순히 XML/텍스트 evidence로 추가하는 방식이었다. 이 방식은 표 관련 subset에서 가능성을 보였지만 full45에서는 불안정했다. Phase2에서는 이를 반복하지 않고, 실제 표 영역을 찾아 OCR evidence를 구조화한 뒤 기존 corpus를 대체하지 않고 augment하는 방식으로 재설계했다.

핵심 원칙은 다음과 같다.

| 원칙 | 설명 |
|---|---|
| replace 금지 | 기존 processed_data/body chunk를 유지하고 OCR evidence만 추가 |
| augment only | table OCR chunk는 보조 evidence로만 추가 |
| table/body 분리 | table_factual과 table_plus_text의 retrieval/packing 규칙을 분리 |
| 구조화 evidence 우선 | raw OCR text보다 header-value, row, cell, AST evidence를 우선 |
| subset gate | table_15, Group B, Group C를 먼저 통과한 경우에만 full45 평가 |

## 팀 OCR 작업의 위치

팀 OCR 작업은 HWP 구조 추출과 table 후보를 만드는 데 중요한 기반이 되었다. 다만 최종 Phase2 기준에서는 팀 OCR 결과를 full replace로 병합하지 않고, structural support layer로 사용했다.

| 팀 OCR 산출물 | 프로젝트 내 사용 방향 |
|---|---|
| `structural_table` | OCR 대상 후보 선정 및 table metadata support |
| `final_review_table` | 수동 검토/고품질 후보 추적 |
| `section_header_block` | table과 parent/section body 연결 |
| `image_ocr_candidate` | 이미지 OCR 보조 후보 |
| `discarded_shortlist` | 수동 승격 후보로만 유지 |

## Phase2 OCR 흐름

```text
HWP/PDF 원본
  -> structural candidate 추출
  -> HWP render/PDF/page image 또는 fallback
  -> table region/bbox 후보 매칭
  -> OCR/table structure 추출
  -> table AST/header-value/row/cell evidence 생성
  -> linked_parent_text / nearby_body pairing
  -> OCR augment corpus 생성
  -> 별도 BM25/Chroma index 생성
  -> table_15 -> group_bc -> full45 평가
```

## Evidence 구조

최종 OCR evidence는 raw text만 넣는 방식이 아니라, retrieval과 answer layer가 활용할 수 있도록 구조화했다.

| 필드 | 역할 |
|---|---|
| `source_doc` | 원문 문서명 |
| `chunk_type` | `table_ocr`, `header_value_pair`, `row_summary`, `paired_body` 등 |
| `table_index` | 문서 내 table 식별자 |
| `section_label` | table이 속한 section |
| `linked_parent_text` | 연결된 parent/nearby 본문 |
| `structural_text` | 구조 추출 기반 table text |
| `table_ocr_text` | OCR 결과 text |
| `table_html_or_markdown` | 가능 시 table 구조 출력 |
| `ocr_status` | 성공/실패/대체 경로 |
| `ocr_confidence` | OCR 또는 matching confidence |
| `render_source` | COM/PDF/bbox/fallback 등 render 경로 |
| `bbox/page metadata` | page, bbox, region id |

## v1~v4 변화

| 버전 | 핵심 변화 | 결과/판정 |
|---|---|---|
| OCR pilot v1 | OCR pilot corpus를 baseline_v1 위에 적용 | table_15/group_bc 하락, 미채택 |
| true_table_ocr_v1 | 실제 table OCR 가능성 검토, shadow candidate | full45/coverage 가능성은 있으나 subset 미충족 |
| true_table_ocr_v2/v2.1 | HWP render 경로, Group B/C 분리, table/body pairing 검토 | 렌더 경로보다 Group C/table+body packing 충돌이 핵심 병목 |
| true_table_ocr_v3 | question-type gated OCR routing, HybridQA-style pairing | Group C는 개선됐지만 table_factual/table_15 부족 |
| true_table_ocr_v4 | table_factual exact answer mode, structured evidence 우선순위, header/row scoring | table_15/full45 개선, baseline_v3 후보 |

## OCR v4 핵심 개선

OCR v4는 표 관련 질문에 서비스형 5단 답변 포맷을 그대로 적용하지 않고, table_factual 전용 answer mode를 추가했다. 표에 있는 값을 짧게 추출해야 하는 질문에서는 구조화 evidence와 header/row match를 우선했다.

| 항목 | 개선 내용 |
|---|---|
| answer mode | table_factual에서 긴 요약/근거 섹션 제거 |
| retrieval unit | header_value_pair, cell_row_block, row_summary를 raw OCR보다 우선 |
| scoring | header_match, row_match, entity_overlap, definition/list pattern 반영 |
| pollution 방지 | 일반적인 회의/보고 row 등 generic row block penalty |
| table_plus_text 보호 | paired_body 필수, same/parent section body 강제 |

## OCR v4 주요 결과

| 지표 | baseline_v2 | OCR v3 | OCR v4 |
|---|---:|---:|---:|
| table_15 mean | 3.7500 | 3.5333 | 3.7833 |
| Group B mean | 3.8000 | 3.1500 | 3.2500 |
| Group C mean | 4.0500 | 3.9000 | 4.1000 |
| full45 overall | 4.3611 | 4.3667 | 4.4389 |
| Type4 rejection success | 1.0000 | 1.0000 | 1.0000 |
| latency | 32.89초 | 34.68초 | 30.15초 |

## 왜 v4가 승격 후보인가

OCR v4는 table_15가 baseline_v2보다 높고, Group C도 baseline_v2보다 높으며, full45 overall도 4.3611에서 4.4389로 상승했다. Type4 rejection success도 1.0을 유지했다. Group B는 baseline_v2보다 낮지만 v3 대비 개선되었고, 전체 기준에서는 OCR 트랙 중 가장 균형 잡힌 결과였다. 따라서 최종 채택 확정이라기보다 `baseline_v3 후보`로 유지하는 것이 타당하다.


# Final Bundle Manifest

## 참조한 원본 파일 목록

| 구분 | 원본 경로 |
|---|---|
| Bundle root | `C:\Users\UserK\Downloads\중급프로젝트\final_bundle` |
| Project root | `C:\Users\UserK\Downloads\중급프로젝트` |
| Team docs root | `C:\Users\UserK\Downloads\github\docs` |

## 생성한 번들 파일

| 번들 파일 | 주요 내용 | 주요 원본 |
|---|---|---|
| `00_project_overview_and_timeline.md` | 프로젝트 목표, 전체 흐름, timeline | Stage1 compare, Phase2 reports, eval summaries |
| `01_system_architecture_and_core_code_map.md` | 시스템 구조와 핵심 코드 역할 | `src\scenario_a`, `src\scenario_b_phase2`, `src\streamlit_qa` |
| `02_stage1_b00_to_b06_summary.md` | B-00~B-06 목적/기법/판단 | `b02_b01_compare.csv`, `b03_b02_compare.csv`, `b04_full_compare.csv`, `b05_full_compare.csv`, `b06_compare.csv` |
| `03_stage1_b00_to_b06_compare.csv` | Stage1 핵심 수치 비교 | Stage1 compare CSV, eval pipeline judge refs |
| `04_phase2_progression_summary.md` | baseline_v1/v2/v3, comparison/OCR/soft-crag/router | Phase2 run outputs |
| `05_phase2_key_compare_tables.csv` | Phase2 핵심 비교표 | baseline_v2 confirm, OCR v4, Soft CRAG, router smoke |
| `06_ocr_pipeline_and_evidence_summary.md` | OCR 설계와 evidence 구조 | 팀 OCR docs, true HWP table OCR 코드/결과 |
| `07_evaluation_framework_summary.md` | auto grader, LLM Judge, qbank v4, factual mode | 팀 평가 docs, eval pipeline outputs |
| `08_evaluation_key_results.csv` | old auto/new auto/judge 핵심 결과 | `auto_grader_v41_compare.csv`, `llm_judge_compare.csv` |
| `09_auto_grader_improvement_summary.md` | qbank v4, grader v4/v4.1, 오류 유형 | improved grader outputs |
| `10_finetuning_and_dataset_summary.md` | Qwen/Gemma, SFT dataset, FT 결과 | Qwen SFT reports, Scenario A outputs |
| `11_team_contributions_summary.md` | 팀원별 산출물과 프로젝트 반영 | `github\docs\하은,윤민`, `github\docs\성현`, `github\docs\건호` |
| `12_streamlit_and_router_summary.md` | Streamlit 앱과 answer_type router | `app.py`, `src\streamlit_qa`, router smoke |
| `13_limitations_and_future_work.md` | 한계와 향후 과제 | Stage1/Phase2/eval/FT reports |
| `14_reference_list.md` | 논문/방법론/모델/라이브러리 정리 | config, requirements, 코드/문서 전반 |
| `bundle_manifest.md` | 번들-원본 매핑 | 본 파일 |
| `bundle_missing_items.md` | 못 찾은 항목과 불완전 항목 | 검색 결과 및 현재 산출물 상태 |

## 번들 작성 원칙

| 원칙 | 적용 방식 |
|---|---|
| 원본 대량 복사 금지 | raw log와 detail CSV를 그대로 넣지 않고 핵심 수치만 요약 |
| trace 가능성 | 각 문서 상단에 원본 파일 경로 기재 |
| 최종 보고서 중심 | markdown + 핵심 compare CSV 형태로 재구성 |
| 불확실성 분리 | 미완료/partial 결과는 `bundle_missing_items.md`에 별도 기록 |
| 평가 해석 분리 | LLM Judge와 auto grader를 같은 척도로 해석하지 않음 |


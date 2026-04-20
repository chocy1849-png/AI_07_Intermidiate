# 11. 팀 기여 요약

## 참조한 원본 파일 목록

| 담당 | 원본 경로 |
|---|---|
| 하은/윤민 평가 파이프라인 | `C:\Users\UserK\Downloads\github\docs\하은,윤민\최종보고서_평가파이프라인_강하은&박윤민.md` |
| 하은/윤민 실행 안내 | `C:\Users\UserK\Downloads\github\docs\하은,윤민\RUN_EVAL_PIPELINE.md` |
| 하은/윤민 qbank | `C:\Users\UserK\Downloads\github\docs\하은,윤민\(260417) PartA_RFP_AutoGrading_QBank_v3_fixed.json` |
| 하은 작업 정리 | `C:\Users\UserK\Downloads\github\docs\하은,윤민\하은_작업정리.md` |
| 성현 OCR 최종 리포트 | `C:\Users\UserK\Downloads\github\docs\성현\table_pipeline_final_report.md` |
| 성현 OCR 작업 요약 | `C:\Users\UserK\Downloads\github\docs\성현\table_pipeline_work_summary.md` |
| 건호 retrieval 분석 | `C:\Users\UserK\Downloads\github\docs\건호\part04_report.md` |
| 건호 type 분석 | `C:\Users\UserK\Downloads\github\docs\건호\part04_type_analysis.md` |
| 건호 개선 아이디어 | `C:\Users\UserK\Downloads\github\docs\건호\part04_improvement_ideas.md` |
| 건호 dense/sparse 결과 | `C:\Users\UserK\Downloads\github\docs\건호\dense_sparse_results` |

## 전체 기여 구조

프로젝트는 한 명이 모든 코드를 독립적으로 만든 구조가 아니라, 팀원이 만든 평가/문제은행/OCR/retrieval 분석 자산을 중심 RAG 고도화 흐름에 통합한 형태다.

| workstream | 담당 | 프로젝트 내 반영 위치 |
|---|---|---|
| RAG baseline 및 Stage1/Phase2 고도화 | 조찬영 | B-00~B-06, Phase2 baseline_v2/v3, Scenario A, FT, Streamlit 통합 |
| 문제은행 및 평가 파이프라인 | 박윤민, 강하은 | 340문항 auto grader, 20문항 LLM Judge, qbank v4 개선 기반 |
| OCR/table extraction | 윤성현 | structural table 후보, OCR 후보, table enrichment support layer |
| Dense/Sparse retrieval 분석 | 이건호 | hybrid retrieval 해석, dense/sparse hit 분석, Type별 retrieval 병목 파악 |

## 하은 & 윤민 기여

하은과 윤민은 평가 체계와 문제은행 구축을 담당했다. 특히 340문항 auto grader와 20문항 LLM Judge subset은 후반부 평가 파이프라인의 기준이 되었다.

| 산출물 | 역할 |
|---|---|
| qbank v3 fixed | 객관식/단답형/서술형/거부형 자동평가 문제은행 |
| auto_grader.py | 초기 exact/contains 기반 자동채점기 |
| llm_judge.py | GPT-5 judge 기반 축소셋 평가 |
| judge_subset_20.json | 빠른 LLM Judge 추세 확인용 subset |
| RUN_EVAL_PIPELINE.md | 팀원이 재현 가능한 평가 실행 안내 |

프로젝트 후반에는 이 평가 자산을 기반으로 qbank v4, improved auto grader v4/v4.1, factual mode 설계가 진행되었다. 즉 하은/윤민의 작업은 최종 “평가 체계”의 원천 자산이다.

## 성현 기여

성현은 OCR 및 table extraction 트랙을 담당했다. 팀 OCR 결과는 최종 OCR v4에 직접 full replace로 들어간 것은 아니지만, table 후보와 structural support를 만드는 데 중요한 기반이 되었다.

| 산출물 | 역할 |
|---|---|
| table_pipeline_final_report.md | HWP table extraction과 OCR 구조 요약 |
| table_pipeline_work_summary.md | 작업 절차 및 사용 방식 정리 |
| structural_table / final_review_table | OCR 대상 후보 및 검증 후보 |
| section_header_block | table과 parent section 연결 |
| image_ocr_candidate | 이미지 OCR 후보 |

Phase2에서는 성현의 결과를 replace가 아니라 augment 후보로 해석했고, structural table candidate를 true HWP table OCR 대상 선정과 metadata support에 활용했다.

## 건호 기여

건호는 dense/sparse retrieval 결과와 Type별 retrieval 병목 분석을 담당했다.

| 산출물 | 역할 |
|---|---|
| part04_report.md | dense/sparse/hybrid retrieval 분석 종합 |
| part04_type_analysis.md | 질문 유형별 retrieval 특성 분석 |
| part04_improvement_ideas.md | 후속 retrieval 개선 아이디어 |
| dense_sparse_results/*.csv | Dense/Sparse/Hybrid hit와 agreement 분석 |

핵심 해석은 Type1에서는 dense/sparse가 대체로 안정적이지만, Type2 비교형에서는 sparse drift와 multi-doc evidence coverage 문제가 크다는 점이었다. 이 분석은 Stage1의 hybrid 유지와 Phase2의 comparison helper 설계 근거가 되었다.

## 팀원 작업이 최종 구조에 포함된 방식

| 팀원 작업 | 최종 반영 |
|---|---|
| 문제은행/평가 | `eval_pipeline`, `qbank v4`, `improved_auto_grader`, `LLM Judge` |
| OCR/table 후보 | `true_table_ocr_v4`, `OCR evidence`, `table/body pairing` |
| Dense/Sparse 분석 | `B-01 hybrid`, `metadata-aware retrieval`, `comparison helper` |
| 문항 검수 | Qwen SFT dataset generation/filtering의 품질 기준 |

## 발표용 핵심 메시지

팀원 작업은 단순 부가자료가 아니라, 전체 프로젝트에서 다음 위치를 담당한다.

| 메시지 | 설명 |
|---|---|
| 평가가 먼저 정리되었기 때문에 고도화 판단이 가능했다 | 하은/윤민 평가셋과 Judge subset |
| 표/OCR 병목을 실제 데이터 기준으로 볼 수 있었다 | 성현 OCR/table 후보 |
| retrieval 개선 방향이 경험이 아니라 분석 기반이었다 | 건호 dense/sparse/type 분석 |
| 최종 RAG 구조는 팀원 산출물을 통합한 결과다 | Stage1/Phase2/Scenario A/Streamlit |


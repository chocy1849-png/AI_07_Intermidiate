# 07. 평가 프레임워크 요약

## 참조한 원본 파일 목록

| 구분 | 원본 경로 |
|---|---|
| 팀 평가 최종 문서 | `C:\Users\UserK\Downloads\github\docs\하은,윤민\최종보고서_평가파이프라인_강하은&박윤민.md` |
| 팀 평가 실행 안내 | `C:\Users\UserK\Downloads\github\docs\하은,윤민\RUN_EVAL_PIPELINE.md` |
| 팀 auto grader | `C:\Users\UserK\Downloads\github\docs\하은,윤민\auto_grader.py` |
| 팀 LLM Judge | `C:\Users\UserK\Downloads\github\docs\하은,윤민\llm_judge.py` |
| 20문항 Judge subset | `C:\Users\UserK\Downloads\github\docs\하은,윤민\judge_subset_20.json` |
| qbank v4 | `C:\Users\UserK\Downloads\중급프로젝트\src\PartA_RFP_AutoGrading_QBank_v4_reviewed.json` |
| qbank v4 리포트 | `C:\Users\UserK\Downloads\중급프로젝트\src\PartA_RFP_AutoGrading_QBank_v4_review_report.md` |
| improved grader v4 | `C:\Users\UserK\Downloads\중급프로젝트\src\scenario_b_phase2\improved_auto_grader.py` |
| improved grader v4.1 | `C:\Users\UserK\Downloads\중급프로젝트\src\scenario_b_phase2\improved_auto_grader_v41.py` |
| LLM Judge 결과 | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\eval_pipeline\llm_judge_compare.csv` |
| Auto grader 결과 | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\eval_pipeline\auto_grader_v41_compare.csv` |

## 평가 체계의 목적

본 프로젝트의 평가는 하나의 점수만으로 판단하지 않고, 목적이 다른 평가 축을 분리했다.

| 평가 축 | 목적 | 장점 | 한계 |
|---|---|---|---|
| full45 LLM Judge | 서비스형 RAG 답변 품질 평가 | faithfulness/completeness/groundedness/relevancy를 종합 평가 | 비용과 시간이 큼 |
| 20문항 LLM Judge subset | 빠른 단계별 추세 확인 | 동일 judge 기준으로 stage progression 확인 | 문항 수가 적어 전체 대표성 제한 |
| 340문항 auto grader | factual retrieval/fact extraction 보조 평가 | 대량 자동 평가 가능 | 장문 답변과 exact grading이 충돌 |
| mini subset | 빠른 후보 압축 | 실험 시간 절감 | 최종 채택 근거로는 부족 |

## LLM Judge

LLM Judge는 GPT-5 계열 judge를 사용해 다음 네 항목을 1~5점으로 평가했다.

| 항목 | 의미 |
|---|---|
| Faithfulness | 답변이 제공된 문맥에 충실한가 |
| Completeness | 질문이 요구한 정보를 빠짐없이 포함했는가 |
| Groundedness | 답변 근거가 실제 검색 문맥에 기반하는가 |
| Relevancy | 질문 의도와 직접 관련된 답변인가 |

이 지표는 최종 서비스형 답변 품질의 주 판단 기준이다. 특히 Stage1/Phase2 채택 여부는 retrieval hit보다 manual mean과 항목별 품질을 우선했다.

## Auto grader

초기 auto grader는 340문항 문제은행에 대해 exact/contains 기반 자동채점을 수행했다. 그러나 서비스형 RAG 답변은 “한줄 요약, 핵심 내용, 요구사항, 일정/예산, 참고근거” 같은 장문 포맷을 출력하기 때문에, 정답이 포함되어 있어도 auto grader가 오답으로 처리하는 경우가 많았다.

따라서 auto grader는 LLM Judge의 대체물이 아니라, 다음 목적의 보조 지표로 재정의했다.

| 재정의 | 설명 |
|---|---|
| 서비스형 품질 평가 아님 | 장문 답변의 전반 품질은 LLM Judge가 담당 |
| factual 측정기 | retrieval 결과에서 정답 slot을 찾을 수 있는지 측정 |
| extraction 중심 | 답변 전체 비교가 아니라 핵심 span 추출 후 비교 |
| 유형별 분석 | short_answer, number/date/currency, list_set, slot_pair 등으로 분리 |

## qbank v4 개선

qbank v4에서는 채점에 필요한 필드를 더 명확히 했다.

| 필드 | 역할 |
|---|---|
| `answer_eval_type` | choice, number, currency, date, duration, list_set, slot_pair, free_string 등 |
| `canonical_answer` | 표준 정답 |
| `answer_aliases` | 허용 가능한 별칭 |
| `choice_index` | 객관식 정답 번호 |
| `scoring_mode` | strict/tolerant/partial 등 채점 모드 |
| `choices` | 보기 텍스트 |

## improved auto grader v4/v4.1

v4는 qbank v4 schema를 실제 채점에 반영했다. v4.1은 더 tolerant하게 푸는 대신 extraction quality를 높이는 방향이었다.

| 버전 | 핵심 개선 |
|---|---|
| v4 | qbank v4 필드 반영, 유형별 extractor, strict/tolerant/partial 출력 |
| v4.1 | long-form short_answer mining, slot_pair label-aware extraction, list_set alias/canonicalization, free_string phrase mining |

## 왜 factual mode가 필요했는가

v4.1은 old auto보다 공정했지만, LLM Judge 추세와의 방향 정렬은 여전히 약했다. 원인은 grader 자체의 tolerance 부족만이 아니라, 서비스형 장문 answer path가 정답형 평가에 맞지 않는다는 점이었다.

따라서 후속 설계는 다음 두 평가 전용 path로 정리했다.

| 평가 path | 설명 |
|---|---|
| Constrained Factual Answer Mode | 모델 생성은 유지하되 한 줄/값 중심으로 답변 형식을 강하게 제한 |
| Retrieval + Extractor Mode | generation을 최소화하고 retrieved context에서 answer_eval_type에 맞는 값을 직접 추출 |

이 설계는 auto grader를 “최종 서비스 품질 평가”가 아니라 “retrieval/fact extraction 개선 측정”에 맞게 재배치하기 위한 것이다.


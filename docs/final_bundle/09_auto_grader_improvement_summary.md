# 09. Auto Grader 개선 요약

## 참조한 원본 파일 목록

| 구분 | 원본 경로 |
|---|---|
| qbank v4 | `C:\Users\UserK\Downloads\중급프로젝트\src\PartA_RFP_AutoGrading_QBank_v4_reviewed.json` |
| qbank v4 리포트 | `C:\Users\UserK\Downloads\중급프로젝트\src\PartA_RFP_AutoGrading_QBank_v4_review_report.md` |
| improved grader v4 | `C:\Users\UserK\Downloads\중급프로젝트\src\scenario_b_phase2\improved_auto_grader.py` |
| improved grader v4.1 | `C:\Users\UserK\Downloads\중급프로젝트\src\scenario_b_phase2\improved_auto_grader_v41.py` |
| v4 비교 | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\eval_pipeline\auto_grader_old_vs_new_compare.csv` |
| v4.1 비교 | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\eval_pipeline\auto_grader_v41_compare.csv` |
| v4.1 trend report | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\eval_pipeline\auto_trend_vs_reference_judge_report_v41.md` |
| representative cases | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\eval_pipeline\representative_error_cases_v41.md` |
| regression cases | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\eval_pipeline\regression_case_results_v41.md` |
| factual mode 설계 | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\eval_pipeline\auto_eval_factual_mode_design.md` |

## 문제 정의

초기 auto grader는 340문항 문제은행에 대해 자동채점이 가능했지만, 서비스형 RAG 답변의 장문 포맷과 맞지 않았다. 모델 답변 안에 정답이 포함되어 있어도 부연설명, 섹션 제목, bullet, 표기 차이 때문에 오답으로 처리되는 경우가 많았다.

따라서 개선 방향은 “LLM Judge 점수를 억지로 따라가게 만들기”가 아니라, auto grader를 정답형 factual 측정기로 재정의하는 것이었다.

## qbank v4 수정 내용

qbank v4는 채점기가 정답을 더 정확히 해석할 수 있도록 다음 필드를 source of truth로 사용한다.

| 필드 | 반영 방식 |
|---|---|
| `answer_eval_type` | choice/number/currency/date/duration/list_set/slot_pair/free_string 분기 |
| `canonical_answer` | 표준 정답 비교 |
| `answer_aliases` | 허용 별칭 |
| `choice_index` | 객관식 번호 기반 canonicalize |
| `scoring_mode` | strict/tolerant/partial 점수 분기 |

## improved auto grader v4

v4는 유형별 extractor와 채점기를 추가했다.

| 유형 | 개선 내용 |
|---|---|
| choice | 번호, 보기 텍스트, 번호+텍스트 모두 허용 |
| number/currency | 원/천원/백만원 등 단위 normalize |
| date/duration | 날짜 표기와 “계약일로부터 N일” normalize |
| email/phone/url | punctuation/spacing normalize |
| list_set/slot_pair | unordered set, slot-wise 비교, partial coverage |
| free_string | exact 실패 시 alias/contains 허용 |

## improved auto grader v4.1

v4.1은 tolerance를 무작정 넓힌 것이 아니라 extraction 품질을 높이는 방향이었다.

| 병목 | 개선 |
|---|---|
| short_answer long-form | 번호 제목 제거, answer cue 탐색, 핵심 phrase mining |
| slot_pair | heading 번호 제거, label-aware extraction, noisy pair 방지 |
| list_set | prefix/suffix 제거, alias dictionary, unordered compare |
| free_string factual | head noun + value phrase 중심의 짧은 canonical phrase 추출 |

회귀 테스트로 `DOC005_Q001`, `DOC037_Q002`, `DOC019_Q004`, `DOC020_Q005`, `DOC050_Q001`, `DOC046_Q001`, `DOC085_Q001` 계열을 고정했다.

## 핵심 결과

| 버전 | old auto | v4.1 tolerant | 변화 |
|---|---:|---:|---:|
| B-00 | 0.6471 | 0.6441 | -0.0030 |
| B-01 | 0.4676 | 0.6706 | +0.2030 |
| B-02 | 0.4735 | 0.6471 | +0.1736 |
| B-06 exact | 0.3529 | 0.5647 | +0.2118 |
| phase2 baseline_v2 | 0.3588 | 0.5706 | +0.2118 |
| baseline_v3 | 0.3441 | 0.5618 | +0.2177 |

v4.1은 후반 버전이 old auto에서 과도하게 깎이던 현상을 완화했다. 특히 baseline_v3는 old auto 0.3441에서 v4.1 tolerant 0.5618로 회복되었다.

## 대표 오류 패턴

| 오류 유형 | 원인 | 개선 방향 |
|---|---|---|
| 객관식 번호/텍스트 혼용 | “2번”, “차장”, “2. 차장”을 다르게 처리 | choice canonicalize |
| 금액/날짜 표기 차이 | 원/천원/백만원, 날짜 포맷 차이 | numeric/date normalize |
| 정답 포함 + 부연설명 | 정답 이후 설명 때문에 exact 실패 | 핵심 span extraction |
| 목록형 일부 정답 | prefix나 alias 차이로 partial 처리 | alias dictionary, unordered set |
| slot_pair 오염 | section 번호가 값으로 추출 | heading 제거, label-aware extraction |

## Judge와 완전히 정렬되지 않은 이유

v4.1은 공정성을 높였지만, LLM Judge 추세와의 방향 정렬은 여전히 약했다. 이유는 두 평가가 보는 대상이 다르기 때문이다.

| 평가 | 보는 것 |
|---|---|
| LLM Judge | 긴 서비스형 답변의 충실성, 완결성, 근거성, 관련성 |
| Auto grader | 특정 정답 slot을 정확히 추출했는지 |

따라서 장문 answer layer를 그대로 둔 auto grader는 Stage2의 실제 장점을 충분히 반영하지 못한다. 이 결론 때문에 auto grader는 LLM Judge 대체가 아니라 `retrieval/fact extraction 측정기`로 재배치했다.

## Constrained factual mode / Retrieval+Extractor 필요성

후속 설계는 두 가지 평가 전용 answer path다.

| path | 목적 |
|---|---|
| Constrained Factual Answer Mode | generation은 유지하되 답변을 한 줄/값 중심으로 제한 |
| Retrieval + Extractor Mode | retrieved context에서 answer_eval_type별 값을 직접 추출 |

이 두 path는 서비스용 prompt와 분리되어야 한다. 목적은 최종 서비스 답변 품질 평가가 아니라, hybrid/prefix/helper/OCR 같은 기법이 factual extraction에 실제로 도움이 되는지를 보는 것이다.


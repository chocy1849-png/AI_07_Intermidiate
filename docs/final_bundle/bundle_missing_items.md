# Bundle Missing / Incomplete Items

## 참조한 원본 파일 목록

| 구분 | 원본 경로 |
|---|---|
| 검색 대상 root | `C:\Users\UserK\Downloads\중급프로젝트` |
| 팀 문서 root | `C:\Users\UserK\Downloads\github\docs` |
| eval pipeline | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\eval_pipeline` |
| Scenario A outputs | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs` |

## 못 찾았거나 최종 파일로 확정되지 않은 항목

| 항목 | 상태 | 처리 |
|---|---|---|
| `qwen_ft_final_report.md` | 명시적 파일명으로는 확인되지 않음 | `qwen_ft_main_experiment_v1\train_report.md`, `train_summary.json`, FT compare CSV로 대체 |
| Gemma4 FT 최종 결과 | 완료 산출물 확인 불가 | Scenario A Gemma raw/Stage1 비교까지만 반영 |
| factual mode full 340 final run | 일부 run이 partial/incomplete 상태로 확인됨 | 최종 점수 표에는 넣지 않고 설계/한계로만 기록 |
| B-00 full45 manual 결과 | full45 동일 형식의 B-00 결과 확인 불가 | eval pipeline 20문항 Judge reference와 raw baseline 설명으로 대체 |
| Soft CRAG-lite 채택 결과 | judge-on full run은 있으나 승격 조건 미충족 | 보류로 기록 |
| follow-up/source-state 완성형 구현 | 최종 baseline에 포함되지 않음 | TODO/향후 과제로 기록 |
| post-generation verification 완성형 | 구현 완료 산출물 없음 | TODO/향후 과제로 기록 |

## 주의해서 해석해야 할 항목

| 항목 | 주의점 |
|---|---|
| Stage1 B-02 manual mean과 B-06 gpt5mini manual mean | 평가 runner/pipeline 조건이 완전히 동일하지 않을 수 있으므로 “B-06이 B-02보다 낮다”를 단순 성능 하락으로 해석하지 말 것 |
| Auto grader 결과 | 서비스형 장문 답변 품질이 아니라 factual extraction 보조 지표로 해석 |
| OCR v4 | baseline_v3 후보로 정리했지만, Group B가 baseline_v2보다 낮아 최종 운영 채택 전 추가 검증 필요 |
| Router 19/20 | smoke set 기준이며 실제 운영 전체 정확도를 의미하지 않음 |
| Qwen FT | B02 고정에서는 소폭 개선이나 전체 stage1 조합에서는 하락. FT 자체를 실패로 단정하기보다 retrieval 구조 우선으로 해석 |

## 문서 품질 관련 메모

일부 팀원 문서 또는 과거 산출물은 인코딩 문제로 내용 일부가 깨져 보일 수 있었다. 해당 경우에는 파일명, 코드 구조, 결과 CSV, 이후 통합 실험 결과를 기준으로 의미를 복원했다. 최종 번들에는 깨진 원문을 그대로 복사하지 않고, 프로젝트에서 실제로 사용된 결론과 수치 중심으로 정리했다.

## 다음에 보강하면 좋은 원본

| 보강 항목 | 이유 |
|---|---|
| Qwen FT 최종 평가 report 단일 파일 | FT 결론을 한 파일에서 trace하기 쉬움 |
| baseline_v3 serving manifest | 웹 데모에서 실제 어떤 config를 쓰는지 명확화 |
| factual mode full run 결과 | auto grader를 factual 측정기로 재정의한 결론 강화 |
| OCR v4 evidence sample sheet | 발표 시 OCR evidence가 어떻게 답변에 쓰이는지 시각적으로 설명 가능 |


# 10. 파인튜닝 및 데이터셋 요약

## 참조한 원본 파일 목록

| 구분 | 원본 경로 |
|---|---|
| SFT v3 report | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\qwen_ft_instruction_expansion_v3_refine\qwen_sft_v3_refine_report.md` |
| SFT v3 stats | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\qwen_ft_instruction_expansion_v3_refine\qwen_sft_v3_refine_stats.json` |
| SFT train | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\qwen_ft_instruction_expansion_v3_refine\data\sft\qwen_sft_train_v3.jsonl` |
| SFT val | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\qwen_ft_instruction_expansion_v3_refine\data\sft\qwen_sft_val_v3.jsonl` |
| Qwen FT train report | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\qwen_ft_main_experiment_v1\train_report.md` |
| Qwen FT summary | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\qwen_ft_main_experiment_v1\train_summary.json` |
| Qwen FT 평가 | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\scenario_a_bgem3_qwen_ft_v1_vs_qwen_compare.csv` |
| B02 fixed FT 비교 | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\scenario_a_b02fixed_bgem3_qwen_vs_qwenft_full_compare.csv` |
| Gemma raw vs Stage1 | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\scenario_a_raw_baseline_gemma4_e4b\scenario_a_gemma4_raw_vs_stage1_compare.csv` |
| Scenario A 요약 | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\scenario_a_ablation_summary.md` |

## Scenario A 목적

Scenario A는 OpenAI generator 대신 로컬/HF 모델을 적용했을 때 현재 RAG pipeline이 어느 정도 성능을 유지하는지 검증하는 실험이다. 핵심 질문은 두 가지였다.

| 질문 | 확인 방식 |
|---|---|
| 모델 자체 성능이 좋은가 | raw baseline 또는 동일 retrieval에서 모델만 교체 |
| RAG 고도화가 모델을 바꿔도 효과가 있는가 | raw baseline vs Stage1 pipeline 비교 |

## Qwen SFT 데이터셋 구축 흐름

Qwen SFT 데이터셋은 한 번에 생성하지 않고, 검수와 유형별 보강을 거쳤다.

```text
질문 생성/검수
  -> approved 질문셋 생성
  -> expansion v1
  -> refinement v2
  -> targeted v3
  -> train/val split
  -> Qwen FT 본실험
```

초기에는 single_doc_factual 편향이 강했고 comparison/rejection/table_plus_body/follow_up이 부족했다. 이후 유형별 완화와 추가 생성, retrieval 보정을 통해 FT 가능한 분포로 복구했다.

## SFT v3 최종 분포

| 항목 | 수치 |
|---|---:|
| 전체 accepted | 2,952 |
| train | 2,660 |
| val | 292 |
| single_doc_factual | 1,389 |
| comparison | 919 |
| rejection | 257 |
| table_plus_body | 197 |
| follow_up | 190 |

목표였던 `accepted >= 2400`, `single_doc_factual >= 1000`, `table_plus_body >= 100`을 모두 충족했다.

## Qwen FT 설정

| 항목 | 값 |
|---|---|
| base model | `Qwen/Qwen2.5-7B-Instruct` |
| train rows | 2,660 |
| val rows | 292 |
| max_length | 4,608 |
| epoch | 1.0 |
| train loss | 0.5046 |
| eval loss | 0.3403 |
| LoRA r/alpha/dropout | 16 / 32 / 0.05 |
| target modules | q/k/v/o/gate/up/down proj |
| train truncation ratio | 0.5289 |
| val truncation ratio | 0.4932 |

## Qwen FT 결과 해석

Qwen FT는 일부 조건에서는 소폭 개선이 있었지만, 전체 운영 관점에서는 기대보다 제한적이었다.

| 비교 | base | FT | 해석 |
|---|---:|---:|---|
| bge-m3 + Qwen stage1 | 3.9611 | 3.6444 | FT 후 전체 하락 |
| B02 fixed Qwen vs Qwen FT | 3.8222 | 3.8944 | B02 고정에서는 +0.0722 소폭 개선 |
| B02 fixed Type2 | 3.3750 | 3.2000 | 비교형은 하락 |
| B02 fixed Type4 | 4.7000 | 4.5500 | rejection도 하락 |

결론적으로 FT는 생성 스타일이나 특정 factual 대응을 일부 바꿀 수 있지만, retrieval miss, 문서 coverage, OCR/table evidence 같은 구조적 병목을 해결하지 못했다.

## Gemma4 관련 결과

Gemma4-E4B는 raw baseline도 강했고, Stage1 pipeline을 적용했을 때 추가 상승했다.

| 조합 | manual mean |
|---|---:|
| Gemma4 raw baseline | 4.1667 |
| Gemma4 Stage1 | 4.4278 |
| Qwen Stage1 | 3.7556 |
| operational gpt-5-mini | 4.4222 |

이 결과는 “모델 자체가 좋은 것”과 “RAG pipeline이 좋은 것”을 분리해 해석해야 한다. Gemma4는 모델 자체 성능이 높지만, raw에서 Stage1로 상승했기 때문에 RAG 고도화 효과도 존재한다고 판단했다.

## 왜 FT보다 RAG 구조 개선이 더 유의미했는가

| 이유 | 설명 |
|---|---|
| 병목 위치 | 핵심 문제는 generator가 아니라 retrieval coverage, table evidence, 비교형 evidence organization에 있었음 |
| FT 데이터 한계 | SFT는 답변 스타일을 학습하지만 누락된 근거를 만들 수 없음 |
| 평가 결과 | Qwen FT는 전체/Type2/Type4에서 불안정했고, RAG 구조 개선은 여러 모델에서 효과 확인 |
| 실무 적용성 | OCR, comparison helper, router는 모델을 바꿔도 재사용 가능 |

따라서 최종 우선순위는 FT 추가 반복보다 RAG 구조 개선과 평가 체계 정리로 잡았다.


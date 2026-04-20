# 02. Stage1 B-00 ~ B-06 요약

## 참조한 원본 파일 목록

| 구분 | 원본 경로 |
|---|---|
| B01/B02 비교 | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\b02_b01_compare.csv` |
| B02/B03 비교 | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\b03_b02_compare.csv` |
| B04 chunking 비교 | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\b04_full_compare.csv` |
| B05 표 enrichment 비교 | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\b05_full_compare.csv` |
| B06 모델 비교 | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\b06_compare.csv` |
| Scenario A 요약 | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\scenario_a_ablation_summary.md` |
| 평가 파이프라인 | `C:\Users\UserK\Downloads\중급프로젝트\rag_outputs\eval_pipeline\llm_judge_compare.csv` |

## Stage1 목적

Stage1의 목적은 RFP RAG의 기본 성능 병목을 찾고, 비교형 질문과 문서 맥락 손실을 줄이는 안정적 운영 기준선을 만드는 것이었다. 실험은 기법을 누적 적용하되, 각 단계에서 채택/보류/기각을 분리해 판단했다.

## 단계별 요약

| 단계 | 목적 | 적용 기법 | 핵심 결과 | 판단 |
|---|---|---|---|---|
| B-00 | 기본 RAG baseline 구축 | similarity search + GPT 응답 | 문서 QA는 가능하지만 비교형 completeness와 후속 문맥이 약함 | 기준점 |
| B-01 | 검색 안정성 개선 | Dense + BM25 Hybrid | B-02 비교의 직전 기준, overall 4.4944 | 채택 기반 |
| B-02 | 청크 맥락 강화 | 구조화 metadata/prefix v2 | overall 4.6611, B-01 대비 +0.1667 | Stage1 기본선 채택 |
| B-03a | hard case rescue | CRAG 계열 조건부 retrieval 보정 | Type2는 상승했지만 전체/Type3/Type4 하락 | 조건부 채택 |
| B-03b | 대화형 확장 | history-aware rewrite/source-state 실험 | follow-up 안정성 개선 의도였으나 최종 기본선에는 미포함 | 보류 |
| B-04 | chunk 크기 재탐색 | 1200/160 chunk | mini에서는 유망했지만 full45에서 B-02보다 낮음 | 미채택 |
| B-05 | 초기 표 enrichment | table/XML evidence augment | 표 특화 일부 개선 가능성, full45와 Type4 하락 | 미채택 |
| B-06 | 모델 크기 비교 | gpt-5-mini / gpt-5 / nano 비교 | gpt-5-mini 운영 기준, gpt-5 ceiling | 기준 확정 |

## B-02가 기본선이 된 이유

B-02는 단순 retrieval hit만 높인 것이 아니라, generator가 청크를 읽을 때 필요한 문서 맥락을 prefix로 제공했다. 사업명, 발주기관, 예산, 기간, 계약방식, section role 같은 구조화 필드가 포함되면서 청크 단독 맥락 손실이 줄었다. full45 기준 B-01 대비 overall manual mean이 4.4944에서 4.6611로 상승했다.

## B-03a가 조건부 채택인 이유

B-03a는 Type2 비교형에서 B-02 대비 4.10에서 4.50으로 개선되었다. 그러나 전체 평균은 4.6611에서 4.5444로 낮아졌고, Type3 후속 질문과 Type4 rejection에서 하락이 발생했다. 따라서 전역 적용이 아니라 비교형 또는 low-confidence hard case에만 개입하는 보조 계층으로 정리했다.

## B-04가 미채택된 이유

1200/160 chunk는 더 긴 문맥을 보존하는 장점이 있었지만, full45에서는 B-02보다 overall, completeness, relevancy가 낮았다. top1/topk hit가 유사한 상태에서 latency는 29.91초에서 41.60초로 증가했다. 이는 긴 청크가 필요한 문맥 외의 노이즈를 함께 넣어 답변 초점을 흐린 것으로 해석했다.

## B-05가 미채택된 이유

초기 표 enrichment는 table 관련 subset에서는 가능성을 보였지만, full45에서 B-02 대비 overall이 하락했고 Type4 rejection도 4.85에서 4.275로 낮아졌다. 표 정보를 추가하는 방향 자체는 유지하되, Stage1 기본선에는 반영하지 않고 Phase2 OCR 트랙으로 이관했다.

## B-06의 의미

B-06은 Stage1의 “새로운 retrieval 기법”이 아니라, generator 크기 변화가 성능과 latency에 미치는 영향을 확인한 ceiling check다.

| 모델 | manual mean | latency | 해석 |
|---|---:|---:|---|
| gpt-5-mini | 4.4222 | 28.58초 | 운영 기준 |
| gpt-5 | 4.6278 | 24.11초 | 최고 성능 ceiling |

운영 기준은 비용과 실험 연속성을 고려해 gpt-5-mini로 유지했고, gpt-5는 retrieval 구조 개선의 상한을 확인하는 비교 기준으로 사용했다.


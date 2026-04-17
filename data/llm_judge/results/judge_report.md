# LLM Judge 평가 보고서

> 평가일: 2026-04-16  
> 모델: gpt-4o-mini  
> 문항 수: 20개  

---

## 1. 전체 결과

- Pass: 20개
- Review: 0개
- Fail: 0개
- Avg Score: 4.99

| 지표 | 평균 점수 |
|------|-----------|
| Faithfulness | 5.00 |
| Completeness | 4.95 |
| Groundedness | 5.00 |
| Relevancy | 5.00 |

---

## 2. 유형별 집계

| 그룹 | Pass | Review | Fail | Avg |
|------|------|--------|------|-----|
| judge_type:comparison | 5 | 0 | 0 | 4.95 |
| judge_type:follow_up | 6 | 0 | 0 | 5.00 |
| judge_type:rejection | 4 | 0 | 0 | 5.00 |
| judge_type:single_doc | 5 | 0 | 0 | 5.00 |
| difficulty:advanced | 12 | 0 | 0 | 4.98 |
| difficulty:intermediate | 8 | 0 | 0 | 5.00 |

---

## 3. 재검토/실패 문항

재검토 또는 실패 문항 없음
---

## 4. 해석

- 이 결과는 Exact Match 대체가 아니라 서술형 품질 진단용이다.
- Review/Fail 문항은 검색 실패인지, 응답 생성 실패인지, Judge 기준상 문제인지 분리해서 다시 보면 된다.
- 멀티턴과 거부응답은 동일 질문셋을 파이프라인 비교용으로 재사용할 수 있다.
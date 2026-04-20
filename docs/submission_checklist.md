# 제출 전 체크리스트

## 포함해야 하는 것

| 항목 | 상태 |
|---|---|
| 재현 가능한 코드 | `src/`, `experiments/`, `app.py`에 포함 |
| 핵심 설정 파일 | `config/`에 포함 |
| 평가 질문/로직 | `evaluation/`, `src/scenario_b_phase2/`에 포함 |
| 보고서 작성용 요약 문서 | `docs/final_bundle/`에 포함 |
| 팀원 기여 문서 | `docs/건호`, `docs/성현`, `docs/하은,윤민`에 포함 |
| 최종 보고서 PDF | `docs/report/final_report.pdf` 추가 필요 |
| 협업일지 링크 | `docs/collaboration_logs.md` 업데이트 필요 |

## 포함하면 안 되는 것

| 항목 | 이유 |
|---|---|
| `.env` | API key 포함 가능 |
| 원본 `.hwp`, `.pdf`, `.hwpx` | NDA 및 원본 데이터 외부 공유 금지 |
| `data_list.csv` | 원본 데이터 목록 |
| `processed_data/` | 원문 파싱 결과 포함 가능 |
| `rag_outputs/` | 대용량 결과물 및 중간 산출물 |
| Chroma DB, BM25 pickle | 대용량 로컬 인덱스 |
| model checkpoint/adapter | 대용량 모델 산출물 |

## 제출 직전 확인 명령

```bash
git status --short
git ls-files | findstr /i ".env .hwp .hwpx .pdf data_list processed_data rag_outputs .pkl .pickle .safetensors"
```

보고서 PDF를 추가하는 경우에는 `docs/report/final_report.pdf`만 예외적으로 포함할 수 있습니다.

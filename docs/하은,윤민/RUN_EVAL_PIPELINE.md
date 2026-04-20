# 평가 파이프라인 실행 가이드

## 1) 작업 경로

```powershell
cd C:\Users\UserK\Downloads\github\docs\하은
```

## 2) 의존성 설치

```powershell
python -m pip install -r requirements_eval_pipeline.txt
```

## 3) API 키 설정

둘 중 하나로 설정합니다.

```powershell
# 현재 세션만
$env:OPENAI_API_KEY="sk-..."
```

또는 `.env` 파일에 `OPENAI_API_KEY=...` 저장

## 4) 실행 전 점검

```powershell
# 자동채점 엔진 CLI 확인
python .\auto_grader.py --help

# LLM Judge CLI 확인 (실행 전 dry-run 권장)
python .\llm_judge.py --dry-run --limit 2
```

## 5) 자동채점 실행

### A. 로직 검증 모드 (API 호출 없음)

```powershell
python .\auto_grader.py `
  --mode verify `
  --question-bank ".\(260417) PartA_RFP_AutoGrading_QBank_v3_fixed.json" `
  --limit 20 `
  --output-dir ".\data\eval_results_verify"
```

### B. 실제 RAG 모드 (API + 벡터 인덱스 필요)

```powershell
python .\auto_grader.py `
  --mode rag `
  --question-bank ".\(260417) PartA_RFP_AutoGrading_QBank_v3_fixed.json" `
  --limit 50 `
  --output-dir ".\data\eval_results_rag"
```

## 6) LLM Judge 실행

```powershell
python .\llm_judge.py `
  --input ".\judge_subset_20.json" `
  --model "gpt-5" `
  --output-dir ".\data\llm_judge\results_current"
```

## 7) 산출물

- 자동채점
  - `data/eval_results_*/eval_detail.csv`
  - `data/eval_results_*/eval_summary.csv`
- LLM Judge
  - `data/llm_judge/results_*/judge_detail.csv`
  - `data/llm_judge/results_*/judge_summary.csv`
  - `data/llm_judge/results_*/judge_report.md`

## 8) 주의사항

- `--mode rag` / `llm_judge.py`는 파싱 데이터(`data/parsed`)와 Chroma 인덱스(`data/chroma_db`)가 준비되어 있어야 합니다.
- 현재 폴더 기준으로는 `data/parsed`, `data/chroma_db`, `data/metadata.db`가 비어 있으므로, 필요하면 먼저 문서 파싱/임베딩을 수행해야 합니다.

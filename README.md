# RFP 기반 RAG Q&A 시스템

## 개요
100개의 실제 RFP(제안요청서) 문서를 대상으로 한국어 RAG 챗봇을 구축하는 프로젝트입니다.  
현재 리포지토리에는 다음 범위가 반영되어 있습니다.

- 베이스라인 RAG 구축
- 고도화 1단계 실험 코드
  - `B-01` Hybrid Retrieval
  - `B-02` Prefix-v2
  - `B-03a` CRAG
  - `B-04` Chunking 실험
  - `B-05` Table enrichment / OCR 실험
  - `B-06` Adopted pipeline + generator ceiling check

## 현재 채택 상태
- 기본선: `B-02`
- 조건부 보강: `B-03a`
- Generator ceiling: `gpt-5`

## 실행 전 준비
이 리포지토리에는 원본 데이터, 전처리 결과, 평가 결과물, Chroma DB가 포함되지 않습니다.

필수 준비물:
- `.env` 파일
  - `OPENAI_API_KEY=...`
- `processed_data/processed_documents.jsonl`
- 필요 시 `files/files/` 및 `data_list.csv`

설치:

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## 주요 경로
```text
.
├─ evaluation/
│  ├─ day3_partA_eval_questions_v1.txt
│  ├─ day4_mini_eval_question_ids_v1.txt
│  ├─ day4_smoke_eval_question_ids_v1.txt
│  ├─ day4_b05_group_bc_question_ids_v1.txt
│  └─ eval_questions_table_v1.txt
├─ experiments/
│  ├─ RAG_베이스라인_재현.ipynb
│  └─ RAG_고도화1단계_재현.ipynb
├─ 01_컨텍스트_청킹.py
├─ 02_임베딩_생성_크로마적재.py
├─ 03_나이브_RAG_베이스라인.py
├─ ...
├─ 34_B06_채택파이프라인_평가.py
├─ 35_B06_비교.py
├─ rag_공통.py
├─ rag_bm25.py
├─ eval_utils.py
└─ requirements.txt
```

## 재현 노트북
- 베이스라인 재현: `experiments/RAG_베이스라인_재현.ipynb`
- 고도화 1단계 재현: `experiments/RAG_고도화1단계_재현.ipynb`

두 노트북 모두 `PROJECT_ROOT = Path.cwd()`를 기본값으로 사용합니다.  
노트북 실행 전 작업 폴더가 리포지토리 루트인지 먼저 확인해야 합니다.

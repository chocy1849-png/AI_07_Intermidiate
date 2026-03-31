# RFP 기반 RAG Q&A 시스템

## 프로젝트 개요
100개의 실제 RFP(제안요청서) 문서를 기반으로 RAG 시스템을 구축하여,
사용자 질문에 맞는 정보를 추출·요약·응답하는 Q&A 서비스입니다.

현재 프로젝트에는 원본 문서, 메타데이터 CSV, 프로젝트 가이드 문서,
EDA 노트북 초안이 포함되어 있으며 이후 Retrieval, Generation, 평가 파이프라인을 순차적으로 확장할 예정입니다.

## 협업일지 링크
https://www.notion.so/33478f95df758097a75dfb773fbfd123?v=33478f95df75803eaca8000ca8f26df2

## 팀 구성
| 이름 | 역할 |
|------|------|
| 조찬영 | Project Manager |
| 강하은 |  |
| 박윤민 |  |
| 윤성현 |  |
| 이건호 |  |

## 기술 스택
- Python, Jupyter Notebook
- LangChain
- OpenAI API (gpt-5-mini, text-embedding-3-small)
- FAISS / Chroma
- GCP VM (L4 GPU)

## 현재 파일 구조
```text
.
├─ README.md
├─ data_list.csv
├─ 개요_원본.txt
├─ 프로젝트연속성브리프.txt
├─ rfp_rag_eda.ipynb
├─ build_eda_notebook.py
└─ files/
   └─ files/
      ├─ *.hwp
      ├─ *.pdf
      └─ *.docx
```

## 데이터 구성
- `data_list.csv`
  - 공고 번호, 공고 차수, 사업명, 사업 금액, 발주 기관, 공개 일자
  - 입찰 참여 시작일, 입찰 참여 마감일, 사업 요약, 파일형식, 파일명, 텍스트
- `files/files/`
  - 실제 RFP 원본 문서 파일 모음
  - HWP, PDF, 일부 DOCX 파일 포함
- `개요_원본.txt`
  - 프로젝트 과제 설명 및 운영 가이드 원본
- `프로젝트연속성브리프.txt`
  - 작업 방식, 이전 미션 교훈, 이번 프로젝트 진행 전략 정리
- `rfp_rag_eda.ipynb`
  - 메타데이터/텍스트/키워드/클러스터링 기반 EDA 노트북

## 설치 및 실행 방법
현재는 데이터 탐색 및 설계 단계이며, 애플리케이션 실행용 코드와 패키지 설정 파일은 추후 추가 예정입니다.

```bash
# 1. 리포지토리 클론
git clone [리포지토리 URL]
cd rfp-rag-system

# 2. 가상환경 생성 및 활성화
python -m venv venv
# Windows
venv\\Scripts\\activate

# 3. 패키지 설치
# requirements.txt 추가 후 아래 명령 실행 예정
# pip install -r requirements.txt

# 4. 환경 변수 설정
# .env.example 추가 후 .env 파일 생성 예정
# OpenAI API Key 등 환경 변수 설정 예정
```

## 현재 진행 상태
- 원본 데이터 및 메타데이터 확보 완료
- 프로젝트 가이드 및 브리프 정리 완료
- EDA 노트북 초안 작성 완료
- Retrieval / Generation / 평가 파이프라인은 이후 단계에서 구현 예정

## 향후 계획
- 문서 파싱 및 청킹 전략 수립
- 임베딩 및 Vector DB 구축
- 메타데이터 필터링 기반 Retrieval 설계
- 답변 생성 프롬프트 및 모델 비교
- 평가셋 구축 및 성능 검증

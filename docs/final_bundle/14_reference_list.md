# 14. 참고 자료 및 사용 기술 목록

## 참조한 원본 파일 목록

| 구분 | 원본 경로 |
|---|---|
| Scenario A 설정 | `C:\Users\UserK\Downloads\중급프로젝트\config\scenario_a_models.yaml` |
| Requirements | `C:\Users\UserK\Downloads\중급프로젝트\requirements_rag.txt` |
| Phase2 코드 | `C:\Users\UserK\Downloads\중급프로젝트\src\scenario_b_phase2` |
| Scenario A 코드 | `C:\Users\UserK\Downloads\중급프로젝트\src\scenario_a` |
| Streamlit 코드 | `C:\Users\UserK\Downloads\중급프로젝트\src\streamlit_qa` |
| 팀 평가 문서 | `C:\Users\UserK\Downloads\github\docs\하은,윤민\최종보고서_평가파이프라인_강하은&박윤민.md` |
| 팀 OCR 문서 | `C:\Users\UserK\Downloads\github\docs\성현\table_pipeline_final_report.md` |
| 팀 retrieval 문서 | `C:\Users\UserK\Downloads\github\docs\건호\part04_report.md` |

## 사용 모델

| 모델/계열 | 사용 위치 | 목적 |
|---|---|---|
| `gpt-5-mini` | Stage1/Phase2 운영 generator | 운영 기준선 |
| `gpt-5` | B-06 ceiling, LLM Judge | 최고 성능 확인 및 judge |
| `text-embedding-3-small` | 초기 embedding/Chroma 적재 | OpenAI embedding baseline |
| `bge-m3` | Scenario A embedding | Qwen/Gemma local stack 평가 |
| `KoE5` 계열 | Scenario A embedding screening | 한국어 embedding 후보 비교 |
| `Qwen/Qwen2.5-7B-Instruct` | Scenario A / FT | 로컬 generator 및 LoRA FT |
| `Gemma4-E4B` | Scenario A | 로컬 generator 비교 |
| `EXAONE` 계열 | Scenario A 후보 | 한국어 특화 모델 후보, 환경 제약으로 제한적 검토 |

## 사용 라이브러리/도구

| 도구 | 사용 목적 |
|---|---|
| Python | 전체 RAG/평가/FT pipeline 구현 |
| ChromaDB | vector store |
| BM25 / rank_bm25 | sparse retrieval |
| OpenAI API | embedding, generation, judge |
| sentence-transformers | HF embedding |
| transformers | Qwen/Gemma 로딩 및 추론 |
| PEFT/LoRA | Qwen SFT |
| bitsandbytes | 양자화/저메모리 학습 |
| pandas | 결과 집계 및 compare csv |
| Streamlit | 웹 데모 |
| pdfplumber / PyMuPDF | PDF 파싱/렌더링 보조 |
| hwp5txt / olefile | HWP 텍스트 추출 |
| win32com / HWP COM | HWP 렌더링 및 PDF 변환 경로 검토 |
| PaddleOCR / PP-Structure 계열 | true table OCR 구조 검토 |

## 참고한 RAG/QA 방법론

| 방법론 | 프로젝트 내 반영 |
|---|---|
| Hybrid Retrieval | Dense + BM25 결합, B-01 및 이후 기본 검색 |
| Contextual Retrieval | chunk prefix에 사업/기관/section metadata 추가, B-02 |
| CRAG | retrieval 결과를 평가하고 보정하는 조건부 rescue 개념, B-03a/Soft CRAG-lite |
| HybridQA | table과 본문을 함께 보는 pairing/stagewise evaluation 아이디어 |
| TAPAS-style schema | header_path, row_id, col_id, value/date 등 table factual evidence 구조 |
| TableFormer-style table AST | OCR table 구조를 retrieval unit으로 쓰는 방향 |
| LLM-as-Judge | Faithfulness/Completeness/Groundedness/Relevancy 평가 |

## 보고서/PPT 출처 표기용 요약

| 분류 | 표기 예시 |
|---|---|
| Vector DB | ChromaDB |
| Sparse retrieval | BM25 / rank_bm25 |
| Embedding | OpenAI `text-embedding-3-small`, BGE-M3, KoE5 |
| Generator | GPT-5 mini, GPT-5, Qwen2.5-7B-Instruct, Gemma4-E4B |
| Fine-tuning | LoRA/PEFT, transformers, bitsandbytes |
| OCR/table | HWP COM rendering, PaddleOCR/PP-Structure, table AST |
| Evaluation | LLM-as-Judge, 45문항 full set, 20문항 judge subset, 340문항 qbank |


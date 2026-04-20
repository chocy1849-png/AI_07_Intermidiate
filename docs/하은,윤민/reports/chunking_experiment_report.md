# Chunking Experiment Report

- 생성 시각: 2026-04-03T15:21:33
- 추천 전략: langchain_recursive / chunk_size=1000 / overlap=200

## 추천 근거

- 1000/200은 평균 청크 수와 문맥 보존 사이 균형이 좋다.
- LangChain RecursiveCharacterTextSplitter를 기준선으로 포함해 비교했다.
- section_aware 전략은 제안요청서의 목차/조항 구조를 살리지만 현재 구현에서는 청크가 과도하게 잘게 나뉜다.
- 500 단위는 검색 정밀도는 높지만 청크 수가 많아져 후속 임베딩 비용이 커진다.

## 결과 표

- langchain_recursive / 500/100: total=17609, avg_doc=176.09, avg_chars=463.78, p95=498.0
- langchain_recursive / 1000/200: total=8800, avg_doc=88, avg_chars=948.12, p95=998.0
- langchain_recursive / 1500/300: total=5882, avg_doc=58.82, avg_chars=1427.31, p95=1498.0
- recursive / 500/100: total=14943, avg_doc=149.43, avg_chars=561.98, p95=600.0
- recursive / 1000/200: total=7346, avg_doc=73.46, avg_chars=1141.1, p95=1200.0
- recursive / 1500/300: total=4902, avg_doc=49.02, avg_chars=1709.74, p95=1800.0
- section_aware / 500/100: total=32561, avg_doc=325.61, avg_chars=280.12, p95=601.0
- section_aware / 1000/200: total=26955, avg_doc=269.55, avg_chars=351.89, p95=1201.0
- section_aware / 1500/300: total=25193, avg_doc=251.93, avg_chars=386.45, p95=1800.0
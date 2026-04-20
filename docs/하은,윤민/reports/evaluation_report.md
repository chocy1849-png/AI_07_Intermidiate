# Evaluation Report

현재 평가 산출물은 아래 파일로 분리되어 있다.

- `parser_comparison_report.md`
- `parsing_quality_report.md`
- `chunking_experiment_report.md`

## 현재 결론

- 파싱: 100개 문서가 JSON으로 저장되었고, HWP/PDF 모두 실사용 가능한 수준의 텍스트가 확보되었다.
- 품질: 일부 구조 손실 위험은 있으나, 검색과 요약 실험을 시작할 만큼의 품질은 확보되었다.
- 청킹: `langchain_recursive + chunk_size=1000 + overlap=200` 조합을 1차 기준선으로 사용하고, `section_aware`는 심화 실험 대상으로 유지한다.

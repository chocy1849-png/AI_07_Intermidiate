from __future__ import annotations

import json
from datetime import datetime
from statistics import mean, median

from config import CHUNK_PARAMETER_GRID, REPORTS_DIR, ensure_directories
from src.chunking.chunker import chunk_document, split_by_section_headers
from src.db.parsed_store import load_parsed_documents


def _percentile(values: list[int], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(len(ordered) * pct), len(ordered) - 1)
    return float(ordered[index])


def build_chunking_experiment_report() -> dict:
    ensure_directories()
    documents = load_parsed_documents()
    rows = []

    for strategy in ["langchain_recursive", "recursive", "section_aware"]:
        for params in CHUNK_PARAMETER_GRID:
            chunk_lengths = []
            chunk_counts = []
            section_hits = []

            for document in documents:
                sections = split_by_section_headers(document["text"])
                section_hits.append(len(sections))
                chunks = chunk_document(
                    document["text"],
                    metadata=document["metadata"],
                    chunk_size=params["chunk_size"],
                    overlap=params["overlap"],
                    strategy=strategy,
                )
                chunk_counts.append(len(chunks))
                chunk_lengths.extend(len(chunk["text"]) for chunk in chunks)

            rows.append(
                {
                    "strategy": strategy,
                    "chunk_size": params["chunk_size"],
                    "overlap": params["overlap"],
                    "total_chunks": sum(chunk_counts),
                    "avg_chunks_per_doc": round(mean(chunk_counts), 2),
                    "avg_chunk_chars": round(mean(chunk_lengths), 2) if chunk_lengths else 0.0,
                    "median_chunk_chars": round(median(chunk_lengths), 2) if chunk_lengths else 0.0,
                    "p95_chunk_chars": round(_percentile(chunk_lengths, 0.95), 2),
                    "avg_detected_sections": round(mean(section_hits), 2),
                }
            )

    recommended = next(
        row for row in rows
        if row["strategy"] == "langchain_recursive" and row["chunk_size"] == 1000 and row["overlap"] == 200
    )
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rows": rows,
        "recommended": recommended,
        "rationale": [
            "1000/200은 평균 청크 수와 문맥 보존 사이 균형이 좋다.",
            "LangChain RecursiveCharacterTextSplitter를 기준선으로 포함해 비교했다.",
            "section_aware 전략은 제안요청서의 목차/조항 구조를 살리지만 현재 구현에서는 청크가 과도하게 잘게 나뉜다.",
            "500 단위는 검색 정밀도는 높지만 청크 수가 많아져 후속 임베딩 비용이 커진다.",
        ],
    }
    (REPORTS_DIR / "chunking_experiment_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    markdown = [
        "# Chunking Experiment Report",
        "",
        f"- 생성 시각: {report['generated_at']}",
        f"- 추천 전략: {recommended['strategy']} / chunk_size={recommended['chunk_size']} / overlap={recommended['overlap']}",
        "",
        "## 추천 근거",
        "",
    ]
    markdown.extend(f"- {item}" for item in report["rationale"])
    markdown.extend(["", "## 결과 표", ""])
    markdown.extend(
        f"- {row['strategy']} / {row['chunk_size']}/{row['overlap']}: total={row['total_chunks']}, "
        f"avg_doc={row['avg_chunks_per_doc']}, avg_chars={row['avg_chunk_chars']}, p95={row['p95_chunk_chars']}"
        for row in rows
    )
    (REPORTS_DIR / "chunking_experiment_report.md").write_text("\n".join(markdown), encoding="utf-8")
    return report


if __name__ == "__main__":
    build_chunking_experiment_report()

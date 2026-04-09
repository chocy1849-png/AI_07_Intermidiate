import argparse
from pathlib import Path
from time import perf_counter

from rag_system.qa import answer_query


TEST_QUERIES = [
    "사업의 주요 요구사항을 요약해줘",
    "발주기관과 제출 방식은 무엇인지 알려줘",
    "예산이나 사업 목적이 문서에 있으면 정리해줘",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation queries against the RAG system.")
    parser.add_argument(
        "--retrieval-mode",
        choices=["baseline", "mmr"],
        default="baseline",
        help="Retrieval strategy to evaluate",
    )
    args = parser.parse_args()

    retrieval_mode = args.retrieval_mode
    output_path = Path(f"llm_answer_{retrieval_mode}.txt")
    results = []
    started_at = perf_counter()

    for query in TEST_QUERIES:
        query_started_at = perf_counter()
        try:
            result = answer_query(query, retrieval_mode=retrieval_mode)
            titles = sorted({doc.metadata.get("title", "") for doc in result["documents"]})
            results.append(
                {
                    "query": query,
                    "answer": result["answer"],
                    "titles": titles,
                    "docs_count": len(result["documents"]),
                    "elapsed_seconds": perf_counter() - query_started_at,
                    "status": "success",
                }
            )
        except Exception as exc:
            results.append(
                {
                    "query": query,
                    "answer": f"ERROR: {exc}",
                    "titles": [],
                    "docs_count": 0,
                    "elapsed_seconds": perf_counter() - query_started_at,
                    "status": "failed",
                }
            )

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("RAG evaluation snapshot\n")
        handle.write(f"Retrieval mode: {retrieval_mode}\n")
        handle.write("=" * 60 + "\n\n")
        for index, item in enumerate(results, start=1):
            handle.write(f"[Test {index}] {item['query']}\n")
            handle.write(f"Status: {item['status']}\n")
            handle.write(f"Elapsed: {item['elapsed_seconds']:.2f}s\n")
            handle.write(f"Retrieved chunks: {item['docs_count']}\n")
            handle.write(f"Projects: {', '.join(item['titles']) or 'N/A'}\n")
            handle.write(f"Answer:\n{item['answer']}\n")
            handle.write("\n" + "-" * 60 + "\n\n")

        total_elapsed = perf_counter() - started_at
        success_count = sum(1 for item in results if item["status"] == "success")
        handle.write(f"Successful tests: {success_count}/{len(results)}\n")
        handle.write(f"Total elapsed: {total_elapsed:.2f}s\n")

    print(f"Saved evaluation results to {output_path.resolve()}.")


if __name__ == "__main__":
    main()

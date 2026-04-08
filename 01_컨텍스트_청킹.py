from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import Any

from rag_공통 import (
    CHUNK_JSONL_PATH,
    CHUNK_SUMMARY_CSV_PATH,
    INPUT_JSONL_PATH,
    csv_저장,
    jsonl_불러오기,
    jsonl_저장,
    메타데이터_접두어,
    문서_본문_선택,
    청크_ID_생성,
    텍스트_청킹,
)


def 청크_레코드_생성(
    document: dict[str, Any],
    chunk_texts: list[str],
) -> list[dict[str, Any]]:
    metadata = document.get("metadata", {}) or {}
    prefix = 메타데이터_접두어(document)
    rows: list[dict[str, Any]] = []

    for index, chunk_text in enumerate(chunk_texts, start=1):
        contextual_chunk = (
            f"{prefix}\n\n"
            f"[본문 청크]\n{chunk_text}\n[/본문 청크]"
        )
        rows.append(
            {
                "chunk_id": 청크_ID_생성(document.get("document_id", "문서"), index),
                "document_id": document.get("document_id"),
                "chunk_index": index,
                "source_file_name": document.get("source_file_name"),
                "source_path": document.get("source_path"),
                "source_extension": document.get("source_extension"),
                "사업명": metadata.get("사업명"),
                "발주 기관": metadata.get("발주 기관"),
                "공고 번호": metadata.get("공고 번호"),
                "공개 일자": metadata.get("공개 일자"),
                "파일형식": metadata.get("파일형식", document.get("source_extension")),
                "metadata_prefix": prefix,
                "raw_chunk_text": chunk_text,
                "contextual_chunk_text": contextual_chunk,
                "raw_chunk_chars": len(chunk_text),
                "contextual_chunk_chars": len(contextual_chunk),
            }
        )

    return rows


def 요약행_생성(chunk_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in chunk_rows:
        grouped.setdefault(str(row["document_id"]), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for document_id, rows in grouped.items():
        raw_lengths = [int(row["raw_chunk_chars"]) for row in rows]
        contextual_lengths = [int(row["contextual_chunk_chars"]) for row in rows]
        first = rows[0]
        summary_rows.append(
            {
                "document_id": document_id,
                "source_file_name": first["source_file_name"],
                "사업명": first["사업명"],
                "발주 기관": first["발주 기관"],
                "chunk_count": len(rows),
                "raw_chunk_chars_avg": round(mean(raw_lengths), 2),
                "raw_chunk_chars_max": max(raw_lengths),
                "contextual_chunk_chars_avg": round(mean(contextual_lengths), 2),
                "contextual_chunk_chars_max": max(contextual_lengths),
            }
        )
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="정부 제안서 문서를 Contextual Retrieval용 청크로 분할합니다."
    )
    parser.add_argument(
        "--입력경로",
        default=str(INPUT_JSONL_PATH),
        help="전처리된 문서 JSONL 경로",
    )
    parser.add_argument(
        "--출력경로",
        default=str(CHUNK_JSONL_PATH),
        help="컨텍스트 청크 JSONL 출력 경로",
    )
    parser.add_argument(
        "--요약CSV경로",
        default=str(CHUNK_SUMMARY_CSV_PATH),
        help="문서별 청크 요약 CSV 출력 경로",
    )
    parser.add_argument(
        "--청크크기",
        type=int,
        default=500,
        help="본문 기준 청크 최대 문자 수",
    )
    parser.add_argument(
        "--겹침크기",
        type=int,
        default=80,
        help="청크 간 겹침 문자 수",
    )
    parser.add_argument(
        "--최대문서수",
        type=int,
        default=None,
        help="테스트용 문서 개수 제한",
    )
    args = parser.parse_args()

    input_path = Path(args.입력경로)
    output_path = Path(args.출력경로)
    summary_csv_path = Path(args.요약CSV경로)

    documents = jsonl_불러오기(path=input_path)
    if args.최대문서수 is not None:
        documents = documents[: args.최대문서수]

    all_chunk_rows: list[dict[str, Any]] = []
    skipped_documents = 0

    for document in documents:
        document_text = 문서_본문_선택(document)
        if not document_text:
            skipped_documents += 1
            continue

        chunk_texts = 텍스트_청킹(
            text=document_text,
            chunk_size=args.청크크기,
            overlap_size=args.겹침크기,
        )
        all_chunk_rows.extend(청크_레코드_생성(document, chunk_texts))

    if not all_chunk_rows:
        raise RuntimeError("생성된 청크가 없습니다. 입력 파일과 전처리 결과를 확인하세요.")

    jsonl_저장(path=output_path, rows=all_chunk_rows)
    summary_rows = 요약행_생성(all_chunk_rows)
    csv_저장(path=summary_csv_path, rows=summary_rows)

    print("[완료] 컨텍스트 청킹이 끝났습니다.")
    print(f"- 입력 문서 수: {len(documents)}")
    print(f"- 건너뛴 문서 수: {skipped_documents}")
    print(f"- 생성 청크 수: {len(all_chunk_rows)}")
    print(f"- 청크 저장 경로: {output_path}")
    print(f"- 요약 CSV 경로: {summary_csv_path}")


if __name__ == "__main__":
    main()

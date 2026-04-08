from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

from rag_공통 import INPUT_JSONL_PATH, OUTPUT_DIR


BASE_DIR = Path(__file__).resolve().parent
A04_DIR = OUTPUT_DIR / "b04_candidates"
A04_MANIFEST_PATH = A04_DIR / "b04_candidate_manifest.json"
SMOKE_IDS_PATH = BASE_DIR / "evaluation" / "day4_smoke_eval_question_ids_v1.txt"


CANDIDATES: list[dict[str, Any]] = [
    {"key": "c500_80", "label": "500/80", "chunk_size": 500, "overlap": 80, "mode": "standard"},
    {"key": "c1000_120", "label": "1000/120", "chunk_size": 1000, "overlap": 120, "mode": "standard"},
    {"key": "c1200_160", "label": "1200/160", "chunk_size": 1200, "overlap": 160, "mode": "standard"},
    {"key": "c1300_180", "label": "1300/180", "chunk_size": 1300, "overlap": 180, "mode": "standard"},
    {"key": "c1200_160_struct", "label": "1200/160 struct", "chunk_size": 1200, "overlap": 160, "mode": "struct"},
]


SMOKE_QUESTION_IDS = [
    "Q02",
    "Q06",
    "Q10",
    "Q13",
    "Q19",
    "Q29",
    "Q32",
    "Q36",
]


def load_b02_module():
    module_path = BASE_DIR / "10_B02_prefix_v2_청킹.py"
    spec = importlib.util.spec_from_file_location("b02_prefix_v2_chunking", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"B-02 chunking module load failed: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def split_structured_text(text: str, chunk_size: int, overlap_size: int, b02) -> list[str]:
    cleaned = b02.clean_text(text)
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    lines = cleaned.splitlines()
    blocks: list[str] = []
    current: list[str] = []
    current_mode = "text"

    def flush() -> None:
        nonlocal current, current_mode
        block = "\n".join(current).strip()
        if block:
            blocks.append(block)
        current = []
        current_mode = "text"

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()
        is_table_line = stripped.startswith("[TABLE BLOCK]") or stripped.startswith("<표>") or stripped.startswith("|")
        is_list_line = bool(stripped) and (stripped.startswith("- ") or stripped.startswith("•") or stripped.startswith("ㅇ "))
        next_mode = "table" if is_table_line else "list" if is_list_line else "text"

        if not stripped:
            flush()
            continue

        if current and next_mode != current_mode and not (current_mode == "table" and next_mode == "table"):
            flush()

        current.append(line)
        current_mode = next_mode

    flush()

    normalized_blocks: list[str] = []
    for block in blocks:
        if len(block) > int(chunk_size * 1.3):
            normalized_blocks.extend(b02.split_text(block, chunk_size=chunk_size, overlap_size=overlap_size))
        else:
            normalized_blocks.append(block)

    chunks: list[str] = []
    current_blocks: list[str] = []
    current_len = 0

    for block in normalized_blocks:
        block_len = len(block) + (2 if current_blocks else 0)
        if current_blocks and current_len + block_len > chunk_size:
            chunks.append("\n\n".join(current_blocks).strip())
            overlap_blocks: list[str] = []
            overlap_len = 0
            for prior_block in reversed(current_blocks):
                if overlap_len + len(prior_block) > overlap_size and overlap_blocks:
                    break
                overlap_blocks.insert(0, prior_block)
                overlap_len += len(prior_block)
            current_blocks = overlap_blocks + [block]
            current_len = sum(len(item) for item in current_blocks) + max(0, len(current_blocks) - 1) * 2
        else:
            current_blocks.append(block)
            current_len += block_len

    if current_blocks:
        chunks.append("\n\n".join(current_blocks).strip())

    deduped: list[str] = []
    for chunk in chunks:
        value = b02.clean_text(chunk)
        if value and (not deduped or value != deduped[-1]):
            deduped.append(value)
    return deduped


def build_chunk_texts(text: str, candidate: dict[str, Any], b02) -> list[str]:
    if candidate["mode"] == "struct":
        return split_structured_text(text, candidate["chunk_size"], candidate["overlap"], b02)
    return b02.split_text(text, chunk_size=candidate["chunk_size"], overlap_size=candidate["overlap"])


def candidate_paths(candidate_key: str) -> dict[str, Path]:
    candidate_dir = A04_DIR / candidate_key
    return {
        "dir": candidate_dir,
        "chunks_jsonl": candidate_dir / f"{candidate_key}_chunks.jsonl",
        "summary_csv": candidate_dir / f"{candidate_key}_chunk_summary.csv",
        "document_fields_csv": candidate_dir / f"{candidate_key}_document_fields.csv",
        "manifest_json": candidate_dir / f"{candidate_key}_manifest.json",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="A-04 chunk candidate generation")
    parser.add_argument("--input-path", default=str(INPUT_JSONL_PATH))
    parser.add_argument("--max-docs", type=int, default=0)
    args = parser.parse_args()

    b02 = load_b02_module()
    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"input jsonl not found: {input_path}")

    documents = b02.read_jsonl(input_path)
    if args.max_docs > 0:
        documents = documents[: args.max_docs]
    if not documents:
        raise RuntimeError("no documents found")

    A04_DIR.mkdir(parents=True, exist_ok=True)
    SMOKE_IDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SMOKE_IDS_PATH.write_text("\n".join(SMOKE_QUESTION_IDS) + "\n", encoding="utf-8")

    candidate_manifest_rows: list[dict[str, Any]] = []

    for candidate in CANDIDATES:
        paths = candidate_paths(candidate["key"])
        chunk_rows: list[dict[str, Any]] = []
        document_field_rows: list[dict[str, Any]] = []
        skipped_documents = 0

        for document in documents:
            document_text = b02.choose_document_text(document)
            if not document_text:
                skipped_documents += 1
                continue
            document_fields = b02.build_document_fields(document, document_text)
            document_field_rows.append(document_fields)
            chunk_texts = build_chunk_texts(document_text, candidate, b02)
            chunk_rows.extend(b02.build_chunk_rows(document, chunk_texts, document_fields))

        if not chunk_rows:
            raise RuntimeError(f"no chunk rows built for candidate: {candidate['key']}")

        write_jsonl(paths["chunks_jsonl"], chunk_rows)
        write_csv(paths["summary_csv"], b02.build_summary_rows(chunk_rows))
        write_csv(paths["document_fields_csv"], document_field_rows)

        manifest = {
            "candidate_key": candidate["key"],
            "candidate_label": candidate["label"],
            "mode": candidate["mode"],
            "chunk_size": candidate["chunk_size"],
            "overlap": candidate["overlap"],
            "collection_name": f"rfp_contextual_chunks_b04_{candidate['key']}",
            "bm25_index_path": str(paths["dir"] / f"{candidate['key']}_bm25.pkl"),
            "chunk_jsonl_path": str(paths["chunks_jsonl"]),
            "summary_csv_path": str(paths["summary_csv"]),
            "document_fields_csv_path": str(paths["document_fields_csv"]),
            "document_count": len(documents),
            "skipped_documents": skipped_documents,
            "chunk_count": len(chunk_rows),
            "prefix_version": "v2_structured_fields",
        }
        paths["manifest_json"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        candidate_manifest_rows.append(manifest)

        print(
            f"[완료] {candidate['key']} | chunk_size={candidate['chunk_size']} "
            f"| overlap={candidate['overlap']} | mode={candidate['mode']} | chunk_count={len(chunk_rows)}"
        )

    A04_MANIFEST_PATH.write_text(json.dumps(candidate_manifest_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[완료] A-04 후보 청킹 생성 완료: {A04_MANIFEST_PATH}")


if __name__ == "__main__":
    main()

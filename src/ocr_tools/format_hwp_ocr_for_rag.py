import argparse
import json
from pathlib import Path


def build_table_chunk(table: dict, source_name: str) -> dict:
    parent_prefix = ""
    if table.get("linked_parent_text"):
        parent_prefix = f"[SECTION_HEADER] {table['linked_parent_text']}\n"

    # Until rendered HWP table crops are available, the most faithful table
    # content we have is the extracted structural text itself.
    body = table.get("ocr_text") or table.get("source_text") or ""
    content = f"[SOURCE] {source_name}\n[TABLE_INDEX] {table['table_index']}\n{parent_prefix}[STRUCTURAL_TABLE]\n{body}".strip()

    return {
        "chunk_type": "structural_table",
        "source_name": source_name,
        "table_index": table["table_index"],
        "linked_parent_table_index": table.get("linked_parent_table_index"),
        "linked_parent_text": table.get("linked_parent_text"),
        "ocr_status": table.get("status"),
        "ocr_candidate_score": table.get("ocr_candidate_score"),
        "ocr_candidate_priority": table.get("ocr_candidate_priority"),
        "content": content,
    }


def build_image_chunk(image: dict, source_name: str) -> dict | None:
    ocr_text = (image.get("ocr_text") or "").strip()
    if not ocr_text:
        return None

    content = (
        f"[SOURCE] {source_name}\n"
        f"[IMAGE_NAME] {image['name']}\n"
        "[IMAGE_OCR]\n"
        f"{ocr_text}"
    ).strip()

    return {
        "chunk_type": "image_ocr",
        "source_name": source_name,
        "image_name": image["name"],
        "saved_path": image.get("saved_path"),
        "ocr_status": image.get("status"),
        "ocr_candidate_score": image.get("ocr_candidate_score"),
        "ocr_candidate_priority": image.get("ocr_candidate_priority"),
        "content": content,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Format HWP OCR results into RAG-ready chunks.")
    parser.add_argument("--input-json", type=Path, required=True, help="OCR results JSON path.")
    parser.add_argument("--output-json", type=Path, required=True, help="Output RAG-ready JSON path.")
    args = parser.parse_args()

    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    source_name = payload["source_name"]

    chunks = []
    for table in payload.get("table_results", []):
        chunks.append(build_table_chunk(table, source_name))

    for image in payload.get("image_results", []):
        image_chunk = build_image_chunk(image, source_name)
        if image_chunk:
            chunks.append(image_chunk)

    output = {
        "source_name": source_name,
        "source_path": payload["source_path"],
        "chunk_count": len(chunks),
        "chunks": chunks,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"source: {source_name}")
    print(f"chunk_count: {len(chunks)}")
    print(f"table_chunks: {len(payload.get('table_results', []))}")
    print(f"image_chunks_with_text: {sum(1 for image in payload.get('image_results', []) if (image.get('ocr_text') or '').strip())}")


if __name__ == "__main__":
    main()

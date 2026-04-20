import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ocr.extract_hwp_artifacts import extract_hwp_artifacts


def build_ocr_payload(file_path: Path, save_images: bool = False, output_dir: Path | None = None) -> dict:
    payload = extract_hwp_artifacts(file_path, save_images=save_images, output_dir=output_dir)

    table_candidates = []
    for table in payload["table_ocr_candidates"]:
        table_candidates.append(
            {
                "table_index": table["table_index"],
                "section": table["section"],
                "record_start_index": table["record_start_index"],
                "linked_parent_table_index": table["linked_parent_table_index"],
                "linked_parent_text": table["linked_parent_text"],
                "ocr_candidate_score": table["ocr_candidate_score"],
                "ocr_candidate_priority": table["ocr_candidate_priority"],
                "ocr_candidate_reasons": table["ocr_candidate_reasons"],
                "text": table["text"],
            }
        )

    image_candidates = []
    for image in payload["image_ocr_candidates"]:
        image_candidates.append(
            {
                "name": image["name"],
                "ext": image["ext"],
                "size_bytes": image["size_bytes"],
                "saved_path": image["saved_path"],
                "ocr_candidate_score": image["ocr_candidate_score"],
                "ocr_candidate_priority": image["ocr_candidate_priority"],
                "ocr_candidate_reasons": image["ocr_candidate_reasons"],
            }
        )

    return {
        "source_path": payload["source_path"],
        "source_name": payload["source_name"],
        "structural_table_count": payload["structural_table_count"],
        "section_header_block_count": payload["section_header_block_count"],
        "explanatory_block_count": payload["explanatory_block_count"],
        "table_ocr_candidate_count": len(table_candidates),
        "image_ocr_candidate_count": len(image_candidates),
        "table_ocr_candidates": table_candidates,
        "image_ocr_candidates": image_candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an OCR-ready payload from extracted HWP artifacts.")
    parser.add_argument("--file", type=Path, required=True, help="HWP file path.")
    parser.add_argument("--json", type=Path, required=True, help="Output JSON path.")
    parser.add_argument("--save-images", action="store_true", help="Save embedded HWP images while building the payload.")
    args = parser.parse_args()

    output_dir = args.json.parent
    payload = build_ocr_payload(args.file, save_images=args.save_images, output_dir=output_dir)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"source: {payload['source_name']}")
    print(f"structural_table_count: {payload['structural_table_count']}")
    print(f"section_header_block_count: {payload['section_header_block_count']}")
    print(f"explanatory_block_count: {payload['explanatory_block_count']}")
    print(f"table_ocr_candidate_count: {payload['table_ocr_candidate_count']}")
    print(f"image_ocr_candidate_count: {payload['image_ocr_candidate_count']}")


if __name__ == "__main__":
    main()

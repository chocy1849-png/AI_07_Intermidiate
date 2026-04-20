import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

WINDOWS_OCR_SCRIPT = Path(__file__).resolve().with_name("windows_ocr_fallback.ps1")


def load_runtime():
    try:
        from PIL import Image  # type: ignore
        from rapidocr_onnxruntime import RapidOCR  # type: ignore

        return Image, RapidOCR
    except Exception:
        return None, None


def find_windows_ocr_candidate_paths(image_path: Path) -> list[Path]:
    candidates = []
    seen = set()
    for path in [image_path, image_path.with_suffix(".jpg"), image_path.with_suffix(".jpeg"), image_path.with_suffix(".png")]:
        resolved = path.resolve()
        key = str(resolved).lower()
        if path.exists() and key not in seen:
            candidates.append(path)
            seen.add(key)
    return candidates


def run_windows_ocr(image_path: Path):
    candidate_paths = find_windows_ocr_candidate_paths(image_path)
    if not candidate_paths:
        return {
            "status": "missing_saved_image",
            "ocr_text": "",
            "ocr_lines": [],
            "reason": "No readable image path exists for Windows OCR fallback",
        }

    last_reason = "Windows OCR fallback did not run"
    for candidate in candidate_paths:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(WINDOWS_OCR_SCRIPT), str(candidate)],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        payload = (completed.stdout or "").strip()
        if not payload:
            last_reason = (completed.stderr or "").strip() or "Windows OCR fallback returned no output"
            continue
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            last_reason = payload
            continue
        if data.get("status") == "ok":
            lines = [line.strip() for line in (data.get("ocr_text") or "").splitlines() if line.strip()]
            return {
                "status": "ok",
                "ocr_text": "\n".join(lines).strip(),
                "ocr_lines": lines,
                "reason": "",
            }
        last_reason = data.get("reason") or last_reason

    return {
        "status": "ocr_failed",
        "ocr_text": "",
        "ocr_lines": [],
        "reason": last_reason,
    }


def run_image_ocr(image_path: Path):
    Image, RapidOCR = load_runtime()
    if Image is None or RapidOCR is None:
        return run_windows_ocr(image_path)

    engine = RapidOCR()
    try:
        result, _ = engine(str(image_path))
    except Exception as exc:
        return {
            "status": "ocr_failed",
            "ocr_text": "",
            "ocr_lines": [],
            "reason": f"{type(exc).__name__}: {exc}",
        }
    if not result:
        return {
            "status": "ok",
            "ocr_text": "",
            "ocr_lines": [],
            "reason": "no text detected",
        }

    lines = []
    for item in result:
        if len(item) >= 2:
            text = item[1]
            if isinstance(text, str) and text.strip():
                lines.append(text.strip())

    return {
        "status": "ok",
        "ocr_text": "\n".join(lines).strip(),
        "ocr_lines": lines,
        "reason": "",
    }


def build_table_result(table: dict) -> dict:
    # We still do not have rendered HWP table crops here, so tables remain
    # OCR-ready rather than OCR-complete. Keeping this explicit makes the
    # pipeline honest and lets us plug in renderer output later without
    # changing downstream RAG formatting.
    return {
        "table_index": table["table_index"],
        "source_type": "structural_table",
        "status": "ready_for_table_ocr",
        "ocr_text": "",
        "source_text": table["text"],
        "linked_parent_text": table.get("linked_parent_text"),
        "ocr_candidate_score": table["ocr_candidate_score"],
        "ocr_candidate_priority": table["ocr_candidate_priority"],
        "ocr_candidate_reasons": table["ocr_candidate_reasons"],
        "reason": "Rendered table image is not available yet, so OCR is deferred",
    }


def build_image_result(image: dict) -> dict:
    saved_path = image.get("saved_path")
    if not saved_path:
        return {
            "name": image["name"],
            "source_type": "embedded_image",
            "status": "missing_saved_image",
            "ocr_text": "",
            "ocr_lines": [],
            "ocr_candidate_score": image["ocr_candidate_score"],
            "ocr_candidate_priority": image["ocr_candidate_priority"],
            "ocr_candidate_reasons": image["ocr_candidate_reasons"],
            "reason": "saved_path is missing, rerun payload build with --save-images",
        }

    ocr_result = run_image_ocr(Path(saved_path))
    return {
        "name": image["name"],
        "source_type": "embedded_image",
        "saved_path": saved_path,
        "status": ocr_result["status"],
        "ocr_text": ocr_result["ocr_text"],
        "ocr_lines": ocr_result["ocr_lines"],
        "ocr_candidate_score": image["ocr_candidate_score"],
        "ocr_candidate_priority": image["ocr_candidate_priority"],
        "ocr_candidate_reasons": image["ocr_candidate_reasons"],
        "reason": ocr_result["reason"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OCR on HWP OCR payload candidates.")
    parser.add_argument("--input-json", type=Path, required=True, help="OCR payload JSON path.")
    parser.add_argument("--output-json", type=Path, required=True, help="Output OCR results JSON path.")
    args = parser.parse_args()

    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    table_results = [build_table_result(table) for table in payload.get("table_ocr_candidates", [])]
    image_results = [build_image_result(image) for image in payload.get("image_ocr_candidates", [])]

    output = {
        "source_path": payload["source_path"],
        "source_name": payload["source_name"],
        "table_ocr_candidate_count": len(table_results),
        "image_ocr_candidate_count": len(image_results),
        "table_results": table_results,
        "image_results": image_results,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"source: {output['source_name']}")
    print(f"table_results: {len(table_results)}")
    print(f"image_results: {len(image_results)}")
    print("note: table OCR remains deferred until rendered table images are available")


if __name__ == "__main__":
    main()

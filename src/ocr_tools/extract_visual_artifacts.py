import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import fitz

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


@dataclass
class TableArtifact:
    page: int
    bbox: list[float]
    bbox_area_ratio: float
    row_count: int
    col_count: int
    text_length: int
    original_text: str
    markdown: str
    text_recovery_percent: float


@dataclass
class ImageArtifact:
    page: int
    bbox: list[float]
    bbox_area_ratio: float
    width: int
    height: int
    ext: str
    xref: int | None
    saved_path: str | None


@dataclass
class PageArtifacts:
    page: int
    page_width: float
    page_height: float
    tables: list[TableArtifact]
    images: list[ImageArtifact]


def rect_area_ratio(bbox: tuple[float, float, float, float] | list[float], page_rect: fitz.Rect) -> float:
    x0, y0, x1, y1 = bbox
    area = max(0.0, (x1 - x0)) * max(0.0, (y1 - y0))
    page_area = max(1.0, page_rect.width * page_rect.height)
    return round(area / page_area, 4)


def normalize_cell(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def compute_text_recovery_percent(markdown: str, table_text: str) -> float:
    original = "".join(table_text.split())
    extracted = "".join(markdown.split())
    if not original:
        return 0.0
    return round(min((len(extracted) / len(original)) * 100, 100.0), 2)


def extract_tables_from_page(page: fitz.Page) -> list[TableArtifact]:
    finder = page.find_tables()
    artifacts: list[TableArtifact] = []

    for table in finder.tables:
        try:
            matrix = table.extract()
        except Exception:
            matrix = []

        normalized_rows = [[normalize_cell(cell) for cell in row] for row in matrix if row]
        row_count = len(normalized_rows)
        col_count = max((len(row) for row in normalized_rows), default=0)

        try:
            markdown = table.to_markdown()
        except Exception:
            markdown = "\n".join(" | ".join(row) for row in normalized_rows)

        table_text = "\n".join(" ".join(row) for row in normalized_rows)
        bbox = [round(value, 2) for value in table.bbox]

        artifacts.append(
            TableArtifact(
                page=page.number + 1,
                bbox=bbox,
                bbox_area_ratio=rect_area_ratio(bbox, page.rect),
                row_count=row_count,
                col_count=col_count,
                text_length=len(table_text),
                original_text=table_text,
                markdown=markdown,
                text_recovery_percent=compute_text_recovery_percent(markdown, table_text),
            )
        )

    return artifacts


def extract_images_from_page(doc: fitz.Document, page: fitz.Page, image_dir: Path | None) -> list[ImageArtifact]:
    page_dict = page.get_text("dict")
    artifacts: list[ImageArtifact] = []
    image_index = 0

    for block in page_dict.get("blocks", []):
        if block.get("type") != 1:
            continue

        bbox = [round(value, 2) for value in block.get("bbox", (0, 0, 0, 0))]
        width = int(block.get("width", 0))
        height = int(block.get("height", 0))
        xref = block.get("xref")
        ext = "png"
        saved_path: str | None = None

        if image_dir is not None:
            try:
                image_bytes = None

                # Prefer xref-backed extraction when available because it usually
                # returns the embedded image object instead of transient mask-like
                # block bytes.
                if xref:
                    image_info = doc.extract_image(xref)
                    ext = image_info.get("ext", ext)
                    image_bytes = image_info["image"]
                elif block.get("image"):
                    # Some PDF image blocks have no xref and expose only raw block
                    # bytes, which can be mask-like or solid black. In that case
                    # render the visible page area instead of trusting the raw bytes.
                    clip = fitz.Rect(block["bbox"])
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip, alpha=False)
                    image_bytes = pix.tobytes("png")
                    ext = "png"

                if image_bytes is None:
                    raise ValueError("No usable image bytes available for this block.")

                image_path = image_dir / f"page_{page.number + 1:03d}_image_{image_index:02d}.{ext}"
                image_path.write_bytes(image_bytes)
                saved_path = str(image_path.resolve())
            except Exception:
                saved_path = None

        artifacts.append(
            ImageArtifact(
                page=page.number + 1,
                bbox=bbox,
                bbox_area_ratio=rect_area_ratio(bbox, page.rect),
                width=width,
                height=height,
                ext=ext,
                xref=int(xref) if isinstance(xref, int) else None,
                saved_path=saved_path,
            )
        )
        image_index += 1

    return artifacts


def filter_tables(tables: list[TableArtifact]) -> list[TableArtifact]:
    filtered: list[TableArtifact] = []
    seen: set[tuple] = set()

    for table in tables:
        if table.row_count < 2:
            continue
        if table.col_count < 2:
            continue
        if table.bbox_area_ratio < 0.01:
            continue
        if table.text_length < 20:
            continue

        dedupe_key = (
            table.page,
            round(table.bbox[0], 0),
            round(table.bbox[1], 0),
            round(table.bbox[2], 0),
            round(table.bbox[3], 0),
            table.row_count,
            table.col_count,
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        filtered.append(table)

    return filtered


def filter_images(images: list[ImageArtifact]) -> list[ImageArtifact]:
    filtered: list[ImageArtifact] = []
    seen: set[tuple] = set()

    for image in images:
        if image.width < 80 or image.height < 80:
            continue
        if image.bbox_area_ratio < 0.01:
            continue

        dedupe_key = (
            image.width,
            image.height,
            round(image.bbox[0], 0),
            round(image.bbox[1], 0),
            round(image.bbox[2], 0),
            round(image.bbox[3], 0),
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        filtered.append(image)

    return filtered


def extract_pdf_artifacts(
    file_path: Path,
    save_images: bool = False,
    output_dir: Path | None = None,
    apply_filters: bool = False,
) -> dict:
    doc = fitz.open(file_path)
    pages: list[PageArtifacts] = []

    image_dir: Path | None = None
    if save_images and output_dir is not None:
        image_dir = output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

    raw_table_count = 0
    raw_image_count = 0
    kept_table_count = 0
    kept_image_count = 0

    for page in doc:
        raw_tables = extract_tables_from_page(page)
        raw_images = extract_images_from_page(doc, page, image_dir=image_dir)
        tables = filter_tables(raw_tables) if apply_filters else raw_tables
        images = filter_images(raw_images) if apply_filters else raw_images
        raw_table_count += len(raw_tables)
        raw_image_count += len(raw_images)
        kept_table_count += len(tables)
        kept_image_count += len(images)
        pages.append(
            PageArtifacts(
                page=page.number + 1,
                page_width=round(page.rect.width, 2),
                page_height=round(page.rect.height, 2),
                tables=tables,
                images=images,
            )
        )

    return {
        "source_path": str(file_path.resolve()),
        "source_name": file_path.name,
        "file_type": file_path.suffix.lower(),
        "status": "ok",
        "filters_applied": apply_filters,
        "page_count": len(pages),
        "raw_table_count": raw_table_count,
        "raw_image_count": raw_image_count,
        "kept_table_count": kept_table_count,
        "kept_image_count": kept_image_count,
        "table_keep_percent": round((kept_table_count / raw_table_count) * 100, 2) if raw_table_count else 0.0,
        "image_keep_percent": round((kept_image_count / raw_image_count) * 100, 2) if raw_image_count else 0.0,
        "avg_table_recovery_percent": round(
            sum(table.text_recovery_percent for page in pages for table in page.tables) /
            max(1, sum(len(page.tables) for page in pages)),
            2,
        ),
        "pages": [asdict(page) for page in pages],
    }


def extract_artifacts(
    file_path: Path,
    save_images: bool = False,
    output_dir: Path | None = None,
    apply_filters: bool = False,
) -> dict:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_artifacts(
            file_path,
            save_images=save_images,
            output_dir=output_dir,
            apply_filters=apply_filters,
        )

    return {
        "source_path": str(file_path.resolve()),
        "source_name": file_path.name,
        "file_type": suffix,
        "status": "unsupported_for_visual_extraction",
        "reason": "Current environment can render and inspect PDF pages with PyMuPDF, but does not have an HWP page renderer installed.",
        "page_count": 0,
        "pages": [],
    }


def summarize_payload(payload: dict) -> str:
    if payload["status"] != "ok":
        return (
            f"source: {payload['source_name']}\n"
            f"status: {payload['status']}\n"
            f"reason: {payload['reason']}"
        )

    page_count = payload["page_count"]
    table_count = sum(len(page["tables"]) for page in payload["pages"])
    image_count = sum(len(page["images"]) for page in payload["pages"])

    lines = [
        f"source: {payload['source_name']}",
        f"pages: {page_count}",
        f"filters applied: {payload['filters_applied']}",
        f"tables: {table_count} kept / {payload['raw_table_count']} raw ({payload['table_keep_percent']}%)",
        f"images: {image_count} kept / {payload['raw_image_count']} raw ({payload['image_keep_percent']}%)",
        f"avg table recovery: {payload['avg_table_recovery_percent']}%",
        "",
    ]

    table_examples = []
    image_examples = []
    for page in payload["pages"]:
        for table in page["tables"]:
            table_examples.append(table)
        for image in page["images"]:
            image_examples.append(image)

    for index, table in enumerate(table_examples[:3], start=1):
        lines.append(
            f"table {index}: page={table['page']} rows={table['row_count']} cols={table['col_count']} "
            f"recovery={table['text_recovery_percent']} area={table['bbox_area_ratio']} bbox={table['bbox']}"
        )

    for index, image in enumerate(image_examples[:3], start=1):
        lines.append(
            f"image {index}: page={image['page']} size={image['width']}x{image['height']} "
            f"area={image['bbox_area_ratio']} bbox={image['bbox']} saved={image['saved_path']}"
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract table and image artifacts from a document.")
    parser.add_argument("--file", type=Path, required=True, help="PDF or HWP file path.")
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--save-images", action="store_true", help="Save extracted PDF image blocks to disk.")
    parser.add_argument(
        "--apply-filters",
        action="store_true",
        help="Apply size/shape post-filters. By default this script now returns raw table/image extraction results.",
    )
    args = parser.parse_args()

    output_dir = args.json.parent if args.json is not None else Path("ocr/output")
    payload = extract_artifacts(
        args.file,
        save_images=args.save_images,
        output_dir=output_dir,
        apply_filters=args.apply_filters,
    )

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(summarize_payload(payload))


if __name__ == "__main__":
    main()

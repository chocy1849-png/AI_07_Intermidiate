from __future__ import annotations

from pathlib import Path

try:
    import fitz
except ImportError:  # pragma: no cover - dependency handled in runtime
    fitz = None

try:
    import pdfplumber
except ImportError:  # pragma: no cover - dependency handled in runtime
    pdfplumber = None


class PDFParseError(RuntimeError):
    """Raised when a PDF document cannot be parsed."""


def _parse_with_pymupdf(path: Path) -> tuple[str, int]:
    if fitz is None:
        raise PDFParseError("PyMuPDF is not installed.")

    chunks: list[str] = []
    with fitz.open(path) as document:
        for page in document:
            text = page.get_text("text")
            if text.strip():
                chunks.append(text.strip())
        return "\n\n".join(chunks).strip(), document.page_count


def _parse_with_pdfplumber(path: Path) -> tuple[str, int]:
    if pdfplumber is None:
        raise PDFParseError("pdfplumber is not installed.")

    chunks: list[str] = []
    with pdfplumber.open(path) as document:
        for page in document.pages:
            text = page.extract_text() or ""
            if text.strip():
                chunks.append(text.strip())
        return "\n\n".join(chunks).strip(), len(document.pages)


def parse_pdf(file_path: str | Path) -> dict:
    path = Path(file_path)
    parser_name = "pymupdf"

    try:
        text, page_count = _parse_with_pymupdf(path)
    except Exception:
        parser_name = "pdfplumber"
        text, page_count = _parse_with_pdfplumber(path)

    if not text:
        raise PDFParseError(f"No readable text extracted from {path.name}")

    return {
        "text": text,
        "metadata": {
            "source_path": str(path.resolve()),
            "file_name": path.name,
            "file_format": "pdf",
            "parser": parser_name,
            "page_count": page_count,
            "char_count": len(text),
        },
    }


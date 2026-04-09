from __future__ import annotations

from pathlib import Path

from .docx_parser import parse_docx
from .hwp_parser import parse_hwp
from .pdf_parser import parse_pdf


PARSERS = {
    ".hwp": parse_hwp,
    ".pdf": parse_pdf,
    ".docx": parse_docx,
}


def parse_document(file_path: str | Path) -> dict:
    path = Path(file_path)
    suffix = path.suffix.lower()
    parser = PARSERS.get(suffix)
    if parser is None:
        raise ValueError(f"Unsupported file format: {suffix or 'unknown'}")
    return parser(path)

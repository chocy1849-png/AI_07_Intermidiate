from __future__ import annotations

import zlib
from pathlib import Path
import re

try:
    import olefile
except ImportError:  # pragma: no cover - dependency handled in runtime
    olefile = None


PARA_TEXT_TAG = 67
READABLE_LINE_PATTERN = re.compile(r"[가-힣A-Za-z0-9\s\.,;:()\[\]{}\-/~\"'“”‘’·&<>%=+*#?!ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ○□※]")


class HWPParseError(RuntimeError):
    """Raised when a HWP document cannot be parsed."""


def _is_compressed(ole: "olefile.OleFileIO") -> bool:
    header = ole.openstream("FileHeader").read()
    if len(header) < 40:
        return False
    flags = int.from_bytes(header[36:40], "little")
    return bool(flags & 0x01)


def _inflate_if_needed(payload: bytes, compressed: bool) -> bytes:
    if not compressed:
        return payload
    return zlib.decompress(payload, -15)


def _iter_body_streams(ole: "olefile.OleFileIO") -> list[str]:
    section_names: list[str] = []
    for entry in ole.listdir(streams=True, storages=False):
        if entry and entry[0] == "BodyText" and len(entry) == 2 and entry[1].startswith("Section"):
            section_names.append("/".join(entry))
    return sorted(section_names)


def _extract_text_from_section(section_bytes: bytes) -> str:
    cursor = 0
    parts: list[str] = []

    while cursor + 4 <= len(section_bytes):
        header = int.from_bytes(section_bytes[cursor : cursor + 4], "little")
        cursor += 4

        tag_id = header & 0x3FF
        size = (header >> 20) & 0xFFF
        if size == 0xFFF:
            if cursor + 4 > len(section_bytes):
                break
            size = int.from_bytes(section_bytes[cursor : cursor + 4], "little")
            cursor += 4

        payload = section_bytes[cursor : cursor + size]
        cursor += size

        if tag_id == PARA_TEXT_TAG and payload:
            decoded = payload.decode("utf-16le", errors="ignore").replace("\x00", "")
            if decoded.strip():
                parts.append(decoded.strip())

    return "\n".join(parts)


def _clean_text(text: str) -> str:
    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = "".join(
            char for char in raw_line if char in {"\n", "\t"} or ord(char) >= 32
        ).strip()
        line = "".join(
            char for char in line if char == "\t" or READABLE_LINE_PATTERN.fullmatch(char)
        ).strip()
        if not line:
            continue

        has_core_text = any(
            ("가" <= char <= "힣") or char.isascii() and char.isalnum()
            for char in line
        )
        if len(line) <= 3 and not has_core_text:
            continue

        readable_chars = len(READABLE_LINE_PATTERN.findall(line))
        readability = readable_chars / max(len(line), 1)
        if len(line) >= 6 and readability < 0.45:
            continue
        cleaned_lines.append(line)

    return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned_lines)).strip()


def parse_hwp(file_path: str | Path) -> dict:
    if olefile is None:
        raise HWPParseError("olefile is not installed. Run `pip install -r requirements.txt` first.")

    path = Path(file_path)
    with olefile.OleFileIO(str(path)) as ole:
        compressed = _is_compressed(ole)
        section_texts: list[str] = []

        for section_name in _iter_body_streams(ole):
            stream = ole.openstream(section_name)
            raw_bytes = stream.read()
            section_bytes = _inflate_if_needed(raw_bytes, compressed)
            section_text = _extract_text_from_section(section_bytes)
            if section_text.strip():
                section_texts.append(section_text)

    text = _clean_text("\n\n".join(section_texts))
    if not text:
        raise HWPParseError(f"No readable text extracted from {path.name}")

    return {
        "text": text,
        "metadata": {
            "source_path": str(path.resolve()),
            "file_name": path.name,
            "file_format": "hwp",
            "parser": "olefile-hwp5",
            "page_count": None,
            "char_count": len(text),
        },
    }

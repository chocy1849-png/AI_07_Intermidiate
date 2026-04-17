from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from config import PARSED_DATA_DIR


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_parsed_documents() -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for path in sorted(PARSED_DATA_DIR.glob("*.json")):
        payload = _load_json(path)
        payload["parsed_path"] = str(path)
        documents.append(payload)
    return documents


def get_parsed_document(file_name: str) -> dict[str, Any] | None:
    for document in load_parsed_documents():
        if document["metadata"].get("file_name") == file_name:
            return document
    return None


def refresh_parsed_document_cache() -> None:
    load_parsed_documents.cache_clear()

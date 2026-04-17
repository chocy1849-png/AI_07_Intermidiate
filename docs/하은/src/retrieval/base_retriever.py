from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrieverResult:
    chunk_id: str
    score: float
    text: str
    metadata: dict


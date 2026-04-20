from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class EmbeddingBackendConfig:
    backend_key: str
    backend_name: str
    model_name: str
    collection_name: str
    bm25_index_name: str
    query_prefix: str = ""
    document_prefix: str = ""
    normalize_embeddings: bool = True
    encode_kwargs: dict[str, Any] = field(default_factory=dict)
    screening_only: bool = False
    note: str = ""


class EmbeddingBackendProtocol(Protocol):
    config: EmbeddingBackendConfig

    def embed_queries(self, texts: list[str]) -> list[list[float]]: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

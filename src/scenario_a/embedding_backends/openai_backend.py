from __future__ import annotations

from typing import Any

from scenario_a.embedding_backends.base import EmbeddingBackendConfig


class OpenAIEmbeddingBackend:
    def __init__(self, config: EmbeddingBackendConfig, client: Any | None = None) -> None:
        self.config = config
        self.client = client

    def bind_client(self, client: Any) -> None:
        self.client = client

    def _require_client(self) -> Any:
        if self.client is None:
            raise RuntimeError("OpenAI embedding backend requires a bound OpenAI client.")
        return self.client

    def _prepare(self, texts: list[str], prefix: str) -> list[str]:
        return [f"{prefix}{text}" if prefix else str(text) for text in texts]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        client = self._require_client()
        response = client.embeddings.create(
            model=self.config.model_name,
            input=self._prepare(texts, self.config.query_prefix),
        )
        return [item.embedding for item in response.data]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        client = self._require_client()
        response = client.embeddings.create(
            model=self.config.model_name,
            input=self._prepare(texts, self.config.document_prefix),
        )
        return [item.embedding for item in response.data]

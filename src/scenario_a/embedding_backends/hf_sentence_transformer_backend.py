from __future__ import annotations

from typing import Any

from scenario_a.embedding_backends.base import EmbeddingBackendConfig


class HFSentenceTransformerEmbeddingBackend:
    def __init__(self, config: EmbeddingBackendConfig) -> None:
        self.config = config
        self.model: Any | None = None

    def load(self) -> None:
        if self.model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Scenario A HF embedding backend requires sentence-transformers."
            ) from exc
        self.model = SentenceTransformer(self.config.model_name)

    def _prepare(self, texts: list[str], prefix: str) -> list[str]:
        return [f"{prefix}{text}" if prefix else str(text) for text in texts]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        self.load()
        assert self.model is not None
        vectors = self.model.encode(
            texts,
            normalize_embeddings=self.config.normalize_embeddings,
            **self.config.encode_kwargs,
        )
        return [vector.tolist() for vector in vectors]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._encode(self._prepare(texts, self.config.query_prefix))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(self._prepare(texts, self.config.document_prefix))

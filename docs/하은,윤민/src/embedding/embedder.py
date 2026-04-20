"""OpenAI 임베딩 제공자.

기본 모델: text-embedding-3-small (1536차원)
배치 단위로 API를 호출해 비용과 속도를 최적화한다.
"""
from __future__ import annotations

from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
_BATCH_SIZE = 100


class OpenAIEmbedder:
    def __init__(self, model: str = EMBEDDING_MODEL) -> None:
        self.model = model
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI()
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        """텍스트 목록을 임베딩 벡터 목록으로 변환한다."""
        all_embeddings: list[list[float]] = []
        for offset in range(0, len(texts), _BATCH_SIZE):
            batch = texts[offset : offset + _BATCH_SIZE]
            response = self.client.embeddings.create(model=self.model, input=batch)
            all_embeddings.extend(r.embedding for r in response.data)
        return all_embeddings

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


_embedder: OpenAIEmbedder | None = None


def get_embedder() -> OpenAIEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = OpenAIEmbedder()
    return _embedder

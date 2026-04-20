from .base import EmbeddingBackendConfig, EmbeddingBackendProtocol
from .hf_sentence_transformer_backend import HFSentenceTransformerEmbeddingBackend
from .openai_backend import OpenAIEmbeddingBackend


def create_embedding_backend(config: EmbeddingBackendConfig) -> EmbeddingBackendProtocol:
    registry = {
        "openai": OpenAIEmbeddingBackend,
        "hf_sentence_transformer": HFSentenceTransformerEmbeddingBackend,
    }
    if config.backend_name not in registry:
        raise KeyError(f"Unsupported embedding backend: {config.backend_name}")
    return registry[config.backend_name](config)


__all__ = [
    "EmbeddingBackendConfig",
    "EmbeddingBackendProtocol",
    "OpenAIEmbeddingBackend",
    "HFSentenceTransformerEmbeddingBackend",
    "create_embedding_backend",
]

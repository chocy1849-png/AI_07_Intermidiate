import os
from dataclasses import dataclass
from pathlib import Path


try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


if load_dotenv is not None:
    load_dotenv()


@dataclass(frozen=True)
class Settings:
    base_dir: Path = Path("./files").resolve()
    db_path: Path = Path("./my_rfp_vectordb").resolve()
    metadata_csv_path: Path = Path("./data_list.csv").resolve()
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "jhgan/ko-sroberta-multitask")
    chat_model: str = os.getenv("CHAT_MODEL", "gpt-5-mini")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "5"))
    answer_temperature: float = float(os.getenv("ANSWER_TEMPERATURE", "0.1"))
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "cpu")


SETTINGS = Settings()


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Required environment variable '{name}' is not set. "
            "Configure it in your shell or a local .env file."
        )
    return value

from __future__ import annotations

from dataclasses import dataclass
import re

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - optional dependency at runtime
    RecursiveCharacterTextSplitter = None


SECTION_PATTERN = re.compile(r"(?m)^(?:\d+\.\s+|[가-하]\.\s+|제\s*\d+\s*조)")
TOKEN_PATTERN = re.compile(r"[가-힣A-Za-z0-9]+")
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " "]

# 한국어 조사 목록 (길이 내림차순 — 긴 것부터 매칭해야 부분 제거 방지)
# 단일 자 조사(가/이/에/도/만/로/와/과)는 단어 끝 음절과 혼동 위험이 높아 제외
_PARTICLES_MULTI = (
    "으로부터", "에서부터",
    "이라고", "이라면", "이라는", "이라서", "이어서",
    "에서는", "에게는", "으로는", "에서도", "으로도",
    "처럼", "같이", "부터", "까지", "에게", "보다",
    "에서", "으로", "이나", "이라", "이야", "이랑", "이고", "이며", "이면",
    "인가", "인지",
)
# 단일 자 조사 중 한국어 명사 어미로 잘 안 쓰이는 것만 포함
_PARTICLES_SINGLE = ("는", "은", "를", "을", "의", "로")


def _strip_particle(token: str) -> str:
    """한국어 조사를 제거. 어근이 2글자 이상일 때만 제거."""
    for p in _PARTICLES_MULTI:
        if token.endswith(p) and len(token) - len(p) >= 2:
            return token[: len(token) - len(p)]
    for p in _PARTICLES_SINGLE:
        if token.endswith(p) and len(token) - len(p) >= 2:
            return token[: len(token) - len(p)]
    return token


@dataclass
class ChunkingStats:
    total_chunks: int
    average_chunk_length: float
    section_count: int


def normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def split_by_section_headers(text: str) -> list[str]:
    normalized = normalize_text(text)
    matches = list(SECTION_PATTERN.finditer(normalized))
    if not matches:
        return [normalized] if normalized else []

    sections: list[str] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
        section = normalized[start:end].strip()
        if section:
            sections.append(section)
    return sections


def tokenize(text: str) -> list[str]:
    """공백 분리 + 한국어 조사 제거 토크나이저."""
    return [_strip_particle(token.lower()) for token in TOKEN_PATTERN.findall(text)]


def _split_long_text(text: str, chunk_size: int) -> list[str]:
    return [text[offset : offset + chunk_size].strip() for offset in range(0, len(text), chunk_size)]


def recursive_character_chunks(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
    separators: list[str] | None = None,
) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    separators = separators or DEFAULT_SEPARATORS

    def split_recursively(segment: str, remaining_separators: list[str]) -> list[str]:
        if len(segment) <= chunk_size:
            return [segment.strip()]
        if not remaining_separators:
            return _split_long_text(segment, chunk_size)

        separator = remaining_separators[0]
        if separator and separator in segment:
            pieces = segment.split(separator)
            merged: list[str] = []
            current = ""
            for piece in pieces:
                piece = piece.strip()
                if not piece:
                    continue
                candidate = f"{current}{separator}{piece}".strip() if current else piece
                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    if current:
                        merged.append(current.strip())
                    if len(piece) <= chunk_size:
                        current = piece
                    else:
                        merged.extend(split_recursively(piece, remaining_separators[1:]))
                        current = ""
            if current:
                merged.append(current.strip())
            return [item for item in merged if item]

        return split_recursively(segment, remaining_separators[1:])

    chunks = split_recursively(normalized, separators)
    return _apply_overlap(chunks, overlap)


def langchain_recursive_chunks(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
    separators: list[str] | None = None,
) -> list[str]:
    if RecursiveCharacterTextSplitter is None:
        raise ImportError("langchain-text-splitters is not installed.")

    normalized = normalize_text(text)
    if not normalized:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=separators or DEFAULT_SEPARATORS,
        length_function=len,
    )
    return [chunk.strip() for chunk in splitter.split_text(normalized) if chunk.strip()]


def _split_paragraphs(text: str, chunk_size: int) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    if not paragraphs:
        return [text] if text else []

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(paragraph) <= chunk_size:
            current = paragraph
            continue
        for offset in range(0, len(paragraph), chunk_size):
            chunks.append(paragraph[offset : offset + chunk_size].strip())
        current = ""

    if current:
        chunks.append(current)
    return chunks


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped: list[str] = []
    for index, chunk in enumerate(chunks):
        if index == 0:
            overlapped.append(chunk)
            continue
        prefix = chunks[index - 1][-overlap:].strip()
        merged = f"{prefix}\n{chunk}".strip() if prefix else chunk
        overlapped.append(merged)
    return overlapped


def chunk_document(
    text: str,
    metadata: dict | None = None,
    chunk_size: int = 1000,
    overlap: int = 200,
    strategy: str = "section_aware",
) -> list[dict]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    if strategy == "recursive":
        chunks = recursive_character_chunks(normalized, chunk_size=chunk_size, overlap=overlap)
    elif strategy == "langchain_recursive":
        chunks = langchain_recursive_chunks(normalized, chunk_size=chunk_size, overlap=overlap)
    else:
        candidate_sections = split_by_section_headers(normalized)
        base_segments = candidate_sections if len(candidate_sections) > 1 else [normalized]

        raw_chunks: list[str] = []
        for segment in base_segments:
            raw_chunks.extend(_split_paragraphs(segment, chunk_size))

        chunks = _apply_overlap(raw_chunks, overlap)
    file_name = (metadata or {}).get("file_name", "document")

    return [
        {
            "chunk_id": f"{file_name}::{index}",
            "text": chunk,
            "metadata": {
                **(metadata or {}),
                "chunk_index": index,
                "chunk_size": chunk_size,
                "overlap": overlap,
                "strategy": strategy,
            },
        }
        for index, chunk in enumerate(chunks)
        if chunk.strip()
    ]


def describe_chunking(text: str, chunk_size: int, overlap: int, strategy: str = "section_aware") -> ChunkingStats:
    normalized = normalize_text(text)
    sections = split_by_section_headers(normalized)
    chunks = chunk_document(
        normalized,
        metadata=None,
        chunk_size=chunk_size,
        overlap=overlap,
        strategy=strategy,
    )
    chunk_lengths = [len(item["text"]) for item in chunks]
    average = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0.0
    return ChunkingStats(
        total_chunks=len(chunks),
        average_chunk_length=round(average, 2),
        section_count=len(sections),
    )

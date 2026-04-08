from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
INPUT_JSONL_PATH = PROCESSED_DATA_DIR / "processed_documents.jsonl"

OUTPUT_DIR = BASE_DIR / "rag_outputs"
CHUNK_JSONL_PATH = OUTPUT_DIR / "contextual_chunks.jsonl"
CHUNK_SUMMARY_CSV_PATH = OUTPUT_DIR / "contextual_chunks_summary.csv"
# Windows 환경에서 Chroma/HNSW가 한글 경로 아래에서 인덱스 파일(header.bin 등)을
# 정상 생성하지 못하는 경우가 있어, 영문 전용 경로를 기본 저장소로 사용한다.
CHROMA_DIR = OUTPUT_DIR / "chroma_db"
RESULTS_DIR = OUTPUT_DIR / "query_results"
DEFAULT_COLLECTION_NAME = "rfp_contextual_chunks_v1"

ALLOWED_CHAT_MODELS = ("gpt-5-mini", "gpt-5-nano", "gpt-5")
ALLOWED_EMBEDDING_MODELS = ("text-embedding-3-small",)
DEFAULT_CHAT_MODEL = "gpt-5-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def Chroma_컬렉션명_검증(name: str) -> str:
    value = str(name or "").strip()
    pattern = r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{1,510}[A-Za-z0-9])?$"
    if re.match(pattern, value):
        return value

    raise ValueError(
        "Chroma 컬렉션 이름은 3~512자 길이여야 하며, "
        "영문/숫자/점(.)/밑줄(_)/하이픈(-)만 사용할 수 있습니다. "
        f"현재 값: {value}"
    )


def 디렉토리_준비(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def 텍스트_정리(text: str) -> str:
    value = str(text or "")
    value = value.replace("\x00", " ")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"\n{3,}", "\n\n", value)
    value = re.sub(r"[ \t]{2,}", " ", value)
    return value.strip()


def 표시값(value: Any, 기본값: str = "정보 없음") -> str:
    if value is None:
        return 기본값
    if isinstance(value, float) and str(value) == "nan":
        return 기본값

    text = str(value).strip()
    return text if text else 기본값


def 크로마_메타데이터값(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        return value
    return json.dumps(value, ensure_ascii=False)


def jsonl_불러오기(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def jsonl_저장(path: Path, rows: list[dict[str, Any]]) -> None:
    디렉토리_준비(path.parent)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def csv_저장(path: Path, rows: list[dict[str, Any]]) -> None:
    디렉토리_준비(path.parent)
    if not rows:
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            file.write("")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def 문서_본문_선택(document: dict[str, Any]) -> str:
    rag_ready = 텍스트_정리(document.get("rag_ready_text", ""))
    if rag_ready:
        return rag_ready
    return 텍스트_정리(document.get("text", ""))


def 메타데이터_접두어(document: dict[str, Any]) -> str:
    metadata = document.get("metadata", {}) or {}
    lines = [
        "[문서 메타데이터]",
        f"- 사업명: {표시값(metadata.get('사업명'))}",
        f"- 발주 기관: {표시값(metadata.get('발주 기관'))}",
        f"- 공고 번호: {표시값(metadata.get('공고 번호'))}",
        f"- 공개 일자: {표시값(metadata.get('공개 일자'))}",
        f"- 파일 형식: {표시값(metadata.get('파일형식'), document.get('source_extension', '정보 없음'))}",
        f"- 파일명: {표시값(document.get('source_file_name'))}",
        f"- 문서 ID: {표시값(document.get('document_id'))}",
        "[/문서 메타데이터]",
    ]
    return "\n".join(lines)


def 텍스트_청킹(text: str, chunk_size: int, overlap_size: int) -> list[str]:
    cleaned = 텍스트_정리(text)
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    minimum_split = max(200, int(chunk_size * 0.6))

    while start < len(cleaned):
        tentative_end = min(start + chunk_size, len(cleaned))
        if tentative_end >= len(cleaned):
            chunk = cleaned[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        search_floor = min(len(cleaned), start + minimum_split)
        split_candidates = [
            cleaned.rfind("\n\n", search_floor, tentative_end),
            cleaned.rfind("\n", search_floor, tentative_end),
            cleaned.rfind(". ", search_floor, tentative_end),
            cleaned.rfind("다. ", search_floor, tentative_end),
            cleaned.rfind(" ", search_floor, tentative_end),
        ]
        split_point = max(split_candidates)
        if split_point <= start:
            split_point = tentative_end

        chunk = cleaned[start:split_point].strip()
        if chunk:
            chunks.append(chunk)

        next_start = max(split_point - overlap_size, start + 1)
        if next_start <= start:
            next_start = split_point
        start = next_start

    return chunks


def 청크_ID_생성(document_id: str, chunk_index: int) -> str:
    safe_document_id = re.sub(r"[^0-9A-Za-z가-힣_-]+", "_", str(document_id))
    return f"{safe_document_id}__chunk_{chunk_index:04d}"


def 현재시각_문자열() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def OpenAI_클라이언트_가져오기():
    load_dotenv(BASE_DIR / ".env", override=False)

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            ".env 파일에 OPENAI_API_KEY가 없습니다. "
            "프로젝트 폴더의 .env 파일에 키를 입력한 뒤 다시 실행하세요."
        )

    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "openai 패키지가 설치되어 있지 않습니다. "
            "requirements.txt를 사용해 필요한 패키지를 먼저 설치하세요."
        ) from exc

    kwargs: dict[str, Any] = {"api_key": api_key}
    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip()
    if not base_url:
        kwargs["base_url"] = "https://api.openai.com/v1"
    else:
        if not re.match(r"^https?://", base_url):
            raise RuntimeError(
                "OPENAI_BASE_URL 형식이 잘못되었습니다. "
                "반드시 http:// 또는 https:// 로 시작해야 합니다. "
                f"현재 값: {base_url}"
            )
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def 결과파일_경로(base_name: str, extension: str = ".json") -> Path:
    디렉토리_준비(RESULTS_DIR)
    return RESULTS_DIR / f"{base_name}_{현재시각_문자열()}{extension}"

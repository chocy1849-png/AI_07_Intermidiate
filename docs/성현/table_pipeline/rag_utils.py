import csv
import re
import struct
import sys
import warnings
import zlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    import olefile
except ImportError:  # pragma: no cover
    olefile = None

try:
    import pymupdf4llm
except ImportError:  # pragma: no cover
    pymupdf4llm = None
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_system.config import SETTINGS

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from table_pipeline.ocr_support.extract_hwp_artifacts import extract_hwp_artifacts
from table_pipeline.ocr_support.run_hwp_ocr_pipeline import run_image_ocr
from table_pipeline.table_enrichment import (
    build_doc_context_blocks,
    build_table_block,
    extract_doc_context_hints,
)


warnings.filterwarnings("ignore")

MIN_OCR_LINE_LENGTH = 2
MIN_IMAGE_BLOCK_LENGTH = 20

SECTION_PATTERNS = [
    re.compile(r"^\s*제?\s*\d+\s*장\b.*$"),
    re.compile(r"^\s*\d+(\.\d+){0,3}[\.\)]\s+.+$"),
    re.compile(r"^\s*[가-힣][\.\)]\s+.+$"),
    re.compile(r"^\s*[A-Z][\.\)]\s+.+$"),
    re.compile(
        r"^\s*(사업개요|사업 목적|추진 배경|추진 목적|제안 요청 내용|제안요청내용|제안 범위|과업 내용|과업내용|제출 서류|입찰 참가 자격|평가 기준|평가기준|사업 예산|예산|추진 일정|일정)\s*$"
    ),
]

TAG_RULES = {
    "budget": {
        "strong": ["예산", "사업비", "예산액", "기초금액", "추정금액", "총사업비", "배정예산"],
        "weak": ["금액", "원", "vat", "부가세"],
        "threshold": 2,
    },
    "submission": {
        "strong": ["제출", "제안서 제출", "접수", "제출서류", "제출방법", "방문접수", "전자제출"],
        "weak": ["우편", "이메일", "usb", "마감"],
        "threshold": 2,
    },
    "evaluation": {
        "strong": ["평가", "평가기준", "배점", "기술평가", "정량평가", "정성평가", "종합평가"],
        "weak": ["심사", "점수"],
        "threshold": 2,
    },
    "purpose": {
        "strong": ["목적", "사업목적", "추진목적", "추진배경", "사업개요", "사업목표"],
        "weak": ["배경", "필요성", "기대효과"],
        "threshold": 2,
    },
    "deadline": {
        "strong": ["마감", "마감일", "기한", "제출기한", "접수기간", "공고기간", "제안서 제출일시"],
        "weak": ["일정", "일시", "기간"],
        "threshold": 2,
    },
    "requirement": {
        "strong": ["요구사항", "과업내용", "과업범위", "기능요건", "기술요건", "상세요건"],
        "weak": ["요건", "기능", "성능", "범위", "구현"],
        "threshold": 2,
    },
    "qualification": {
        "strong": ["자격", "참가자격", "입찰참가자격", "제안자격", "자격요건"],
        "weak": ["실적", "인력", "면허", "등록", "인증"],
        "threshold": 2,
    },
}

TAG_KEYWORDS = {tag: rules["strong"] + rules["weak"] for tag, rules in TAG_RULES.items()}

CSV_METADATA_FIELDS = {
    "공고 번호": "notice_id",
    "공고 차수": "notice_round",
    "사업명": "project_name",
    "사업 금액": "bid_amount",
    "발주 기관": "issuer",
    "공개 일자": "posted_at",
    "입찰 참여 시작일": "bid_start_at",
    "입찰 참여 마감일": "bid_end_at",
    "사업 요약": "project_summary",
    "파일형식": "listed_extension",
    "파일명": "listed_filename",
}
def build_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=SETTINGS.embedding_model,
        model_kwargs={
            "device": SETTINGS.embedding_device,
            "local_files_only": True,
        },
    )


def build_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "],
        chunk_size=SETTINGS.chunk_size,
        chunk_overlap=SETTINGS.chunk_overlap,
    )


def clean_text(text: str) -> str:
    text = re.sub(r"\*\*==>.*?<==\*\*", "", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^\S\r\n]{3,}", " ", text)
    text = re.sub(r"\.{4,}", " ", text)
    text = re.sub(r"\n\s*-\d+-\s*\n", "\n", text)
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"^\|[-|\s]+\|\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\|[\s|]*\|\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def is_section_header(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return False
    if len(stripped) < 4:
        return False
    if len(stripped) > 70 and not stripped.startswith("제"):
        return False
    if stripped.endswith((".", ",")):
        return False
    if len(stripped.split()) > 12:
        return False
    return any(pattern.match(stripped) for pattern in SECTION_PATTERNS)


def split_into_sections(text: str) -> List[dict]:
    sections: List[dict] = []
    current_header = "문서 개요"
    current_lines: List[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if current_lines and current_lines[-1] != "":
                current_lines.append("")
            continue

        if is_section_header(line):
            if current_lines:
                body = "\n".join(current_lines).strip()
                if body:
                    sections.append({"header": current_header, "body": body})
            current_header = line
            current_lines = []
            continue

        current_lines.append(line)

    if current_lines:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append({"header": current_header, "body": body})

    if not sections:
        return [{"header": "문서 개요", "body": text}]
    return sections


def chunk_section(section: dict, splitter: RecursiveCharacterTextSplitter) -> List[dict]:
    header = section["header"].strip()
    body = section["body"].strip()
    combined = f"{header}\n{body}" if body else header

    if len(combined) <= SETTINGS.chunk_size:
        return [{"section": header, "content": combined}]

    chunks = splitter.split_text(body)
    results = []
    for index, chunk in enumerate(chunks, start=1):
        chunk = chunk.strip()
        if not chunk:
            continue
        label = header if len(chunks) == 1 else f"{header} (part {index})"
        results.append({"section": header, "content": f"{header}\n{chunk}", "subchunk_label": label})
    return results


def normalize_text(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def normalize_filename(text: str) -> str:
    lowered = normalize_text(text)
    return re.sub(r"[^0-9a-z가-힣]", "", lowered)


def match_keywords(text: str, keywords: List[str]) -> List[str]:
    normalized = normalize_text(text)
    return [keyword for keyword in keywords if normalize_text(keyword) in normalized]


def score_tags(text: str, section: Optional[str] = None) -> dict[str, dict]:
    body = text or ""
    section_text = section or ""
    scored = {}

    for tag, rules in TAG_RULES.items():
        strong_hits = match_keywords(body, rules["strong"])
        weak_hits = match_keywords(body, rules["weak"])
        section_hits = match_keywords(section_text, rules["strong"] + rules["weak"])
        score = len(strong_hits) * 2 + len(weak_hits) + len(section_hits) * 2

        if score >= rules["threshold"]:
            scored[tag] = {
                "score": score,
                "strong_hits": strong_hits,
                "weak_hits": weak_hits,
                "section_hits": section_hits,
            }

    return scored


def infer_tags(text: str, section: Optional[str] = None) -> dict:
    scored_tags = score_tags(text=text, section=section)
    matched_tags = sorted(scored_tags, key=lambda tag: (-scored_tags[tag]["score"], tag))
    primary_tag = matched_tags[0] if matched_tags else "general"
    return {
        "primary_tag": primary_tag,
        "tags": ", ".join(matched_tags) if matched_tags else "",
        "has_budget": "budget" in matched_tags,
        "has_submission": "submission" in matched_tags,
        "has_evaluation": "evaluation" in matched_tags,
        "has_purpose": "purpose" in matched_tags,
        "has_deadline": "deadline" in matched_tags,
        "has_requirement": "requirement" in matched_tags,
        "has_qualification": "qualification" in matched_tags,
    }


def load_csv_metadata(csv_path: Path = SETTINGS.metadata_csv_path) -> Dict[str, dict]:
    if not csv_path.exists():
        return {}

    metadata_by_filename: Dict[str, dict] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            filename = (row.get("파일명") or "").strip()
            if not filename:
                continue

            mapped = {}
            for source_key, target_key in CSV_METADATA_FIELDS.items():
                value = (row.get(source_key) or "").strip()
                if value:
                    mapped[target_key] = value

            metadata_by_filename[normalize_filename(filename)] = mapped

    return metadata_by_filename


def get_document_metadata(file_path: Path, metadata_by_filename: Optional[Dict[str, dict]] = None) -> dict:
    metadata_by_filename = metadata_by_filename or {}
    return metadata_by_filename.get(normalize_filename(file_path.name), {}).copy()


def extract_pdf_text(file_path: Path) -> str:
    if pymupdf4llm is None:
        raise RuntimeError("pymupdf4llm is required to read PDF files.")
    return pymupdf4llm.to_markdown(str(file_path))


def extract_hwp_text(file_path: Path) -> str:
    if olefile is None:
        raise RuntimeError("olefile is required to read HWP files.")
    try:
        hwp = olefile.OleFileIO(str(file_path))
        if not hwp.exists("FileHeader"):
            hwp.close()
            return ""

        header_data = hwp.openstream("FileHeader").read()
        is_compressed = (header_data[36] & 1) == 1
        text_parts: List[str] = []

        for entry in hwp.listdir():
            if not entry or entry[0] != "BodyText":
                continue

            data = hwp.openstream("/".join(entry)).read()
            if is_compressed:
                try:
                    data = zlib.decompress(data, -15)
                except zlib.error:
                    continue

            i = 0
            size = len(data)
            while i < size:
                if i + 4 > size:
                    break

                header = struct.unpack_from("<I", data, i)[0]
                rec_type = header & 0x3FF
                rec_len = (header >> 20) & 0xFFF

                if rec_len == 0xFFF:
                    if i + 8 > size:
                        break
                    rec_len = struct.unpack_from("<I", data, i + 4)[0]
                    i += 4

                if rec_type == 67:
                    rec_data = data[i + 4 : i + 4 + rec_len]
                    chars: List[str] = []
                    j = 0
                    while j < len(rec_data) - 1:
                        code = struct.unpack_from("<H", rec_data, j)[0]

                        if code in (1, 2, 3, 11, 12, 14, 15, 16, 17, 18, 21, 22, 23):
                            j += 14
                            continue
                        if code in (4, 5, 6, 7, 8, 9, 19, 20):
                            j += 10
                            continue
                        if code in (10, 13):
                            chars.append("\n")
                            j += 2
                            continue
                        if code == 24:
                            chars.append("\t")
                            j += 14
                            continue
                        if code == 0 or 0xD800 <= code <= 0xDFFF:
                            j += 2
                            continue

                        chars.append(chr(code))
                        j += 2

                    line = "".join(chars).strip()
                    if line:
                        text_parts.append(line)

                i += 4 + rec_len

        if not text_parts and hwp.exists("PrvText"):
            preview = hwp.openstream("PrvText").read().decode("utf-16-le", errors="ignore").strip()
            if preview:
                text_parts.append(preview)

        hwp.close()
        return "\n".join(text_parts)
    except Exception:
        return ""


def normalize_ocr_lines(lines: List[str]) -> List[str]:
    cleaned_lines: List[str] = []
    for line in lines:
        normalized = re.sub(r"\s+", " ", line).strip()
        if len(normalized) < MIN_OCR_LINE_LENGTH:
            continue
        cleaned_lines.append(normalized)
    return cleaned_lines


def append_labeled_block(blocks: List[str], label: str, body: str) -> None:
    body = body.strip()
    if not body:
        return
    blocks.append(f"{label}\n{body}")


def build_image_blocks(payload: dict) -> List[str]:
    blocks: List[str] = []

    for image in payload.get("image_ocr_candidates", []):
        saved_path = image.get("saved_path")
        if not saved_path:
            continue

        ocr_result = run_image_ocr(Path(saved_path))
        cleaned_lines = normalize_ocr_lines(ocr_result.get("ocr_lines", []))
        if not cleaned_lines:
            continue

        joined_text = "\n".join(cleaned_lines)
        if len(joined_text) < MIN_IMAGE_BLOCK_LENGTH:
            continue

        block_lines = [
            "[IMAGE_OCR]",
            f"[IMAGE_NAME] {image['name']}",
            f"[OCR_PRIORITY] {image.get('ocr_candidate_priority', 'low')}",
            joined_text,
        ]
        blocks.append("\n".join(block_lines).strip())

    return blocks


def build_hwp_semantic_text(file_path: Path) -> str:
    payload = extract_hwp_artifacts(file_path, save_images=True, output_dir=ROOT_DIR / "ocr")

    blocks: List[str] = []
    doc_title = file_path.stem
    hint_map = extract_doc_context_hints(payload)

    blocks.extend(build_doc_context_blocks(doc_title, payload))

    for table in payload.get("structural_tables", []):
        table_block = build_table_block(table, doc_title, hint_map=hint_map)
        if table_block:
            blocks.append(table_block)

    for block in payload.get("explanatory_blocks", []):
        append_labeled_block(blocks, "[EXPLANATORY_BLOCK]", block.get("text", ""))

    for header in payload.get("section_header_blocks", []):
        if header.get("linked_child_table_indices"):
            child_labels = ", ".join(str(idx) for idx in header["linked_child_table_indices"])
            append_labeled_block(
                blocks,
                f"[SECTION_HEADER] children={child_labels}",
                header.get("text", ""),
            )

    blocks.extend(build_image_blocks(payload))

    return "\n\n".join(blocks).strip()


def extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_text(file_path)
    if suffix == ".hwp":
        semantic_text = build_hwp_semantic_text(file_path)
        if semantic_text:
            return semantic_text
        return extract_hwp_text(file_path)
    return ""


def make_documents(
    file_path: Path,
    splitter: RecursiveCharacterTextSplitter,
    metadata_by_filename: Optional[Dict[str, dict]] = None,
) -> List[Document]:
    raw_text = extract_text(file_path)
    if not raw_text.strip():
        return []

    cleaned = clean_text(raw_text)
    sections = split_into_sections(cleaned)
    title = file_path.stem
    mapped_metadata = get_document_metadata(file_path, metadata_by_filename)
    docs: List[Document] = []

    chunk_index = 0
    for section in sections:
        for chunk_info in chunk_section(section, splitter):
            chunk = chunk_info["content"].strip()
            if len(chunk) < 20:
                continue
            section_name: Optional[str] = chunk_info.get("section")
            docs.append(
                Document(
                    page_content=f"[사업명: {title}]\n[섹션: {section_name or '문서 개요'}]\n---\n{chunk}",
                    metadata={
                        "title": title,
                        "source": file_path.name,
                        "file_path": str(file_path.resolve()),
                        "chunk_id": chunk_index,
                        "extension": file_path.suffix.lower(),
                        "section": section_name or "문서 개요",
                        **mapped_metadata,
                        **infer_tags(chunk, section_name),
                    },
                )
            )
            chunk_index += 1
    return docs


def iter_source_files(base_dir: Path) -> Iterable[Path]:
    for pattern in ("*.pdf", "*.hwp"):
        yield from sorted(base_dir.glob(pattern))

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scenario_a.common_pipeline import PipelinePaths, ScenarioACommonPipeline

try:
    import fitz  # type: ignore
except Exception as exc:  # noqa: BLE001
    raise RuntimeError("PyMuPDF(fitz) is required. Install with: pip install pymupdf") from exc

try:
    from PIL import Image  # type: ignore
except Exception as exc:  # noqa: BLE001
    raise RuntimeError("Pillow is required. Install with: pip install pillow") from exc

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception as exc:  # noqa: BLE001
    raise RuntimeError("beautifulsoup4 is required. Install with: pip install beautifulsoup4") from exc

try:
    from rank_bm25 import BM25Okapi  # type: ignore
except Exception as exc:  # noqa: BLE001
    raise RuntimeError("rank-bm25 is required. Install with: pip install rank-bm25") from exc

try:
    import win32com.client as win32  # type: ignore
except Exception as exc:  # noqa: BLE001
    raise RuntimeError("pywin32 is required for HWP COM rendering on Windows.") from exc

try:
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    from paddleocr import PPStructureV3, TableStructureRecognition  # type: ignore
except Exception as exc:  # noqa: BLE001
    raise RuntimeError("paddleocr >= 3.x is required for PP-StructureV3.") from exc

try:
    from ocr.extract_hwp_artifacts import extract_hwp_artifacts
    from ocr.run_hwp_ocr_pipeline import run_image_ocr
except Exception as exc:  # noqa: BLE001
    raise RuntimeError("Failed to import OCR helpers from ocr/ package.") from exc


EMBEDDING_BACKEND_KEY = "openai_text_embedding_3_small_true_table_ocr_v2"
COLLECTION_NAME = "rfp_contextual_chunks_v2_true_table_ocr_v2"
BM25_INDEX_NAME = "bm25_index_phase2_true_table_ocr_v2.pkl"


@dataclass(slots=True)
class HtmlTable:
    index: int
    text: str
    rows: list[list[str]]
    normalized: str


@dataclass(slots=True)
class RenderedPage:
    page_index: int
    page_text: str
    image_path: Path
    width: int
    height: int


@dataclass(slots=True)
class TableRegion:
    region_id: str
    page_index: int
    label: str
    score: float
    reading_order: int
    bbox: list[float]
    crop_path: Path
    ocr_text: str
    ocr_status: str
    table_html: str
    structure_score: float
    render_source: str
    parser_source: str


def _require_openai() -> Any:
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError("openai package is required.") from exc
    return OpenAI


def _require_chromadb() -> Any:
    try:
        import chromadb
    except ModuleNotFoundError as exc:
        raise RuntimeError("chromadb package is required.") from exc
    return chromadb


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            raw = line.strip()
            if not raw:
                continue
            rows.append(json.loads(raw))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _doc_hash(document_id: str) -> str:
    return hashlib.sha1(document_id.encode("utf-8")).hexdigest()[:10]


def _norm_name(value: str) -> str:
    lowered = re.sub(r"\s+", "", str(value or "").lower())
    return re.sub(r"[^0-9a-z가-힣._-]", "", lowered)


def _normalize_text(value: str) -> str:
    text = str(value or "").lower()
    text = re.sub(r"\s+", "", text)
    return re.sub(r"[^0-9a-z가-힣._-]", "", text)


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    value = _normalize_text(text)
    if len(value) < n:
        return {value} if value else set()
    return {value[i : i + n] for i in range(len(value) - n + 1)}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[0-9A-Za-z가-힣]{2,}", str(text or "").lower())


def _split_text(text: str, max_chars: int = 1400, overlap: int = 120) -> list[str]:
    value = str(text or "").strip()
    if not value:
        return []
    if len(value) <= max_chars:
        return [value]
    chunks: list[str] = []
    start = 0
    while start < len(value):
        end = min(len(value), start + max_chars)
        snippet = value[start:end].strip()
        if snippet:
            chunks.append(snippet)
        if end >= len(value):
            break
        start = max(0, end - overlap)
    return chunks


def _contains_budget(text: str) -> int:
    return int(bool(re.search(r"(예산|금액|사업비|추정금액|원)", str(text or ""))))


def _contains_schedule(text: str) -> int:
    return int(bool(re.search(r"(기간|일정|마감|착수|완료|기한|납기|개월|일 이내)", str(text or ""))))


def _contains_contract(text: str) -> int:
    return int(bool(re.search(r"(계약|입찰|방식|협상|수의)", str(text or ""))))


def _simple_metadata(row: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in row.items():
        if key == "contextual_chunk_text":
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            output[key] = value if value is not None else ""
        else:
            output[key] = json.dumps(value, ensure_ascii=False)
    return output


def _build_bm25_index(chunk_rows: list[dict[str, Any]]) -> dict[str, Any]:
    tokenized = [re.findall(r"[0-9a-zA-Z가-힣]+", str(row.get("contextual_chunk_text", "")).lower()) for row in chunk_rows]
    model = BM25Okapi(tokenized)
    return {"tokenized_corpus": tokenized, "chunk_rows": chunk_rows, "model": model}


def _write_bm25_index(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(payload, file)


def _load_baseline_chunks(project_root: Path) -> list[dict[str, Any]]:
    baseline_path = project_root / "rag_outputs" / "b02_prefix_v2_chunks.jsonl"
    if not baseline_path.exists():
        raise FileNotFoundError(f"baseline chunks not found: {baseline_path}")
    return _read_jsonl(baseline_path)


def _load_processed_documents(project_root: Path) -> list[dict[str, Any]]:
    path = project_root / "processed_data" / "processed_documents.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"processed_documents.jsonl not found: {path}")
    return _read_jsonl(path)


def _read_target_doc_names(eval_set_paths: list[Path]) -> set[str]:
    names: set[str] = set()
    for path in eval_set_paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                one = str(row.get("ground_truth_doc", "")).strip()
                if one.lower().endswith(".hwp"):
                    names.add(_norm_name(one))
                many = [x.strip() for x in str(row.get("ground_truth_docs", "")).split("|") if x.strip()]
                for item in many:
                    if item.lower().endswith(".hwp"):
                        names.add(_norm_name(item))
    return names


def _select_target_hwp_docs(processed_rows: list[dict[str, Any]], target_names: set[str]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in processed_rows:
        ext = str(row.get("source_extension", "")).strip().lower()
        if ext != ".hwp":
            continue
        file_name = str(row.get("source_file_name", "")).strip()
        if not file_name:
            continue
        norm_file = _norm_name(file_name)
        matched = False
        for target in target_names:
            if not target:
                continue
            if min(len(target), len(norm_file)) >= 8 and (target in norm_file or norm_file in target):
                matched = True
                break
            ratio = difflib.SequenceMatcher(None, target, norm_file).ratio()
            if ratio >= 0.78:
                matched = True
                break
        if not matched:
            continue
        source_path = Path(str(row.get("source_path", "")).strip())
        if not source_path.exists():
            continue
        doc_id = str(row.get("document_id", "")).strip()
        if not doc_id or doc_id in seen:
            continue
        selected.append(row)
        seen.add(doc_id)
    return selected


def _find_collection_dir(project_root: Path, collection_name: str) -> Path:
    chromadb = _require_chromadb()
    candidates = [
        project_root / "rag_outputs" / "chroma_db",
        project_root.parent / "rfp_rag_chroma_db",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            client = chromadb.PersistentClient(path=str(path))
            client.get_collection(collection_name)
            return path
        except Exception:  # noqa: BLE001
            continue
    raise FileNotFoundError(
        f"collection {collection_name} not found. checked: {', '.join(str(path) for path in candidates)}"
    )


def _copy_baseline_collection(
    *,
    baseline_client: Any,
    baseline_collection_name: str,
    target_client: Any,
    target_collection_name: str,
    batch_size: int,
) -> int:
    baseline_collection = baseline_client.get_collection(baseline_collection_name)
    target_collection = target_client.get_or_create_collection(name=target_collection_name)
    total = int(baseline_collection.count() or 0)
    copied = 0
    for offset in range(0, total, max(1, batch_size)):
        batch = baseline_collection.get(
            include=["documents", "metadatas", "embeddings"],
            limit=max(1, batch_size),
            offset=offset,
        )
        ids = batch.get("ids")
        if ids is None:
            ids = []
        if not ids:
            continue
        documents = batch.get("documents")
        if documents is None:
            documents = []
        metadatas = batch.get("metadatas")
        if metadatas is None:
            metadatas = []
        embeddings = batch.get("embeddings")
        if embeddings is None:
            embeddings = []
        target_collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        copied += len(ids)
    return copied


def _embed_and_add_rows(
    *,
    target_collection: Any,
    rows: list[dict[str, Any]],
    openai_client: Any,
    embedding_model: str,
    batch_size: int,
) -> int:
    added = 0
    for start in range(0, len(rows), max(1, batch_size)):
        batch = rows[start : start + max(1, batch_size)]
        texts = [str(row.get("contextual_chunk_text", ""))[:6000] for row in batch]
        response = openai_client.embeddings.create(model=embedding_model, input=texts)
        vectors = [item.embedding for item in response.data]
        target_collection.add(
            ids=[str(row.get("chunk_id", "")) for row in batch],
            documents=texts,
            metadatas=[_simple_metadata(row) for row in batch],
            embeddings=vectors,
        )
        added += len(batch)
    return added


def _build_chroma(
    *,
    project_root: Path,
    augment_rows: list[dict[str, Any]],
    chroma_output_dir: Path,
    baseline_collection_name: str,
    target_collection_name: str,
    embedding_model: str,
    embedding_batch_size: int,
    copy_batch_size: int,
    reset_collection: bool,
) -> dict[str, Any]:
    chromadb = _require_chromadb()
    OpenAI = _require_openai()

    load_dotenv(project_root / ".env", override=False)
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing in .env")
    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip() or "https://api.openai.com/v1"
    openai_client = OpenAI(api_key=api_key, base_url=base_url)

    baseline_dir = _find_collection_dir(project_root, baseline_collection_name)
    baseline_client = chromadb.PersistentClient(path=str(baseline_dir))
    chroma_output_dir.mkdir(parents=True, exist_ok=True)
    target_client = chromadb.PersistentClient(path=str(chroma_output_dir))
    if reset_collection:
        try:
            target_client.delete_collection(target_collection_name)
        except Exception:  # noqa: BLE001
            pass

    copied = _copy_baseline_collection(
        baseline_client=baseline_client,
        baseline_collection_name=baseline_collection_name,
        target_client=target_client,
        target_collection_name=target_collection_name,
        batch_size=copy_batch_size,
    )
    target_collection = target_client.get_collection(target_collection_name)
    added = _embed_and_add_rows(
        target_collection=target_collection,
        rows=augment_rows,
        openai_client=openai_client,
        embedding_model=embedding_model,
        batch_size=embedding_batch_size,
    )
    return {
        "baseline_chroma_dir": str(baseline_dir),
        "target_chroma_dir": str(chroma_output_dir),
        "collection_name": target_collection_name,
        "baseline_rows_copied": copied,
        "augment_rows_added": added,
        "collection_count": int(target_collection.count() or 0),
    }


def _run_hwp5html(*, source_path: Path, html_path: Path, refresh: bool) -> tuple[bool, str]:
    if html_path.exists() and not refresh:
        return True, ""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "hwp5html",
        "--loglevel",
        "error",
        "--html",
        "--output",
        str(html_path),
        str(source_path),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        tail = "\n".join((completed.stderr or completed.stdout or "").splitlines()[-20:])
        return False, tail.strip()
    return True, ""


def _parse_html_tables(html_path: Path) -> list[HtmlTable]:
    text = html_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(text, "html.parser")
    output: list[HtmlTable] = []
    for index, table in enumerate(soup.find_all("table"), start=1):
        rows: list[list[str]] = []
        for tr in table.find_all("tr"):
            cells = [re.sub(r"\s+", " ", cell.get_text(" ", strip=True)).strip() for cell in tr.find_all(["th", "td"])]
            cells = [cell for cell in cells if cell]
            if cells:
                rows.append(cells)
        if rows:
            lines = [" | ".join(cells) for cells in rows]
            table_text = "\n".join(lines).strip()
        else:
            table_text = re.sub(r"\s+", " ", table.get_text(" ", strip=True)).strip()
        if not table_text:
            continue
        output.append(HtmlTable(index=index, text=table_text, rows=rows, normalized=_normalize_text(table_text)))
    return output


def _match_html_table(*, structural_text: str, linked_parent_text: str, html_tables: list[HtmlTable]) -> tuple[HtmlTable | None, float]:
    norm_struct = _normalize_text(structural_text)
    if not norm_struct or not html_tables:
        return None, 0.0
    struct_grams = _char_ngrams(norm_struct, n=3)
    parent_norm = _normalize_text(linked_parent_text)

    best_table: HtmlTable | None = None
    best_score = 0.0
    for table in html_tables:
        if not table.normalized:
            continue
        table_grams = _char_ngrams(table.normalized, n=3)
        if not table_grams:
            continue
        inter = len(struct_grams & table_grams)
        union = len(struct_grams | table_grams)
        jaccard = inter / union if union else 0.0
        prefix_score = 0.0
        if norm_struct[:80] and norm_struct[:80] in table.normalized:
            prefix_score = 0.2
        elif table.normalized[:80] and table.normalized[:80] in norm_struct:
            prefix_score = 0.15
        parent_boost = 0.15 if parent_norm and parent_norm in table.normalized else 0.0
        score = jaccard + prefix_score + parent_boost
        if score > best_score:
            best_score = score
            best_table = table
    return best_table, round(best_score, 4)


def _build_doc_body_index(baseline_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_file: dict[str, list[dict[str, Any]]] = {}
    for row in baseline_rows:
        file_name = str(row.get("source_file_name", "")).strip()
        if not file_name:
            continue
        if int(row.get("has_table", 0) or 0) == 1:
            continue
        by_file.setdefault(file_name, []).append(row)
    return by_file


def _norm_text_for_match(text: str) -> str:
    value = re.sub(r"\s+", " ", str(text or "").strip().lower())
    return re.sub(r"[^0-9a-z가-힣 ]", "", value)


def _lexical_overlap_score(query_tokens: set[str], body_text: str) -> float:
    if not query_tokens:
        return 0.0
    tokens = set(_tokenize(body_text))
    if not tokens:
        return 0.0
    inter = len(query_tokens & tokens)
    return inter / max(1, len(query_tokens))


def _pair_body_text(
    *,
    body_rows: list[dict[str, Any]],
    table_row: dict[str, Any],
    structural_text: str,
    linked_parent_text: str,
) -> dict[str, Any]:
    query_tokens = set(_tokenize(linked_parent_text))
    if len(query_tokens) < 3:
        query_tokens.update(_tokenize(structural_text)[:18])
    table_section = _norm_text_for_match(
        str(table_row.get("linked_parent_text", "") or table_row.get("section", "") or "")
    )

    method_best: dict[str, tuple[float, dict[str, Any] | None]] = {
        "caption_window": (0.0, None),
        "same_section_window": (0.0, None),
        "semantic_nearest_paragraph": (0.0, None),
    }

    for row in body_rows:
        body_text = str(row.get("raw_chunk_text", "") or row.get("contextual_chunk_text", "")).strip()
        if not body_text:
            continue
        body_norm = _norm_text_for_match(body_text)
        body_section = _norm_text_for_match(
            str(row.get("section_title", "") or row.get("section_label", "") or row.get("parent_section_label", ""))
        )
        body_path = _norm_text_for_match(str(row.get("section_path", "")))
        base_overlap = _lexical_overlap_score(query_tokens, body_text)

        caption_score = base_overlap
        if linked_parent_text and _norm_text_for_match(linked_parent_text) and _norm_text_for_match(linked_parent_text) in body_norm:
            caption_score += 0.45
        if table_section and table_section in body_norm:
            caption_score += 0.20

        section_score = base_overlap
        if table_section and body_section and (table_section == body_section):
            section_score += 0.55
        if table_section and body_path and table_section in body_path:
            section_score += 0.30

        semantic_score = base_overlap
        if table_section and body_norm:
            table_terms = set(_tokenize(table_section))
            body_terms = set(_tokenize(body_norm))
            if table_terms and body_terms:
                semantic_score += min(0.35, len(table_terms & body_terms) * 0.06)

        current = {
            "text": body_text[:1500],
            "section_title": str(row.get("section_title", "") or ""),
            "section_path": str(row.get("section_path", "") or ""),
            "chunk_id": str(row.get("chunk_id", "") or ""),
        }
        for method, score in [
            ("caption_window", caption_score),
            ("same_section_window", section_score),
            ("semantic_nearest_paragraph", semantic_score),
        ]:
            best_score, _ = method_best[method]
            if score > best_score:
                method_best[method] = (float(score), current)

    chosen_method = "semantic_nearest_paragraph"
    chosen_score = 0.0
    chosen_row: dict[str, Any] | None = None
    for method in ["caption_window", "same_section_window", "semantic_nearest_paragraph"]:
        score, item = method_best[method]
        if score > chosen_score and item is not None:
            chosen_method = method
            chosen_score = float(score)
            chosen_row = item

    method_scores = {key: round(val[0], 4) for key, val in method_best.items()}
    return {
        "text": (chosen_row or {}).get("text", "") if chosen_row else "",
        "method": chosen_method if chosen_row else "",
        "score": round(chosen_score, 4),
        "method_scores": method_scores,
        "source_chunk_id": (chosen_row or {}).get("chunk_id", "") if chosen_row else "",
        "source_section_title": (chosen_row or {}).get("section_title", "") if chosen_row else "",
        "source_section_path": (chosen_row or {}).get("section_path", "") if chosen_row else "",
    }


def _find_nearby_body_text(*, body_rows: list[dict[str, Any]], structural_text: str, linked_parent_text: str) -> str:
    query_tokens = set(_tokenize(linked_parent_text))
    if len(query_tokens) < 3:
        query_tokens.update(_tokenize(structural_text)[:18])
    if not query_tokens:
        return ""
    best_row: dict[str, Any] | None = None
    best_score = 0.0
    for row in body_rows:
        body_text = str(row.get("raw_chunk_text", "") or row.get("contextual_chunk_text", "")).strip()
        if not body_text:
            continue
        tokens = set(_tokenize(body_text))
        if not tokens:
            continue
        inter = len(query_tokens & tokens)
        if inter <= 0:
            continue
        score = inter / max(1, len(query_tokens))
        if linked_parent_text and linked_parent_text in body_text:
            score += 0.2
        if score > best_score:
            best_score = score
            best_row = row
    if best_row is None:
        return ""
    return str(best_row.get("raw_chunk_text", "") or best_row.get("contextual_chunk_text", ""))[:1500]


def _table_context_text(row: dict[str, Any]) -> str:
    parts = [
        "[문서 구조 요약]",
        f"- source_doc: {row.get('source_doc', '')}",
        f"- section_label: {row.get('section_label', '')}",
        f"- chunk_type: {row.get('chunk_type', '')}",
        f"- linked_parent_text: {row.get('linked_parent_text', '')}",
        f"- render_source: {row.get('render_source', '')}",
        f"- parser_source: {row.get('parser_source', '')}",
        f"- match_source: {row.get('match_source', '')}",
        f"- pairing_method: {row.get('pairing_method', '')}",
        f"- pairing_score: {row.get('pairing_score', '')}",
        f"- page_index: {row.get('bbox_page_index', '')}",
        f"- bbox: ({row.get('bbox_x1', '')}, {row.get('bbox_y1', '')}, {row.get('bbox_x2', '')}, {row.get('bbox_y2', '')})",
        "[/문서 구조 요약]",
        "",
        "[STRUCTURAL_TABLE]",
        str(row.get("structural_text", "")).strip(),
        "[/STRUCTURAL_TABLE]",
        "",
        "[TABLE_OCR_TEXT]",
        str(row.get("table_ocr_text", "")).strip(),
        "[/TABLE_OCR_TEXT]",
        "",
        "[TABLE_STRUCTURE]",
        str(row.get("table_html_or_markdown", "")).strip(),
        "[/TABLE_STRUCTURE]",
        "",
        "[NEARBY_BODY]",
        str(row.get("nearby_body_text", "")).strip(),
        "[/NEARBY_BODY]",
    ]
    return "\n".join(parts).strip()


def _render_hwp_to_pdf_com(*, source_path: Path, pdf_path: Path, timeout_sec: int = 180) -> tuple[bool, str]:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    helper = r"""
import sys
from pathlib import Path
import win32com.client as win32
import os
src = Path(sys.argv[1]).resolve()
dst = Path(sys.argv[2]).resolve()
hwp = None
try:
    hwp = win32.gencache.EnsureDispatch("HWPFrame.HwpObject")
    try:
        hwp.XHwpWindows.Item(0).Visible = False
    except Exception:
        pass
    module_candidates = []
    env_name = (os.getenv("HWP_FILEPATH_CHECKER_MODULE") or "").strip()
    if env_name:
        module_candidates.append(env_name)
    module_candidates.extend(["FilePathCheckerModuleExample", "FilePathCheckerModule"])
    registered = False
    for module_name in module_candidates:
        if not module_name:
            continue
        try:
            hwp.RegisterModule("FilePathCheckDLL", module_name)
            registered = True
            break
        except Exception:
            continue
    if not registered:
        print("WARN:RegisterModule failed for all module candidates")
    ok = hwp.Open(str(src), "", "forceopen:true;")
    if not bool(ok):
        print("OPEN_FALSE")
        sys.exit(2)
    hwp.SaveAs(str(dst), "PDF")
    if (not dst.exists()) or dst.stat().st_size <= 0:
        print("EMPTY_PDF")
        sys.exit(3)
    print("OK")
    sys.exit(0)
except Exception as e:
    print(f"ERR:{type(e).__name__}:{e}")
    sys.exit(1)
finally:
    if hwp is not None:
        try:
            hwp.Clear(1)
        except Exception:
            pass
        try:
            hwp.Quit()
        except Exception:
            pass
"""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", helper, str(source_path), str(pdf_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(30, int(timeout_sec)),
        )
    except subprocess.TimeoutExpired:
        return False, f"HWP COM render timeout ({timeout_sec}s)"

    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    if completed.returncode == 0 and pdf_path.exists() and pdf_path.stat().st_size > 0:
        return True, ""
    if stdout:
        return False, stdout
    if stderr:
        return False, stderr
    return False, f"HWP COM render failed (code={completed.returncode})"


def _classify_render_failure(message: str) -> str:
    text = str(message or "").lower()
    if "timeout" in text:
        return "timeout"
    popup_keywords = [
        "security",
        "filepathcheck",
        "filepathcheckdll",
        "permission",
        "권한",
        "보안",
        "차단",
        "open_false",
    ]
    if any(keyword in text for keyword in popup_keywords):
        return "popup_blocked"
    return "render_error"


def _extract_pdf_page_texts(pdf_path: Path) -> list[str]:
    doc = fitz.open(str(pdf_path))
    texts: list[str] = []
    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        texts.append(page.get_text("text"))
    return texts


def _score_table_to_page(structural_text: str, linked_parent_text: str, page_text: str) -> float:
    q_tokens = set(_tokenize(linked_parent_text))
    if len(q_tokens) < 3:
        q_tokens.update(_tokenize(structural_text)[:16])
    if not q_tokens:
        return 0.0
    p_tokens = set(_tokenize(page_text))
    if not p_tokens:
        return 0.0
    overlap = len(q_tokens & p_tokens)
    score = overlap / max(1, len(q_tokens))
    if linked_parent_text and linked_parent_text in page_text:
        score += 0.2
    return score


def _pick_target_pages(
    *,
    candidates: list[dict[str, Any]],
    page_texts: list[str],
    top_n_per_table: int,
    neighbor_window: int,
    min_score: float,
) -> tuple[set[int], dict[int, list[int]], dict[int, dict[int, float]]]:
    selected: set[int] = set()
    table_to_pages: dict[int, list[int]] = {}
    table_to_page_scores: dict[int, dict[int, float]] = {}
    for table in candidates:
        table_idx = int(table.get("table_index", 0) or 0)
        scored: list[tuple[int, float]] = []
        structural_text = str(table.get("text", "") or "")
        linked_parent = str(table.get("linked_parent_text", "") or "")
        for page_i, page_text in enumerate(page_texts, start=1):
            score = _score_table_to_page(structural_text, linked_parent, page_text)
            if score >= min_score:
                scored.append((page_i, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        top_pages = [item[0] for item in scored[: max(1, top_n_per_table)]]
        table_to_pages[table_idx] = top_pages
        table_to_page_scores[table_idx] = {page: score for page, score in scored[: max(1, top_n_per_table)]}
        for page in top_pages:
            for delta in range(-neighbor_window, neighbor_window + 1):
                p = page + delta
                if 1 <= p <= len(page_texts):
                    selected.add(p)
    if not selected:
        for page in range(1, min(8, len(page_texts)) + 1):
            selected.add(page)
    return selected, table_to_pages, table_to_page_scores


def _render_pdf_pages(
    *,
    pdf_path: Path,
    page_numbers: set[int],
    output_dir: Path,
    zoom: float,
) -> list[RenderedPage]:
    output_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    pages: list[RenderedPage] = []
    for page_number in sorted(page_numbers):
        page = doc.load_page(page_number - 1)
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        image_path = output_dir / f"page_{page_number:04d}.png"
        pix.save(str(image_path))
        pages.append(
            RenderedPage(
                page_index=page_number,
                page_text=page.get_text("text"),
                image_path=image_path,
                width=pix.width,
                height=pix.height,
            )
        )
    return pages


def _crop_bbox(*, page_image_path: Path, bbox: list[float], out_path: Path) -> tuple[bool, str]:
    try:
        with Image.open(page_image_path) as image:
            width, height = image.size
            x1 = max(0, min(width - 1, int(round(float(bbox[0])))))
            y1 = max(0, min(height - 1, int(round(float(bbox[1])))))
            x2 = max(0, min(width, int(round(float(bbox[2])))))
            y2 = max(0, min(height, int(round(float(bbox[3])))))
            if x2 <= x1 or y2 <= y1:
                return False, "invalid bbox"
            crop = image.crop((x1, y1, x2, y2))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(out_path)
            return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def _extract_table_regions_from_pp(
    *,
    result: Any,
    page_index: int,
    render_source: str,
    parser_source: str,
    min_region_score: float,
) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    if result is None:
        return regions
    try:
        layout = result.get("layout_det_res")
    except Exception:  # noqa: BLE001
        layout = None

    boxes: list[dict[str, Any]] = []
    if isinstance(layout, dict):
        maybe_boxes = layout.get("boxes", [])
        if isinstance(maybe_boxes, list):
            boxes = [item for item in maybe_boxes if isinstance(item, dict)]
    for order, box in enumerate(boxes, start=1):
        label = str(box.get("label", "")).strip().lower()
        if "table" not in label:
            continue
        score = float(box.get("score", 0.0) or 0.0)
        if score < float(min_region_score):
            continue
        coord = box.get("coordinate")
        if not isinstance(coord, list) or len(coord) != 4:
            continue
        regions.append(
            {
                "region_id": f"p{page_index}_layout_table_{order:03d}",
                "page_index": page_index,
                "label": label or "table",
                "score": score,
                "reading_order": order,
                "bbox": [float(coord[0]), float(coord[1]), float(coord[2]), float(coord[3])],
                "render_source": render_source,
                "parser_source": parser_source,
            }
        )
    return regions


def _table_structure_to_html(table_model: Any, crop_path: Path) -> tuple[str, float]:
    try:
        result = next(iter(table_model.predict(str(crop_path))))
        tokens = result.get("structure", []) if isinstance(result, dict) else []
        if isinstance(tokens, list):
            html = "".join(str(token) for token in tokens)
        else:
            html = str(tokens or "")
        score = float(result.get("structure_score", 0.0) or 0.0) if isinstance(result, dict) else 0.0
        return html.strip(), score
    except Exception:  # noqa: BLE001
        return "", 0.0


def _collect_pp_table_regions(
    *,
    rendered_pages: list[RenderedPage],
    output_dir: Path,
    pp_pipeline: Any,
    table_model: Any,
    min_ocr_region_score: float,
) -> tuple[list[TableRegion], dict[str, Any]]:
    regions: list[TableRegion] = []
    summary = {
        "page_count": len(rendered_pages),
        "pp_page_processed_count": 0,
        "pp_table_region_count": 0,
        "bbox_crop_success_count": 0,
        "bbox_crop_fail_count": 0,
        "table_ocr_ok_count": 0,
        "table_ocr_fail_count": 0,
        "table_structure_nonempty_count": 0,
        "pp_errors": [],
    }

    for page in rendered_pages:
        try:
            pp_result = next(iter(pp_pipeline.predict(input=str(page.image_path))))
        except Exception as exc:  # noqa: BLE001
            summary["pp_errors"].append(
                {
                    "page_index": page.page_index,
                    "error_type": "ppstructure_failed",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        summary["pp_page_processed_count"] += 1
        table_regions = _extract_table_regions_from_pp(
            result=pp_result,
            page_index=page.page_index,
            render_source="hwp_com_pdf",
            parser_source="ppstructurev3_layout_table_bbox",
            min_region_score=min_ocr_region_score,
        )
        summary["pp_table_region_count"] += len(table_regions)

        for item in table_regions:
            region_id = str(item["region_id"])
            crop_path = output_dir / "table_crops_bbox" / f"{region_id}.png"
            ok, err = _crop_bbox(page_image_path=page.image_path, bbox=item["bbox"], out_path=crop_path)
            if not ok:
                summary["bbox_crop_fail_count"] += 1
                summary["pp_errors"].append(
                    {
                        "page_index": page.page_index,
                        "error_type": "bbox_crop_failed",
                        "region_id": region_id,
                        "message": err,
                    }
                )
                continue

            summary["bbox_crop_success_count"] += 1
            ocr = run_image_ocr(crop_path)
            ocr_text = str(ocr.get("ocr_text", "")).strip()
            ocr_status = str(ocr.get("status", "")).strip() or "ocr_failed"
            if ocr_status == "ok" and ocr_text:
                summary["table_ocr_ok_count"] += 1
            else:
                summary["table_ocr_fail_count"] += 1
            table_html, structure_score = _table_structure_to_html(table_model=table_model, crop_path=crop_path)
            if table_html:
                summary["table_structure_nonempty_count"] += 1

            regions.append(
                TableRegion(
                    region_id=region_id,
                    page_index=int(item["page_index"]),
                    label=str(item.get("label", "table")),
                    score=float(item.get("score", 0.0) or 0.0),
                    reading_order=int(item.get("reading_order", 0) or 0),
                    bbox=[float(v) for v in item["bbox"]],
                    crop_path=crop_path,
                    ocr_text=ocr_text,
                    ocr_status=ocr_status,
                    table_html=table_html,
                    structure_score=float(structure_score or 0.0),
                    render_source=str(item.get("render_source", "hwp_com_pdf")),
                    parser_source=str(item.get("parser_source", "ppstructurev3_layout_table_bbox")),
                )
            )
    return regions, summary


def _match_table_to_region(
    *,
    table: dict[str, Any],
    regions: list[TableRegion],
    table_to_page_scores: dict[int, dict[int, float]],
) -> tuple[TableRegion | None, float]:
    structural_text = str(table.get("text", "") or "").strip()
    parent_text = str(table.get("linked_parent_text", "") or "").strip()
    struct_grams = _char_ngrams(structural_text, n=3)
    parent_tokens = set(_tokenize(parent_text))
    table_index = int(table.get("table_index", 0) or 0)
    page_scores = table_to_page_scores.get(table_index, {})

    best_region: TableRegion | None = None
    best_score = 0.0
    for region in regions:
        region_text = region.ocr_text
        grams = _char_ngrams(region_text, n=3)
        if not grams:
            continue
        inter = len(struct_grams & grams)
        union = len(struct_grams | grams)
        jaccard = inter / union if union else 0.0
        page_hint = float(page_scores.get(region.page_index, 0.0) or 0.0)
        parent_boost = 0.0
        if parent_tokens:
            overlap = len(parent_tokens & set(_tokenize(region_text)))
            parent_boost = min(0.25, overlap * 0.03)
        score = jaccard + (page_hint * 0.35) + parent_boost + (region.score * 0.1)
        if score > best_score:
            best_score = score
            best_region = region
    return best_region, round(best_score, 4)


def _make_table_row(
    *,
    doc_row: dict[str, Any],
    table: dict[str, Any],
    match_source: str,
    structural_text: str,
    table_ocr_text: str,
    table_html_or_markdown: str,
    nearby_body_text: str,
    ocr_status: str,
    ocr_confidence: float,
    render_source: str,
    parser_source: str,
    bbox_page_index: int,
    bbox: list[float],
    crop_path: str,
    region_match_score: float,
    v1_match_score: float,
    pairing_method: str,
    pairing_score: float,
    pairing_caption_score: float,
    pairing_same_section_score: float,
    pairing_semantic_score: float,
    pairing_source_chunk_id: str,
    pairing_source_section_title: str,
    pairing_source_section_path: str,
    part_index: int,
    part_count: int,
) -> dict[str, Any]:
    metadata = doc_row.get("metadata", {}) or {}
    parser_info = doc_row.get("parser_info", {}) or {}
    document_id = str(doc_row.get("document_id", "")).strip()
    source_file_name = str(doc_row.get("source_file_name", "")).strip()
    table_index = int(table.get("table_index", 0) or 0)
    doc_code = _doc_hash(document_id)
    part_suffix = f"_p{part_index:02d}" if part_count > 1 else ""
    chunk_id = f"trueocrv2_{doc_code}_table_{table_index:04d}{part_suffix}"
    linked_parent_text = str(table.get("linked_parent_text", "") or "").strip()
    section_label = linked_parent_text or (structural_text.splitlines()[0][:140] if structural_text else "")
    row = {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "source_doc": source_file_name,
        "source_file_name": source_file_name,
        "source_path": str(doc_row.get("source_path", "")),
        "source_extension": str(doc_row.get("source_extension", "")),
        "section_title": section_label,
        "section_path": str(table.get("section", "")),
        "section_label": section_label,
        "parent_section_label": linked_parent_text or section_label,
        "item_title": f"table_{table_index}",
        "linked_parent_text": linked_parent_text,
        "linked_parent_table_index": str(table.get("linked_parent_table_index", "") or ""),
        "source_position": str(table.get("record_start_index", "") or ""),
        "chunk_type": "table_true_ocr_v2",
        "chunk_role": "table_ocr",
        "chunk_role_tags": "table_true_ocr_v2",
        "extracted_from": "table",
        "parser_source": parser_source,
        "render_source": render_source,
        "match_source": match_source,
        "table_index": table_index,
        "structural_text": structural_text,
        "table_ocr_text": table_ocr_text,
        "table_html_or_markdown": table_html_or_markdown,
        "nearby_body_text": nearby_body_text,
        "pairing_method": str(pairing_method or ""),
        "pairing_score": round(float(pairing_score), 4),
        "pairing_caption_score": round(float(pairing_caption_score), 4),
        "pairing_same_section_score": round(float(pairing_same_section_score), 4),
        "pairing_semantic_score": round(float(pairing_semantic_score), 4),
        "pairing_source_chunk_id": str(pairing_source_chunk_id or ""),
        "pairing_source_section_title": str(pairing_source_section_title or ""),
        "pairing_source_section_path": str(pairing_source_section_path or ""),
        "ocr_status": ocr_status,
        "ocr_confidence": round(float(ocr_confidence), 4),
        "ocr_crop_path": crop_path,
        "bbox_page_index": int(bbox_page_index or 0),
        "bbox_x1": round(float(bbox[0]), 2) if len(bbox) == 4 else 0.0,
        "bbox_y1": round(float(bbox[1]), 2) if len(bbox) == 4 else 0.0,
        "bbox_x2": round(float(bbox[2]), 2) if len(bbox) == 4 else 0.0,
        "bbox_y2": round(float(bbox[3]), 2) if len(bbox) == 4 else 0.0,
        "region_match_score": round(float(region_match_score), 4),
        "legacy_v1_match_score": round(float(v1_match_score), 4),
        "raw_chunk_chars": len(structural_text),
        "table_part_index": int(part_index),
        "table_part_count": int(part_count),
        "has_table": 1,
        "is_high_priority": 1,
        "source_marker": "true_table_ocr_v2_augment",
        "parser_used": str(parser_info.get("parser_used", "")),
        "agency": str(metadata.get("발주 기관", "")),
        "issuer": str(metadata.get("발주 기관", "")),
        "project_name": str(metadata.get("사업명", "")),
        "budget_text": str(metadata.get("사업 금액", "")),
        "period_raw": str(metadata.get("사업 기간", "")),
        "deadline_text": "",
        "evaluation_text": "",
        "contract_method": "",
        "bid_method": "",
    }
    row["contextual_chunk_text"] = _table_context_text(row)
    row["contextual_chunk_chars"] = len(row["contextual_chunk_text"])
    row["has_budget_signal"] = _contains_budget(row["contextual_chunk_text"])
    row["has_schedule_signal"] = _contains_schedule(row["contextual_chunk_text"])
    row["has_contract_signal"] = _contains_contract(row["contextual_chunk_text"])
    return row


def _extract_cells_for_ast(table_html_or_markdown: str, table_ocr_text: str) -> tuple[list[str], list[dict[str, Any]]]:
    headers: list[str] = []
    cells: list[dict[str, Any]] = []
    html = str(table_html_or_markdown or "").strip()
    if html and ("<table" in html.lower() or "<tr" in html.lower()):
        try:
            soup = BeautifulSoup(html, "html.parser")
            table_tag = soup.find("table") or soup
            rows = table_tag.find_all("tr")
            for row_idx, row in enumerate(rows, start=1):
                col_idx = 0
                for cell in row.find_all(["th", "td"]):
                    col_idx += 1
                    value_text = " ".join(cell.get_text(" ", strip=True).split())
                    if not value_text:
                        continue
                    if row_idx == 1 and cell.name == "th":
                        headers.append(value_text)
                    cells.append(
                        {
                            "row_id": row_idx,
                            "col_id": col_idx,
                            "header_path": headers[col_idx - 1] if col_idx - 1 < len(headers) else "",
                            "span": {
                                "rowspan": int(cell.get("rowspan", 1) or 1),
                                "colspan": int(cell.get("colspan", 1) or 1),
                            },
                            "value_text": value_text,
                        }
                    )
            if cells:
                return headers, cells
        except Exception:
            pass

    fallback_lines = [line.strip() for line in str(table_ocr_text or "").splitlines() if line.strip()]
    for row_idx, line in enumerate(fallback_lines, start=1):
        cols = [part.strip() for part in re.split(r"[|,\t]", line) if part.strip()]
        if not cols:
            cols = [line]
        for col_idx, value_text in enumerate(cols, start=1):
            cells.append(
                {
                    "row_id": row_idx,
                    "col_id": col_idx,
                    "header_path": "",
                    "span": {"rowspan": 1, "colspan": 1},
                    "value_text": value_text,
                }
            )
    return headers, cells


def _coerce_value_num_date(value_text: str) -> tuple[float | None, str]:
    value = str(value_text or "").strip()
    num_match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", value.replace(" ", ""))
    value_num: float | None = None
    if num_match:
        raw_num = num_match.group(0).replace(",", "")
        try:
            value_num = float(raw_num)
        except Exception:
            value_num = None

    date_match = re.search(r"\d{4}[./-]\d{1,2}[./-]\d{1,2}", value)
    value_date = date_match.group(0) if date_match else ""
    return value_num, value_date


def _build_structured_evidence_records(
    *,
    doc_row: dict[str, Any],
    table: dict[str, Any],
    structural_text: str,
    table_ocr_text: str,
    table_html_or_markdown: str,
    nearby_body_text: str,
    ocr_status: str,
    ocr_confidence: float,
    render_source: str,
    parser_source: str,
    bbox_page_index: int,
    bbox: list[float],
    crop_path: str,
) -> dict[str, Any]:
    document_id = str(doc_row.get("document_id", "")).strip()
    source_file_name = str(doc_row.get("source_file_name", "")).strip()
    table_index = int(table.get("table_index", 0) or 0)
    section_label = str(table.get("linked_parent_text", "") or "").strip()
    if not section_label:
        section_label = (structural_text.splitlines()[0][:140] if structural_text else "")
    parent_section_label = section_label or str(table.get("section", "")).strip()

    headers, cells = _extract_cells_for_ast(table_html_or_markdown=table_html_or_markdown, table_ocr_text=table_ocr_text)
    grouped_rows: dict[int, list[dict[str, Any]]] = {}
    for cell in cells:
        grouped_rows.setdefault(int(cell["row_id"]), []).append(cell)

    ast_rows: list[dict[str, Any]] = []
    row_summary_chunks: list[dict[str, Any]] = []
    cell_row_blocks: list[dict[str, Any]] = []
    header_value_pairs: list[dict[str, Any]] = []

    for row_id in sorted(grouped_rows):
        row_cells = sorted(grouped_rows[row_id], key=lambda item: int(item["col_id"]))
        values = [str(cell["value_text"]) for cell in row_cells if str(cell["value_text"]).strip()]
        row_text = " | ".join(values)
        ast_rows.append({"row_id": row_id, "values": values})
        row_summary_chunks.append(
            {
                "document_id": document_id,
                "source_doc": source_file_name,
                "table_index": table_index,
                "row_id": row_id,
                "chunk_type": "row_summary",
                "section_label": section_label,
                "parent_section_label": parent_section_label,
                "text": row_text,
            }
        )
        cell_row_blocks.append(
            {
                "document_id": document_id,
                "source_doc": source_file_name,
                "table_index": table_index,
                "row_id": row_id,
                "chunk_type": "cell_row_block",
                "section_label": section_label,
                "parent_section_label": parent_section_label,
                "cells": row_cells,
                "text": row_text,
            }
        )
        for cell in row_cells:
            value_num, value_date = _coerce_value_num_date(str(cell["value_text"]))
            header_value_pairs.append(
                {
                    "document_id": document_id,
                    "source_doc": source_file_name,
                    "table_index": table_index,
                    "section_label": section_label,
                    "parent_section_label": parent_section_label,
                    "header_path": str(cell.get("header_path", "")),
                    "row_id": int(cell["row_id"]),
                    "col_id": int(cell["col_id"]),
                    "span": cell.get("span", {"rowspan": 1, "colspan": 1}),
                    "value_text": str(cell["value_text"]),
                    "value_num": value_num,
                    "value_date": value_date,
                }
            )

    table_ast = {
        "document_id": document_id,
        "source_doc": source_file_name,
        "table_index": table_index,
        "section_label": section_label,
        "parent_section_label": parent_section_label,
        "parser_source": parser_source,
        "render_source": render_source,
        "ocr_status": ocr_status,
        "ocr_confidence": round(float(ocr_confidence), 4),
        "bbox_page_index": int(bbox_page_index or 0),
        "bbox": [round(float(v), 2) for v in bbox] if len(bbox) == 4 else [0.0, 0.0, 0.0, 0.0],
        "crop_path": str(crop_path),
        "headers": headers,
        "rows": ast_rows,
        "structural_text": structural_text,
        "table_ocr_text": table_ocr_text,
        "table_html_or_markdown": table_html_or_markdown,
    }
    paired_body_chunk = {
        "document_id": document_id,
        "source_doc": source_file_name,
        "table_index": table_index,
        "section_label": section_label,
        "parent_section_label": parent_section_label,
        "chunk_type": "paired_body",
        "linked_parent_text": str(table.get("linked_parent_text", "") or ""),
        "text": str(nearby_body_text or ""),
    }
    return {
        "table_ast": table_ast,
        "header_value_pairs": header_value_pairs,
        "row_summary_chunks": row_summary_chunks,
        "cell_row_blocks": cell_row_blocks,
        "paired_body_chunks": [paired_body_chunk],
    }


def _structured_chunk_text(parts: list[str]) -> str:
    return "\n".join(str(part or "").strip() for part in parts if str(part or "").strip()).strip()


def _build_structured_chunk_rows(
    *,
    doc_row: dict[str, Any],
    table: dict[str, Any],
    table_row_template: dict[str, Any],
    evidence: dict[str, Any],
    pairing_method: str,
    pairing_score: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    doc_code = _doc_hash(str(doc_row.get("document_id", "")).strip())
    table_index = int(table.get("table_index", 0) or 0)
    table_prefix = f"trueocrv3_{doc_code}_table_{table_index:04d}"

    def base_row(chunk_id: str, chunk_type: str, has_table: int, text: str) -> dict[str, Any]:
        row = dict(table_row_template)
        row["chunk_id"] = chunk_id
        row["chunk_type"] = chunk_type
        row["chunk_role"] = "table_structured_evidence"
        row["chunk_role_tags"] = f"table_structured_evidence,{chunk_type}"
        row["has_table"] = int(has_table)
        row["is_high_priority"] = 1
        row["extracted_from"] = "table_structured"
        row["contextual_chunk_text"] = text[:6000]
        row["contextual_chunk_chars"] = len(row["contextual_chunk_text"])
        row["raw_chunk_chars"] = len(text)
        row["pairing_method"] = str(pairing_method or "")
        row["pairing_score"] = round(float(pairing_score), 4)
        row["has_budget_signal"] = _contains_budget(row["contextual_chunk_text"])
        row["has_schedule_signal"] = _contains_schedule(row["contextual_chunk_text"])
        row["has_contract_signal"] = _contains_contract(row["contextual_chunk_text"])
        return row

    for idx, hv in enumerate(evidence.get("header_value_pairs", []), start=1):
        header_path = str(hv.get("header_path", "") or "").strip()
        value_text = str(hv.get("value_text", "") or "").strip()
        if not value_text:
            continue
        value_num = hv.get("value_num")
        value_date = str(hv.get("value_date", "") or "")
        text = _structured_chunk_text(
            [
                "[HEADER_VALUE_PAIR]",
                f"header_path: {header_path}",
                f"row_id: {hv.get('row_id', '')}, col_id: {hv.get('col_id', '')}",
                f"value_text: {value_text}",
                f"value_num: {value_num if value_num is not None else ''}",
                f"value_date: {value_date}",
                f"section_label: {table_row_template.get('section_label', '')}",
                f"paired_body: {table_row_template.get('nearby_body_text', '')}",
                "[/HEADER_VALUE_PAIR]",
            ]
        )
        row = base_row(f"{table_prefix}_hv_{idx:04d}", "header_value_pair", 1, text)
        row["header_path"] = header_path
        row["row_id"] = int(hv.get("row_id", 0) or 0)
        row["col_id"] = int(hv.get("col_id", 0) or 0)
        row["span"] = json.dumps(hv.get("span", {"rowspan": 1, "colspan": 1}), ensure_ascii=False)
        row["value_text"] = value_text
        row["value_num"] = value_num if value_num is not None else ""
        row["value_date"] = value_date
        rows.append(row)

    for idx, block in enumerate(evidence.get("cell_row_blocks", []), start=1):
        row_text = str(block.get("text", "") or "").strip()
        if not row_text:
            continue
        text = _structured_chunk_text(
            [
                "[CELL_ROW_BLOCK]",
                f"row_id: {block.get('row_id', '')}",
                row_text,
                f"section_label: {table_row_template.get('section_label', '')}",
                f"paired_body: {table_row_template.get('nearby_body_text', '')}",
                "[/CELL_ROW_BLOCK]",
            ]
        )
        row = base_row(f"{table_prefix}_cellrow_{idx:04d}", "cell_row_block", 1, text)
        row["row_id"] = int(block.get("row_id", 0) or 0)
        rows.append(row)

    for idx, block in enumerate(evidence.get("row_summary_chunks", []), start=1):
        row_text = str(block.get("text", "") or "").strip()
        if not row_text:
            continue
        text = _structured_chunk_text(
            [
                "[ROW_SUMMARY]",
                f"row_id: {block.get('row_id', '')}",
                row_text,
                f"section_label: {table_row_template.get('section_label', '')}",
                "[/ROW_SUMMARY]",
            ]
        )
        row = base_row(f"{table_prefix}_rowsum_{idx:04d}", "row_summary_chunk", 1, text)
        row["row_id"] = int(block.get("row_id", 0) or 0)
        rows.append(row)

    paired_body_text = ""
    paired_rows = list(evidence.get("paired_body_chunks", []) or [])
    if paired_rows:
        paired_body_text = str(paired_rows[0].get("text", "") or "").strip()
    if paired_body_text:
        paired = base_row(f"{table_prefix}_pairbody", "paired_body_chunk", 0, _structured_chunk_text(
            [
                "[PAIRED_BODY]",
                f"pairing_method: {pairing_method}",
                f"pairing_score: {round(float(pairing_score), 4)}",
                paired_body_text,
                "[/PAIRED_BODY]",
            ]
        ))
        rows.append(paired)

    # HybridQA/TableFormer-like packed evidence chunks for table_plus_text retrieval.
    header_pairs = list(evidence.get("header_value_pairs", []) or [])
    if header_pairs and paired_body_text:
        top_pairs = header_pairs[:2]
        pair_lines = [
            f"- {str(item.get('header_path', '') or '')}: {str(item.get('value_text', '') or '')[:180]}"
            for item in top_pairs
        ]
        packed = base_row(
            f"{table_prefix}_tablebodypack",
            "table_body_pack",
            1,
            _structured_chunk_text(
                [
                    "[TABLE_BODY_PACK]",
                    f"pairing_method: {pairing_method}",
                    f"pairing_score: {round(float(pairing_score), 4)}",
                    "header_value_pairs:",
                    *pair_lines,
                    "paired_body:",
                    paired_body_text,
                    "[/TABLE_BODY_PACK]",
                ]
            ),
        )
        rows.append(packed)

    row_summaries = list(evidence.get("row_summary_chunks", []) or [])
    if row_summaries and paired_body_text:
        top_summary = row_summaries[0]
        packed = base_row(
            f"{table_prefix}_rowbodypack",
            "row_body_pack",
            1,
            _structured_chunk_text(
                [
                    "[ROW_BODY_PACK]",
                    f"pairing_method: {pairing_method}",
                    f"pairing_score: {round(float(pairing_score), 4)}",
                    f"row_summary: {str(top_summary.get('text', '') or '')[:600]}",
                    "paired_body:",
                    paired_body_text,
                    "[/ROW_BODY_PACK]",
                ]
            ),
        )
        rows.append(packed)

    return rows


def build_true_table_ocr_v2_rows(
    *,
    hwp_docs: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    output_root: Path,
    max_table_candidates_per_doc: int,
    min_region_match_score: float,
    refresh_html: bool,
    top_n_pages_per_table: int,
    page_neighbor_window: int,
    min_page_score: float,
    pdf_zoom: float,
    min_ocr_region_score: float,
    render_timeout_sec: int,
    include_structured_evidence_chunks: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    augment_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "target_hwp_doc_count": len(hwp_docs),
        "processed_doc_count": 0,
        "render_manifest": [],
        "render_success_count": 0,
        "render_fail_count": 0,
        "render_fallback_count": 0,
        "render_mode_counts": {"com_pdf_bbox": 0, "fallback_html": 0},
        "render_failure_reason_counts": {"popup_blocked": 0, "timeout": 0, "render_error": 0},
        "table_candidate_count": 0,
        "pp_table_region_count": 0,
        "pp_bbox_crop_success_count": 0,
        "pp_bbox_crop_fail_count": 0,
        "table_matched_to_pp_count": 0,
        "table_fallback_v1_count": 0,
        "table_fallback_structural_count": 0,
        "table_ocr_ok_count": 0,
        "table_ocr_fallback_count": 0,
        "structured_chunk_count": 0,
        "errors": [],
        "docs": [],
    }
    table_ast_records: list[dict[str, Any]] = []
    header_value_pairs: list[dict[str, Any]] = []
    row_summary_chunks: list[dict[str, Any]] = []
    cell_row_blocks: list[dict[str, Any]] = []
    paired_body_chunks: list[dict[str, Any]] = []
    body_by_file = _build_doc_body_index(baseline_rows)
    pp_pipeline = PPStructureV3(device="cpu")
    table_model = TableStructureRecognition(model_name="SLANet_plus")

    for doc_row in hwp_docs:
        t0 = time.time()
        document_id = str(doc_row.get("document_id", "")).strip()
        source_file_name = str(doc_row.get("source_file_name", "")).strip()
        source_path = Path(str(doc_row.get("source_path", "")).strip())
        doc_hash = _doc_hash(document_id)
        doc_dir = output_root / "ocr_work_v2" / doc_hash
        html_path = doc_dir / "rendered.html"
        pdf_path = doc_dir / "rendered_from_hwp_com.pdf"
        pages_dir = doc_dir / "pages"
        pp_dir = doc_dir / "pp_outputs"

        render_info = {
            "document_id": document_id,
            "source_file_name": source_file_name,
            "render_tool": "hwp_com_saveas_pdf",
            "render_mode": "com_pdf_bbox",
            "render_source": "com_pdf_bbox",
            "render_ok": 0,
            "render_error": "",
            "render_failure_reason": "",
            "fallback_used": 0,
            "fallback_tool": "",
            "fallback_error": "",
        }
        render_ok, render_err = _render_hwp_to_pdf_com(
            source_path=source_path,
            pdf_path=pdf_path,
            timeout_sec=max(30, int(render_timeout_sec)),
        )
        if render_ok:
            summary["render_success_count"] += 1
            summary["render_mode_counts"]["com_pdf_bbox"] += 1
            render_info["render_ok"] = 1
        else:
            summary["render_fail_count"] += 1
            failure_reason = _classify_render_failure(render_err)
            summary["render_failure_reason_counts"][failure_reason] += 1
            render_info["render_error"] = render_err
            render_info["render_failure_reason"] = failure_reason
            render_info["fallback_used"] = 1
            render_info["fallback_tool"] = "hwp5html_text_page_render"
            render_info["render_mode"] = "fallback_html"
            render_info["render_source"] = "fallback_html"
            summary["render_fallback_count"] += 1
            summary["render_mode_counts"]["fallback_html"] += 1

        hwp5html_ok, hwp5html_err = _run_hwp5html(source_path=source_path, html_path=html_path, refresh=refresh_html)
        if not hwp5html_ok:
            summary["errors"].append(
                {
                    "document_id": document_id,
                    "source_file_name": source_file_name,
                    "error_type": "hwp5html_failed",
                    "message": hwp5html_err,
                }
            )
        html_tables = _parse_html_tables(html_path) if hwp5html_ok else []
        payload = extract_hwp_artifacts(source_path, save_images=False, output_dir=doc_dir)
        candidates = sorted(
            payload.get("table_ocr_candidates", []),
            key=lambda item: int(item.get("ocr_candidate_score", 0) or 0),
            reverse=True,
        )
        if max_table_candidates_per_doc > 0:
            candidates = candidates[: max_table_candidates_per_doc]
        summary["table_candidate_count"] += len(candidates)

        table_to_pages: dict[int, list[int]] = {}
        table_to_page_scores: dict[int, dict[int, float]] = {}
        rendered_pages: list[RenderedPage] = []
        pp_regions: list[TableRegion] = []
        pp_stats: dict[str, Any] = {}

        if render_ok and pdf_path.exists():
            try:
                page_texts = _extract_pdf_page_texts(pdf_path)
                page_numbers, table_to_pages, table_to_page_scores = _pick_target_pages(
                    candidates=candidates,
                    page_texts=page_texts,
                    top_n_per_table=max(1, top_n_pages_per_table),
                    neighbor_window=max(0, page_neighbor_window),
                    min_score=max(0.0, min_page_score),
                )
                rendered_pages = _render_pdf_pages(
                    pdf_path=pdf_path,
                    page_numbers=page_numbers,
                    output_dir=pages_dir,
                    zoom=max(1.0, pdf_zoom),
                )
                pp_regions, pp_stats = _collect_pp_table_regions(
                    rendered_pages=rendered_pages,
                    output_dir=pp_dir,
                    pp_pipeline=pp_pipeline,
                    table_model=table_model,
                    min_ocr_region_score=min_ocr_region_score,
                )
                summary["pp_table_region_count"] += int(pp_stats.get("pp_table_region_count", 0) or 0)
                summary["pp_bbox_crop_success_count"] += int(pp_stats.get("bbox_crop_success_count", 0) or 0)
                summary["pp_bbox_crop_fail_count"] += int(pp_stats.get("bbox_crop_fail_count", 0) or 0)
            except Exception as exc:  # noqa: BLE001
                summary["errors"].append(
                    {
                        "document_id": document_id,
                        "source_file_name": source_file_name,
                        "error_type": "pp_render_or_infer_failed",
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                )

        doc_body_rows = body_by_file.get(source_file_name, [])
        added_before = len(augment_rows)
        doc_table_matched_pp = 0
        doc_table_fallback_v1 = 0
        doc_table_fallback_structural = 0

        for table in candidates:
            structural_text = str(table.get("text", "") or "").strip()
            linked_parent = str(table.get("linked_parent_text", "") or "").strip()
            pairing = _pair_body_text(
                body_rows=doc_body_rows,
                table_row=table,
                structural_text=structural_text,
                linked_parent_text=linked_parent,
            )
            nearby_body = str(pairing.get("text", "") or "").strip()
            method_scores = dict(pairing.get("method_scores", {}) or {})

            best_region, region_score = _match_table_to_region(
                table=table,
                regions=pp_regions,
                table_to_page_scores=table_to_page_scores,
            )
            matched_html_table, v1_match_score = _match_html_table(
                structural_text=structural_text,
                linked_parent_text=linked_parent,
                html_tables=html_tables,
            )

            match_source = "structural_only"
            table_ocr_text = structural_text
            table_html_or_markdown = structural_text
            ocr_status = "fallback_structural_text"
            ocr_confidence = 0.25
            render_source = render_info["render_source"]
            parser_source = "extract_hwp_artifacts"
            bbox_page_index = 0
            bbox = [0.0, 0.0, 0.0, 0.0]
            crop_path = ""

            if best_region is not None and region_score >= min_region_match_score and best_region.ocr_text.strip():
                match_source = "ppstructure_bbox_crop"
                table_ocr_text = best_region.ocr_text.strip()
                table_html_or_markdown = best_region.table_html.strip() or table_ocr_text
                ocr_status = best_region.ocr_status or "ok"
                ocr_confidence = min(0.99, max(0.35, (best_region.score * 0.6) + (best_region.structure_score * 0.4)))
                render_source = best_region.render_source
                parser_source = f"{best_region.parser_source}+table_structure_recognition_v2"
                bbox_page_index = best_region.page_index
                bbox = best_region.bbox
                crop_path = str(best_region.crop_path)
                summary["table_matched_to_pp_count"] += 1
                summary["table_ocr_ok_count"] += int(ocr_status == "ok")
                doc_table_matched_pp += 1
            elif matched_html_table is not None and v1_match_score >= 0.22:
                match_source = "v1_hwp5html_table_match_fallback"
                table_ocr_text = matched_html_table.text.strip()
                table_html_or_markdown = matched_html_table.text.strip()
                ocr_status = "fallback_html_table_text"
                ocr_confidence = 0.5
                render_source = "hwp5html_table_text"
                parser_source = "hwp5html_v1_legacy_fallback"
                summary["table_fallback_v1_count"] += 1
                summary["table_ocr_fallback_count"] += 1
                doc_table_fallback_v1 += 1
            else:
                summary["table_fallback_structural_count"] += 1
                summary["table_ocr_fallback_count"] += 1
                doc_table_fallback_structural += 1

            merged_text = "\n".join(
                part for part in [structural_text, table_ocr_text, table_html_or_markdown, nearby_body] if part.strip()
            ).strip()
            parts = _split_text(merged_text, max_chars=1400, overlap=120)
            if not parts:
                continue
            for part_index, _ in enumerate(parts, start=1):
                row = _make_table_row(
                    doc_row=doc_row,
                    table=table,
                    match_source=match_source,
                    structural_text=structural_text,
                    table_ocr_text=table_ocr_text,
                    table_html_or_markdown=table_html_or_markdown,
                    nearby_body_text=nearby_body,
                    ocr_status=ocr_status,
                    ocr_confidence=ocr_confidence,
                    render_source=render_source,
                    parser_source=parser_source,
                    bbox_page_index=bbox_page_index,
                    bbox=bbox,
                    crop_path=crop_path,
                    region_match_score=region_score,
                    v1_match_score=v1_match_score,
                    pairing_method=str(pairing.get("method", "") or ""),
                    pairing_score=float(pairing.get("score", 0.0) or 0.0),
                    pairing_caption_score=float(method_scores.get("caption_window", 0.0) or 0.0),
                    pairing_same_section_score=float(method_scores.get("same_section_window", 0.0) or 0.0),
                    pairing_semantic_score=float(method_scores.get("semantic_nearest_paragraph", 0.0) or 0.0),
                    pairing_source_chunk_id=str(pairing.get("source_chunk_id", "") or ""),
                    pairing_source_section_title=str(pairing.get("source_section_title", "") or ""),
                    pairing_source_section_path=str(pairing.get("source_section_path", "") or ""),
                    part_index=part_index,
                    part_count=len(parts),
                )
                augment_rows.append(row)

            evidence = _build_structured_evidence_records(
                doc_row=doc_row,
                table=table,
                structural_text=structural_text,
                table_ocr_text=table_ocr_text,
                table_html_or_markdown=table_html_or_markdown,
                nearby_body_text=nearby_body,
                ocr_status=ocr_status,
                ocr_confidence=ocr_confidence,
                render_source=render_source,
                parser_source=parser_source,
                bbox_page_index=bbox_page_index,
                bbox=bbox,
                crop_path=crop_path,
            )
            table_ast_records.append(evidence["table_ast"])
            header_value_pairs.extend(evidence["header_value_pairs"])
            row_summary_chunks.extend(evidence["row_summary_chunks"])
            cell_row_blocks.extend(evidence["cell_row_blocks"])
            paired_body_chunks.extend(evidence["paired_body_chunks"])

            if include_structured_evidence_chunks:
                template_row = {
                    "document_id": str(doc_row.get("document_id", "")).strip(),
                    "source_doc": source_file_name,
                    "source_file_name": source_file_name,
                    "source_path": str(doc_row.get("source_path", "")),
                    "source_extension": str(doc_row.get("source_extension", "")),
                    "section_title": str(table.get("linked_parent_text", "") or ""),
                    "section_path": str(table.get("section", "") or ""),
                    "section_label": str(table.get("linked_parent_text", "") or ""),
                    "parent_section_label": str(table.get("linked_parent_text", "") or ""),
                    "item_title": f"table_{int(table.get('table_index', 0) or 0)}",
                    "linked_parent_text": str(table.get("linked_parent_text", "") or ""),
                    "linked_parent_table_index": str(table.get("linked_parent_table_index", "") or ""),
                    "source_position": str(table.get("record_start_index", "") or ""),
                    "extracted_from": "table_structured",
                    "parser_source": parser_source,
                    "render_source": render_source,
                    "match_source": match_source,
                    "table_index": int(table.get("table_index", 0) or 0),
                    "structural_text": structural_text,
                    "table_ocr_text": table_ocr_text,
                    "table_html_or_markdown": table_html_or_markdown,
                    "nearby_body_text": nearby_body,
                    "ocr_status": ocr_status,
                    "ocr_confidence": round(float(ocr_confidence), 4),
                    "ocr_crop_path": crop_path,
                    "bbox_page_index": int(bbox_page_index or 0),
                    "bbox_x1": round(float(bbox[0]), 2) if len(bbox) == 4 else 0.0,
                    "bbox_y1": round(float(bbox[1]), 2) if len(bbox) == 4 else 0.0,
                    "bbox_x2": round(float(bbox[2]), 2) if len(bbox) == 4 else 0.0,
                    "bbox_y2": round(float(bbox[3]), 2) if len(bbox) == 4 else 0.0,
                    "region_match_score": round(float(region_score), 4),
                    "legacy_v1_match_score": round(float(v1_match_score), 4),
                    "source_marker": "true_table_ocr_v3_structured_augment",
                    "parser_used": str((doc_row.get("parser_info", {}) or {}).get("parser_used", "")),
                    "agency": str((doc_row.get("metadata", {}) or {}).get("발주 기관", "")),
                    "issuer": str((doc_row.get("metadata", {}) or {}).get("발주 기관", "")),
                    "project_name": str((doc_row.get("metadata", {}) or {}).get("사업명", "")),
                    "budget_text": str((doc_row.get("metadata", {}) or {}).get("사업 금액", "")),
                    "period_raw": str((doc_row.get("metadata", {}) or {}).get("사업 기간", "")),
                    "deadline_text": "",
                    "evaluation_text": "",
                    "contract_method": "",
                    "bid_method": "",
                }
                structured_rows = _build_structured_chunk_rows(
                    doc_row=doc_row,
                    table=table,
                    table_row_template=template_row,
                    evidence=evidence,
                    pairing_method=str(pairing.get("method", "") or ""),
                    pairing_score=float(pairing.get("score", 0.0) or 0.0),
                )
                augment_rows.extend(structured_rows)
                summary["structured_chunk_count"] += len(structured_rows)

        summary["processed_doc_count"] += 1
        summary["render_manifest"].append(render_info)
        summary["docs"].append(
            {
                "document_id": document_id,
                "source_file_name": source_file_name,
                "table_candidate_count": len(candidates),
                "pdf_render_ok": render_info["render_ok"],
                "render_mode": render_info["render_mode"],
                "render_failure_reason": render_info["render_failure_reason"],
                "render_source": render_info["render_source"],
                "selected_page_count": len(rendered_pages),
                "pp_region_count": len(pp_regions),
                "table_matched_to_pp_count": doc_table_matched_pp,
                "table_fallback_v1_count": doc_table_fallback_v1,
                "table_fallback_structural_count": doc_table_fallback_structural,
                "augment_rows_added": len(augment_rows) - added_before,
                "elapsed_sec": round(time.time() - t0, 2),
                "table_to_pages": {str(k): v for k, v in table_to_pages.items()},
                "pp_stats": pp_stats,
            }
        )

    deduped: dict[str, dict[str, Any]] = {}
    for row in augment_rows:
        deduped[row["chunk_id"]] = row
    augment_rows = list(deduped.values())
    summary["augment_row_count"] = len(augment_rows)
    summary["com_pdf_bbox_success_rate"] = round(
        float(summary["render_success_count"]) / max(1, int(summary["target_hwp_doc_count"])),
        4,
    )
    summary["render_stability_gate_threshold"] = 0.8
    summary["render_stability_gate_passed"] = int(summary["com_pdf_bbox_success_rate"] >= 0.8)
    summary["structured_evidence"] = {
        "table_ast": table_ast_records,
        "header_value_pairs": header_value_pairs,
        "row_summary_chunks": row_summary_chunks,
        "cell_row_blocks": cell_row_blocks,
        "paired_body_chunks": paired_body_chunks,
    }
    return augment_rows, summary


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Build true_table_ocr_v2 augment corpus: structural extraction + HWP COM PDF render + PP-StructureV3 bbox crop + "
            "table OCR + table structure recognition."
        )
    )
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument("--output-root", default=str(root / "rag_outputs" / "phase2_true_table_ocr_v2_assets"))
    parser.add_argument("--chroma-output-dir", default=str(root.parent / "rfp_rag_chroma_db_phase2_true_table_ocr_v2"))
    parser.add_argument(
        "--table-eval-set-path",
        default=str(root / "rag_outputs" / "eval_sets" / "b05_table_eval_questions_v2.csv"),
    )
    parser.add_argument(
        "--groupbc-eval-set-path",
        default=str(root / "rag_outputs" / "eval_sets" / "b05_group_bc_questions_v1.csv"),
    )
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--embedding-backend-key", default=EMBEDDING_BACKEND_KEY)
    parser.add_argument("--collection-name", default=COLLECTION_NAME)
    parser.add_argument("--bm25-index-name", default=BM25_INDEX_NAME)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--copy-batch-size", type=int, default=256)
    parser.add_argument("--max-table-candidates-per-doc", type=int, default=12)
    parser.add_argument("--min-region-match-score", type=float, default=0.08)
    parser.add_argument("--top-n-pages-per-table", type=int, default=2)
    parser.add_argument("--page-neighbor-window", type=int, default=1)
    parser.add_argument("--min-page-score", type=float, default=0.03)
    parser.add_argument("--min-ocr-region-score", type=float, default=0.35)
    parser.add_argument("--pdf-zoom", type=float, default=1.6)
    parser.add_argument("--render-timeout-sec", type=int, default=180)
    parser.add_argument("--refresh-html", action="store_true")
    parser.add_argument("--reset-collection", action="store_true")
    parser.add_argument("--reuse-existing-assets", action="store_true")
    parser.add_argument("--include-structured-evidence-chunks", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_root).resolve()
    chroma_output_dir = Path(args.chroma_output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    baseline_rows = _load_baseline_chunks(project_root)
    processed_rows = _load_processed_documents(project_root)
    table_eval_path = Path(args.table_eval_set_path).resolve()
    group_eval_path = Path(args.groupbc_eval_set_path).resolve()
    target_names = _read_target_doc_names([table_eval_path, group_eval_path])
    target_hwp_docs = _select_target_hwp_docs(processed_rows, target_names)

    baseline_pipeline = ScenarioACommonPipeline(PipelinePaths(project_root=project_root))
    embedding_config = baseline_pipeline.load_embedding_config("openai_text_embedding_3_small")
    baseline_collection_name = embedding_config.collection_name

    augment_jsonl = output_root / "true_table_ocr_v2_augment.jsonl"
    combined_jsonl = output_root / "processed_data_hwp_true_table_ocr_v2.jsonl"
    build_summary: dict[str, Any] = {}
    if args.reuse_existing_assets and augment_jsonl.exists() and combined_jsonl.exists():
        augment_rows = _read_jsonl(augment_jsonl)
        combined_rows = _read_jsonl(combined_jsonl)
        build_summary = {
            "reuse_existing_assets": True,
            "augment_jsonl_loaded": str(augment_jsonl),
            "combined_jsonl_loaded": str(combined_jsonl),
        }
    else:
        augment_rows, build_summary = build_true_table_ocr_v2_rows(
            hwp_docs=target_hwp_docs,
            baseline_rows=baseline_rows,
            output_root=output_root,
            max_table_candidates_per_doc=max(1, args.max_table_candidates_per_doc),
            min_region_match_score=max(0.0, args.min_region_match_score),
            refresh_html=bool(args.refresh_html),
            top_n_pages_per_table=max(1, args.top_n_pages_per_table),
            page_neighbor_window=max(0, args.page_neighbor_window),
            min_page_score=max(0.0, args.min_page_score),
            pdf_zoom=max(1.0, args.pdf_zoom),
            min_ocr_region_score=max(0.0, args.min_ocr_region_score),
            render_timeout_sec=max(30, int(args.render_timeout_sec)),
            include_structured_evidence_chunks=bool(args.include_structured_evidence_chunks),
        )
        combined_rows = [*baseline_rows, *augment_rows]
        _write_jsonl(augment_jsonl, augment_rows)
        _write_jsonl(combined_jsonl, combined_rows)

    structured = build_summary.get("structured_evidence")
    if isinstance(structured, dict):
        evidence_dir = output_root / "evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        table_ast_path = evidence_dir / "table_ast.json"
        header_value_pairs_path = evidence_dir / "header_value_pairs.jsonl"
        row_summary_chunks_path = evidence_dir / "row_summary_chunks.jsonl"
        cell_row_blocks_path = evidence_dir / "cell_row_blocks.jsonl"
        paired_body_chunks_path = evidence_dir / "paired_body_chunks.jsonl"

        _write_json(table_ast_path, {"tables": structured.get("table_ast", [])})
        _write_jsonl(header_value_pairs_path, list(structured.get("header_value_pairs", [])))
        _write_jsonl(row_summary_chunks_path, list(structured.get("row_summary_chunks", [])))
        _write_jsonl(cell_row_blocks_path, list(structured.get("cell_row_blocks", [])))
        _write_jsonl(paired_body_chunks_path, list(structured.get("paired_body_chunks", [])))
        build_summary["structured_evidence_files"] = {
            "table_ast_json": str(table_ast_path),
            "header_value_pairs_jsonl": str(header_value_pairs_path),
            "row_summary_chunks_jsonl": str(row_summary_chunks_path),
            "cell_row_blocks_jsonl": str(cell_row_blocks_path),
            "paired_body_chunks_jsonl": str(paired_body_chunks_path),
        }
        build_summary.pop("structured_evidence", None)

    bm25_path = project_root / "rag_outputs" / str(args.bm25_index_name)
    if args.reuse_existing_assets and bm25_path.exists():
        pass
    else:
        bm25_payload = _build_bm25_index(combined_rows)
        _write_bm25_index(bm25_path, bm25_payload)

    chroma = _build_chroma(
        project_root=project_root,
        augment_rows=augment_rows,
        chroma_output_dir=chroma_output_dir,
        baseline_collection_name=baseline_collection_name,
        target_collection_name=str(args.collection_name),
        embedding_model=args.embedding_model,
        embedding_batch_size=max(1, args.embedding_batch_size),
        copy_batch_size=max(1, args.copy_batch_size),
        reset_collection=bool(args.reset_collection),
    )

    summary = {
        "project_root": str(project_root),
        "embedding_backend_key": str(args.embedding_backend_key),
        "collection_name": str(args.collection_name),
        "bm25_index_name": str(args.bm25_index_name),
        "baseline_row_count": len(baseline_rows),
        "augment_row_count": len(augment_rows),
        "combined_row_count": len(combined_rows),
        "target_hwp_docs": [str(row.get("source_file_name", "")) for row in target_hwp_docs],
        "table_eval_set_path": str(table_eval_path),
        "groupbc_eval_set_path": str(group_eval_path),
        "augment_jsonl": str(augment_jsonl),
        "combined_jsonl": str(combined_jsonl),
        "bm25_index_path": str(bm25_path),
        "chroma": chroma,
        "build_summary": build_summary,
        "true_table_ocr_v2_manifest": {
            "render_pipeline": [
                "extract_hwp_artifacts structural_table/final_review_table",
                "HWP COM SaveAs PDF render (primary)",
                "PP-StructureV3 layout table bbox detection",
                "bbox crop OCR + TableStructureRecognition(SLANet_plus)",
            ],
            "fallback_pipeline": [
                "hwp5html table text matching(v1 compatible)",
                "structural text fallback",
            ],
            "replace_policy": "augment_only",
            "output_format": {
                "ppstructure": "json(dict-like result fields)",
                "table_structure": "html token sequence",
                "bbox_metadata": "page_index + x1,y1,x2,y2",
            },
        },
    }
    summary_path = output_root / "phase2_true_table_ocr_v2_summary.json"
    _write_json(summary_path, summary)

    print(f"[done] target_hwp_docs={len(target_hwp_docs)}")
    print(f"[done] augment_rows={len(augment_rows)} combined_rows={len(combined_rows)}")
    print(f"[done] bm25={bm25_path}")
    print(f"[done] collection={args.collection_name} chroma_dir={chroma_output_dir}")
    print(f"[done] summary={summary_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import base64
import csv
import hashlib
import html as html_lib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scenario_a.common_pipeline import PipelinePaths
from scenario_b_phase2.answer_type_router import (
    ANSWER_TYPES,
    apply_answer_type_override,
    predict as predict_answer_type,
)
from scenario_b_phase2.phase2_pipeline import ScenarioBPhase2Pipeline
from streamlit_qa.config import (
    DEFAULT_PHASE2_PROFILE,
    build_phase2_options,
    build_pipeline_settings,
    load_embedding_backend_keys,
    load_local_model_specs,
    load_phase2_profiles,
)
from streamlit_qa.providers import ProviderSelection, create_provider_adapter
from streamlit_qa.rag_service import (
    RunResult,
    build_attachment_context,
    build_history_for_adapter,
    run_chat_turn,
    run_rag_turn_with_pipeline,
)
from streamlit_qa.storage import SessionStore


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_filename(name: str) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(name or ""))
    return sanitized or f"upload_{uuid4().hex[:8]}"


def decode_bytes(raw: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def extract_pdf_preview(path: Path, page_limit: int = 4) -> str:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError:
        return "pypdf가 설치되지 않아 PDF 본문 미리보기를 생성하지 못했습니다."
    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # noqa: BLE001
        return f"PDF 파싱 실패: {exc}"
    blocks: list[str] = []
    for index, page in enumerate(reader.pages[:page_limit], start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:  # noqa: BLE001
            text = ""
        if text:
            blocks.append(f"[page {index}]\n{text}")
    return "\n\n".join(blocks)[:6000]


def extract_preview_text(path: Path, ext: str) -> str:
    if ext in {".txt", ".md", ".json", ".csv"}:
        raw = path.read_bytes()
        text = decode_bytes(raw)
        if ext == ".json":
            try:
                payload = json.loads(text)
                text = json.dumps(payload, ensure_ascii=False, indent=2)
            except Exception:  # noqa: BLE001
                pass
        if ext == ".csv":
            rows = list(csv.reader(text.splitlines()))
            lines = [", ".join(row) for row in rows[:40]]
            text = "\n".join(lines)
        return text[:6000]
    if ext == ".pdf":
        return extract_pdf_preview(path)
    return ""


def ingest_uploads(session: dict[str, Any], store: SessionStore, uploaded_files: list[Any]) -> int:
    if not uploaded_files:
        return 0
    uploaded = 0
    existing_hashes = {str(item.get("sha1", "")) for item in session.get("attachments", []) or []}
    target_dir = store.session_upload_path(session["id"])
    for item in uploaded_files:
        raw = item.getvalue()
        sha1 = hashlib.sha1(raw).hexdigest()
        if sha1 in existing_hashes:
            continue
        original_name = item.name or f"file_{uuid4().hex[:8]}"
        file_name = f"{sha1[:10]}_{safe_filename(original_name)}"
        path = target_dir / file_name
        path.write_bytes(raw)
        ext = path.suffix.lower()
        preview = extract_preview_text(path, ext)
        session.setdefault("attachments", [])
        session["attachments"].append(
            {
                "id": f"att_{uuid4().hex[:10]}",
                "name": original_name,
                "saved_path": str(path),
                "size_bytes": len(raw),
                "ext": ext,
                "sha1": sha1,
                "included": True,
                "preview_text": preview,
                "created_at": now_iso(),
            }
        )
        existing_hashes.add(sha1)
        uploaded += 1
    return uploaded


def build_provider_selection(payload: dict[str, Any]) -> ProviderSelection:
    return ProviderSelection(
        provider=str(payload.get("provider", "openai_api")),
        model_name=str(payload.get("model_name", "gpt-5-mini")),
        api_key=str(payload.get("api_key", "")),
        base_url=str(payload.get("base_url", "")),
        max_new_tokens=int(payload.get("max_new_tokens", 768)),
        temperature=float(payload.get("temperature", 0.0)),
        top_p=float(payload.get("top_p", 1.0)),
        local_model_ui_key=str(payload.get("local_model_ui_key", "")),
    )


def model_label(selection: ProviderSelection) -> str:
    return f"{selection.provider}:{selection.model_name}"


def run_result_to_payload(result: RunResult, label: str) -> dict[str, Any]:
    return {
        "id": f"run_{uuid4().hex[:10]}",
        "label": label,
        "content": result.answer_text,
        "latency_sec": result.latency_sec,
        "mode": result.mode,
        "model_label": result.model_label,
        "route": result.route,
        "profile": result.profile,
        "top_k": result.top_k,
        "chunks": result.chunks,
        "retrieval_context": result.retrieval_context,
    }


def extract_artifacts_from_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for run in runs:
        content = str(run.get("content", ""))
        is_long = len(content) >= 1000
        looks_structured = ("```" in content) or ("\n|" in content)
        if not (is_long or looks_structured):
            continue
        artifacts.append(
            {
                "id": f"artifact_{uuid4().hex[:10]}",
                "title": f"{run.get('label', 'Result')} · {run.get('model_label', '')}",
                "content": content,
                "created_at": now_iso(),
            }
        )
    return artifacts


def get_last_user_prompt(messages: list[dict[str, Any]]) -> str:
    for row in reversed(messages):
        if str(row.get("role", "")) == "user" and str(row.get("content", "")).strip():
            return str(row.get("content", "")).strip()
    return ""


def session_messages_for_history(messages: list[dict[str, Any]], *, drop_last_assistant: bool) -> list[dict[str, Any]]:
    copied = list(messages)
    if drop_last_assistant:
        for index in range(len(copied) - 1, -1, -1):
            if str(copied[index].get("role", "")) == "assistant":
                copied.pop(index)
                break
    return copied


def stream_text_chunks(text: str, *, chunk_words: int = 5, delay_sec: float = 0.02):
    words = str(text or "").split(" ")
    if not words:
        return
    for index in range(0, len(words), chunk_words):
        piece = " ".join(words[index : index + chunk_words]).strip()
        if piece:
            yield piece + " "
            time.sleep(delay_sec)


def get_source_file_index() -> dict[str, str]:
    cache = st.session_state.get("_source_file_index")
    if isinstance(cache, dict):
        return cache

    index: dict[str, str] = {}
    for base in [PROJECT_ROOT / "files", PROJECT_ROOT / "processed_data", PROJECT_ROOT / "docs"]:
        if not base.exists():
            continue
        try:
            for path in base.rglob("*"):
                if not path.is_file():
                    continue
                key = path.name.lower()
                index.setdefault(key, str(path.resolve()))
        except Exception:  # noqa: BLE001
            continue

    st.session_state["_source_file_index"] = index
    return index


def resolve_source_path(source_name: str) -> str:
    name = str(source_name or "").strip()
    if not name:
        return ""
    key = Path(name).name.lower()
    index = get_source_file_index()
    if key in index:
        return index[key]
    for file_name, path in index.items():
        if key and key in file_name:
            return path
    return ""


def compute_run_scores(run: dict[str, Any]) -> tuple[int, int]:
    chunks = list(run.get("chunks", []) or [])
    if not chunks:
        return 24, 0

    raw_scores: list[float] = []
    crag_hits = 0
    for row in chunks:
        value = row.get("adjusted_score", row.get("fusion_score", 0.0))
        try:
            raw_scores.append(float(value))
        except (TypeError, ValueError):
            raw_scores.append(0.0)
        label = str(row.get("crag_label", "")).upper().strip()
        if label in {"CORRECT", "SOFT", "AMBIGUOUS"}:
            crag_hits += 1

    top_scores = sorted(raw_scores, reverse=True)[:3]
    avg_top = sum(top_scores) / max(1, len(top_scores))
    relevance = int(max(0.0, min(100.0, (1 - math.exp(-avg_top * 220.0)) * 100)))

    crag_ratio = crag_hits / max(1, len(chunks))
    coverage = min(1.0, len(chunks) / max(1, int(run.get("top_k", len(chunks)) or len(chunks))))
    confidence = int(max(0.0, min(100.0, (0.58 * (relevance / 100.0) + 0.26 * crag_ratio + 0.16 * coverage) * 100)))
    return confidence, relevance


def render_source_citations(run: dict[str, Any]) -> None:
    chunks = list(run.get("chunks", []) or [])
    if not chunks:
        return
    docs: dict[str, list[float]] = {}
    for row in chunks:
        name = str(row.get("source_file_name", "")).strip()
        if not name:
            continue
        score_value = row.get("adjusted_score", row.get("fusion_score", 0.0))
        try:
            score = float(score_value)
        except (TypeError, ValueError):
            score = 0.0
        docs.setdefault(name, []).append(score)

    if not docs:
        return

    st.markdown("**출처**")
    shown = 0
    for source_name, scores in docs.items():
        if shown >= 4:
            break
        shown += 1
        doc_relevance = int(max(0.0, min(100.0, (1 - math.exp(-max(scores) * 220.0)) * 100)))
        resolved = resolve_source_path(source_name)
        if resolved:
            uri = Path(resolved).as_uri()
            st.markdown(f"- [{source_name}]({uri}) · 관련도 {doc_relevance}%")
        else:
            st.markdown(f"- `{source_name}` · 관련도 {doc_relevance}%")


def friendly_error_message(exc: Exception) -> str:
    detail = str(exc).strip() or exc.__class__.__name__
    lowered = detail.lower()
    if "api key" in lowered or "openai_api_key" in lowered or "unauthorized" in lowered or "401" in lowered:
        return (
            "인증 정보 확인이 필요합니다.\n\n"
            "사이드바의 제공자 설정에서 API Key/Base URL을 확인해 주세요."
        )
    if "connection" in lowered or "timeout" in lowered or "temporarily unavailable" in lowered:
        return (
            "모델 서버 연결이 일시적으로 불안정합니다.\n\n"
            "잠시 후 다시 시도하거나 다른 provider/model 조합으로 테스트해 주세요."
        )
    if "chroma" in lowered or "bm25" in lowered or "collection" in lowered:
        return (
            "RAG 인덱스 자원을 찾지 못했습니다.\n\n"
            "`rag_outputs`의 Chroma/BM25 파일 경로와 고급 설정의 `Chroma dir`를 확인해 주세요."
        )
    return (
        "요청 처리 중 문제가 발생했습니다.\n\n"
        "설정을 점검한 뒤 다시 시도해 주세요.\n\n"
        f"세부 정보: `{detail}`"
    )


def render_run_meta(run: dict[str, Any]) -> None:
    confidence, relevance = compute_run_scores(run)
    st.markdown(
        (
            "<div class='qa-score-row'>"
            f"<span class='qa-score qa-score-confidence'>신뢰도 {confidence}%</span>"
            f"<span class='qa-score qa-score-relevance'>관련도 {relevance}%</span>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    st.caption(
        f"model={run.get('model_label')} · mode={run.get('mode')} · route={run.get('route')} · "
        f"latency={run.get('latency_sec')}s · top_k={run.get('top_k')}"
    )
    render_source_citations(run)


def render_assistant_message(message: dict[str, Any], *, animate: bool = False) -> None:
    runs = list(message.get("runs", []) or [])
    message_id = str(message.get("id", "")).strip()
    animated_ids = st.session_state.setdefault("_animated_message_ids", set())
    should_animate = bool(
        animate
        and message.get("typing_effect", False)
        and message_id
        and message_id not in animated_ids
    )
    if not runs:
        st.markdown(str(message.get("content", "")))
        if should_animate:
            animated_ids.add(message_id)
        return
    if len(runs) == 1:
        run = runs[0]
        content = str(run.get("content", ""))
        if should_animate and content:
            st.write_stream(stream_text_chunks(content))
        else:
            st.markdown(content)
        render_run_meta(run)
        if should_animate:
            animated_ids.add(message_id)
        return
    labels = [str(run.get("label", f"Run {index + 1}")) for index, run in enumerate(runs)]
    tabs = st.tabs(labels)
    for index, (tab, run) in enumerate(zip(tabs, runs)):
        with tab:
            content = str(run.get("content", ""))
            run_should_animate = bool(should_animate and index == 0 and content)
            if run_should_animate:
                st.write_stream(stream_text_chunks(content))
            else:
                st.markdown(content)
            render_run_meta(run)
    if should_animate:
        animated_ids.add(message_id)


def render_source_panel(message: dict[str, Any]) -> None:
    runs = list(message.get("runs", []) or [])
    if not runs:
        st.info("표시할 source가 없습니다.")
        return
    for run in runs:
        confidence, relevance = compute_run_scores(run)
        st.markdown(f"**{run.get('label')} · {run.get('model_label')}**")
        st.caption(f"신뢰도 {confidence}% · 관련도 {relevance}%")
        chunks = list(run.get("chunks", []) or [])
        if not chunks:
            st.caption("RAG source 없음 (chat-only 모드 또는 검색 결과 없음)")
            continue
        for chunk in chunks:
            header = (
                f"#{chunk.get('rank')} | {chunk.get('source_file_name', '')} | "
                f"chunk={chunk.get('chunk_id', '')} | score={chunk.get('adjusted_score') or chunk.get('fusion_score')}"
            )
            with st.expander(header):
                st.caption(
                    f"section={chunk.get('section_title', '')} · role={chunk.get('chunk_role', '')} "
                    f"· crag={chunk.get('crag_label', '')}"
                )
                source_name = str(chunk.get("source_file_name", "")).strip()
                resolved = resolve_source_path(source_name)
                if source_name:
                    if resolved:
                        st.markdown(f"[원문 열기: {source_name}]({Path(resolved).as_uri()})")
                    else:
                        st.caption(f"원문 경로 미확인: {source_name}")
                st.markdown(str(chunk.get("text", "")))


def render_debug_panel(message: dict[str, Any], *, show_router: bool = False) -> None:
    runs = list(message.get("runs", []) or [])
    if not runs:
        st.info("디버그 정보가 없습니다.")
        return
    for run in runs:
        with st.expander(f"{run.get('label')} · debug", expanded=False):
            debug_payload = {
                "mode": run.get("mode"),
                "model_label": run.get("model_label"),
                "latency_sec": run.get("latency_sec"),
                "route": run.get("route"),
                "top_k": run.get("top_k"),
                "profile": run.get("profile"),
            }
            if show_router:
                router = dict(run.get("router", {}) or {})
                debug_payload["answer_type_router"] = {
                    "predicted": router.get("predicted", router),
                    "answer_type_used": run.get("answer_type_used") or router.get("answer_type"),
                    "route_used": run.get("router_route_used") or router.get("route"),
                    "override_applied": bool(router.get("override_applied", False)),
                }
            st.json(debug_payload)


def render_artifact_panel(session: dict[str, Any]) -> None:
    artifacts = list(session.get("artifacts", []) or [])
    if not artifacts:
        st.info("아직 생성된 artifact가 없습니다.")
        return
    for item in artifacts[-12:][::-1]:
        with st.expander(f"{item.get('title', 'artifact')} · {item.get('created_at', '')}", expanded=False):
            st.markdown(str(item.get("content", "")))


def load_icon_svg(name: str) -> str:
    """static/icons/{name}.svg 파일을 읽어 인라인 SVG 문자열로 반환"""
    path = PROJECT_ROOT / "static" / "icons" / f"{name}.svg"
    if path.exists():
        try:
            return path.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            return ""
    return ""


def load_image_b64(relative_path: str | Path) -> str:
    """이미지 파일을 base64 data URI 문자열로 반환"""
    path = Path(relative_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return ""
    try:
        data = path.read_bytes()
        ext = path.suffix.lower().lstrip(".")
        mime = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "svg": "image/svg+xml",
        }.get(ext, "image/png")
        b64 = base64.b64encode(data).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    except Exception:  # noqa: BLE001
        return ""


def render_navbar() -> None:
    logo_b64 = load_image_b64("static/logo.png")
    avatar_b64 = load_image_b64("static/avatar.png") or logo_b64
    logo_img = (
        f'<img src="{logo_b64}" class="qa-navbar-logo" />'
        if logo_b64
        else '<div class="qa-navbar-logo qa-navbar-logo-fallback">RQ</div>'
    )
    avatar_img = (
        f'<img src="{avatar_b64}" class="qa-navbar-avatar" />'
        if avatar_b64
        else '<div class="qa-navbar-avatar qa-navbar-logo-fallback">U</div>'
    )
    st.markdown(
        f"""
<div class="qa-navbar">
  <div class="qa-navbar-left">
    {logo_img}
    <span class="qa-navbar-brand">RAG QA</span>
  </div>
  <div class="qa-navbar-center">
    <span class="qa-nav-tab">Training Data</span>
    <span class="qa-nav-tab qa-nav-tab-active">Instant Chat</span>
    <span class="qa-nav-tab">Embed Chatbot</span>
    <span class="qa-nav-tab">Leads</span>
  </div>
  <div class="qa-navbar-right">
    <div class="qa-navbar-search"><input type="text" placeholder="Search..." disabled /></div>
    <span class="qa-navbar-bell">🔔</span>
    {avatar_img}
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


MODE_PRESET_MAP: dict[str, tuple[str, str]] = {
    "fast": ("chat", "일반 채팅"),
    "in_depth": ("rag", "RAG QA"),
    "in_depth_compare": ("rag", "비교형 질문 우선"),
    "holistic": ("rag", "표/본문 질문"),
}


def render_status_pills(
    *,
    mode: str,
    provider: str,
    model_name: str,
    rag_profile_key: str,
    compare_mode: bool,
    compare_model_name: str,
    compare_profile_key: str,
) -> None:
    pills = [
        f"<span class='qa-pill'>mode: {mode}</span>",
        f"<span class='qa-pill'>provider: {provider}</span>",
        f"<span class='qa-pill'>model: {model_name}</span>",
    ]
    if mode == "rag":
        pills.append(f"<span class='qa-pill'>profile: {rag_profile_key}</span>")
    if compare_mode:
        pills.append(f"<span class='qa-pill qa-pill-compare'>compare: {compare_model_name or '-'} / {compare_profile_key or '-'}</span>")
    st.markdown(f"<div class='qa-pill-row'>{''.join(pills)}</div>", unsafe_allow_html=True)


def render_main_mode_pills(*, session: dict[str, Any], state: dict[str, Any], store: SessionStore) -> None:
    st.caption("Mode / Preset")
    st.markdown("<div class='qa-mode-pills-sentinel'></div>", unsafe_allow_html=True)
    mode_defs = [
        ("fast", "⚡ Fast"),
        ("in_depth", "📖 In-depth"),
        ("in_depth_compare", "✨ In-depth"),
        ("holistic", "🌐 Holistic"),
    ]
    current_settings = dict(session.get("settings", {}) or {})
    current_mode = str(current_settings.get("mode", "rag"))
    current_preset = str(current_settings.get("prompt_preset", "RAG QA"))
    cols = st.columns(4)
    for index, (mode_key, label) in enumerate(mode_defs):
        mapped_mode, mapped_preset = MODE_PRESET_MAP[mode_key]
        is_active_mode = bool(mapped_mode == current_mode and mapped_preset == current_preset)
        if cols[index].button(
            label,
            key=f"main_mode_{mode_key}_{session['id']}",
            use_container_width=True,
            type="primary" if is_active_mode else "secondary",
        ):
            session_settings = dict(session.get("settings", {}) or {})
            session_settings["mode"] = mapped_mode
            session_settings["prompt_preset"] = mapped_preset
            session["settings"] = session_settings
            store.update_session(state, session)
            store.save(state)
            st.rerun()


def render_empty_state(
    *,
    session: dict[str, Any],
    state: dict[str, Any],
    store: SessionStore,
) -> str:
    logo_b64 = load_image_b64("static/logo.png")
    logo_html = (
        f'<img src="{logo_b64}" class="qa-greeting-logo-image" />'
        if logo_b64
        else '<span class="qa-greeting-logo-fallback">RQ</span>'
    )
    st.markdown(
        f"""
<div class="qa-greeting-container">
  <div class="qa-greeting-logo">{logo_html}</div>
  <div class="qa-greeting-title">Hi, I'm RAG QA</div>
  <div class="qa-greeting-sub">How can I help you today?</div>
</div>
        """,
        unsafe_allow_html=True,
    )

    input_card = st.container(border=True)
    with input_card:
        st.markdown("<div class='qa-main-input-card-sentinel'></div>", unsafe_allow_html=True)

        hero_key = f"hero_prompt_{session.get('id', 'default')}"
        hero_prompt = st.text_area(
            "Ask anything...",
            value=st.session_state.get(hero_key, ""),
            key=hero_key,
            height=110,
            label_visibility="collapsed",
            placeholder="Ask anything...",
        )

        top_actions = st.columns([1, 1, 4.7, 1.3])
        top_actions[0].button("Deep search", key=f"hero_deep_{session['id']}", use_container_width=True)
        top_actions[1].button("Search", key=f"hero_search_{session['id']}", use_container_width=True)

        send_cols = st.columns([6, 1.3])
        submit_from_hero = send_cols[1].button(
            "전송",
            key=f"hero_send_{session['id']}",
            use_container_width=True,
            type="primary",
        )
    if submit_from_hero and str(hero_prompt or "").strip():
        return str(hero_prompt).strip()
    return ""


def get_or_create_adapter(
    selection: ProviderSelection,
    local_specs: dict[str, Any],
) -> Any:
    cache = st.session_state.setdefault("_adapter_cache", {})
    cache_key = json.dumps(
        {
            "provider": selection.provider,
            "model_name": selection.model_name,
            "local_model_ui_key": selection.local_model_ui_key,
            "base_url": selection.base_url,
            "api_key": selection.api_key,
            "max_new_tokens": selection.max_new_tokens,
            "temperature": selection.temperature,
            "top_p": selection.top_p,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    if cache_key not in cache:
        cache[cache_key] = create_provider_adapter(selection, local_specs)
    return cache[cache_key]


def get_or_create_pipeline(
    *,
    project_root: Path,
    rag_cfg: dict[str, Any],
    rag_profile_options: dict[str, Any],
) -> ScenarioBPhase2Pipeline:
    cache = st.session_state.setdefault("_pipeline_cache", {})
    cache_key = json.dumps(
        {
            "project_root": str(project_root),
            "embedding_backend_key": rag_cfg["embedding_backend_key"],
            "routing_model": rag_cfg["routing_model"],
            "candidate_k": rag_cfg["candidate_k"],
            "top_k": rag_cfg["top_k"],
            "crag_top_n": rag_cfg["crag_top_n"],
            "vector_weight": rag_cfg["vector_weight"],
            "bm25_weight": rag_cfg["bm25_weight"],
            "chroma_dir": rag_cfg.get("chroma_dir", ""),
            "options": rag_profile_options,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    if cache_key in cache:
        return cache[cache_key]

    settings = build_pipeline_settings(
        embedding_backend_key=rag_cfg["embedding_backend_key"],
        routing_model=rag_cfg["routing_model"],
        candidate_k=rag_cfg["candidate_k"],
        top_k=rag_cfg["top_k"],
        crag_top_n=rag_cfg["crag_top_n"],
        vector_weight=rag_cfg["vector_weight"],
        bm25_weight=rag_cfg["bm25_weight"],
    )
    options = build_phase2_options(rag_profile_options)
    pipeline = ScenarioBPhase2Pipeline(
        PipelinePaths(
            project_root=project_root,
            chroma_dir=Path(rag_cfg["chroma_dir"]).resolve() if str(rag_cfg.get("chroma_dir", "")).strip() else None,
        ),
        settings=settings,
        options=options,
    )
    cache[cache_key] = pipeline
    return pipeline


def run_one_turn(
    *,
    session: dict[str, Any],
    prompt: str,
    history_messages: list[dict[str, Any]],
    provider_payload: dict[str, Any],
    mode: str,
    rag_cfg: dict[str, Any],
    rag_profile_key: str,
    rag_profile_options: dict[str, Any],
    local_specs: dict[str, Any],
    preset_name: str,
    router_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # RAG pipeline internals read OPENAI_* from env; sync from current provider input.
    api_key_override = str(provider_payload.get("api_key", "") or "").strip()
    base_url_override = str(provider_payload.get("base_url", "") or "").strip()
    env_changed = False
    if api_key_override and os.getenv("OPENAI_API_KEY", "") != api_key_override:
        os.environ["OPENAI_API_KEY"] = api_key_override
        env_changed = True
    if base_url_override and os.getenv("OPENAI_BASE_URL", "") != base_url_override:
        os.environ["OPENAI_BASE_URL"] = base_url_override
        env_changed = True
    if env_changed:
        st.session_state.pop("_pipeline_cache", None)

    selection = build_provider_selection(provider_payload)
    adapter = get_or_create_adapter(selection, local_specs)
    history = build_history_for_adapter(history_messages, max_turns=14)
    attachment_context = build_attachment_context(session.get("attachments", []) or [], max_chars=4000)
    if mode == "chat":
        result = run_chat_turn(
            adapter=adapter,
            question=prompt,
            history=history,
            attachment_context=attachment_context,
            model_label=model_label(selection),
            preset_name=preset_name,
        )
        payload = run_result_to_payload(result, label="A")
        if router_result:
            payload["router"] = dict(router_result)
            payload["answer_type_used"] = str(router_result.get("answer_type", ""))
            payload["router_route_used"] = str(router_result.get("route", ""))
        return payload

    pipeline = get_or_create_pipeline(
        project_root=PROJECT_ROOT,
        rag_cfg=rag_cfg,
        rag_profile_options=rag_profile_options,
    )
    result = run_rag_turn_with_pipeline(
        pipeline=pipeline,
        adapter=adapter,
        question=prompt,
        history=history,
        attachment_context=attachment_context,
        model_label=model_label(selection),
        rag_profile_key=rag_profile_key,
        answer_type=str((router_result or {}).get("answer_type", "")) or None,
        router_result=router_result,
    )
    payload = run_result_to_payload(result, label="A")
    if router_result:
        payload["router"] = dict(router_result)
        payload["answer_type_used"] = str(router_result.get("answer_type", ""))
        payload["router_route_used"] = str(router_result.get("route", ""))
    return payload


@st.cache_data(show_spinner=False)
def cached_phase2_profiles(project_root_str: str) -> dict[str, dict[str, Any]]:
    return load_phase2_profiles(Path(project_root_str))


@st.cache_data(show_spinner=False)
def cached_local_specs(project_root_str: str) -> dict[str, Any]:
    return load_local_model_specs(Path(project_root_str))


@st.cache_data(show_spinner=False)
def cached_embedding_keys(project_root_str: str) -> list[str]:
    return load_embedding_backend_keys(Path(project_root_str))


def apply_custom_style() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&display=swap');

:root {
  --qa-bg: #f8fafc;
  --qa-panel: #ffffff;
  --qa-border: #e2e8f0;
  --qa-muted: #64748b;
  --qa-text: #0f172a;
  --qa-primary: #8B5CF6;
  --qa-primary-soft: #A78BFA;
  --qa-primary-faint: #f3edff;
}

.stApp {
  font-family: "Manrope", "Inter", "Noto Sans KR", sans-serif;
  background: linear-gradient(135deg, #faf5ff 0%, #f8fafc 30%, #ffffff 60%, #faf5ff 100%);
  color: var(--qa-text);
}
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
  background: transparent !important;
}
[data-testid="stDecoration"] {
  display: none !important;
}
[data-testid="stHeader"] {
  background: transparent !important;
  border-bottom: none !important;
}
[data-testid="stToolbar"] {
  top: 4.05rem;
  right: 1rem;
  z-index: 1002;
}
[data-testid="stToolbar"] button,
[data-testid="stToolbar"] [role="button"] {
  background: rgba(255, 255, 255, 0.9) !important;
  border: 1px solid #e2e8f0 !important;
  border-radius: 10px !important;
}
[data-testid="stSidebar"] {
  background: #ffffff;
  border-right: 1px solid #e2e8f0;
}
[data-testid="stSidebar"] > div:first-child {
  padding-top: 3.6rem;
}
[data-testid="stSidebar"] * {
  color: var(--qa-text);
}
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] {
  top: 4.05rem !important;
  z-index: 1002 !important;
}
[data-testid="collapsedControl"] {
  margin-top: 0 !important;
}
[data-testid="stSidebar"] h1 {
  font-size: 1.1rem !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] [data-baseweb="select"] * {
  font-size: 0.86rem !important;
}
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stCheckbox label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stSlider label {
  font-size: 0.84rem !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] button,
[data-testid="stSidebar"] [data-testid="baseButton-secondary"],
[data-testid="stSidebar"] [data-testid="baseButton-primary"] {
  font-size: 0.86rem !important;
  line-height: 1.15 !important;
  min-height: 1.9rem !important;
  padding: 0.2rem 0.5rem !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] {
  margin-bottom: 0.15rem !important;
}
[data-testid="stSidebar"] [data-testid="stTextInput"] input,
[data-testid="stSidebar"] [data-testid="stNumberInput"] input,
[data-testid="stSidebar"] [data-baseweb="select"] > div {
  font-size: 0.86rem !important;
  min-height: 2rem !important;
}
[data-testid="stSidebar"] .stExpander details summary p {
  font-size: 0.86rem !important;
}
.block-container {
  padding-top: 5.2rem;
  max-width: 1560px;
}

@media (max-width: 900px) {
  .block-container {
    padding-top: 6rem;
  }
}

.qa-navbar {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 999;
  height: 56px;
  background: #ffffff;
  border-bottom: 1px solid #e2e8f0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 2rem;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}

.qa-navbar-left,
.qa-navbar-center,
.qa-navbar-right {
  display: flex;
  align-items: center;
  gap: 0.65rem;
}

.qa-navbar-brand {
  font-size: 1.05rem;
  font-weight: 700;
  margin-left: 0.5rem;
  color: var(--qa-text);
}

.qa-navbar-logo,
.qa-navbar-avatar {
  width: 32px;
  height: 32px;
  border-radius: 10px;
  object-fit: cover;
}

.qa-navbar-logo-fallback {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: var(--qa-primary);
  color: #ffffff;
  font-size: 0.84rem;
  font-weight: 700;
}

.qa-nav-tab {
  font-size: 0.85rem;
  font-weight: 500;
  color: var(--qa-muted);
  padding: 0.4rem 0.8rem;
  border-radius: 8px;
  border: 1px solid transparent;
  cursor: pointer;
  transition: all 0.2s;
}

.qa-nav-tab:hover {
  background: #faf5ff;
  color: #7c3aed;
}

.qa-nav-tab-active {
  color: #7c3aed;
  background: #faf5ff;
  border-color: #ede9fe;
  font-weight: 600;
}

.qa-navbar-search {
  min-width: 180px;
  height: 34px;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  background: #ffffff;
  color: #94a3b8;
  display: inline-flex;
  align-items: center;
  padding: 0 0.2rem;
  font-size: 0.8rem;
}

.qa-navbar-search input {
  width: 100%;
  border: none;
  outline: none;
  background: transparent;
  color: #94a3b8;
  font-size: 0.8rem;
  padding: 0 0.7rem;
}

.qa-navbar-bell {
  font-size: 1rem;
  opacity: 0.8;
}

@media (max-width: 1100px) {
  .qa-navbar {
    padding: 0 0.85rem;
    height: 56px;
    flex-wrap: wrap;
    gap: 0.45rem;
  }
  .qa-navbar-center {
    width: 52%;
    overflow-x: auto;
    scrollbar-width: none;
  }
  .qa-navbar-search {
    min-width: 120px;
    width: 140px;
  }
}

@media (max-width: 768px) {
  .qa-navbar-center { display: none; }
  .qa-navbar-search { display: none; }
  .qa-greeting-title { font-size: 1.2rem; }
  .qa-main-input-card { margin: 1rem; max-width: 100%; }
  .qa-mode-pills { gap: 0.4rem; }
  .qa-mode-pill { font-size: 0.78rem; padding: 0.35rem 0.7rem; }

  div[data-testid="stVerticalBlockBorderWrapper"]:has(.qa-main-input-card-sentinel) {
    margin: 1rem !important;
    max-width: 100% !important;
  }
  div[data-testid="stVerticalBlockBorderWrapper"]:has(.qa-mode-pills-sentinel) [data-testid="stBaseButton-secondary"],
  div[data-testid="stVerticalBlockBorderWrapper"]:has(.qa-mode-pills-sentinel) [data-testid="stBaseButton-primary"] {
    font-size: 0.78rem !important;
    padding: 0.35rem 0.7rem !important;
  }
}

[data-testid="stTabs"] [role="tablist"] {
  gap: 0.36rem;
}

[data-testid="stTabs"] button[role="tab"] {
  border-radius: 999px;
  border: 1px solid var(--qa-border);
  background: #ffffff;
  color: #475569;
  padding: 0.25rem 0.75rem;
  font-size: 0.8rem;
  font-weight: 500;
}

[data-testid="stTabs"] button[aria-selected="true"] {
  border-color: #d8c8ff;
  background: var(--qa-primary-faint);
  color: var(--qa-primary);
}

[data-testid="stExpander"] details {
  border: 1px solid var(--qa-border);
  border-radius: 12px;
  background: #ffffff;
}

[data-testid="stBaseButton-secondary"] {
  border-radius: 999px;
  border-color: var(--qa-border);
}

[data-testid="stBaseButton-secondary"]:hover {
  border-color: #d8c8ff !important;
  color: var(--qa-primary) !important;
  background: #faf7ff !important;
}

div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input,
div[data-baseweb="select"] > div {
  border-radius: 10px !important;
}

[data-testid="stChatInput"] {
  background: transparent;
}

[data-testid="stBottom"] {
  background: transparent !important;
}

[data-testid="stBottomBlockContainer"] {
  background: linear-gradient(180deg, rgba(250, 245, 255, 0) 0%, rgba(250, 245, 255, 0.8) 100%);
  border-top: 1px solid #efe7ff;
  padding-top: 0.46rem;
  padding-bottom: 0.62rem;
}

[data-testid="stChatInput"] textarea {
  border: 1px solid #e2e8f0 !important;
  background: #ffffff !important;
  border-radius: 24px !important;
  padding: 0.8rem 1.2rem !important;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
  font-size: 0.9rem;
}

.qa-greeting-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 52vh;
  text-align: center;
  padding: 2rem 1rem 0.8rem 1rem;
}

.qa-greeting-logo {
  width: 56px;
  height: 56px;
  border-radius: 14px;
  background: linear-gradient(135deg, #c084fc, #8b5cf6);
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 1.2rem;
  box-shadow: 0 8px 24px rgba(139, 92, 246, 0.2);
  overflow: hidden;
}

.qa-greeting-logo-image {
  width: 56px;
  height: 56px;
  border-radius: 14px;
  object-fit: cover;
}

.qa-greeting-logo-fallback {
  color: #ffffff;
  font-weight: 700;
  font-size: 1rem;
}

.qa-greeting-title {
  font-size: 1.6rem;
  font-weight: 700;
  color: #0f172a;
  margin-bottom: 0.3rem;
}

.qa-greeting-sub {
  font-size: 0.92rem;
  color: #64748b;
}

.qa-main-input-card-sentinel {
  display: none;
}

div[data-testid="stVerticalBlockBorderWrapper"]:has(.qa-main-input-card-sentinel) {
  background: rgba(255, 255, 255, 0.75) !important;
  backdrop-filter: blur(12px);
  border: 1px solid #e2e8f0 !important;
  border-radius: 20px !important;
  padding: 1.2rem !important;
  max-width: 680px;
  margin: 1.5rem auto;
  box-shadow: 0 12px 40px rgba(139, 92, 246, 0.06) !important;
}

div[data-testid="stVerticalBlockBorderWrapper"]:has(.qa-main-input-card-sentinel) [data-testid="stBaseButton-secondary"] {
  padding: 0.45rem 1rem;
  border-radius: 999px;
  border: 1px solid #e2e8f0 !important;
  background: #ffffff !important;
  font-size: 0.84rem;
  font-weight: 500;
  color: #334155 !important;
  transition: all 0.2s;
}

div[data-testid="stVerticalBlockBorderWrapper"]:has(.qa-main-input-card-sentinel) [data-testid="stBaseButton-secondary"]:hover {
  border-color: #c4b5fd !important;
  background: #faf5ff !important;
  color: #7c3aed !important;
}

div[data-testid="stVerticalBlockBorderWrapper"]:has(.qa-main-input-card-sentinel) [data-testid="stBaseButton-primary"] {
  border-color: #a78bfa !important;
  background: #ede9fe !important;
  color: #6d28d9 !important;
  border-radius: 999px !important;
}

div[data-testid="stVerticalBlockBorderWrapper"]:has(.qa-main-input-card-sentinel) [data-testid="stTextArea"] textarea {
  border: 1px solid #e2e8f0 !important;
  border-radius: 16px !important;
  background: rgba(255, 255, 255, 0.9) !important;
  min-height: 110px;
}

.qa-hero {
  border: 1px solid var(--qa-border);
  border-radius: 20px;
  background: #ffffff;
  padding: 0.95rem 1rem 0.65rem 1rem;
  box-shadow: 0 16px 34px rgba(15, 23, 42, 0.05);
  margin-bottom: 0.8rem;
}

.qa-hero-title {
  font-size: 1.08rem;
  font-weight: 700;
  color: var(--qa-text);
}

.qa-hero-meta {
  font-size: 0.82rem;
  color: var(--qa-muted);
  margin-top: 0.2rem;
}

.qa-pill-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  margin: 0.2rem 0 0.85rem 0;
}

.qa-pill {
  display: inline-flex;
  align-items: center;
  padding: 0.18rem 0.56rem;
  border: 1px solid var(--qa-border);
  border-radius: 999px;
  font-size: 0.76rem;
  color: #334155;
  background: #ffffff;
  font-weight: 500;
}

.qa-pill-compare {
  border-color: #d8c8ff;
  background: var(--qa-primary-faint);
  color: var(--qa-primary);
}

.qa-empty {
  border: 1px solid var(--qa-border);
  border-radius: 20px;
  background: #ffffff;
  padding: 1.2rem 1.05rem;
  box-shadow: 0 18px 40px rgba(15, 23, 42, 0.06);
  margin: 0.4rem 0 1rem 0;
}

.qa-empty-title {
  font-weight: 700;
  font-size: 1.5rem;
  color: var(--qa-text);
  margin-bottom: 0.2rem;
}

.qa-empty-body {
  color: var(--qa-muted);
  font-size: 0.9rem;
  margin-bottom: 0.75rem;
}

.qa-home-empty {
  text-align: center;
  background:
    radial-gradient(circle at 50% 0%, rgba(167, 139, 250, 0.14) 0%, rgba(255,255,255,0.88) 58%),
    #ffffff;
}

.qa-empty-logo-wrap {
  display: flex;
  justify-content: center;
  margin-bottom: 0.4rem;
}

.qa-empty-logo {
  width: 56px;
  height: 56px;
  border-radius: 14px;
  object-fit: cover;
}

.qa-empty-mic {
  color: var(--qa-primary);
  font-size: 1.05rem;
  margin-top: -0.15rem;
}

.qa-score-row {
  display: flex;
  gap: 0.35rem;
  margin: 0.1rem 0 0.25rem 0;
}

.qa-score {
  display: inline-flex;
  align-items: center;
  padding: 0.12rem 0.52rem;
  border-radius: 999px;
  font-size: 0.74rem;
  font-weight: 600;
  border: 1px solid transparent;
}

.qa-score-confidence {
  background: #f3f0ff;
  color: #6d28d9;
  border-color: #ddd6fe;
}

.qa-score-relevance {
  background: #f5f3ff;
  color: #7c3aed;
  border-color: #ddd6fe;
}

.qa-context-widget {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 20px;
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.06);
  padding: 0;
  overflow: hidden;
}

.qa-context-widget-header {
  padding: 0.8rem 0.15rem 0.5rem 0.15rem;
  border-bottom: 1px solid #e2e8f0;
  font-weight: 600;
  font-size: 0.9rem;
  margin-bottom: 0.2rem;
}

.qa-context-widget-title {
  font-weight: 600;
  font-size: 0.9rem;
  color: #0f172a;
  line-height: 1.25;
}

.qa-context-widget-sub {
  margin-top: 0.12rem;
  color: #64748b;
  font-size: 0.78rem;
  font-weight: 500;
}

.qa-send-btn {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  background: #8B5CF6;
  border: none;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
}

.qa-send-btn svg {
  fill: #ffffff;
  width: 18px;
  height: 18px;
}

[data-testid="stChatMessage"] {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
  margin-bottom: 0.56rem !important;
}

[data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {
  border: 1px solid var(--qa-border);
  background: #ffffff;
  border-radius: 18px 18px 18px 4px;
  padding: 0.52rem 0.78rem;
  max-width: 80%;
  width: fit-content;
  overflow-wrap: anywhere;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  justify-content: flex-end;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
  justify-content: flex-end;
}

[data-testid="stChatMessage"]:has([aria-label*="user"]) {
  justify-content: flex-end;
}

[data-testid="stChatMessage"]:has(.qa-user-bubble) {
  justify-content: flex-end;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"],
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"],
[data-testid="stChatMessage"]:has([aria-label*="user"]) [data-testid="stChatMessageContent"],
[data-testid="stChatMessage"]:has(.qa-user-bubble) [data-testid="stChatMessageContent"] {
  background: transparent;
  border: none;
  padding: 0;
  max-width: 100%;
  width: 100%;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageAvatar"],
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageAvatar"],
[data-testid="stChatMessage"]:has(.qa-user-bubble) [data-testid="stChatMessageAvatar"] {
  display: none;
}

.qa-user-bubble {
  background: var(--qa-primary);
  color: #ffffff;
  border-color: var(--qa-primary);
  border: none;
  border-radius: 18px 18px 4px 18px;
  padding: 0.52rem 0.78rem;
  max-width: min(75%, 700px);
  width: fit-content;
  margin-left: auto;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  word-break: keep-all;
  line-height: 1.62;
  font-size: 0.88rem;
}

[data-testid="stChatMessage"] [data-testid="stChatMessageContent"] p {
  line-height: 1.62;
  font-size: 0.88rem;
}

div[data-testid="stVerticalBlockBorderWrapper"]:has(.qa-context-widget-header) {
  border: 1px solid #e2e8f0 !important;
  border-radius: 20px !important;
  background: #ffffff !important;
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.06) !important;
  position: fixed !important;
  right: 0.85rem;
  bottom: 6.2rem;
  width: min(410px, calc(100vw - 2.4rem));
  max-height: min(62vh, 680px);
  overflow: auto;
  z-index: 995;
}

div[data-testid="stVerticalBlockBorderWrapper"]:has(.qa-fab-sentinel) {
  position: fixed !important;
  right: 0.9rem;
  bottom: 1.9rem;
  z-index: 996;
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
  width: auto !important;
}

div[data-testid="stVerticalBlockBorderWrapper"]:has(.qa-fab-sentinel) [data-testid="stButton"] button {
  width: 54px !important;
  height: 54px !important;
  border-radius: 50% !important;
  border: none !important;
  padding: 0 !important;
  font-size: 1.35rem !important;
  background: linear-gradient(135deg, #9f67ff 0%, #7c3aed 100%) !important;
  color: #ffffff !important;
  box-shadow: 0 10px 30px rgba(124, 58, 237, 0.35) !important;
}

div[data-testid="stVerticalBlockBorderWrapper"]:has(.qa-fab-sentinel) [data-testid="stButton"] button:hover {
  filter: brightness(1.03);
  transform: translateY(-1px);
}

@media (max-width: 768px) {
  div[data-testid="stVerticalBlockBorderWrapper"]:has(.qa-fab-sentinel) {
    right: 0.85rem;
    bottom: 1.3rem;
  }
  div[data-testid="stVerticalBlockBorderWrapper"]:has(.qa-context-widget-header) {
    right: 0.65rem;
    left: 0.65rem;
    width: auto;
    bottom: 5.8rem;
    max-height: 56vh;
  }
}

/* Omago-style final UI pass: visual-only overrides. */
.stApp {
  background:
    linear-gradient(135deg, rgba(250, 245, 255, 0.96) 0%, rgba(248, 250, 252, 0.98) 32%, #ffffff 62%, rgba(250, 245, 255, 0.92) 100%) !important;
  overflow-x: hidden;
}

.stApp::before {
  content: none;
  display: none;
}

.block-container {
  position: relative;
  z-index: 1;
  max-width: 1060px !important;
  padding-top: 5.6rem !important;
  padding-left: 2.4rem !important;
  padding-right: 2.4rem !important;
  padding-bottom: 7.4rem !important;
}

.qa-navbar {
  height: 58px;
  padding: 0 2.15rem;
  background: rgba(255, 255, 255, 0.9);
  backdrop-filter: blur(14px);
  border-bottom: 1px solid rgba(226, 232, 240, 0.82);
}

.qa-navbar-logo,
.qa-navbar-avatar {
  width: 30px;
  height: 30px;
}

.qa-nav-tab {
  font-size: 0.78rem;
  border-radius: 999px;
}

.qa-navbar-search {
  width: 190px;
  height: 32px;
  background: rgba(255, 255, 255, 0.72);
}

.qa-greeting-container {
  min-height: 39vh;
  padding-top: 3.8rem;
  padding-bottom: 0.25rem;
}

.qa-greeting-logo {
  width: 48px;
  height: 48px;
  margin-bottom: 0.82rem;
  border-radius: 14px;
}

.qa-greeting-logo-image {
  width: 48px;
  height: 48px;
  border-radius: 14px;
}

.qa-greeting-title {
  font-size: 1.42rem;
  margin-bottom: 0.18rem;
}

.qa-greeting-sub {
  font-size: 0.84rem;
}

div[data-testid="stVerticalBlockBorderWrapper"]:has(.qa-main-input-card-sentinel) {
  max-width: 700px !important;
  margin: 1.05rem auto 0.72rem auto !important;
  padding: 1rem !important;
  border-radius: 18px !important;
  border: 1px solid rgba(226, 232, 240, 0.86) !important;
  background: rgba(255, 255, 255, 0.58) !important;
  box-shadow: 0 18px 60px rgba(124, 58, 237, 0.07) !important;
}

div[data-testid="stVerticalBlockBorderWrapper"]:has(.qa-main-input-card-sentinel) [data-testid="stTextArea"] textarea {
  min-height: 92px !important;
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
  padding: 0.25rem 0.25rem 0.4rem 0.25rem !important;
  font-size: 0.92rem !important;
}

div[data-testid="stVerticalBlockBorderWrapper"]:has(.qa-main-input-card-sentinel) [data-testid="stButton"] button {
  min-height: 2rem !important;
  border-radius: 999px !important;
  font-size: 0.78rem !important;
}

div[data-testid="stVerticalBlockBorderWrapper"]:has(.qa-mode-pills-sentinel) {
  max-width: 560px;
  margin: 0 auto 1rem auto;
}

div[data-testid="stVerticalBlockBorderWrapper"]:has(.qa-mode-pills-sentinel) [data-testid="stButton"] button {
  min-height: 2.15rem !important;
  border-radius: 999px !important;
  border: 1px solid rgba(226, 232, 240, 0.95) !important;
  background: rgba(255, 255, 255, 0.74) !important;
  color: #334155 !important;
  font-size: 0.8rem !important;
  font-weight: 600 !important;
  box-shadow: 0 5px 18px rgba(15, 23, 42, 0.035) !important;
}

div[data-testid="stVerticalBlockBorderWrapper"]:has(.qa-mode-pills-sentinel) [data-testid="stBaseButton-primary"] {
  background: #ede9fe !important;
  border-color: #c4b5fd !important;
  color: #6d28d9 !important;
}

.qa-hero {
  max-width: 960px;
  margin: 0 auto 0.75rem auto;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.68);
}

.qa-pill-row {
  max-width: 960px;
  margin-left: auto;
  margin-right: auto;
}

[data-testid="stChatMessage"] {
  max-width: 960px;
  margin-left: auto !important;
  margin-right: auto !important;
}

[data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {
  border-radius: 18px 18px 18px 5px;
  padding: 0.68rem 0.92rem;
  box-shadow: 0 10px 32px rgba(15, 23, 42, 0.045);
}

.qa-user-bubble {
  max-width: min(72%, 720px);
  padding: 0.68rem 0.92rem;
  background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
  box-shadow: 0 10px 28px rgba(124, 58, 237, 0.18);
}

[data-testid="stBottomBlockContainer"] {
  background: linear-gradient(180deg, rgba(250, 245, 255, 0) 0%, rgba(250, 245, 255, 0.78) 52%, rgba(255, 255, 255, 0.96) 100%) !important;
}

[data-testid="stChatInput"] {
  max-width: 900px;
  margin: 0 auto;
}

[data-testid="stChatInput"] textarea {
  min-height: 48px !important;
  border-radius: 24px !important;
  border: 1px solid rgba(226, 232, 240, 0.94) !important;
  box-shadow: 0 10px 34px rgba(15, 23, 42, 0.065) !important;
}

div[data-testid="stVerticalBlockBorderWrapper"]:has(.qa-context-widget-header) {
  right: 1.05rem !important;
  bottom: 6.1rem !important;
  width: min(390px, calc(100vw - 2.1rem)) !important;
  border-radius: 16px !important;
  box-shadow: 0 20px 60px rgba(15, 23, 42, 0.13) !important;
}

.qa-context-widget-header {
  padding: 0.85rem 0.35rem 0.55rem 0.35rem;
}

div[data-testid="stVerticalBlockBorderWrapper"]:has(.qa-fab-sentinel) {
  right: 1.05rem !important;
  bottom: 1.65rem !important;
}

div[data-testid="stVerticalBlockBorderWrapper"]:has(.qa-fab-sentinel) [data-testid="stButton"] button {
  width: 52px !important;
  height: 52px !important;
  background: linear-gradient(135deg, #a855f7 0%, #7c3aed 100%) !important;
}

@media (max-width: 768px) {
  .block-container {
    padding-left: 0.9rem !important;
    padding-right: 0.9rem !important;
    padding-top: 4.8rem !important;
  }

  .qa-navbar {
    padding: 0 0.85rem;
  }

  .qa-navbar-right {
    gap: 0.35rem;
  }

  .qa-greeting-container {
    min-height: 35vh;
    padding-top: 2.6rem;
  }

  div[data-testid="stVerticalBlockBorderWrapper"]:has(.qa-main-input-card-sentinel) {
    margin: 0.85rem 0 0.65rem 0 !important;
  }

  div[data-testid="stVerticalBlockBorderWrapper"]:has(.qa-mode-pills-sentinel) {
    max-width: 100%;
  }

  .qa-user-bubble {
    max-width: min(86%, 720px);
  }

  div[data-testid="stVerticalBlockBorderWrapper"]:has(.qa-context-widget-header) {
    left: 0.75rem !important;
    right: 0.75rem !important;
    width: auto !important;
    bottom: 5.45rem !important;
    max-height: 58vh !important;
  }
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
  color: #334155;
}
</style>
        """,
        unsafe_allow_html=True,
    )

st.set_page_config(page_title="RAG QA", layout="wide", initial_sidebar_state="collapsed")
apply_custom_style()
render_navbar()

store = SessionStore(PROJECT_ROOT)
if "qa_state" not in st.session_state:
    st.session_state.qa_state = store.load()
state = st.session_state.qa_state

phase2_profiles = cached_phase2_profiles(str(PROJECT_ROOT))
local_specs = cached_local_specs(str(PROJECT_ROOT))
embedding_backend_keys = cached_embedding_keys(str(PROJECT_ROOT))

if not phase2_profiles:
    st.error("phase2 profile config를 찾지 못했습니다. config/phase2_experiments.yaml을 확인하세요.")
    st.stop()

active_project_id = str(state.get("active_project_id", "default") or "default")
projects = state.get("projects", {}) or {}
if active_project_id not in projects:
    active_project_id = "default"
    state["active_project_id"] = active_project_id

session_ids_for_project = [
    sid
    for sid in state.get("session_order", []) or []
    if sid in (state.get("sessions", {}) or {}) and state["sessions"][sid].get("project_id") == active_project_id
]
if not session_ids_for_project:
    store.create_session(state, project_id=active_project_id, settings=projects[active_project_id].get("defaults", {}))
    session_ids_for_project = [
        sid
        for sid in state.get("session_order", []) or []
        if sid in (state.get("sessions", {}) or {}) and state["sessions"][sid].get("project_id") == active_project_id
    ]

active_session_id = str(state.get("active_session_id", "") or "")
if active_session_id not in session_ids_for_project:
    active_session_id = session_ids_for_project[0]
    state["active_session_id"] = active_session_id

session = state["sessions"][active_session_id]
session.setdefault("settings", {})
session.setdefault("messages", [])
session.setdefault("attachments", [])
session.setdefault("artifacts", [])

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    with st.expander("Open Settings", expanded=False):
        project_order = [pid for pid in state.get("project_order", []) if pid in projects]
        selected_project = st.selectbox(
            "Project",
            options=project_order,
            index=project_order.index(active_project_id),
            format_func=lambda pid: projects[pid].get("name", pid),
            help="프로젝트 단위로 세션/기본 설정을 분리합니다.",
        )
        if selected_project != active_project_id:
            state["active_project_id"] = selected_project
            sessions_of_new_project = [
                sid
                for sid in state.get("session_order", []) or []
                if sid in (state.get("sessions", {}) or {}) and state["sessions"][sid].get("project_id") == selected_project
            ]
            if not sessions_of_new_project:
                store.create_session(state, project_id=selected_project, settings=projects[selected_project].get("defaults", {}))
            else:
                state["active_session_id"] = sessions_of_new_project[0]
            store.save(state)
            st.rerun()

        with st.expander("Project 관리", expanded=False):
            new_project_name = st.text_input("새 프로젝트 이름", key="new_project_name", help="내부 테스트 목적에 맞는 프로젝트명을 입력하세요.")
            if st.button("프로젝트 생성", use_container_width=True, help="새 프로젝트를 만들고 해당 프로젝트로 전환합니다."):
                store.create_project(state, new_project_name)
                store.create_session(state, project_id=state["active_project_id"], settings=state["projects"][state["active_project_id"]]["defaults"])
                store.save(state)
                st.rerun()

        sessions_for_sidebar = [
            sid
            for sid in state.get("session_order", []) or []
            if sid in state.get("sessions", {}) and state["sessions"][sid].get("project_id") == state["active_project_id"]
        ]
        selected_session_id = st.radio(
            "채팅 세션",
            options=sessions_for_sidebar,
            index=sessions_for_sidebar.index(state["active_session_id"]),
            format_func=lambda sid: state["sessions"][sid].get("title", sid),
            help="같은 프로젝트 안에서 테스트 대화를 분리합니다.",
        )
        if selected_session_id != state["active_session_id"]:
            state["active_session_id"] = selected_session_id
            store.save(state)
            st.rerun()

        if st.button("새 채팅", use_container_width=True, help="현재 프로젝트에 새 대화 세션을 생성합니다."):
            store.create_session(
                state,
                project_id=state["active_project_id"],
                settings=state["projects"][state["active_project_id"]].get("defaults", {}),
            )
            store.save(state)
            st.rerun()
        col_b, col_c = st.columns(2)
        if col_b.button("초기화", use_container_width=True, help="현재 세션의 메시지/아티팩트를 비웁니다."):
            store.clear_session_messages(state, state["active_session_id"])
            store.save(state)
            st.rerun()
        if col_c.button("삭제", use_container_width=True, help="현재 채팅 세션을 삭제합니다."):
            target_session_id = str(state.get("active_session_id", "") or "")
            target_project_id = str(state.get("active_project_id", "default") or "default")
            if target_session_id:
                store.delete_session(state, target_session_id)
                remaining_sessions = [
                    sid
                    for sid in state.get("session_order", []) or []
                    if sid in state.get("sessions", {}) and state["sessions"][sid].get("project_id") == target_project_id
                ]
                if remaining_sessions:
                    state["active_session_id"] = remaining_sessions[0]
                else:
                    store.create_session(
                        state,
                        project_id=target_project_id,
                        settings=state["projects"][target_project_id].get("defaults", {}),
                    )
                store.save(state)
                st.rerun()

        regen_clicked = st.button("마지막 답변 재생성", use_container_width=True, help="직전 사용자 질문으로 다시 생성합니다.")
        if st.button("Stop (다음 요청 취소)", use_container_width=True, help="다음 실행 요청을 1회 취소합니다."):
            st.session_state.stop_requested = True

        export_payload = json.dumps(session, ensure_ascii=False, indent=2)
        st.download_button(
            "현재 세션 Export (.json)",
            data=export_payload,
            file_name=f"{session['id']}.json",
            mime="application/json",
            use_container_width=True,
            help="현재 세션 기록과 설정을 JSON으로 내보냅니다.",
        )

        with st.expander("파일 업로드", expanded=False):
            uploads = st.file_uploader(
                "세션 컨텍스트 파일",
                type=["pdf", "txt", "md", "json", "csv"],
                accept_multiple_files=True,
                key=f"uploader_{session['id']}",
                help="업로드한 파일의 미리보기 텍스트를 현재 세션 컨텍스트로 주입합니다.",
            )
            if uploads:
                uploaded_count = ingest_uploads(session, store, uploads)
                if uploaded_count > 0:
                    store.update_session(state, session)
                    store.save(state)
                    st.success(f"{uploaded_count}개 파일을 세션 컨텍스트에 추가했습니다.")
            for idx, att in enumerate(session.get("attachments", []) or []):
                cols = st.columns([0.12, 0.63, 0.25])
                included = cols[0].checkbox(
                    "사용",
                    value=bool(att.get("included", True)),
                    key=f"att_used_{session['id']}_{idx}",
                    label_visibility="collapsed",
                    help="체크하면 답변 생성 시 이 파일 컨텍스트를 포함합니다.",
                )
                att["included"] = included
                cols[1].caption(f"{att.get('name')} ({att.get('size_bytes')} bytes)")
                if cols[2].button("삭제", key=f"att_del_{session['id']}_{idx}", use_container_width=True, help="현재 세션에서 이 첨부를 제거합니다."):
                    session["attachments"].pop(idx)
                    store.update_session(state, session)
                    store.save(state)
                    st.rerun()
                preview = str(att.get("preview_text", "")).strip()
                if preview:
                    with st.expander(f"미리보기 · {att.get('name')}", expanded=False):
                        st.text(preview[:1500])

        st.divider()
        st.subheader("실험 설정")

        settings = session.get("settings", {})
        mode = str(settings.get("mode", "rag"))
        if mode not in {"rag", "chat"}:
            mode = "rag"
        preset_options = ["일반 채팅", "RAG QA", "비교형 질문 우선", "표/본문 질문"]
        preset_name = str(settings.get("prompt_preset", "RAG QA"))
        if preset_name not in preset_options:
            preset_name = "RAG QA"
        st.caption("Mode / Prompt preset 전환은 메인 상단 pill 버튼에서 변경합니다.")
        compare_mode = st.toggle("Compare mode", value=bool(settings.get("compare_mode", False)), help="같은 질문을 A/B 설정으로 동시에 실행합니다.")
        router_debug = st.toggle(
            "Router debug",
            value=bool(settings.get("router_debug", False)),
            help="켜면 답변의 Debug 패널에서 answer_type, route, confidence, signals를 확인합니다.",
        )
        override_options = ["Auto", *ANSWER_TYPES]
        default_override = str(settings.get("answer_type_override", "Auto") or "Auto")
        if default_override not in override_options:
            default_override = "Auto"
        answer_type_override = st.selectbox(
            "Answer type override",
            options=override_options,
            index=override_options.index(default_override),
            help="Auto는 router 예측을 사용합니다. 발표/데모 중 오분기 시 수동 answer_type으로 실행할 수 있습니다.",
        )

        provider_options = ["openai_api", "local", "openai_compatible"]
        selected_provider = st.selectbox(
            "Provider",
            options=provider_options,
            index=provider_options.index(str(settings.get("provider", "openai_api")))
            if str(settings.get("provider", "openai_api")) in provider_options
            else 0,
            key="primary_provider",
            help="모델 제공자를 선택합니다. local은 로컬 HF 모델, openai_compatible은 vLLM/Ollama 게이트웨이 등에 연결할 때 사용합니다.",
        )

        primary_payload: dict[str, Any] = {"provider": selected_provider}
        primary_api_key_default = str(settings.get("api_key", ""))
        primary_base_url_default = str(settings.get("base_url", ""))
        if selected_provider == "local":
            local_keys = list(local_specs.keys())
            if not local_keys:
                st.error("로컬 모델 스펙을 찾지 못했습니다. config/scenario_a_models.yaml을 확인하세요.")
                st.stop()
            default_local = str(settings.get("local_model_ui_key", local_keys[0]))
            if default_local not in local_keys:
                default_local = local_keys[0]
            local_choice = st.selectbox(
                "Local model",
                options=local_keys,
                index=local_keys.index(default_local),
                format_func=lambda key: local_specs[key].label,
                help="로컬에서 실행할 모델을 선택합니다.",
            )
            primary_payload["model_name"] = local_specs[local_choice].label
            primary_payload["local_model_ui_key"] = local_choice
            primary_api_key_default = ""
            primary_base_url_default = ""
        elif selected_provider == "openai_api":
            model_name = st.selectbox(
                "API model",
                options=["gpt-5", "gpt-5-mini"],
                index=0 if str(settings.get("model_name", "gpt-5-mini")) == "gpt-5" else 1,
                help="OpenAI API 모델을 선택합니다.",
            )
            primary_payload.update({"model_name": model_name, "local_model_ui_key": ""})
            primary_api_key_default = primary_api_key_default or os.getenv("OPENAI_API_KEY", "")
            primary_base_url_default = primary_base_url_default or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        else:
            model_name = st.text_input(
                "Compatible model",
                value=str(settings.get("model_name", "gpt-5-mini")),
                help="OpenAI Responses API 호환 모델명을 입력합니다. 예: `qwen2.5-32b-instruct`",
            )
            primary_payload.update({"model_name": model_name, "local_model_ui_key": ""})
            primary_base_url_default = primary_base_url_default or "http://localhost:8000/v1"

        profile_keys = list(phase2_profiles.keys())
        default_profile_key = str(settings.get("rag_profile_key", DEFAULT_PHASE2_PROFILE))
        if default_profile_key not in profile_keys:
            default_profile_key = DEFAULT_PHASE2_PROFILE if DEFAULT_PHASE2_PROFILE in profile_keys else profile_keys[0]
        rag_profile_key = st.selectbox(
            "RAG profile",
            options=profile_keys,
            index=profile_keys.index(default_profile_key),
            disabled=(mode != "rag"),
            help="RAG 파이프라인 옵션 묶음입니다. 기본은 `phase2_baseline_v2`입니다.",
        )
        profile_description = str(phase2_profiles.get(rag_profile_key, {}).get("description", ""))
        if mode == "rag" and profile_description:
            st.caption(profile_description)

        compare_payload: dict[str, Any] = {}
        compare_profile_key = rag_profile_key
        compare_api_key_default = str(settings.get("compare_api_key", ""))
        compare_base_url_default = str(settings.get("compare_base_url", ""))
        if compare_mode:
            st.markdown("---")
            st.markdown("**Compare 기본 설정 (B)**")
            compare_provider = st.selectbox(
                "Provider (B)",
                options=provider_options,
                index=provider_options.index(str(settings.get("compare_provider", selected_provider)))
                if str(settings.get("compare_provider", selected_provider)) in provider_options
                else 0,
                key="compare_provider",
                help="비교 대상 B의 모델 제공자를 선택합니다.",
            )
            compare_payload["provider"] = compare_provider
            if compare_provider == "local":
                local_keys = list(local_specs.keys())
                default_compare_local = str(settings.get("compare_local_model_ui_key", local_keys[0]))
                if default_compare_local not in local_keys:
                    default_compare_local = local_keys[0]
                compare_local_choice = st.selectbox(
                    "Local model (B)",
                    options=local_keys,
                    index=local_keys.index(default_compare_local),
                    format_func=lambda key: local_specs[key].label,
                    key="compare_local_choice",
                    help="비교 대상 B의 로컬 모델을 선택합니다.",
                )
                compare_payload.update(
                    {
                        "model_name": local_specs[compare_local_choice].label,
                        "local_model_ui_key": compare_local_choice,
                    }
                )
                compare_api_key_default = ""
                compare_base_url_default = ""
            elif compare_provider == "openai_api":
                compare_model_name = st.selectbox(
                    "API model (B)",
                    options=["gpt-5", "gpt-5-mini"],
                    index=1,
                    key="compare_api_model",
                    help="비교 대상 B의 OpenAI 모델을 선택합니다.",
                )
                compare_payload.update(
                    {
                        "model_name": compare_model_name,
                        "local_model_ui_key": "",
                    }
                )
                compare_api_key_default = compare_api_key_default or os.getenv("OPENAI_API_KEY", "")
                compare_base_url_default = compare_base_url_default or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            else:
                compare_payload.update(
                    {
                        "model_name": st.text_input(
                            "Compatible model (B)",
                            value=str(settings.get("compare_model_name", "gpt-5-mini")),
                            key="compare_comp_model",
                            help="비교 대상 B의 OpenAI-compatible 모델명을 입력합니다.",
                        ),
                        "local_model_ui_key": "",
                    }
                )
                compare_base_url_default = compare_base_url_default or "http://localhost:8000/v1"
            compare_profile_key = st.selectbox(
                "RAG profile (B)",
                options=profile_keys,
                index=profile_keys.index(str(settings.get("compare_rag_profile_key", rag_profile_key)))
                if str(settings.get("compare_rag_profile_key", rag_profile_key)) in profile_keys
                else profile_keys.index(rag_profile_key),
                disabled=(mode != "rag"),
                help="비교 대상 B에서 사용할 RAG profile입니다.",
            )

        primary_max_new_tokens = int(settings.get("max_new_tokens", 768))
        primary_temperature = float(settings.get("temperature", 0.0))
        primary_top_p = float(settings.get("top_p", 1.0))
        compare_max_new_tokens = int(settings.get("compare_max_new_tokens", primary_max_new_tokens))
        compare_temperature = float(settings.get("compare_temperature", primary_temperature))
        compare_top_p = float(settings.get("compare_top_p", primary_top_p))

        backend_default = str(settings.get("embedding_backend_key", "openai_text_embedding_3_small"))
        if backend_default not in embedding_backend_keys:
            backend_default = embedding_backend_keys[0] if embedding_backend_keys else "openai_text_embedding_3_small"
        embedding_backend_key = backend_default
        routing_model = str(settings.get("routing_model", "gpt-5-mini"))
        candidate_k = int(settings.get("candidate_k", 10))
        top_k = int(settings.get("top_k", 5))
        crag_top_n = int(settings.get("crag_top_n", 5))
        vector_weight = float(settings.get("vector_weight", 0.7))
        bm25_weight = float(settings.get("bm25_weight", 0.3))
        chroma_dir = str(settings.get("chroma_dir", ""))

        with st.expander("고급 설정", expanded=False):
            st.caption("핵심 설정 외 세부 파라미터를 조정합니다.")
            if selected_provider in {"openai_api", "openai_compatible"}:
                primary_api_key_default = st.text_input(
                    "API Key",
                    type="password",
                    value=primary_api_key_default,
                    help="OpenAI/OpenAI-compatible endpoint 인증 키입니다.",
                )
                primary_base_url_default = st.text_input(
                    "Base URL",
                    value=primary_base_url_default or "https://api.openai.com/v1",
                    help="OpenAI-compatible 서버 주소입니다.",
                )
            else:
                st.caption("Local provider는 API Key/Base URL이 필요 없습니다.")

            primary_max_new_tokens = st.slider(
                "Max new tokens",
                min_value=128,
                max_value=2048,
                value=primary_max_new_tokens,
                step=32,
                help="응답 최대 길이 제한입니다.",
            )
            primary_temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=primary_temperature,
                step=0.05,
                help="높을수록 다양하게 생성합니다. QA는 낮게 권장합니다.",
            )
            primary_top_p = st.slider(
                "Top P",
                min_value=0.1,
                max_value=1.0,
                value=primary_top_p,
                step=0.05,
                help="샘플링 범위를 제한합니다.",
            )

            embedding_backend_key = st.selectbox(
                "Embedding backend",
                options=embedding_backend_keys,
                index=embedding_backend_keys.index(backend_default) if embedding_backend_keys else 0,
                disabled=(mode != "rag"),
                help="질문 임베딩에 사용할 백엔드입니다.",
            )
            routing_model = st.text_input(
                "Routing model",
                value=routing_model,
                disabled=(mode != "rag"),
                help="CRAG/라우팅 판단에 쓰는 모델입니다.",
            )

            col_c, col_d = st.columns(2)
            candidate_k = int(
                col_c.number_input(
                    "Candidate K",
                    min_value=1,
                    max_value=30,
                    value=candidate_k,
                    step=1,
                    disabled=(mode != "rag"),
                    help="초기 검색 후보 개수입니다.",
                )
            )
            top_k = int(
                col_d.number_input(
                    "Top K",
                    min_value=1,
                    max_value=20,
                    value=top_k,
                    step=1,
                    disabled=(mode != "rag"),
                    help="최종 답변에 쓰는 evidence 개수입니다.",
                )
            )
            crag_top_n = int(
                st.number_input(
                    "CRAG Top N",
                    min_value=1,
                    max_value=20,
                    value=crag_top_n,
                    step=1,
                    disabled=(mode != "rag"),
                    help="CRAG 판단 대상 상위 후보 개수입니다.",
                )
            )
            col_e, col_f = st.columns(2)
            vector_weight = float(
                col_e.slider(
                    "Vector weight",
                    min_value=0.0,
                    max_value=1.0,
                    value=vector_weight,
                    step=0.05,
                    disabled=(mode != "rag"),
                    help="벡터 유사도 비중입니다.",
                )
            )
            bm25_weight = float(
                col_f.slider(
                    "BM25 weight",
                    min_value=0.0,
                    max_value=1.0,
                    value=bm25_weight,
                    step=0.05,
                    disabled=(mode != "rag"),
                    help="키워드 위주로 더 검색합니다.",
                )
            )

            chroma_dir = st.text_input(
                "Chroma dir override (optional)",
                value=chroma_dir,
                disabled=(mode != "rag"),
                help="기본 Chroma 경로 대신 특정 DB 경로를 사용합니다.",
            )

            if compare_mode:
                st.markdown("---")
                st.markdown("**Compare 대상(B) 고급 설정**")
                if compare_payload.get("provider") in {"openai_api", "openai_compatible"}:
                    compare_api_key_default = st.text_input(
                        "API Key (B)",
                        type="password",
                        value=compare_api_key_default,
                        key="compare_api_key",
                        help="비교 대상 B endpoint 인증 키입니다.",
                    )
                    compare_base_url_default = st.text_input(
                        "Base URL (B)",
                        value=compare_base_url_default or "https://api.openai.com/v1",
                        key="compare_base_url",
                        help="비교 대상 B endpoint 주소입니다.",
                    )
                else:
                    st.caption("Compare B가 local provider이면 API Key/Base URL이 필요 없습니다.")

                compare_max_new_tokens = int(
                    st.slider(
                        "Max new tokens (B)",
                        min_value=128,
                        max_value=2048,
                        value=compare_max_new_tokens,
                        step=32,
                        help="비교 대상 B의 응답 최대 길이입니다.",
                    )
                )
                compare_temperature = float(
                    st.slider(
                        "Temperature (B)",
                        min_value=0.0,
                        max_value=1.0,
                        value=compare_temperature,
                        step=0.05,
                        help="비교 대상 B의 샘플링 온도입니다.",
                    )
                )
                compare_top_p = float(
                    st.slider(
                        "Top P (B)",
                        min_value=0.1,
                        max_value=1.0,
                        value=compare_top_p,
                        step=0.05,
                        help="비교 대상 B의 top-p 값입니다.",
                    )
                )

        primary_payload.update(
            {
                "api_key": primary_api_key_default,
                "base_url": primary_base_url_default,
                "max_new_tokens": primary_max_new_tokens,
                "temperature": primary_temperature,
                "top_p": primary_top_p,
            }
        )
        if compare_mode:
            compare_payload.update(
                {
                    "api_key": compare_api_key_default,
                    "base_url": compare_base_url_default,
                    "max_new_tokens": compare_max_new_tokens,
                    "temperature": compare_temperature,
                    "top_p": compare_top_p,
                }
            )

        if st.button("현재 설정을 프로젝트 기본값으로 저장", use_container_width=True, help="이 프로젝트에서 새 세션 생성 시 적용될 기본값으로 저장합니다."):
            store.set_project_defaults(
                state,
                state["active_project_id"],
                {
                    "mode": mode,
                    "router_debug": router_debug,
                    "answer_type_override": answer_type_override,
                    "provider": primary_payload["provider"],
                    "model_name": primary_payload["model_name"],
                    "local_model_ui_key": primary_payload.get("local_model_ui_key", ""),
                    "base_url": primary_payload.get("base_url", ""),
                    "max_new_tokens": primary_payload["max_new_tokens"],
                    "temperature": primary_payload["temperature"],
                    "top_p": primary_payload["top_p"],
                    "rag_profile_key": rag_profile_key,
                    "embedding_backend_key": embedding_backend_key,
                    "routing_model": routing_model,
                    "candidate_k": int(candidate_k),
                    "top_k": int(top_k),
                    "crag_top_n": int(crag_top_n),
                    "vector_weight": float(vector_weight),
                    "bm25_weight": float(bm25_weight),
                    "chroma_dir": chroma_dir,
                },
            )
            store.save(state)
            st.success("프로젝트 기본값을 저장했습니다.")

        session["settings"] = {
            "mode": mode,
            "compare_mode": compare_mode,
            "router_debug": router_debug,
            "answer_type_override": answer_type_override,
            "provider": primary_payload["provider"],
            "model_name": primary_payload["model_name"],
            "api_key": primary_api_key_default,
            "local_model_ui_key": primary_payload.get("local_model_ui_key", ""),
            "base_url": primary_payload.get("base_url", ""),
            "max_new_tokens": primary_payload["max_new_tokens"],
            "temperature": primary_payload["temperature"],
            "top_p": primary_payload["top_p"],
            "rag_profile_key": rag_profile_key,
            "embedding_backend_key": embedding_backend_key,
            "routing_model": routing_model,
            "candidate_k": int(candidate_k),
            "top_k": int(top_k),
            "crag_top_n": int(crag_top_n),
            "vector_weight": float(vector_weight),
            "bm25_weight": float(bm25_weight),
            "chroma_dir": chroma_dir,
            "compare_provider": compare_payload.get("provider", ""),
            "compare_model_name": compare_payload.get("model_name", ""),
            "compare_api_key": compare_api_key_default,
            "compare_local_model_ui_key": compare_payload.get("local_model_ui_key", ""),
            "compare_base_url": compare_payload.get("base_url", ""),
            "compare_max_new_tokens": compare_payload.get("max_new_tokens", primary_payload["max_new_tokens"]),
            "compare_temperature": compare_payload.get("temperature", primary_payload["temperature"]),
            "compare_top_p": compare_payload.get("top_p", primary_payload["top_p"]),
            "compare_rag_profile_key": compare_profile_key,
            "prompt_preset": preset_name,
        }
selected_quick_prompt = ""
selected_reuse_prompt = ""
hero_submit_prompt = ""
message_rows = list(session.get("messages", []) or [])

latest_assistant_id = next(
    (
        str(row.get("id", ""))
        for row in reversed(message_rows)
        if str(row.get("role", "")) == "assistant"
    ),
    "",
)

if not message_rows:
    hero_submit_prompt = render_empty_state(session=session, state=state, store=store)
    render_main_mode_pills(session=session, state=state, store=store)
else:
    st.markdown(
        f"""
<div class="qa-hero">
  <div class="qa-hero-title">RAG Internal QA Workspace</div>
  <div class="qa-hero-meta">project={projects[state['active_project_id']]['name']} · session={session.get('title', session['id'])} · baseline={DEFAULT_PHASE2_PROFILE}</div>
</div>
        """,
        unsafe_allow_html=True,
    )
    render_status_pills(
        mode=mode,
        provider=primary_payload.get("provider", ""),
        model_name=primary_payload.get("model_name", ""),
        rag_profile_key=rag_profile_key,
        compare_mode=compare_mode,
        compare_model_name=compare_payload.get("model_name", ""),
        compare_profile_key=compare_profile_key,
    )
    render_main_mode_pills(session=session, state=state, store=store)

    for message in message_rows:
        role = str(message.get("role", "assistant"))
        with st.chat_message(role):
            if role == "assistant":
                render_assistant_message(message, animate=(str(message.get("id", "")) == latest_assistant_id))
            else:
                escaped = html_lib.escape(str(message.get("content", ""))).replace("\n", "<br/>")
                st.markdown(f"<div class='qa-user-bubble'>{escaped}</div>", unsafe_allow_html=True)

    st.caption("예제 질문")
    quick_examples = [
        "이번 RFP의 예산/일정/평가 기준을 5줄로 요약해줘.",
        "A 문서와 B 문서의 계약 방식 차이를 표로 비교해줘.",
        "표 정보와 본문 정보를 함께 근거로 답변해줘.",
    ]
    quick_cols = st.columns(3)
    for index, question in enumerate(quick_examples):
        if quick_cols[index].button(
            f"예제 {index + 1}",
            key=f"quick_example_{index}",
            use_container_width=True,
            help=question,
        ):
            selected_quick_prompt = question

    recent_user_prompts: list[str] = []
    for row in reversed(message_rows):
        if str(row.get("role", "")) != "user":
            continue
        prompt_text = str(row.get("content", "")).strip()
        if not prompt_text or prompt_text in recent_user_prompts:
            continue
        recent_user_prompts.append(prompt_text)
        if len(recent_user_prompts) >= 5:
            break
    if recent_user_prompts:
        with st.expander("이전 질문 재사용", expanded=False):
            for index, question in enumerate(recent_user_prompts, start=1):
                label = f"{index}. {question[:48]}{'…' if len(question) > 48 else ''}"
                if st.button(label, key=f"reuse_prompt_{index}", use_container_width=True, help=question):
                    selected_reuse_prompt = question

if "qa_widget_open" not in st.session_state:
    st.session_state.qa_widget_open = False

st.markdown("<div class='qa-fab-sentinel'></div>", unsafe_allow_html=True)
if st.button("✦", key="teammate_fab", help="RAG QA Digital Teammate", use_container_width=False):
    st.session_state.qa_widget_open = not st.session_state.qa_widget_open
    st.rerun()

if st.session_state.qa_widget_open:
    widget_box = st.container(border=True)
    with widget_box:
        header_cols = st.columns([5, 1])
        header_cols[0].markdown(
            (
                "<div class='qa-context-widget-header'>"
                "<div class='qa-context-widget-title'>RAG QA Digital Teammate</div>"
                "<div class='qa-context-widget-sub'>Context Companion</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        if header_cols[1].button("✕", key="toggle_widget_close", help="패널 닫기", use_container_width=True):
            st.session_state.qa_widget_open = False
            st.rerun()

        tabs = st.tabs(["Artifacts", "Sources", "Debug"])
        latest_assistant = next(
            (row for row in reversed(session.get("messages", []) or []) if str(row.get("role", "")) == "assistant"),
            None,
        )
        with tabs[0]:
            render_artifact_panel(session)
        with tabs[1]:
            if latest_assistant:
                render_source_panel(latest_assistant)
            else:
                st.info("아직 assistant 응답이 없습니다.")
        with tabs[2]:
            if latest_assistant:
                render_debug_panel(latest_assistant, show_router=bool(router_debug))
            else:
                st.info("아직 debug 데이터가 없습니다.")
        st.markdown("---")
        widget_mode_tabs = st.columns(2)
        widget_mode_tabs[0].button("Chat", use_container_width=True, key="widget_chat_tab")
        widget_mode_tabs[1].button("Voice", use_container_width=True, key="widget_voice_tab")
        st.caption("Attach · Emoji · Deep Search")

chat_prompt = ""
if message_rows:
    st.caption("Enter로 전송, Shift+Enter로 줄바꿈")
    chat_prompt = st.chat_input("Ask anything...")
pending_prompt = chat_prompt or hero_submit_prompt or selected_quick_prompt or selected_reuse_prompt
is_regeneration = False
if regen_clicked and not pending_prompt:
    pending_prompt = get_last_user_prompt(session.get("messages", []) or [])
    is_regeneration = bool(pending_prompt)

if pending_prompt:
    if st.session_state.get("stop_requested"):
        st.session_state.stop_requested = False
        st.warning("요청 실행이 취소되었습니다.")
        pending_prompt = ""

if pending_prompt:
    history_source = session_messages_for_history(session.get("messages", []) or [], drop_last_assistant=is_regeneration)
    if not is_regeneration:
        session["messages"].append(
            {
                "id": f"msg_{uuid4().hex[:10]}",
                "role": "user",
                "content": pending_prompt,
                "created_at": now_iso(),
            }
        )
    rag_cfg = {
        "embedding_backend_key": embedding_backend_key,
        "routing_model": routing_model,
        "candidate_k": int(candidate_k),
        "top_k": int(top_k),
        "crag_top_n": int(crag_top_n),
        "vector_weight": float(vector_weight),
        "bm25_weight": float(bm25_weight),
        "chroma_dir": chroma_dir,
    }
    primary_profile_options = dict(phase2_profiles.get(rag_profile_key, {}).get("options", {}) or {})
    if not primary_profile_options:
        st.error(f"RAG profile options를 찾지 못했습니다: {rag_profile_key}")
        st.stop()

    router_prediction: dict[str, Any] = {}
    router_used: dict[str, Any] = {}
    try:
        with st.status("생각 중...", expanded=False) as status:
            status.write("answer type router 실행 중...")
            router_history = build_history_for_adapter(history_source, max_turns=14)
            router_prediction = predict_answer_type(
                question=pending_prompt,
                history=router_history,
                confidence_threshold=0.56,
                low_conf_fallback="comparison_safe",
            )
            router_used = apply_answer_type_override(
                router_prediction,
                None if answer_type_override == "Auto" else answer_type_override,
            )
            st.session_state.latest_user_question = pending_prompt
            st.session_state.latest_router_result = router_prediction
            st.session_state.latest_answer_type_used = str(router_used.get("answer_type", ""))
            st.session_state.latest_route_used = str(router_used.get("route", ""))
            st.session_state.latest_model_used = f"{primary_payload.get('provider', '')}:{primary_payload.get('model_name', '')}"
            status.write(
                "router: "
                f"{router_used.get('answer_type', '-')} -> {router_used.get('route', '-')} "
                f"(confidence={float(router_used.get('confidence', 0.0) or 0.0):.2f})"
            )
            status.write("질문 분석 완료. 답변 생성 중...")
            run_a = run_one_turn(
                session=session,
                prompt=pending_prompt,
                history_messages=history_source,
                provider_payload=primary_payload,
                mode=mode,
                rag_cfg=rag_cfg,
                rag_profile_key=rag_profile_key,
                rag_profile_options=primary_profile_options,
                local_specs=local_specs,
                preset_name=preset_name,
                router_result=router_used,
            )
            run_a["label"] = "A"
            status.write("1차 답변 생성 완료")

            runs = [run_a]
            if compare_mode:
                status.write("비교 대상(B) 실행 중...")
                compare_profile_options = dict(phase2_profiles.get(compare_profile_key, {}).get("options", {}) or {})
                run_b = run_one_turn(
                    session=session,
                    prompt=pending_prompt,
                    history_messages=history_source,
                    provider_payload=compare_payload,
                    mode=mode,
                    rag_cfg=rag_cfg,
                    rag_profile_key=compare_profile_key,
                    rag_profile_options=compare_profile_options or primary_profile_options,
                    local_specs=local_specs,
                    preset_name=preset_name,
                    router_result=router_used,
                )
                run_b["label"] = "B"
                runs.append(run_b)
            status.update(label="응답 준비 완료", state="complete")
    except Exception as exc:  # noqa: BLE001
        runs = [
            {
                "id": f"run_{uuid4().hex[:10]}",
                "label": "A",
                "content": friendly_error_message(exc),
                "latency_sec": 0.0,
                "mode": mode,
                "model_label": "error",
                "route": "error",
                "profile": {"error": str(exc)},
                "top_k": 0,
                "chunks": [],
                "retrieval_context": "",
                "router": dict(router_used or router_prediction or {}),
                "answer_type_used": str((router_used or router_prediction or {}).get("answer_type", "")),
                "router_route_used": str((router_used or router_prediction or {}).get("route", "")),
            }
        ]

    assistant_message = {
        "id": f"msg_{uuid4().hex[:10]}",
        "role": "assistant",
        "created_at": now_iso(),
        "compare_mode": bool(compare_mode),
        "typing_effect": True,
        "runs": runs,
    }
    session["messages"].append(assistant_message)
    artifacts = extract_artifacts_from_runs(runs)
    if artifacts:
        session.setdefault("artifacts", [])
        session["artifacts"].extend(artifacts)

    store.update_session(state, session)
    store.save(state)
    st.rerun()

store.update_session(state, session)
store.save(state)

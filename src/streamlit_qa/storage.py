from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from streamlit_qa.config import DEFAULT_PHASE2_PROFILE


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_name(value: str, fallback: str) -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    return " ".join(text.split())


class SessionStore:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.base_dir = (self.project_root / "rag_outputs" / "streamlit_internal_qa").resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir = self.base_dir / "uploads"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.base_dir / "state.json"

    def default_state(self) -> dict[str, Any]:
        return {
            "projects": {
                "default": {
                    "id": "default",
                    "name": "RAG Internal QA",
                    "created_at": _now_iso(),
                    "updated_at": _now_iso(),
                    "defaults": {
                        "mode": "rag",
                        "provider": "openai_api",
                        "model_name": "gpt-5-mini",
                        "rag_profile_key": DEFAULT_PHASE2_PROFILE,
                        "embedding_backend_key": "openai_text_embedding_3_small",
                        "routing_model": "gpt-5-mini",
                        "candidate_k": 10,
                        "top_k": 5,
                        "crag_top_n": 5,
                        "vector_weight": 0.7,
                        "bm25_weight": 0.3,
                    },
                }
            },
            "project_order": ["default"],
            "sessions": {},
            "session_order": [],
            "active_project_id": "default",
            "active_session_id": "",
        }

    def load(self) -> dict[str, Any]:
        if not self.state_path.exists():
            state = self.default_state()
            self.save(state)
            return state
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            payload = self.default_state()
            self.save(payload)
            return payload
        return self._normalize(payload)

    def save(self, state: dict[str, Any]) -> None:
        normalized = self._normalize(state)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    def _normalize(self, state: dict[str, Any]) -> dict[str, Any]:
        base = self.default_state()
        base.update({key: value for key, value in state.items() if key in base})
        projects = dict(base.get("projects", {}) or {})
        if "default" not in projects:
            projects["default"] = self.default_state()["projects"]["default"]
        base["projects"] = projects

        project_order = [pid for pid in (base.get("project_order", []) or []) if pid in projects]
        if "default" not in project_order:
            project_order.insert(0, "default")
        for pid in projects:
            if pid not in project_order:
                project_order.append(pid)
        base["project_order"] = project_order

        sessions = dict(base.get("sessions", {}) or {})
        base["sessions"] = sessions
        session_order = [sid for sid in (base.get("session_order", []) or []) if sid in sessions]
        for sid in sessions:
            if sid not in session_order:
                session_order.append(sid)
        base["session_order"] = session_order

        active_project_id = str(base.get("active_project_id", "default") or "default")
        if active_project_id not in projects:
            active_project_id = "default"
        base["active_project_id"] = active_project_id

        active_session_id = str(base.get("active_session_id", "") or "")
        if active_session_id not in sessions:
            active_session_id = ""
        base["active_session_id"] = active_session_id
        return base

    def create_project(self, state: dict[str, Any], name: str) -> str:
        project_id = f"project_{uuid4().hex[:10]}"
        project_name = _sanitize_name(name, "Untitled Project")
        state.setdefault("projects", {})
        state["projects"][project_id] = {
            "id": project_id,
            "name": project_name,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "defaults": dict(state["projects"]["default"]["defaults"]),
        }
        state.setdefault("project_order", [])
        state["project_order"].append(project_id)
        state["active_project_id"] = project_id
        return project_id

    def set_project_defaults(self, state: dict[str, Any], project_id: str, defaults: dict[str, Any]) -> None:
        if project_id not in state.get("projects", {}):
            return
        project = state["projects"][project_id]
        merged = dict(project.get("defaults", {}) or {})
        merged.update(defaults or {})
        project["defaults"] = merged
        project["updated_at"] = _now_iso()

    def create_session(
        self,
        state: dict[str, Any],
        *,
        project_id: str,
        title: str = "",
        settings: dict[str, Any] | None = None,
    ) -> str:
        if project_id not in state.get("projects", {}):
            project_id = "default"
        session_id = f"session_{uuid4().hex[:12]}"
        merged_settings = dict(state["projects"][project_id].get("defaults", {}) or {})
        if settings:
            merged_settings.update(settings)
        session_payload = {
            "id": session_id,
            "project_id": project_id,
            "title": _sanitize_name(title, "새 채팅"),
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "settings": merged_settings,
            "messages": [],
            "attachments": [],
            "artifacts": [],
        }
        state.setdefault("sessions", {})
        state["sessions"][session_id] = session_payload
        state.setdefault("session_order", [])
        state["session_order"].insert(0, session_id)
        state["active_session_id"] = session_id
        state["active_project_id"] = project_id
        return session_id

    def touch_session_title(self, session: dict[str, Any]) -> None:
        title = str(session.get("title", "")).strip()
        if title and title != "새 채팅":
            return
        for row in session.get("messages", []) or []:
            if str(row.get("role", "")) != "user":
                continue
            text = str(row.get("content", "")).strip()
            if not text:
                continue
            snippet = text.replace("\n", " ")[:38].strip()
            if snippet:
                session["title"] = snippet
                break

    def update_session(self, state: dict[str, Any], session: dict[str, Any]) -> None:
        session_id = str(session.get("id", "")).strip()
        if not session_id:
            return
        session["updated_at"] = _now_iso()
        self.touch_session_title(session)
        state.setdefault("sessions", {})
        state["sessions"][session_id] = session
        order = state.setdefault("session_order", [])
        if session_id in order:
            order.remove(session_id)
        order.insert(0, session_id)
        state["active_session_id"] = session_id
        state["active_project_id"] = session.get("project_id", state.get("active_project_id", "default"))

    def clear_session_messages(self, state: dict[str, Any], session_id: str) -> None:
        session = state.get("sessions", {}).get(session_id)
        if not session:
            return
        session["messages"] = []
        session["artifacts"] = []
        session["updated_at"] = _now_iso()

    def delete_session(self, state: dict[str, Any], session_id: str) -> None:
        if session_id in state.get("sessions", {}):
            state["sessions"].pop(session_id, None)
        order = state.get("session_order", []) or []
        state["session_order"] = [sid for sid in order if sid != session_id]
        if state.get("active_session_id") == session_id:
            state["active_session_id"] = state["session_order"][0] if state["session_order"] else ""

    def session_upload_path(self, session_id: str) -> Path:
        path = (self.upload_dir / session_id).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

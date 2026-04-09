from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Any

from config import SQLITE_PATH, ensure_directories, get_metadata_csv_path, get_raw_document_root


TABLE_NAME = "documents"
CSV_ENCODING = "utf-8-sig"


def _connect() -> sqlite3.Connection:
    ensure_directories()
    connection = sqlite3.connect(SQLITE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def _safe_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _normalize_row(row: dict[str, str], raw_root: Path) -> dict[str, Any]:
    file_name = row.get("파일명", "").strip()
    source_path = raw_root / file_name if file_name else None

    return {
        "notice_number": (row.get("공고 번호") or "").strip() or None,
        "notice_round": _safe_float(row.get("공고 차수")),
        "project_name": (row.get("사업명") or "").strip(),
        "project_budget": _safe_float(row.get("사업 금액")),
        "agency": (row.get("발주 기관") or "").strip(),
        "published_at": (row.get("공개 일자") or "").strip() or None,
        "bid_start_at": (row.get("입찰 참여 시작일") or "").strip() or None,
        "bid_end_at": (row.get("입찰 참여 마감일") or "").strip() or None,
        "project_summary": (row.get("사업 요약") or "").strip(),
        "file_format": (row.get("파일형식") or "").strip().lower(),
        "file_name": file_name,
        "csv_text": (row.get("텍스트") or "").strip(),
        "source_path": str(source_path.resolve()) if source_path and source_path.exists() else None,
    }


def bootstrap_metadata_db(force: bool = False) -> Path:
    csv_path = get_metadata_csv_path()
    raw_root = get_raw_document_root()

    with _connect() as connection:
        connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                notice_number TEXT,
                notice_round REAL,
                project_name TEXT NOT NULL,
                project_budget REAL,
                agency TEXT NOT NULL,
                published_at TEXT,
                bid_start_at TEXT,
                bid_end_at TEXT,
                project_summary TEXT,
                file_format TEXT,
                file_name TEXT UNIQUE,
                csv_text TEXT,
                source_path TEXT
            )
            """
        )
        connection.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_agency ON {TABLE_NAME}(agency)"
        )
        connection.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_project_name ON {TABLE_NAME}(project_name)"
        )

        existing_count = connection.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
        linked_source_count = connection.execute(
            f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE source_path IS NOT NULL AND source_path != ''"
        ).fetchone()[0]
        if existing_count > 0 and linked_source_count > 0 and not force:
            return SQLITE_PATH

        connection.execute(f"DELETE FROM {TABLE_NAME}")
        with csv_path.open("r", encoding=CSV_ENCODING, newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            rows = [_normalize_row(row, raw_root) for row in reader]

        connection.executemany(
            f"""
            INSERT INTO {TABLE_NAME} (
                notice_number, notice_round, project_name, project_budget, agency,
                published_at, bid_start_at, bid_end_at, project_summary, file_format,
                file_name, csv_text, source_path
            )
            VALUES (
                :notice_number, :notice_round, :project_name, :project_budget, :agency,
                :published_at, :bid_start_at, :bid_end_at, :project_summary, :file_format,
                :file_name, :csv_text, :source_path
            )
            """,
            rows,
        )
        connection.commit()
    return SQLITE_PATH


def get_dataset_summary() -> dict[str, Any]:
    bootstrap_metadata_db()
    with _connect() as connection:
        document_count = connection.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
        agency_count = connection.execute(
            f"SELECT COUNT(DISTINCT agency) FROM {TABLE_NAME}"
        ).fetchone()[0]
        rows = connection.execute(
            f"SELECT file_format, COUNT(*) AS count FROM {TABLE_NAME} GROUP BY file_format ORDER BY count DESC"
        ).fetchall()

    return {
        "document_count": document_count,
        "agency_count": agency_count,
        "file_format_counts": {row["file_format"]: row["count"] for row in rows},
    }


def list_agencies() -> list[str]:
    bootstrap_metadata_db()
    with _connect() as connection:
        rows = connection.execute(
            f"SELECT DISTINCT agency FROM {TABLE_NAME} WHERE agency != '' ORDER BY agency"
        ).fetchall()
    return [row["agency"] for row in rows]


def query_documents(
    agency: str | None = None,
    keyword: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    bootstrap_metadata_db()

    sql = f"""
        SELECT
            notice_number,
            notice_round,
            project_name,
            project_budget,
            agency,
            published_at,
            bid_start_at,
            bid_end_at,
            project_summary,
            file_format,
            file_name,
            csv_text,
            source_path
        FROM {TABLE_NAME}
        WHERE 1 = 1
    """
    params: list[Any] = []

    if agency:
        sql += " AND agency = ?"
        params.append(agency)
    if keyword:
        sql += " AND (project_name LIKE ? OR project_summary LIKE ? OR csv_text LIKE ?)"
        like_value = f"%{keyword}%"
        params.extend([like_value, like_value, like_value])

    sql += " ORDER BY published_at DESC, project_name ASC LIMIT ?"
    params.append(limit)

    with _connect() as connection:
        rows = connection.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def get_document_by_file_name(file_name: str) -> dict[str, Any] | None:
    bootstrap_metadata_db()
    with _connect() as connection:
        row = connection.execute(
            f"SELECT * FROM {TABLE_NAME} WHERE file_name = ?",
            (file_name,),
        ).fetchone()
    return dict(row) if row else None


def list_source_documents() -> list[dict[str, Any]]:
    bootstrap_metadata_db()
    with _connect() as connection:
        rows = connection.execute(
            f"""
            SELECT
                file_name,
                file_format,
                source_path,
                project_name,
                agency,
                published_at
            FROM {TABLE_NAME}
            WHERE source_path IS NOT NULL AND source_path != ''
            ORDER BY published_at DESC, project_name ASC
            """
        ).fetchall()
    return [dict(row) for row in rows]

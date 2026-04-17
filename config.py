from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PARSED_DATA_DIR = DATA_DIR / "parsed"
REPORTS_DIR = BASE_DIR / "reports"
SQLITE_PATH = DATA_DIR / "metadata.db"

METADATA_CSV_CANDIDATES = [
    DATA_DIR / "data_list.csv",
    BASE_DIR / "data_list.csv",
]

RAW_DOCUMENT_DIR_CANDIDATES = [
    BASE_DIR / "files",
    RAW_DATA_DIR,
]

CHUNK_PARAMETER_GRID = [
    {"chunk_size": 500, "overlap": 100},
    {"chunk_size": 1000, "overlap": 200},
    {"chunk_size": 1500, "overlap": 300},
]


def _first_existing_path(candidates: list[Path]) -> Path:
    non_empty_dirs: list[Path] = []
    for candidate in candidates:
        if candidate.is_dir():
            try:
                next(candidate.iterdir())
                non_empty_dirs.append(candidate)
            except StopIteration:
                continue
        elif candidate.exists():
            return candidate
    if non_empty_dirs:
        return non_empty_dirs[0]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def get_metadata_csv_path() -> Path:
    return _first_existing_path(METADATA_CSV_CANDIDATES)


def get_raw_document_root() -> Path:
    return _first_existing_path(RAW_DOCUMENT_DIR_CANDIDATES)


def ensure_directories() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    RAW_DATA_DIR.mkdir(exist_ok=True)
    PARSED_DATA_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

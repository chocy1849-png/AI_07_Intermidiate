import argparse
import shutil
from pathlib import Path

from langchain_chroma import Chroma

from rag_system.config import SETTINGS
from table_pipeline.rag_utils import (
    build_embeddings,
    build_text_splitter,
    iter_source_files,
    load_csv_metadata,
    make_documents,
)


def ingest_documents(
    base_dir: Path = SETTINGS.base_dir,
    db_path: Path = SETTINGS.db_path,
    rebuild: bool = False,
) -> int:
    if not base_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {base_dir}")

    files = list(iter_source_files(base_dir))
    if not files:
        raise FileNotFoundError(f"No PDF or HWP files found in: {base_dir}")

    if rebuild and db_path.exists():
        shutil.rmtree(db_path)

    splitter = build_text_splitter()
    metadata_by_filename = load_csv_metadata()
    documents = []

    for file_path in files:
        documents.extend(make_documents(file_path, splitter, metadata_by_filename=metadata_by_filename))

    if not documents:
        raise RuntimeError("Text extraction produced no chunks. Check the source documents.")

    Chroma.from_documents(
        documents=documents,
        embedding=build_embeddings(),
        persist_directory=str(db_path),
    )
    return len(documents)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or update the RFP vector database.")
    parser.add_argument("--rebuild", action="store_true", help="Delete the existing vector DB before ingesting.")
    args = parser.parse_args()

    count = ingest_documents(rebuild=args.rebuild)
    print(f"Ingestion completed. Stored {count} chunks in {SETTINGS.db_path}.")


if __name__ == "__main__":
    main()

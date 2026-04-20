from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from eval_utils import write_csv, write_json
from scenario_b_phase2.corpus_schema import schema_field_spec
from scenario_b_phase2.parser_routing import build_parser_routing_rows


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Preview parser routing decisions for Phase2 Track A")
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument(
        "--processed-documents-path",
        default=str(root / "processed_data" / "processed_documents.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(root / "rag_outputs" / "phase2_trackA_preview"),
    )
    parser.add_argument("--limit", type=int, default=0, help="Read first N docs only (0 means all).")
    return parser.parse_args()


def _read_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            raw = line.strip()
            if not raw:
                continue
            rows.append(json.loads(raw))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def _build_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    route_counter = Counter(str(row.get("parser_route", "")) for row in rows)
    high_priority_counter = Counter(
        str(row.get("parser_route", ""))
        for row in rows
        if int(row.get("is_high_priority", 0) or 0) == 1
    )
    output: list[dict[str, Any]] = []
    total = len(rows)
    for route, count in sorted(route_counter.items(), key=lambda item: (-item[1], item[0])):
        output.append(
            {
                "parser_route": route,
                "doc_count": count,
                "ratio": round(count / total, 4) if total else 0.0,
                "high_priority_count": high_priority_counter.get(route, 0),
            }
        )
    return output


def _write_schema_markdown(path: Path) -> None:
    lines = [
        "# Enriched Chunk Schema (Phase2 Draft)",
        "",
        "| field | description |",
        "|---|---|",
    ]
    for row in schema_field_spec():
        lines.append(f"| {row['field']} | {row['description']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    processed_documents_path = Path(args.processed_documents_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    documents = _read_jsonl(processed_documents_path, limit=max(0, args.limit))
    routing_rows = build_parser_routing_rows(documents)
    summary_rows = _build_summary_rows(routing_rows)
    high_priority_rows = [row for row in routing_rows if int(row.get("is_high_priority", 0) or 0) == 1]
    high_priority_rows = sorted(high_priority_rows, key=lambda row: int(row.get("route_score", 0) or 0), reverse=True)

    write_csv(output_dir / "parser_routing_decisions.csv", routing_rows)
    write_csv(output_dir / "parser_routing_summary.csv", summary_rows)
    write_csv(output_dir / "parser_routing_high_priority.csv", high_priority_rows[:300])
    _write_schema_markdown(output_dir / "enriched_chunk_schema.md")
    write_json(
        output_dir / "parser_routing_manifest.json",
        {
            "project_root": str(project_root),
            "processed_documents_path": str(processed_documents_path),
            "document_count": len(documents),
            "output_dir": str(output_dir),
        },
    )

    print(f"[done] parser routing preview: {output_dir}")
    print(f"[done] documents: {len(documents)}")


if __name__ == "__main__":
    main()

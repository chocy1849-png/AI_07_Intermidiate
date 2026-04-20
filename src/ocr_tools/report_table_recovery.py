import argparse
import json
from pathlib import Path


def clip(text: str, limit: int = 500) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def safe_console(text: str) -> str:
    return text.encode("cp949", errors="replace").decode("cp949")


def load_tables(payload_path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    tables: list[dict] = []
    for page in payload.get("pages", []):
        for table in page.get("tables", []):
            tables.append(table)
    return payload, tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a human-readable table recovery report.")
    parser.add_argument("--json", type=Path, required=True, help="Artifact JSON path.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of tables to print.")
    parser.add_argument(
        "--sort-by",
        choices=["recovery", "rows", "area"],
        default="recovery",
        help="How to sort printed tables.",
    )
    args = parser.parse_args()

    payload, tables = load_tables(args.json)
    if payload.get("status") != "ok":
        print(f"source: {payload.get('source_name')}")
        print(f"status: {payload.get('status')}")
        print(f"reason: {payload.get('reason')}")
        return

    if args.sort_by == "rows":
        tables = sorted(tables, key=lambda item: (item["row_count"], item["col_count"]), reverse=True)
    elif args.sort_by == "area":
        tables = sorted(tables, key=lambda item: item["bbox_area_ratio"], reverse=True)
    else:
        tables = sorted(tables, key=lambda item: item["text_recovery_percent"], reverse=True)

    print(f"source: {payload['source_name']}")
    print(f"tables kept: {payload['kept_table_count']} / {payload['raw_table_count']} ({payload['table_keep_percent']}%)")
    print(f"avg table recovery: {payload['avg_table_recovery_percent']}%")
    print()

    for index, table in enumerate(tables[: args.top_k], start=1):
        print(
            f"[table {index}] page={table['page']} rows={table['row_count']} cols={table['col_count']} "
            f"area={table['bbox_area_ratio']} recovery={table['text_recovery_percent']}%"
        )
        print("original:")
        print(safe_console(clip(table.get("original_text", ""), 700)))
        print()
        print("markdown:")
        print(safe_console(clip(table.get("markdown", ""), 700)))
        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

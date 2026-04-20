import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ocr.select_documents import score_table_structure
from rag_system.rag_utils import clean_text, extract_text


SPLIT_PATTERN = re.compile(r"[ \t]{2,}")
NUMERIC_PATTERN = re.compile(r"\d")


@dataclass
class TableBlock:
    start_line: int
    end_line: int
    row_count: int
    structured_row_count: int
    modal_column_count: int
    consistent_row_ratio: float
    multi_column_row_ratio: float
    numeric_row_ratio: float
    block_score: float
    text_recovery_percent: float
    rows: list[list[str]]
    original_lines: list[str]


def split_columns(line: str) -> list[str]:
    return [part.strip() for part in SPLIT_PATTERN.split(line.strip()) if part.strip()]


def group_structured_blocks(structured_indices: list[int], gap: int = 2) -> list[list[int]]:
    blocks: list[list[int]] = []
    current: list[int] = []
    for idx in structured_indices:
        if not current or idx <= current[-1] + gap:
            current.append(idx)
        else:
            blocks.append(current)
            current = [idx]
    if current:
        blocks.append(current)
    return blocks


def expand_block(lines: list[str], block: list[int], margin: int = 1) -> tuple[int, int, list[str]]:
    start = max(0, block[0] - margin)
    end = min(len(lines), block[-1] + margin + 1)
    return start, end, lines[start:end]


def score_block(block_lines: list[str], structured_row_count: int) -> TableBlock | None:
    parsed_rows = [split_columns(line) for line in block_lines]
    multi_column_rows = [row for row in parsed_rows if len(row) >= 2]

    if len(block_lines) < 3 or len(multi_column_rows) < 2:
        return None

    column_counts = [len(row) for row in multi_column_rows]
    modal_column_count, modal_count = Counter(column_counts).most_common(1)[0]
    consistent_row_ratio = modal_count / len(multi_column_rows)
    multi_column_row_ratio = len(multi_column_rows) / len(block_lines)
    numeric_row_ratio = sum(1 for line in block_lines if NUMERIC_PATTERN.search(line)) / len(block_lines)

    # This score prefers blocks that look like repeated rows with stable column counts.
    block_score = (
        len(multi_column_rows) * 1.5
        + modal_column_count * 1.0
        + consistent_row_ratio * 4.0
        + multi_column_row_ratio * 3.0
        + numeric_row_ratio * 1.5
        + structured_row_count * 0.3
    )

    original_compact = "".join(re.sub(r"\s+", "", line) for line in block_lines)
    extracted_compact = "".join(re.sub(r"\s+", "", "".join(row)) for row in multi_column_rows)
    if original_compact:
        text_recovery_percent = round((len(extracted_compact) / len(original_compact)) * 100, 2)
    else:
        text_recovery_percent = 0.0

    return TableBlock(
        start_line=0,
        end_line=0,
        row_count=len(block_lines),
        structured_row_count=structured_row_count,
        modal_column_count=modal_column_count,
        consistent_row_ratio=round(consistent_row_ratio, 2),
        multi_column_row_ratio=round(multi_column_row_ratio, 2),
        numeric_row_ratio=round(numeric_row_ratio, 2),
        block_score=round(block_score, 2),
        text_recovery_percent=text_recovery_percent,
        rows=multi_column_rows,
        original_lines=block_lines,
    )


def extract_likely_tables(file_path: Path, min_block_score: float = 10.0) -> list[TableBlock]:
    text = clean_text(extract_text(file_path))
    lines = text.splitlines()
    _, _, structured_indices = score_table_structure(lines)
    raw_blocks = group_structured_blocks(structured_indices)

    table_blocks: list[TableBlock] = []
    for raw_block in raw_blocks:
        start, end, block_lines = expand_block(lines, raw_block)
        candidate = score_block(block_lines, structured_row_count=len(raw_block))
        if candidate is None or candidate.block_score < min_block_score:
            continue
        candidate.start_line = start + 1
        candidate.end_line = end
        table_blocks.append(candidate)

    return sorted(table_blocks, key=lambda item: item.block_score, reverse=True)


def load_top_document(priority_json: Path, rank: int) -> Path:
    data = json.loads(priority_json.read_text(encoding="utf-8"))
    return Path(data[rank - 1]["source_path"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract likely table blocks from a high-priority document.")
    parser.add_argument("--priority-json", type=Path, default=Path("ocr/document_priorities.json"))
    parser.add_argument("--rank", type=int, default=1, help="Rank from the priority JSON to inspect.")
    parser.add_argument("--file", type=Path, default=None, help="Optional direct file path instead of using rank.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of table blocks to print.")
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    file_path = args.file if args.file is not None else load_top_document(args.priority_json, args.rank)
    blocks = extract_likely_tables(file_path)

    payload = {
        "source_path": str(file_path.resolve()),
        "table_block_count": len(blocks),
        "blocks": [asdict(block) for block in blocks],
    }

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"source: {file_path.name}")
    print(f"table blocks: {len(blocks)}")
    print()
    for index, block in enumerate(blocks[: args.top_k], start=1):
        print(
            f"{index}. lines {block.start_line}-{block.end_line} | score={block.block_score} | "
            f"rows={block.row_count} | modal_cols={block.modal_column_count} | "
            f"recovery={block.text_recovery_percent}%"
        )
        print("   original:")
        for line in block.original_lines[:6]:
            print(f"   {line.encode('unicode_escape').decode('ascii')}")
        print("   parsed rows:")
        for row in block.rows[:6]:
            safe_row = [cell.encode("unicode_escape").decode("ascii") for cell in row]
            print(f"   {safe_row}")
        print()


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "rag_outputs"
B04_MANIFEST_PATH = OUTPUT_DIR / "b04_candidates" / "b04_candidate_manifest.json"
SMOKE_COMPARE_CSV = OUTPUT_DIR / "b04_smoke_compare.csv"
MINI_COMPARE_CSV = OUTPUT_DIR / "b04_mini_compare.csv"
FULL_COMPARE_CSV = OUTPUT_DIR / "b04_full_compare.csv"
B04_REPORT_JSON = OUTPUT_DIR / "b04_report.json"


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def main() -> None:
    parser = argparse.ArgumentParser(description="B-04 compare report builder")
    parser.add_argument("--manifest-path", default=str(B04_MANIFEST_PATH))
    args = parser.parse_args()

    manifest_rows = json.loads(Path(args.manifest_path).read_text(encoding="utf-8"))
    by_key = {row["candidate_key"]: row for row in manifest_rows}
    smoke_rows = read_csv(SMOKE_COMPARE_CSV)
    mini_rows = read_csv(MINI_COMPARE_CSV)
    full_rows = read_csv(FULL_COMPARE_CSV)

    report = {
        "candidates": manifest_rows,
        "smoke_compare": smoke_rows,
        "mini_compare": mini_rows,
        "full_compare": full_rows,
    }
    B04_REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if mini_rows:
        winner = mini_rows[0]
        candidate = by_key.get(winner["candidate_key"], {})
        print(
            f"[요약] mini winner={winner['candidate_key']} "
            f"| chunk={candidate.get('chunk_size')} overlap={candidate.get('overlap')} mode={candidate.get('mode')}"
        )
    if full_rows:
        print(f"[요약] full compare rows={len(full_rows)}")
    print(f"[완료] report: {B04_REPORT_JSON}")


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path

from extract_hwp_artifacts import extract_hwp_artifacts


ROOT_DIR = Path(__file__).resolve().parent.parent
FILES_DIR = ROOT_DIR / "files"
OUTPUT_JSON = ROOT_DIR / "ocr" / "hwp_review_pack.json"
OUTPUT_MD = ROOT_DIR / "ocr" / "hwp_review_pack.md"

SAMPLE_PATTERNS = (
    "벤처기업협회",
    "국방과학연구소_기록관리시스템",
    "한국사학진흥재단",
    "인천일자리플랫폼 정보시스템 구축 ISP",
    "보험개발원_실손보험 청구 전산화",
)


def select_files() -> list[Path]:
    all_hwps = sorted(FILES_DIR.glob("*.hwp"))
    selected: list[Path] = []
    for pattern in SAMPLE_PATTERNS:
        match = next((path for path in all_hwps if pattern in path.name), None)
        if match is not None:
            selected.append(match)
    return selected


def preview(text: str, limit: int = 280) -> str:
    flat = text.replace("\n", " / ").strip()
    if len(flat) <= limit:
        return flat
    return flat[:limit] + "..."


def bucket_rows(payload: dict, bucket: str, limit: int = 8) -> list[dict]:
    rows = [table for table in payload["tables"] if table["storage_bucket"] == bucket]
    if bucket == "structural_table":
        rows.sort(key=lambda t: (-t["data_score"], -t["header_group_hits"], -t["row_hint_count"]))
    elif bucket == "section_header_block":
        rows.sort(key=lambda t: (t["table_index"]))
    elif bucket == "explanatory_block":
        rows.sort(key=lambda t: (-t["data_score"], -t["row_hint_count"], t["table_index"]))
    else:
        rows.sort(key=lambda t: (-t["data_score"], -t["row_hint_count"], t["table_index"]))
    return [
        {
            "table_index": row["table_index"],
            "section": row["section"],
            "record_start_index": row["record_start_index"],
            "first_pass_classification": row["first_pass_classification"],
            "final_classification": row["final_classification"],
            "storage_bucket": row["storage_bucket"],
            "row_hint_count": row["row_hint_count"],
            "paragraph_count": row["paragraph_count"],
            "data_score": row["data_score"],
            "header_group_hits": row["header_group_hits"],
            "missing_signals": row["missing_signals"],
            "section_header_role": row.get("section_header_role"),
            "linked_parent_table_index": row.get("linked_parent_table_index"),
            "linked_child_table_indices": row.get("linked_child_table_indices", []),
            "preview": preview(row["text"]),
        }
        for row in rows[:limit]
    ]


def build_file_report(file_path: Path) -> dict:
    payload = extract_hwp_artifacts(file_path, save_images=False, output_dir=ROOT_DIR / "ocr")
    return {
        "source_name": payload["source_name"],
        "source_path": payload["source_path"],
        "counts": {
            "table_count": payload["table_count"],
            "structural_table_count": payload["structural_table_count"],
            "section_header_block_count": payload["section_header_block_count"],
            "explanatory_block_count": payload["explanatory_block_count"],
            "discarded_table_count": payload["discarded_table_count"],
            "excluded_table_count": payload["excluded_table_count"],
            "image_count": payload["image_count"],
        },
        "structural_samples": bucket_rows(payload, "structural_table"),
        "section_header_samples": bucket_rows(payload, "section_header_block"),
        "explanatory_samples": bucket_rows(payload, "explanatory_block"),
        "discarded_samples": bucket_rows(payload, "discarded"),
    }


def build_markdown(report: dict) -> str:
    lines: list[str] = []
    lines.append("# HWP Review Pack")
    lines.append("")
    lines.append("이 파일은 원본 HWP와 추출 결과를 사람이 비교하기 쉽게 만든 검토 팩입니다.")
    lines.append("현재 추출기는 페이지 번호를 갖고 있지 않으므로, `table_index`, `section`, `record_start_index`, `preview`를 기준으로 원본 HWP에서 대응 블록을 찾습니다.")
    lines.append("")

    for item in report["files"]:
        lines.append(f"## {item['source_name']}")
        lines.append("")
        lines.append(f"- source_path: {item['source_path']}")
        lines.append(f"- table_count: {item['counts']['table_count']}")
        lines.append(f"- structural_table_count: {item['counts']['structural_table_count']}")
        lines.append(f"- section_header_block_count: {item['counts']['section_header_block_count']}")
        lines.append(f"- explanatory_block_count: {item['counts']['explanatory_block_count']}")
        lines.append(f"- discarded_table_count: {item['counts']['discarded_table_count']}")
        lines.append(f"- excluded_table_count: {item['counts']['excluded_table_count']}")
        lines.append(f"- image_count: {item['counts']['image_count']}")
        lines.append("")

        for section_name, rows in (
            ("Structural Samples", item["structural_samples"]),
            ("Section Header Samples", item["section_header_samples"]),
            ("Explanatory Samples", item["explanatory_samples"]),
            ("Discarded Samples", item["discarded_samples"]),
        ):
            lines.append(f"### {section_name}")
            lines.append("")
            if not rows:
                lines.append("- none")
                lines.append("")
                continue
            for row in rows:
                lines.append(f"- table_index {row['table_index']}")
                lines.append(f"  section: {row['section']}")
                lines.append(f"  record_start_index: {row['record_start_index']}")
                lines.append(f"  first_pass: {row['first_pass_classification']}")
                lines.append(f"  final: {row['final_classification']}")
                lines.append(f"  bucket: {row['storage_bucket']}")
                lines.append(f"  row_hint_count: {row['row_hint_count']}")
                lines.append(f"  paragraph_count: {row['paragraph_count']}")
                lines.append(f"  data_score: {row['data_score']}")
                lines.append(f"  header_group_hits: {row['header_group_hits']}")
                if row.get("section_header_role"):
                    lines.append(f"  section_header_role: {row['section_header_role']}")
                if row.get("linked_parent_table_index") is not None:
                    lines.append(f"  linked_parent_table_index: {row['linked_parent_table_index']}")
                if row.get("linked_child_table_indices"):
                    lines.append(f"  linked_child_table_indices: {row['linked_child_table_indices']}")
                lines.append(
                    f"  missing_signals: {', '.join(row['missing_signals']) if row['missing_signals'] else '-'}"
                )
                lines.append(f"  preview: {row['preview']}")
            lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a 5-file HWP review pack for manual comparison with the original files.")
    parser.add_argument("--json", type=Path, default=OUTPUT_JSON, help="Output JSON path.")
    parser.add_argument("--md", type=Path, default=OUTPUT_MD, help="Output markdown path.")
    args = parser.parse_args()

    selected_files = select_files()
    files = [build_file_report(path) for path in selected_files]
    report = {
        "sample_patterns": list(SAMPLE_PATTERNS),
        "files": files,
    }

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.md.write_text(build_markdown(report), encoding="utf-8")

    print(f"json={args.json}")
    print(f"md={args.md}")
    print(f"files={len(files)}")
    for item in files:
        print(
            (
                f"{item['source_name']}|structural={item['counts']['structural_table_count']}|"
                f"section_header={item['counts']['section_header_block_count']}|"
                f"explanatory={item['counts']['explanatory_block_count']}|"
                f"discarded={item['counts']['discarded_table_count']}|images={item['counts']['image_count']}"
            ).encode("cp949", errors="replace").decode("cp949")
        )


if __name__ == "__main__":
    main()

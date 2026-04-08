from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from eval_utils import (
    build_auto_summary,
    build_dependency_components,
    build_manual_summary,
    parse_question_rows,
    read_csv,
    sort_result_rows,
    sort_key_for_question_id,
    write_csv,
    write_json,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_QUESTION_SET = BASE_DIR / "evaluation" / "day3_partA_eval_questions_v1.txt"
DEFAULT_OUTPUT_DIR = BASE_DIR / "rag_outputs" / "b05_1_conditional_eval"

TABLE_SIGNAL_KEYWORDS = [
    "예산",
    "사업비",
    "금액",
    "기초금액",
    "추정금액",
    "계약금액",
    "총액",
    "평가기준",
    "배점",
    "점수",
    "일정",
    "기간",
    "월별",
    "일자",
    "날짜",
    "납기",
    "계약방식",
    "입찰방식",
    "투입인력",
    "인원",
]


def has_table_signal(question: str) -> list[str]:
    hits = [keyword for keyword in TABLE_SIGNAL_KEYWORDS if keyword in str(question)]
    return hits


def decide_component_route(component_rows: list[dict[str, Any]]) -> tuple[str, str, list[str]]:
    if any(row.get("depends_on_list") for row in component_rows):
        return "b02", "dependency_component", []

    combined_question = " ".join(str(row.get("question", "")) for row in component_rows)
    hits = has_table_signal(combined_question)
    if hits:
        return "b05", "table_signal", hits
    return "b02", "default", []


def run_command(command: list[str], cwd: Path, log_path: Path) -> None:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(
            [
                "[COMMAND]",
                " ".join(command),
                "",
                "[STDOUT]",
                completed.stdout,
                "",
                "[STDERR]",
                completed.stderr,
            ]
        ),
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"command failed: {log_path}")


def build_parallel_command(
    *,
    question_set_path: Path,
    question_id_file: Path,
    collection_name: str,
    bm25_index_path: Path,
    output_dir: Path,
    response_model: str,
    judge_model: str,
    shard_count: int,
    workers: int,
) -> list[str]:
    return [
        "python",
        ".\\28_평가_샤딩_실행.py",
        "--question-set-path",
        str(question_set_path),
        "--question-id-file",
        str(question_id_file),
        "--collection-name",
        collection_name,
        "--bm25-index-path",
        str(bm25_index_path),
        "--response-model",
        response_model,
        "--judge-model",
        judge_model,
        "--output-dir",
        str(output_dir),
        "--shard-count",
        str(shard_count),
        "--workers",
        str(workers),
        "--candidate-k",
        "10",
        "--top-k",
        "5",
        "--vector-weight",
        "0.7",
        "--bm25-weight",
        "0.3",
        "--rrf-k",
        "60",
    ]


def merge_route_outputs(route_output_dirs: list[Path], route_by_qid: dict[str, str], output_dir: Path) -> None:
    parsed_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    manual_template_rows: list[dict[str, Any]] = []
    preview_rows: list[dict[str, Any]] = []
    manual_completed_rows: list[dict[str, Any]] = []

    for route_dir in route_output_dirs:
        parsed_path = route_dir / "baseline_eval_questions_parsed.csv"
        result_path = route_dir / "baseline_eval_results.csv"
        manual_template_path = route_dir / "baseline_eval_manual_template.csv"
        preview_path = route_dir / "b01_hybrid_preview.csv"
        manual_completed_path = route_dir / "baseline_eval_manual_completed.csv"

        if parsed_path.exists():
            rows = read_csv(parsed_path)
            for row in rows:
                row["selected_pipeline"] = route_by_qid.get(row.get("question_id", ""), "")
            parsed_rows.extend(rows)
        if result_path.exists():
            rows = read_csv(result_path)
            for row in rows:
                row["selected_pipeline"] = route_by_qid.get(row.get("question_id", ""), "")
            result_rows.extend(rows)
        if manual_template_path.exists():
            rows = read_csv(manual_template_path)
            for row in rows:
                row["selected_pipeline"] = route_by_qid.get(row.get("question_id", ""), "")
            manual_template_rows.extend(rows)
        if preview_path.exists():
            rows = read_csv(preview_path)
            for row in rows:
                row["selected_pipeline"] = route_by_qid.get(row.get("question_id", ""), "")
            preview_rows.extend(rows)
        if manual_completed_path.exists():
            rows = read_csv(manual_completed_path)
            for row in rows:
                row["selected_pipeline"] = route_by_qid.get(row.get("question_id", ""), "")
            manual_completed_rows.extend(rows)

    parsed_rows = sort_result_rows(parsed_rows)
    result_rows = sort_result_rows(result_rows)
    manual_template_rows = sort_result_rows(manual_template_rows)
    manual_completed_rows = sort_result_rows(manual_completed_rows)
    preview_rows = sorted(
        preview_rows,
        key=lambda row: (sort_key_for_question_id(row.get("question_id", "")), int(float(row.get("rank", 0) or 0))),
    )

    write_csv(output_dir / "baseline_eval_questions_parsed.csv", parsed_rows)
    write_csv(output_dir / "baseline_eval_results.csv", result_rows)
    write_csv(output_dir / "baseline_eval_summary.csv", build_auto_summary(result_rows))
    write_csv(output_dir / "baseline_eval_manual_template.csv", manual_template_rows)
    write_csv(output_dir / "b01_hybrid_preview.csv", preview_rows)
    write_csv(output_dir / "baseline_eval_manual_completed.csv", manual_completed_rows)
    write_csv(output_dir / "baseline_eval_manual_summary.csv", build_manual_summary(manual_completed_rows))


def main() -> None:
    parser = argparse.ArgumentParser(description="A-05.1 조건부 라우팅 평가")
    parser.add_argument("--question-set-path", default=str(DEFAULT_QUESTION_SET))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--response-model", default="gpt-5-mini", choices=("gpt-5-mini", "gpt-5-nano", "gpt-5"))
    parser.add_argument("--judge-model", default="gpt-5", choices=("gpt-5-mini", "gpt-5-nano", "gpt-5"))
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    question_set_path = Path(args.question_set_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    question_rows = parse_question_rows(question_set_path)
    components = build_dependency_components(question_rows)

    route_rows: list[dict[str, Any]] = []
    b02_question_ids: list[str] = []
    b05_question_ids: list[str] = []
    route_by_qid: dict[str, str] = {}

    for component_rows in components:
        route, reason, signal_hits = decide_component_route(component_rows)
        question_ids = [row["question_id"] for row in component_rows]
        for row in component_rows:
            route_by_qid[row["question_id"]] = route
            route_rows.append(
                {
                    "question_id": row["question_id"],
                    "question_index": row["question_index"],
                    "question": row.get("question", ""),
                    "answer_type": row.get("answer_type", ""),
                    "depends_on": row.get("depends_on", ""),
                    "selected_pipeline": route,
                    "route_reason": reason,
                    "signal_hits": " | ".join(signal_hits),
                    "component_question_ids": " | ".join(question_ids),
                }
            )
        if route == "b05":
            b05_question_ids.extend(question_ids)
        else:
            b02_question_ids.extend(question_ids)

    route_rows = sorted(route_rows, key=lambda row: row["question_index"])
    write_csv(output_dir / "b05_1_route_plan.csv", route_rows)

    b02_id_file = output_dir / "b02_question_ids.txt"
    b05_id_file = output_dir / "b05_question_ids.txt"
    b02_id_file.write_text("\n".join(b02_question_ids), encoding="utf-8")
    b05_id_file.write_text("\n".join(b05_question_ids), encoding="utf-8")

    route_output_dirs: list[Path] = []
    if b02_question_ids:
        b02_output_dir = output_dir / "_route_b02"
        route_output_dirs.append(b02_output_dir)
        command = build_parallel_command(
            question_set_path=question_set_path,
            question_id_file=b02_id_file,
            collection_name="rfp_contextual_chunks_v2_b02",
        bm25_index_path=BASE_DIR / "rag_outputs" / "bm25_index_b02.pkl",
            output_dir=b02_output_dir,
            response_model=args.response_model,
            judge_model=args.judge_model,
            shard_count=args.shard_count,
            workers=args.workers,
        )
    run_command(command, BASE_DIR, output_dir / "b02_route.log")

    if b05_question_ids:
        b05_output_dir = output_dir / "_route_b05"
        route_output_dirs.append(b05_output_dir)
        command = build_parallel_command(
            question_set_path=question_set_path,
            question_id_file=b05_id_file,
            collection_name="rfp_contextual_chunks_v2_b05_table",
        bm25_index_path=BASE_DIR / "rag_outputs" / "bm25_index_b05.pkl",
            output_dir=b05_output_dir,
            response_model=args.response_model,
            judge_model=args.judge_model,
            shard_count=max(1, min(args.shard_count, 2)),
            workers=max(1, min(args.workers, 2)),
        )
    run_command(command, BASE_DIR, output_dir / "b05_route.log")

    merge_route_outputs(route_output_dirs, route_by_qid, output_dir)
    write_json(
        output_dir / "b05_1_route_manifest.json",
        {
            "question_set_path": str(question_set_path),
            "question_count": len(question_rows),
            "b02_question_count": len(b02_question_ids),
            "b05_question_count": len(b05_question_ids),
            "response_model": args.response_model,
            "judge_model": args.judge_model,
            "b02_collection": "rfp_contextual_chunks_v2_b02",
            "b05_collection": "rfp_contextual_chunks_v2_b05_table",
        },
    )

    print("[완료] A-05.1 조건부 라우팅 평가가 끝났습니다.")
    print(f"- 전체 질문 수: {len(question_rows)}")
    print(f"- A-02 route 질문 수: {len(b02_question_ids)}")
    print(f"- A-05 route 질문 수: {len(b05_question_ids)}")
    print(f"- 출력 디렉토리: {output_dir}")


if __name__ == "__main__":
    main()

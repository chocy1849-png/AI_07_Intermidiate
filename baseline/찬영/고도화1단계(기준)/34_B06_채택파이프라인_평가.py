from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from eval_utils import (
    build_auto_summary,
    build_dependency_components,
    build_manual_summary,
    pack_components_greedily,
    parse_question_rows,
    read_csv,
    sort_result_rows,
    write_csv,
    write_json,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_QUESTION_SET = BASE_DIR / "evaluation" / "day3_partA_eval_questions_v1.txt"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "rag_outputs"
DEFAULT_CHROMA_COLLECTION = "rfp_contextual_chunks_v2_b02"
DEFAULT_BM25_INDEX = BASE_DIR / "rag_outputs" / "bm25_index_b02.pkl"


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


def decide_component_route(component_rows: list[dict[str, Any]]) -> tuple[str, str]:
    answer_types = {str(row.get("answer_type", "")).strip() for row in component_rows}
    has_dependency = any(row.get("depends_on_list") for row in component_rows)
    if has_dependency:
        return "b02", "dependency_component"
    if "follow_up" in answer_types:
        return "b02", "follow_up"
    if "rejection" in answer_types:
        return "b02", "rejection"
    if answer_types & {"comparison", "factual"}:
        return "b03", "factual_or_comparison"
    return "b02", "default"


def merge_route_outputs(route_output_dirs: dict[str, Path], route_by_qid: dict[str, str], output_dir: Path) -> None:
    parsed_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    manual_template_rows: list[dict[str, Any]] = []
    manual_completed_rows: list[dict[str, Any]] = []

    for route_name, route_dir in route_output_dirs.items():
        parsed_path = route_dir / "baseline_eval_questions_parsed.csv"
        result_path = route_dir / "baseline_eval_results.csv"
        manual_template_path = route_dir / "baseline_eval_manual_template.csv"
        manual_completed_path = route_dir / "baseline_eval_manual_completed.csv"

        if parsed_path.exists():
            rows = read_csv(parsed_path)
            for row in rows:
                row["selected_pipeline"] = route_by_qid.get(row.get("question_id", ""), route_name)
            parsed_rows.extend(rows)
        if result_path.exists():
            rows = read_csv(result_path)
            for row in rows:
                row["selected_pipeline"] = route_by_qid.get(row.get("question_id", ""), route_name)
            result_rows.extend(rows)
        if manual_template_path.exists():
            rows = read_csv(manual_template_path)
            for row in rows:
                row["selected_pipeline"] = route_by_qid.get(row.get("question_id", ""), route_name)
            manual_template_rows.extend(rows)
        if manual_completed_path.exists():
            rows = read_csv(manual_completed_path)
            for row in rows:
                row["selected_pipeline"] = route_by_qid.get(row.get("question_id", ""), route_name)
            manual_completed_rows.extend(rows)

    parsed_rows = sort_result_rows(parsed_rows)
    result_rows = sort_result_rows(result_rows)
    manual_template_rows = sort_result_rows(manual_template_rows)
    manual_completed_rows = sort_result_rows(manual_completed_rows)

    write_csv(output_dir / "baseline_eval_questions_parsed.csv", parsed_rows)
    write_csv(output_dir / "baseline_eval_results.csv", result_rows)
    write_csv(output_dir / "baseline_eval_summary.csv", build_auto_summary(result_rows))
    write_csv(output_dir / "baseline_eval_manual_template.csv", manual_template_rows)
    write_csv(output_dir / "baseline_eval_manual_completed.csv", manual_completed_rows)
    write_csv(output_dir / "baseline_eval_manual_summary.csv", build_manual_summary(manual_completed_rows))


def run_b02_route(
    *,
    question_set_path: Path,
    question_id_file: Path,
    response_model: str,
    judge_model: str,
    output_dir: Path,
    shard_count: int,
    workers: int,
) -> None:
    command = [
        "python",
        ".\\28_평가_샤딩_실행.py",
        "--question-set-path",
        str(question_set_path),
        "--question-id-file",
        str(question_id_file),
        "--collection-name",
        DEFAULT_CHROMA_COLLECTION,
        "--bm25-index-path",
        str(DEFAULT_BM25_INDEX),
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
    run_command(command, BASE_DIR, output_dir / "run.log")


def merge_b03_generation(shard_dirs: list[Path], output_dir: Path) -> None:
    parsed_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    preview_rows: list[dict[str, Any]] = []
    manual_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for shard_dir in shard_dirs:
        parsed_path = shard_dir / "baseline_eval_questions_parsed.csv"
        result_path = shard_dir / "baseline_eval_results.csv"
        preview_path = shard_dir / "b03_crag_preview.csv"
        manual_path = shard_dir / "baseline_eval_manual_template.csv"
        manifest_path = shard_dir / "baseline_eval_manifest.json"
        if parsed_path.exists():
            parsed_rows.extend(read_csv(parsed_path))
        if result_path.exists():
            result_rows.extend(read_csv(result_path))
        if preview_path.exists():
            preview_rows.extend(read_csv(preview_path))
        if manual_path.exists():
            manual_rows.extend(read_csv(manual_path))
        if manifest_path.exists():
            manifest_rows.append(json.loads(manifest_path.read_text(encoding="utf-8")))

    parsed_rows = sort_result_rows(parsed_rows)
    result_rows = sort_result_rows(result_rows)
    manual_rows = sort_result_rows(manual_rows)

    write_csv(output_dir / "baseline_eval_questions_parsed.csv", parsed_rows)
    write_csv(output_dir / "baseline_eval_results.csv", result_rows)
    write_csv(output_dir / "baseline_eval_summary.csv", build_auto_summary(result_rows))
    write_csv(output_dir / "baseline_eval_manual_template.csv", manual_rows)
    write_csv(output_dir / "b03_crag_preview.csv", preview_rows)
    write_json(
        output_dir / "baseline_eval_manifest.json",
        {
            "sharded": True,
            "question_count": len(result_rows),
            "shard_count": len(shard_dirs),
            "source_manifests": manifest_rows,
        },
    )


def run_b03_route(
    *,
    question_set_path: Path,
    question_rows: list[dict[str, Any]],
    response_model: str,
    judge_model: str,
    output_dir: Path,
    shard_count: int,
    workers: int,
) -> None:
    components = build_dependency_components(question_rows)
    shards = pack_components_greedily(components, shard_count)
    generation_root = output_dir / "_shards" / "generation"
    generation_root.mkdir(parents=True, exist_ok=True)

    shard_dirs: list[Path] = []
    futures = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(shards)))) as executor:
        for index, shard_rows in enumerate(shards, start=1):
            shard_dir = generation_root / f"shard_{index:02d}"
            shard_dir.mkdir(parents=True, exist_ok=True)
            shard_dirs.append(shard_dir)

            question_id_file = shard_dir / "question_ids.txt"
            question_id_file.write_text(
                "\n".join(row["question_id"] for row in shard_rows),
                encoding="utf-8",
            )
            command = [
                "python",
                ".\\12_A03_CRAG_평가.py",
                "--질문셋경로",
                str(question_set_path),
                "--질문ID파일",
                str(question_id_file),
                "--출력디렉토리",
                str(shard_dir),
                "--컬렉션이름",
                DEFAULT_CHROMA_COLLECTION,
                "--bm25인덱스경로",
        str(DEFAULT_BM25_INDEX),
                "--응답모델",
                response_model,
                "--평가기모델",
                "gpt-5-mini",
                "--후보개수",
                "10",
                "--평가기입력개수",
                "5",
                "--최종상위개수",
                "3",
                "--비교형최종상위개수",
                "4",
                "--vector가중치",
                "0.7",
                "--bm25가중치",
                "0.3",
                "--2차vector가중치",
                "0.55",
                "--2차bm25가중치",
                "0.45",
                "--rrf_k",
                "60",
            ]
            futures.append(executor.submit(run_command, command, BASE_DIR, shard_dir / "run.log"))

        for future in as_completed(futures):
            future.result()

    merge_b03_generation(shard_dirs, output_dir)
    judge_command = [
        "python",
        ".\\15_자동_Judge_ASCII.py",
        "--eval-dir",
        str(output_dir),
        "--judge-model",
        judge_model,
    ]
    run_command(judge_command, BASE_DIR, output_dir / "judge.log")


def main() -> None:
    parser = argparse.ArgumentParser(description="A-06 현재 채택 파이프라인 평가")
    parser.add_argument("--question-set-path", default=str(DEFAULT_QUESTION_SET))
    parser.add_argument("--response-model", default="gpt-5-mini", choices=("gpt-5-mini", "gpt-5-nano", "gpt-5"))
    parser.add_argument("--judge-model", default="gpt-5", choices=("gpt-5-mini", "gpt-5-nano", "gpt-5"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    question_set_path = Path(args.question_set_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    question_rows = parse_question_rows(question_set_path)
    components = build_dependency_components(question_rows)

    route_rows: list[dict[str, Any]] = []
    route_by_qid: dict[str, str] = {}
    b02_ids: list[str] = []
    b03_ids: list[str] = []
    b03_rows: list[dict[str, Any]] = []

    for component in components:
        route, reason = decide_component_route(component)
        for row in component:
            route_by_qid[row["question_id"]] = route
            route_rows.append(
                {
                    "question_id": row["question_id"],
                    "type_group": row.get("type_group", ""),
                    "answer_type": row.get("answer_type", ""),
                    "selected_pipeline": route,
                    "route_reason": reason,
                }
            )
            if route == "b03":
                b03_ids.append(row["question_id"])
                b03_rows.append(row)
            else:
                b02_ids.append(row["question_id"])

    route_rows = sort_result_rows(route_rows)
    write_csv(output_dir / "b06_route_plan.csv", route_rows)
    write_json(
        output_dir / "b06_route_manifest.json",
        {
            "response_model": args.response_model,
            "judge_model": args.judge_model,
            "b02_question_count": len(b02_ids),
            "b03_question_count": len(b03_ids),
            "route_rules": {
                "b03": "factual_or_comparison_without_dependency",
                "b02": "follow_up_or_rejection_or_dependency_or_default",
            },
        },
    )

    route_output_dirs: dict[str, Path] = {}

    if b02_ids:
        b02_dir = output_dir / "_route_b02"
        b02_dir.mkdir(parents=True, exist_ok=True)
        b02_id_file = b02_dir / "question_ids.txt"
        b02_id_file.write_text("\n".join(b02_ids), encoding="utf-8")
        run_b02_route(
            question_set_path=question_set_path,
            question_id_file=b02_id_file,
            response_model=args.response_model,
            judge_model=args.judge_model,
            output_dir=b02_dir,
            shard_count=args.shard_count,
            workers=args.workers,
        )
        route_output_dirs["b02"] = b02_dir

    if b03_ids:
        b03_dir = output_dir / "_route_b03"
        b03_dir.mkdir(parents=True, exist_ok=True)
        run_b03_route(
            question_set_path=question_set_path,
            question_rows=b03_rows,
            response_model=args.response_model,
            judge_model=args.judge_model,
            output_dir=b03_dir,
            shard_count=args.shard_count,
            workers=args.workers,
        )
        route_output_dirs["b03"] = b03_dir

    merge_route_outputs(route_output_dirs, route_by_qid, output_dir)
    print("[완료] A-06 현재 채택 파이프라인 평가가 끝났습니다.")
    print(f"- 응답 모델: {args.response_model}")
    print(f"- A-02 route 질문 수: {len(b02_ids)}")
    print(f"- A-03 route 질문 수: {len(b03_ids)}")
    print(f"- 출력 디렉토리: {output_dir}")


if __name__ == "__main__":
    main()

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
    expand_selected_with_dependencies,
    parse_question_rows,
    pack_components_greedily,
    read_csv,
    sort_result_rows,
    sort_key_for_question_id,
    write_csv,
    write_json,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_QUESTION_SET = BASE_DIR / "evaluation" / "day3_partA_eval_questions_v1.txt"
DEFAULT_OUTPUT_DIR = BASE_DIR / "rag_outputs" / "parallel_eval"


def load_selected_ids(question_id_file: Path | None) -> set[str]:
    if question_id_file is None or not question_id_file.exists():
        return set()
    return {
        line.strip()
        for line in question_id_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def run_command(command: list[str], cwd: Path, log_path: Path) -> dict[str, Any]:
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
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "command": command,
        "log_path": str(log_path),
    }


def merge_generation_outputs(shard_dirs: list[Path], output_dir: Path) -> None:
    parsed_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    preview_rows: list[dict[str, Any]] = []
    manual_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for shard_dir in shard_dirs:
        parsed_path = shard_dir / "baseline_eval_questions_parsed.csv"
        result_path = shard_dir / "baseline_eval_results.csv"
        preview_path = shard_dir / "b01_hybrid_preview.csv"
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
    preview_rows = sorted(
        preview_rows,
        key=lambda row: (sort_key_for_question_id(row.get("question_id", "")), int(float(row.get("rank", 0) or 0))),
    )
    summary_rows = build_auto_summary(result_rows)

    write_csv(output_dir / "baseline_eval_questions_parsed.csv", parsed_rows)
    write_csv(output_dir / "baseline_eval_results.csv", result_rows)
    write_csv(output_dir / "baseline_eval_summary.csv", summary_rows)
    write_csv(output_dir / "baseline_eval_manual_template.csv", manual_rows)
    write_csv(output_dir / "b01_hybrid_preview.csv", preview_rows)
    (output_dir / "baseline_eval_results.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in result_rows),
        encoding="utf-8",
    )
    write_json(
        output_dir / "baseline_eval_manifest.json",
        {
            "sharded": True,
            "question_count": len(result_rows),
            "shard_count": len(shard_dirs),
            "source_manifests": manifest_rows,
        },
    )


def merge_judge_outputs(shard_dirs: list[Path], output_dir: Path, judge_model: str) -> None:
    completed_rows: list[dict[str, Any]] = []
    judge_manifests: list[dict[str, Any]] = []
    for shard_dir in shard_dirs:
        completed_path = shard_dir / "baseline_eval_manual_completed.csv"
        manifest_path = shard_dir / "baseline_eval_manual_judge_manifest.json"
        if completed_path.exists():
            completed_rows.extend(read_csv(completed_path))
        if manifest_path.exists():
            judge_manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))

    completed_rows = sort_result_rows(completed_rows)
    summary_rows = build_manual_summary(completed_rows)
    write_csv(output_dir / "baseline_eval_manual_completed.csv", completed_rows)
    write_csv(output_dir / "baseline_eval_manual_summary.csv", summary_rows)
    write_json(
        output_dir / "baseline_eval_manual_judge_manifest.json",
        {
            "sharded": True,
            "judge_model": judge_model,
            "question_count": len(completed_rows),
            "shard_count": len(shard_dirs),
            "source_manifests": judge_manifests,
        },
    )


def build_generation_command(args: argparse.Namespace, question_set_path: Path, question_id_file: Path, shard_dir: Path) -> list[str]:
    command = [
        "python",
        ".\\08_A01_Hybrid_평가.py",
        "--질문셋경로",
        str(question_set_path),
        "--컬렉션이름",
        args.collection_name,
        "--bm25인덱스경로",
        str(args.bm25_index_path),
        "--응답모델",
        args.response_model,
        "--출력디렉토리",
        str(shard_dir),
        "--질문ID파일",
        str(question_id_file),
        "--최종상위개수",
        str(args.top_k),
        "--후보개수",
        str(args.candidate_k),
        "--vector가중치",
        str(args.vector_weight),
        "--bm25가중치",
        str(args.bm25_weight),
        "--rrf_k",
        str(args.rrf_k),
    ]
    if args.history_for_follow_up:
        command.append("--후속질문히스토리적용")
    if args.chroma_dir:
        command.extend(["--\ud06c\ub85c\ub9c8\uacbd\ub85c", str(args.chroma_dir)])
    return command


def build_judge_command(eval_dir: Path, judge_model: str) -> list[str]:
    return [
        "python",
        ".\\15_자동_Judge_ASCII.py",
        "--eval-dir",
        str(eval_dir),
        "--judge-model",
        judge_model,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="질문 의존성을 보존한 평가 샤딩/병렬 실행기")
    parser.add_argument("--question-set-path", default=str(DEFAULT_QUESTION_SET))
    parser.add_argument("--question-id-file")
    parser.add_argument("--collection-name", required=True)
    parser.add_argument("--bm25-index-path", required=True)
    parser.add_argument("--chroma-dir")
    parser.add_argument("--response-model", default="gpt-5-mini", choices=("gpt-5-mini", "gpt-5-nano", "gpt-5"))
    parser.add_argument("--judge-model", choices=("gpt-5-mini", "gpt-5-nano", "gpt-5"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--history-for-follow-up", action="store_true")
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--vector-weight", type=float, default=0.7)
    parser.add_argument("--bm25-weight", type=float, default=0.3)
    parser.add_argument("--rrf-k", type=int, default=60)
    args = parser.parse_args()

    question_set_path = Path(args.question_set_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_question_rows = parse_question_rows(question_set_path)
    selected_ids = load_selected_ids(Path(args.question_id_file) if args.question_id_file else None)
    if selected_ids:
        selected_ids = expand_selected_with_dependencies(all_question_rows, selected_ids)
        question_rows = [row for row in all_question_rows if row["question_id"] in selected_ids]
    else:
        question_rows = all_question_rows

    components = build_dependency_components(question_rows)
    shards = pack_components_greedily(components, args.shard_count)

    generation_root = output_dir / "_shards" / "generation"
    generation_root.mkdir(parents=True, exist_ok=True)

    shard_dirs: list[Path] = []
    futures = []
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, len(shards)))) as executor:
        for index, shard_rows in enumerate(shards, start=1):
            shard_dir = generation_root / f"shard_{index:02d}"
            shard_dir.mkdir(parents=True, exist_ok=True)
            shard_dirs.append(shard_dir)

            question_id_file = shard_dir / "question_ids.txt"
            question_id_file.write_text(
                "\n".join(row["question_id"] for row in shard_rows),
                encoding="utf-8",
            )
            command = build_generation_command(args, question_set_path, question_id_file, shard_dir)
            log_path = shard_dir / "run.log"
        futures.append(executor.submit(run_command, command, BASE_DIR, log_path))

        generation_logs = []
        for future in as_completed(futures):
            result = future.result()
            generation_logs.append(result)
            if result["returncode"] != 0:
                raise RuntimeError(f"generation shard failed: {result['log_path']}")

    merge_generation_outputs(shard_dirs, output_dir)

    judge_logs: list[dict[str, Any]] = []
    if args.judge_model:
        judge_root = output_dir / "_shards" / "judge"
        judge_root.mkdir(parents=True, exist_ok=True)
        futures = []
        with ThreadPoolExecutor(max_workers=max(1, min(args.workers, len(shard_dirs)))) as executor:
            for shard_dir in shard_dirs:
                judge_dir = judge_root / shard_dir.name
                judge_dir.mkdir(parents=True, exist_ok=True)
                for filename in ("baseline_eval_results.csv", "baseline_eval_manifest.json"):
                    source_path = shard_dir / filename
                    target_path = judge_dir / filename
                    target_path.write_text(source_path.read_text(encoding="utf-8-sig"), encoding="utf-8-sig")
                command = build_judge_command(judge_dir, args.judge_model)
                log_path = judge_dir / "judge.log"
                futures.append(executor.submit(run_command, command, BASE_DIR, log_path))

            for future in as_completed(futures):
                result = future.result()
                judge_logs.append(result)
                if result["returncode"] != 0:
                    raise RuntimeError(f"judge shard failed: {result['log_path']}")

        merge_judge_outputs([judge_root / shard_dir.name for shard_dir in shard_dirs], output_dir, args.judge_model)

    write_json(
        output_dir / "parallel_eval_manifest.json",
        {
            "question_set_path": str(question_set_path),
            "question_count": len(question_rows),
            "selected_ids_count": len(selected_ids) if selected_ids else len(question_rows),
            "component_count": len(components),
            "shard_count": len(shards),
            "collection_name": args.collection_name,
            "bm25_index_path": str(args.bm25_index_path),
            "response_model": args.response_model,
            "judge_model": args.judge_model,
            "history_for_follow_up": args.history_for_follow_up,
            "generation_logs": generation_logs,
            "judge_logs": judge_logs,
        },
    )

    print("[완료] 평가 샤딩/병렬 실행이 끝났습니다.")
    print(f"- 질문 수: {len(question_rows)}")
    print(f"- 샤드 수: {len(shards)}")
    print(f"- 출력 디렉토리: {output_dir}")
    if args.judge_model:
        print(f"- judge 모델: {args.judge_model}")


if __name__ == "__main__":
    main()


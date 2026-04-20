from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from eval_utils import average, build_auto_summary, build_manual_summary, parse_question_rows, sort_result_rows, write_csv, write_json

from scenario_a.judge import judge_row


def read_question_rows(question_set_path: Path, question_id_file: Path | None = None) -> list[dict[str, Any]]:
    rows = parse_question_rows(question_set_path)
    if question_id_file is None:
        return rows
    selected = {
        line.strip()
        for line in question_id_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    return [row for row in rows if row["question_id"] in selected]


def normalize_doc_name(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").lower())


def compute_doc_hits(question_row: dict[str, Any], source_docs: list[str]) -> tuple[float | None, float | None, float | None]:
    gt_doc = str(question_row.get("ground_truth_doc", "")).strip()
    gt_docs = [x.strip() for x in str(question_row.get("ground_truth_docs", "")).split("|") if x.strip()]
    targets = [*([gt_doc] if gt_doc else []), *gt_docs]
    if not targets:
        return None, None, None

    source_norm = [normalize_doc_name(x) for x in source_docs]
    target_norm = [normalize_doc_name(x) for x in targets]
    top1_hit = 1.0 if source_norm and any(t in source_norm[0] or source_norm[0] in t for t in target_norm) else 0.0
    topk_hit = 1.0 if any(any(t in src or src in t for t in target_norm) for src in source_norm) else 0.0
    hit_count = sum(1 for target in target_norm if any(target in src or src in target for src in source_norm))
    hit_rate = hit_count / len(target_norm) if target_norm else None
    return top1_hit, topk_hit, hit_rate


def build_result_row(
    question_row: dict[str, Any],
    pipeline_result: Any,
    answer_text: str,
    elapsed_sec: float,
    *,
    model_key: str,
    embedding_backend_key: str,
) -> dict[str, Any]:
    source_docs = [str(candidate.metadata.get("source_file_name", "")).strip() for candidate in pipeline_result.candidates]
    top1_hit, topk_hit, hit_rate = compute_doc_hits(question_row, source_docs)
    return {
        "question_id": question_row.get("question_id", ""),
        "question_index": question_row.get("question_index", ""),
        "type_group": question_row.get("type_group", ""),
        "answer_type": question_row.get("answer_type", ""),
        "question": question_row.get("question", ""),
        "ground_truth_doc": question_row.get("ground_truth_doc", ""),
        "ground_truth_docs": question_row.get("ground_truth_docs", ""),
        "ground_truth_hint": question_row.get("ground_truth_hint", ""),
        "expected": question_row.get("expected", ""),
        "eval_focus": question_row.get("eval_focus", ""),
        "selected_pipeline": pipeline_result.route,
        "embedding_backend": embedding_backend_key,
        "model_key": model_key,
        "answer_text": answer_text,
        "answer_chars": len(answer_text),
        "elapsed_sec": round(elapsed_sec, 2),
        "retrieval_context": pipeline_result.context_text,
        "source_docs": " | ".join(source_docs),
        "top1_doc_hit": top1_hit,
        "topk_doc_hit": topk_hit,
        "ground_truth_doc_hit_rate": hit_rate,
    }


def _is_empty_answer(text: str) -> bool:
    return len(str(text or "").strip()) == 0


def _is_malformed_answer(text: str) -> bool:
    value = str(text or "").strip()
    return (not value) or value.startswith("USER:") or value.startswith("SYSTEM:")


def _is_verbose_answer(text: str, limit: int = 1800) -> bool:
    return len(str(text or "")) > limit


def _is_speculative_answer(text: str) -> bool:
    return bool(re.search(r"(아마|추정|예상|가정하면)", str(text or "")))


def _is_rejection_failure(answer_type: str, text: str) -> bool:
    if answer_type != "rejection":
        return False
    return not bool(re.search(r"(없습니다|확인할 수 없습니다|문서에 없습니다|부족합니다|알 수 없습니다)", str(text or "")))


def build_screening_diagnostics(result_rows: list[dict[str, Any]]) -> dict[str, Any]:
    empty_rows = [row for row in result_rows if _is_empty_answer(row.get("answer_text", ""))]
    malformed_rows = [row for row in result_rows if _is_malformed_answer(row.get("answer_text", ""))]
    verbose_rows = [row for row in result_rows if _is_verbose_answer(row.get("answer_text", ""))]
    speculative_rows = [row for row in result_rows if _is_speculative_answer(row.get("answer_text", ""))]
    rejection_fail_rows = [row for row in result_rows if _is_rejection_failure(str(row.get("answer_type", "")), str(row.get("answer_text", "")))]

    def _examples(rows: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
        return [
            {
                "question_id": row.get("question_id", ""),
                "question": row.get("question", ""),
                "answer_preview": str(row.get("answer_text", ""))[:300],
            }
            for row in rows[:limit]
        ]

    stable_rows = [row for row in result_rows if row not in empty_rows and row not in malformed_rows and row not in speculative_rows]
    return {
        "question_count": len(result_rows),
        "empty_response_count": len(empty_rows),
        "malformed_response_count": len(malformed_rows),
        "verbose_response_count": len(verbose_rows),
        "speculative_response_count": len(speculative_rows),
        "rejection_failure_count": len(rejection_fail_rows),
        "avg_latency_sec": average([float(row.get("elapsed_sec", 0) or 0) for row in result_rows]) if result_rows else None,
        "avg_answer_chars": average([float(row.get("answer_chars", 0) or 0) for row in result_rows]) if result_rows else None,
        "failure_examples": _examples(empty_rows + malformed_rows + speculative_rows + rejection_fail_rows),
        "success_examples": _examples(stable_rows),
    }


def screening_pass(diagnostics: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if diagnostics["empty_response_count"] > 0:
        reasons.append("empty_response_detected")
    if diagnostics["malformed_response_count"] > 0:
        reasons.append("format_break_detected")
    if diagnostics["rejection_failure_count"] >= 2:
        reasons.append("rejection_failures_exceed_threshold")
    return len(reasons) == 0, reasons


def write_examples_markdown(path: Path, title: str, diagnostics: dict[str, Any]) -> None:
    lines = [f"# {title}", "", "## Failure Examples", ""]
    if diagnostics["failure_examples"]:
        for item in diagnostics["failure_examples"]:
            lines.append(f"- {item['question_id']}: {item['question']}")
            lines.append(f"  - {item['answer_preview']}")
    else:
        lines.append("- none")
    lines.extend(["", "## Success Examples", ""])
    if diagnostics["success_examples"]:
        for item in diagnostics["success_examples"]:
            lines.append(f"- {item['question_id']}: {item['question']}")
            lines.append(f"  - {item['answer_preview']}")
    else:
        lines.append("- none")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_eval(pipeline: Any, adapter: Any, question_rows: list[dict[str, Any]], output_dir: Path, *, judge_model: str | None = None, run_label: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict[str, Any]] = []

    for row in question_rows:
        started = time.time()
        answered = pipeline.answer(row, adapter)
        elapsed = time.time() - started
        result_rows.append(
            build_result_row(
                row,
                answered,
                answered.answer_text,
                elapsed,
                model_key=adapter.config.model_key,
                embedding_backend_key=pipeline.embedding_backend.config.backend_key,
            )
        )

    result_rows = sort_result_rows(result_rows)
    write_csv(output_dir / "baseline_eval_results.csv", result_rows)
    write_csv(output_dir / "baseline_eval_summary.csv", build_auto_summary(result_rows))
    write_json(
        output_dir / "baseline_eval_manifest.json",
        {
            "run_label": run_label,
            "question_count": len(result_rows),
            "model_key": adapter.config.model_key,
            "embedding_backend": pipeline.embedding_backend.config.backend_key,
            "collection_name": pipeline.embedding_backend.config.collection_name,
            "bm25_index_path": str(pipeline.resolve_bm25_index_path()),
        },
    )

    completed_rows: list[dict[str, Any]] = []
    if judge_model:
        for row in result_rows:
            judged = judge_row(pipeline.openai_client, judge_model, row)
            merged = dict(row)
            merged.update(judged)
            completed_rows.append(merged)
        write_csv(output_dir / "baseline_eval_manual_completed.csv", completed_rows)
        write_csv(output_dir / "baseline_eval_manual_summary.csv", build_manual_summary(completed_rows))

    diagnostics = build_screening_diagnostics(result_rows)
    write_json(output_dir / "screening_diagnostics.json", diagnostics)
    return {"result_rows": result_rows, "manual_rows": completed_rows, "diagnostics": diagnostics}

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

QUESTION_HEADER_RE = re.compile(r"^Q(?P<num>\d+)(?:\s+\[(?P<turn>\d+)(?:턴)?\])?$")
FIELD_RE = re.compile(r"^\s*([A-Za-z_]+)\s*:\s*(.+?)\s*$")
TYPE_RE = re.compile(r"^TYPE\s+(?P<type_num>\d+)\s*:\s*(?P<label>.+)$")
SCENARIO_RE = re.compile(r"^---\s*(?P<label>.+?)\s*---$")


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def average(values: list[float | None]) -> float | None:
    valid = [x for x in values if x is not None]
    if not valid:
        return None
    return round(sum(valid) / len(valid), 4)


def sort_key_for_question_id(question_id: str) -> tuple[int, str]:
    match = re.search(r"(\d+)", str(question_id))
    return (int(match.group(1)) if match else math.inf, str(question_id))


def parse_question_rows(question_set_path: Path) -> list[dict[str, Any]]:
    text = question_set_path.read_text(encoding="utf-8")
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    current_type_num = ""
    current_type_label = ""
    current_scenario = ""

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        type_match = TYPE_RE.match(line)
        if type_match:
            current_type_num = f"TYPE {type_match.group('type_num')}"
            current_type_label = type_match.group("label").strip()
            current_scenario = ""
            continue

        scenario_match = SCENARIO_RE.match(line)
        if scenario_match:
            current_scenario = scenario_match.group("label").strip()
            continue

        question_match = QUESTION_HEADER_RE.match(line)
        if question_match:
            if current is not None:
                rows.append(current)
            qnum = int(question_match.group("num"))
            current = {
                "question_id": f"Q{qnum:02d}",
                "question_index": qnum,
                "turn_index": int(question_match.group("turn")) if question_match.group("turn") else None,
                "type_group": current_type_num,
                "type_label": current_type_label,
                "scenario_label": current_scenario,
            }
            continue

        if current is None:
            continue

        field_match = FIELD_RE.match(raw_line)
        if field_match:
            key = field_match.group(1).strip()
            value = field_match.group(2).strip()
            current[key] = value

    if current is not None:
        rows.append(current)

    for row in rows:
        row.setdefault("question", "")
        row.setdefault("answer_type", "")
        row.setdefault("ground_truth_doc", "")
        row.setdefault("ground_truth_docs", "")
        row.setdefault("ground_truth_hint", "")
        row.setdefault("eval_focus", "")
        row.setdefault("depends_on", "")
        row.setdefault("expected", "")
        if not row["answer_type"] and (row["type_group"] == "TYPE 4" or row["expected"]):
            row["answer_type"] = "rejection"
        row["depends_on_list"] = [
            part.strip()
            for part in str(row.get("depends_on", "")).split(",")
            if part.strip() and part.strip() != "-"
        ]
    return rows


def expand_selected_with_dependencies(
    all_rows: list[dict[str, Any]],
    selected_ids: set[str],
) -> set[str]:
    question_by_id = {row["question_id"]: row for row in all_rows}
    expanded = set(selected_ids)
    changed = True
    while changed:
        changed = False
        for question_id in list(expanded):
            row = question_by_id.get(question_id)
            if not row:
                continue
            for dep_id in row.get("depends_on_list", []):
                if dep_id in question_by_id and dep_id not in expanded:
                    expanded.add(dep_id)
                    changed = True
    return expanded


def build_dependency_components(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    question_by_id = {row["question_id"]: row for row in rows}
    adjacency: dict[str, set[str]] = {row["question_id"]: set() for row in rows}

    for row in rows:
        question_id = row["question_id"]
        for dep_id in row.get("depends_on_list", []):
            if dep_id not in question_by_id:
                continue
            adjacency[question_id].add(dep_id)
            adjacency[dep_id].add(question_id)

    visited: set[str] = set()
    components: list[list[dict[str, Any]]] = []
    for question_id in sorted(question_by_id.keys(), key=sort_key_for_question_id):
        if question_id in visited:
            continue
        stack = [question_id]
        component_ids: list[str] = []
        visited.add(question_id)
        while stack:
            current = stack.pop()
            component_ids.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        component_rows = sorted(
            (question_by_id[qid] for qid in component_ids),
            key=lambda row: row.get("question_index", 0),
        )
        components.append(component_rows)
    return components


def pack_components_greedily(
    components: list[list[dict[str, Any]]],
    shard_count: int,
) -> list[list[dict[str, Any]]]:
    shard_count = max(1, shard_count)
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    bucket_sizes = [0 for _ in range(shard_count)]

    sorted_components = sorted(
        components,
        key=lambda component: (-len(component), component[0].get("question_index", 0)),
    )
    for component in sorted_components:
        index = min(range(shard_count), key=lambda idx: bucket_sizes[idx])
        buckets[index].extend(component)
        bucket_sizes[index] += len(component)

    return [
        sorted(bucket, key=lambda row: row.get("question_index", 0))
        for bucket in buckets
        if bucket
    ]


def build_auto_summary(result_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    group_specs: list[tuple[str, list[dict[str, Any]]]] = [("overall", result_rows)]
    for type_group in sorted({row["type_group"] for row in result_rows if row.get("type_group")}):
        group_specs.append((type_group, [row for row in result_rows if row.get("type_group") == type_group]))
    for answer_type in sorted({row["answer_type"] for row in result_rows if row.get("answer_type")}):
        group_specs.append((f"answer_type:{answer_type}", [row for row in result_rows if row.get("answer_type") == answer_type]))

    summary_rows: list[dict[str, Any]] = []
    for label, rows in group_specs:
        top1_values = [safe_float(row.get("top1_doc_hit")) for row in rows if safe_float(row.get("top1_doc_hit")) is not None]
        topk_values = [safe_float(row.get("topk_doc_hit")) for row in rows if safe_float(row.get("topk_doc_hit")) is not None]
        hit_rate_values = [safe_float(row.get("ground_truth_doc_hit_rate")) for row in rows if safe_float(row.get("ground_truth_doc_hit_rate")) is not None]
        rejection_values = [safe_float(row.get("rejection_success")) for row in rows if safe_float(row.get("rejection_success")) is not None]
        elapsed_values = [safe_float(row.get("elapsed_sec")) for row in rows if safe_float(row.get("elapsed_sec")) is not None]
        answer_char_values = [safe_float(row.get("answer_chars")) for row in rows if safe_float(row.get("answer_chars")) is not None]

        summary_rows.append(
            {
                "group_name": label,
                "question_count": len(rows),
                "top1_doc_hit_rate": average(top1_values),
                "topk_doc_hit_rate": average(topk_values),
                "avg_ground_truth_doc_hit_rate": average(hit_rate_values),
                "rejection_success_rate": average(rejection_values),
                "avg_elapsed_sec": round(sum(elapsed_values) / len(elapsed_values), 2) if elapsed_values else None,
                "avg_answer_chars": round(sum(answer_char_values) / len(answer_char_values), 2) if answer_char_values else None,
            }
        )
    return summary_rows


def build_manual_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    group_specs: list[tuple[str, list[dict[str, Any]]]] = [("overall", rows)]
    for type_group in sorted({row["type_group"] for row in rows if row.get("type_group")}):
        group_specs.append((type_group, [row for row in rows if row.get("type_group") == type_group]))
    for answer_type in sorted({row["answer_type"] for row in rows if row.get("answer_type")}):
        group_specs.append((f"answer_type:{answer_type}", [row for row in rows if row.get("answer_type") == answer_type]))

    summary_rows: list[dict[str, Any]] = []
    for label, group_rows in group_specs:
        summary_rows.append(
            {
                "group_name": label,
                "question_count": len(group_rows),
                "avg_faithfulness_score": average([safe_float(row.get("faithfulness_score")) for row in group_rows]),
                "avg_completeness_score": average([safe_float(row.get("completeness_score")) for row in group_rows]),
                "avg_groundedness_score": average([safe_float(row.get("groundedness_score")) for row in group_rows]),
                "avg_relevancy_score": average([safe_float(row.get("relevancy_score")) for row in group_rows]),
                "avg_manual_eval_score": average(
                    [
                        average(
                            [
                                safe_float(row.get("faithfulness_score")),
                                safe_float(row.get("completeness_score")),
                                safe_float(row.get("groundedness_score")),
                                safe_float(row.get("relevancy_score")),
                            ]
                        )
                        for row in group_rows
                    ]
                ),
            }
        )
    return summary_rows


def sort_result_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def row_key(row: dict[str, Any]) -> tuple[int, str]:
        qindex = safe_float(row.get("question_index"))
        return (int(qindex) if qindex is not None else sort_key_for_question_id(row.get("question_id", ""))[0], str(row.get("question_id", "")))

    return sorted(rows, key=row_key)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

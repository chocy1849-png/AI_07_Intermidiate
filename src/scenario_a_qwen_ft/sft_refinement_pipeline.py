from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scenario_a.common_pipeline import CandidateRow
from scenario_a_qwen_ft.sft_expansion_pipeline import QwenSFTExpansionBuilder, _slot_aliases
from scenario_a_qwen_ft.sft_pipeline import _read_jsonl, _write_json, _write_jsonl, _write_markdown


class QwenSFTRefinementBuilder(QwenSFTExpansionBuilder):
    def __init__(
        self,
        *,
        project_root: Path,
        question_root: Path,
        prior_output_root: Path,
        output_root: Path | None = None,
        retrieval_profile: str = "bge_m3",
        teacher_model: str = "gpt-5",
        judge_model: str = "gpt-5",
        shard_count: int = 8,
        teacher_workers: int = 8,
        judge_workers: int = 8,
        seed: int = 20260410,
    ) -> None:
        super().__init__(
            project_root=project_root,
            question_root=question_root,
            output_root=output_root or (project_root / "rag_outputs" / "qwen_ft_instruction_expansion_v2_refine"),
            retrieval_profile=retrieval_profile,
            teacher_model=teacher_model,
            judge_model=judge_model,
            shard_count=shard_count,
            teacher_workers=teacher_workers,
            judge_workers=judge_workers,
            seed=seed,
        )
        self.prior_output_root = prior_output_root.resolve()
        self.prior_judged_rows = _read_jsonl(self.prior_output_root / f"judged_sft_candidates_{self.retrieval_profile}.jsonl")
        self.prior_stats = json.loads((self.prior_output_root / "qwen_sft_expansion_stats.json").read_text(encoding="utf-8"))

    def select_target_rows(self) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        for row in self.prior_judged_rows:
            if row.get("accepted_for_sft"):
                continue
            task_type = str(row.get("task_type", ""))
            if task_type not in {"single_doc_factual", "table_plus_body"}:
                continue
            selected.append(self._normalize_candidate(row, provenance=f"targeted_refine::{task_type}"))
        selected.sort(key=lambda item: item["qid"])
        _write_jsonl(self.output_root / "target_rows.jsonl", selected)
        self.log_step("select_target_rows", target_count=len(selected), by_task_type=dict(Counter(row["task_type"] for row in selected)))
        return selected

    def _slot_match(self, candidate: CandidateRow, slot: str) -> bool:
        text = " ".join(
            [
                str(candidate.text or ""),
                str(candidate.metadata.get("section_title", "")),
                str(candidate.metadata.get("chunk_role", "")),
                str(candidate.metadata.get("chunk_role_tags", "")),
            ]
        )
        if slot == "budget" and int(candidate.metadata.get("has_budget_signal", 0) or 0):
            return True
        if slot in {"period", "schedule"} and int(candidate.metadata.get("has_schedule_signal", 0) or 0):
            return True
        if slot == "contract" and int(candidate.metadata.get("has_contract_signal", 0) or 0):
            return True
        return any(alias in text for alias in _slot_aliases(slot))

    def _same_section_bonus(self, left: CandidateRow, right: CandidateRow) -> float:
        left_path = str(left.metadata.get("section_path", ""))
        right_path = str(right.metadata.get("section_path", ""))
        if left_path and right_path and left_path == right_path:
            return 1.0
        left_title = str(left.metadata.get("section_title", ""))
        right_title = str(right.metadata.get("section_title", ""))
        if left_title and right_title and left_title == right_title:
            return 0.5
        return 0.0

    def _build_retrieval_candidate_one(self, row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
        task_type = str(row.get("task_type", ""))
        source_docs = list(row.get("source_docs", []))
        source_filter = {source_docs[0]} if source_docs else set()
        query_text = row["question"]
        reranked = self._merge_and_rerank(row=row, query_text=query_text, source_filter=source_filter)

        selected: list[CandidateRow] = []
        seen: set[str] = set()
        if task_type == "single_doc_factual":
            for slot in row.get("target_slots", []):
                for candidate in reranked:
                    if candidate.chunk_id in seen:
                        continue
                    if self._slot_match(candidate, slot):
                        selected.append(candidate)
                        seen.add(candidate.chunk_id)
                        break
            for candidate in reranked:
                if candidate.chunk_id in seen:
                    continue
                selected.append(candidate)
                seen.add(candidate.chunk_id)
                if len(selected) >= 6:
                    break
        else:
            table_candidates = [c for c in reranked if int(c.metadata.get("has_table", 0) or 0)]
            body_candidates = [c for c in reranked if not int(c.metadata.get("has_table", 0) or 0)]
            best_table = table_candidates[0] if table_candidates else None
            best_body = None
            if best_table and body_candidates:
                scored = sorted(
                    body_candidates,
                    key=lambda c: (
                        self._same_section_bonus(best_table, c),
                        c.adjusted_score if c.adjusted_score is not None else c.fusion_score,
                    ),
                    reverse=True,
                )
                best_body = scored[0]
            elif body_candidates:
                best_body = body_candidates[0]

            for candidate in [best_table, best_body]:
                if candidate and candidate.chunk_id not in seen:
                    selected.append(candidate)
                    seen.add(candidate.chunk_id)
            for slot in row.get("target_slots", []):
                for candidate in reranked:
                    if candidate.chunk_id in seen:
                        continue
                    if self._slot_match(candidate, slot):
                        selected.append(candidate)
                        seen.add(candidate.chunk_id)
                        break
            for candidate in reranked:
                if candidate.chunk_id in seen:
                    continue
                selected.append(candidate)
                seen.add(candidate.chunk_id)
                if len(selected) >= 6:
                    break

        contexts = [self._candidate_to_context(candidate, rank=index + 1, route=f"refine::{task_type}") for index, candidate in enumerate(selected[:6])]
        table_hit = any(int(candidate.metadata.get("has_table", 0) or 0) for candidate in selected[:6]) if task_type == "table_plus_body" else None
        nearby_body_hit = any(not int(candidate.metadata.get("has_table", 0) or 0) for candidate in selected[:6]) if task_type == "table_plus_body" else None
        weak_reasons: list[str] = []
        if task_type == "table_plus_body" and not (table_hit and nearby_body_hit):
            weak_reasons.append("table_body_coverage_low")

        retrieval_row = {
            "id": f"{self.retrieval_profile}_qwen_refine_{row['qid']}",
            "qid": row["qid"],
            "question_id": row.get("question_id", row["qid"]),
            "source_docs": source_docs,
            "task_type": task_type,
            "answer_type": row["answer_type"],
            "question": row["question"],
            "expected_answerability": row["expected_answerability"],
            "target_slots": row.get("target_slots", []),
            "official_eval_overlap": bool(row.get("official_eval_overlap", False)),
            "requires_refusal": row["expected_answerability"] == "no_answer",
            "retrieval_profile": self.retrieval_profile,
            "selected_pipeline": f"targeted_refine::{task_type}",
            "contexts": contexts,
            "weak_context": bool(weak_reasons),
            "weak_context_reasons": weak_reasons,
            "metadata": {
                "question_source": row.get("provenance", ""),
                "table_related": task_type == "table_plus_body",
                "comparison": False,
                "follow_up": False,
                "primary_slots": row.get("target_slots", []),
                "section_labels": sorted({ctx["section_label"] for ctx in contexts if ctx["section_label"]}),
                "doc_count": len({ctx["file_name"] for ctx in contexts if ctx["file_name"]}),
                "dual_doc_coverage": None,
                "comparison_axis_detected": None,
                "comparison_context_strength": None,
                "table_hit": table_hit,
                "nearby_body_hit": nearby_body_hit,
                "table_body_coverage": {"table": bool(table_hit), "body": bool(nearby_body_hit)} if task_type == "table_plus_body" else None,
                "direct_support_flag": None,
                "indirect_support_flag": None,
                "ambiguity_flag": None,
                "support_confidence": None,
                "support_judge": None,
                "parent_qid": None,
                "source_state_anchor": None,
                "followup_resolvable_with_context": None,
                "context_length": sum(len(ctx["text"]) for ctx in contexts),
            },
            "conversation_group_id": row.get("conversation_group_id"),
            "depends_on_list": row.get("depends_on_list", []),
        }
        fail_row = None
        if weak_reasons:
            fail_row = {
                "qid": row["qid"],
                "question": row["question"],
                "task_type": task_type,
                "selected_pipeline": retrieval_row["selected_pipeline"],
                "weak_context_reasons": weak_reasons,
                "provenance": row.get("provenance", ""),
            }
        return retrieval_row, fail_row

    def _teacher_user_prompt(self, row: dict[str, Any]) -> str:
        context_blocks = []
        for ctx in row["contexts"]:
            context_blocks.append(
                "\n".join(
                    [
                        f"[문서 {ctx['rank']} | {ctx['file_name']} | {ctx.get('section_label', '')}]",
                        ctx["text"],
                    ]
                )
            )
        slot_labels = []
        for slot in row.get("target_slots", []):
            slot_labels.extend(_slot_aliases(slot)[:2])
        guide = "질문에서 요구한 각 슬롯을 빠짐없이 답하세요. 슬롯별로 문맥에 없으면 '문서에 명시되지 않음'이라고 적으세요."
        if row.get("task_type") == "table_plus_body":
            guide = "표 근거와 본문 근거를 모두 반영해서 답하세요. 표만 보고 답하지 말고, 본문 설명까지 결합하세요. 슬롯이 문맥에 없으면 '문서에 명시되지 않음'이라고 적으세요."
        return "\n".join(
            [
                "[질문]",
                row["question"],
                "[요구 슬롯]",
                ", ".join(slot_labels) or "없음",
                "[문맥]",
                "\n\n".join(context_blocks),
                "[응답 지침]",
                guide,
            ]
        )

    def merge_and_finalize(self, rerun_judged_rows: list[dict[str, Any]]) -> dict[str, Any]:
        rerun_by_qid = {row["qid"]: row for row in rerun_judged_rows}
        merged_judged = [rerun_by_qid.get(row["qid"], row) for row in self.prior_judged_rows]
        merged_judged.sort(key=lambda item: item["qid"])
        _write_jsonl(self.output_root / f"judged_sft_candidates_{self.retrieval_profile}.jsonl", merged_judged)
        raw_rows = self.build_raw_dataset(merged_judged)
        train_rows, val_rows, split_manifest = self.split_raw_dataset(raw_rows)
        train_payload, val_payload = self.format_qwen_dataset(train_rows=train_rows, val_rows=val_rows)

        accepted_by_type = Counter(row.get("task_type", "") for row in raw_rows)
        stats = {
            "accepted_count": len(raw_rows),
            "accepted_count_previous": int(self.prior_stats.get("accepted_count", 0) or 0),
            "accepted_delta": len(raw_rows) - int(self.prior_stats.get("accepted_count", 0) or 0),
            "accepted_by_task_type": dict(accepted_by_type),
            "accepted_by_task_type_previous": dict(self.prior_stats.get("accepted_by_task_type", {}) or {}),
            "train_count": len(train_payload),
            "val_count": len(val_payload),
            "ft_ready_minimum": len(raw_rows) >= 2000
            and accepted_by_type.get("single_doc_factual", 0) >= 1000
            and accepted_by_type.get("comparison", 0) >= 300
            and accepted_by_type.get("rejection", 0) >= 200
            and accepted_by_type.get("table_plus_body", 0) >= 100
            and accepted_by_type.get("follow_up", 0) >= 100,
        }
        _write_json(self.output_root / "qwen_sft_refine_stats.json", stats)
        lines = [
            "# Qwen SFT Refinement v2",
            "",
            f"- accepted: {stats['accepted_count_previous']} -> {stats['accepted_count']} (delta {stats['accepted_delta']})",
            "",
            "## Accepted by Task Type",
        ]
        for task_type, count in sorted(accepted_by_type.items()):
            previous = int((self.prior_stats.get("accepted_by_task_type", {}) or {}).get(task_type, 0) or 0)
            lines.append(f"- {task_type}: {previous} -> {count} (delta {count - previous})")
        lines.extend(["", f"- ft_ready_minimum: {stats['ft_ready_minimum']}"])
        _write_markdown(self.output_root / "qwen_sft_refine_report.md", lines)
        return {"stats": stats, "split_manifest": split_manifest}

    def run(self) -> dict[str, Any]:
        target_rows = self.select_target_rows()
        target_rows, _ = self.build_overlap_blocklist(target_rows)
        retrieval_rows = self.build_retrieval_candidates(target_rows)
        judged_rows = self.generate_teacher_and_judge(retrieval_rows)
        merged = self.merge_and_finalize(judged_rows)
        _write_jsonl(self.output_root / "qwen_sft_refine_build_log.jsonl", self.step_logs)
        return {
            "stats": merged["stats"],
            "split_manifest": merged["split_manifest"],
            "output_root": str(self.output_root),
        }

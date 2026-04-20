from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scenario_a.common_pipeline import CandidateRow
from scenario_a_qwen_ft.sft_expansion_pipeline import QwenSFTExpansionBuilder
from scenario_a_qwen_ft.sft_pipeline import (
    _normalize_question,
    _read_jsonl,
    _sequence_similarity,
    _write_json,
    _write_jsonl,
    _write_markdown,
)


def _split_pipe_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [item.strip() for item in text.split("|") if item.strip()]


def _slot_label(slot: str) -> str:
    labels = {
        "budget": "예산",
        "period": "사업 기간",
        "contract": "계약 방식",
        "purpose": "사업 목적",
        "scope": "수행 범위",
        "security": "보안 요구사항",
        "integration": "연계 대상",
        "evaluation": "평가 기준",
        "eligibility": "참가 자격",
        "deliverables": "납품 산출물",
        "schedule": "추진 일정",
        "operations": "운영 방식",
        "maintenance": "유지보수 범위",
        "modules": "주요 기능",
        "issuer": "발주 기관",
        "submission_docs": "제출 서류",
        "deadline": "마감일",
    }
    return labels.get(slot, slot)


def _slot_aliases_v3(slot: str) -> list[str]:
    mapping = {
        "budget": ["예산", "금액", "사업비", "기초금액", "추정금액"],
        "period": ["사업 기간", "수행 기간", "계약체결일로부터", "기간", "개월", "일 이내"],
        "contract": ["계약 방식", "입찰 방식", "계약", "협상", "낙찰"],
        "purpose": ["목적", "추진 배경", "배경", "필요성"],
        "scope": ["범위", "수행 범위", "과업 범위", "주요 업무", "구축 범위"],
        "security": ["보안", "접근통제", "개인정보", "권한"],
        "integration": ["연계", "연동", "API", "인터페이스"],
        "evaluation": ["평가 기준", "평가 항목", "배점", "정량", "정성"],
        "eligibility": ["참가 자격", "입찰 자격", "자격 요건"],
        "deliverables": ["산출물", "납품", "제출물", "결과물"],
        "schedule": ["일정", "추진 일정", "마감", "제출 기한"],
        "operations": ["운영", "운영 방식", "상주", "모니터링"],
        "maintenance": ["유지보수", "하자보수", "유지관리"],
        "modules": ["기능", "모듈", "구성", "기능 목록"],
        "issuer": ["발주 기관", "발주기관", "수요 기관", "기관명"],
        "submission_docs": ["제출 서류", "제안서", "서류", "증빙"],
        "deadline": ["마감일", "마감", "접수 마감", "제출 기한"],
    }
    return mapping.get(slot, [_slot_label(slot)])


class QwenSFTTargetedV3Builder(QwenSFTExpansionBuilder):
    TARGET_TASK_TYPES = {"single_doc_factual", "table_plus_body"}
    SINGLE_DOC_TARGET = 1100
    TABLE_BODY_TARGET = 420

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
            output_root=output_root or (project_root / "rag_outputs" / "qwen_ft_instruction_expansion_v3_refine"),
            retrieval_profile=retrieval_profile,
            teacher_model=teacher_model,
            judge_model=judge_model,
            shard_count=shard_count,
            teacher_workers=teacher_workers,
            judge_workers=judge_workers,
            seed=seed,
        )
        self.prior_output_root = prior_output_root.resolve()
        self.prior_raw_rows = _read_jsonl(
            self.prior_output_root / "data" / "sft" / f"raw_sft_dataset_{self.retrieval_profile}_qwen.jsonl"
        )
        self.prior_stats = json.loads((self.prior_output_root / "qwen_sft_refine_stats.json").read_text(encoding="utf-8"))
        self.prior_retrieval_rows = _read_jsonl(self.prior_output_root / f"retrieval_candidates_{self.retrieval_profile}.jsonl")
        self.single_doc_output_root = self.project_root / "rag_outputs" / "qwen_ft_single_doc_expansion_v3"
        self.table_body_output_root = self.project_root / "rag_outputs" / "qwen_ft_table_body_expansion_v3"
        self.single_doc_output_root.mkdir(parents=True, exist_ok=True)
        self.table_body_output_root.mkdir(parents=True, exist_ok=True)
        self._official_question_cache: list[dict[str, Any]] | None = None
        self._slot_counts = self._build_single_doc_slot_counts()

    def _build_single_doc_slot_counts(self) -> Counter[str]:
        counter: Counter[str] = Counter()
        for row in self.prior_raw_rows:
            if str(row.get("task_type", "")) != "single_doc_factual":
                continue
            metadata = row.get("metadata", {}) or {}
            for slot in metadata.get("primary_slots", []) or []:
                counter[str(slot)] += 1
        return counter

    def _existing_signature_set(self) -> set[str]:
        signatures: set[str] = set()
        for row in self.prior_raw_rows:
            signatures.add(self._row_signature(row))
        return signatures

    def _row_signature(self, row: dict[str, Any]) -> str:
        return "::".join(
            [
                str(row.get("task_type", "")).strip(),
                "|".join(sorted(_split_pipe_list(row.get("source_docs")))),
                _normalize_question(row.get("question", "")),
            ]
        )

    def _is_official_overlap(self, question: str) -> bool:
        if self._official_question_cache is None:
            self._official_question_cache = self._load_official_eval_bank()
        normalized = _normalize_question(question)
        for item in self._official_question_cache:
            official_question = str(item.get("question", "")).strip()
            if not official_question:
                continue
            official_normalized = _normalize_question(official_question)
            if normalized == official_normalized:
                return True
            if _sequence_similarity(normalized, official_normalized) >= 0.94:
                return True
        return False

    def _new_qid(self, prefix: str, index: int) -> str:
        return f"{prefix}_{index:06d}"

    def _single_doc_slot_priority(self, supported_slots: set[str]) -> list[str]:
        preferred = [
            "maintenance",
            "operations",
            "deliverables",
            "evaluation",
            "eligibility",
            "security",
            "integration",
            "scope",
            "modules",
            "purpose",
            "contract",
            "submission_docs",
            "issuer",
            "schedule",
            "deadline",
            "budget",
            "period",
        ]
        available = [slot for slot in preferred if slot in supported_slots]
        available.sort(key=lambda slot: (self._slot_counts.get(slot, 0), preferred.index(slot)))
        return available

    def _single_doc_templates(self, subject: str, slot_a: str, slot_b: str | None = None) -> list[str]:
        if slot_b:
            return [
                f"{subject}에서 {_slot_label(slot_a)}와 {_slot_label(slot_b)}만 간단히 정리해줘.",
                f"{subject} 기준으로 {_slot_label(slot_a)}와 {_slot_label(slot_b)}가 어떻게 제시됐는지 알려줘.",
            ]
        return [
            f"{subject}의 {_slot_label(slot_a)}은 어떻게 제시되어 있어?",
            f"{subject} 문서에서 {_slot_label(slot_a)}만 정확히 알려줘.",
            f"{subject} 기준으로 {_slot_label(slot_a)} 정보를 정리해줘.",
        ]

    def _table_body_templates(self, subject: str, table_slot: str, body_slot: str) -> list[str]:
        return [
            f"{subject}에서 표에 나온 {_slot_label(table_slot)}와 본문 설명의 {_slot_label(body_slot)}를 함께 반영해서 정리해줘.",
            f"{subject} 문서에서 {_slot_label(table_slot)} 정보와 {_slot_label(body_slot)} 설명을 결합해 요약해줘.",
            f"{subject} 기준으로 표의 {_slot_label(table_slot)} 내용과 본문의 {_slot_label(body_slot)} 조건을 같이 설명해줘.",
        ]

    def _candidate_row(
        self,
        *,
        qid: str,
        question: str,
        task_type: str,
        source_docs: list[str],
        target_slots: list[str],
        provenance: str,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        row = {
            "qid": qid,
            "question_id": qid,
            "question": question.strip(),
            "task_type": task_type,
            "answer_type": "factual",
            "source_docs": source_docs,
            "target_slots": target_slots,
            "depends_on_list": [],
            "official_eval_overlap": False,
            "expected_answerability": "answerable",
            "requires_refusal": False,
            "provenance": provenance,
        }
        if extra:
            row.update(extra)
        return row

    def generate_single_doc_candidates(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        seen = self._existing_signature_set()
        index = 1

        for doc in self.question_generator.documents:
            supported = set(doc.supported_slots) | {"issuer"}
            if doc.metadata.get("입찰 참여 마감일") or doc.metadata.get("입찰 참여 시작일"):
                supported.add("deadline")
            if doc.metadata.get("사업 금액"):
                supported.add("budget")
            ordered_slots = self._single_doc_slot_priority(supported)
            subject = self.question_generator._short_doc_subject(doc)

            for slot in ordered_slots:
                for question in self._single_doc_templates(subject, slot):
                    if self._is_official_overlap(question):
                        continue
                    row = self._candidate_row(
                        qid=self._new_qid("v3_single", index),
                        question=question,
                        task_type="single_doc_factual",
                        source_docs=[doc.source_file_name],
                        target_slots=[slot],
                        provenance="v3::single_doc_generated",
                        extra={"factual_primary_slot": slot},
                    )
                    signature = self._row_signature(row)
                    if signature in seen:
                        continue
                    seen.add(signature)
                    rows.append(row)
                    index += 1
                    if len(rows) >= self.SINGLE_DOC_TARGET:
                        break
                if len(rows) >= self.SINGLE_DOC_TARGET:
                    break
            if len(rows) >= self.SINGLE_DOC_TARGET:
                break

            if len(ordered_slots) >= 2:
                pair_candidates = [
                    (ordered_slots[i], ordered_slots[j])
                    for i in range(len(ordered_slots))
                    for j in range(i + 1, min(len(ordered_slots), i + 4))
                ]
                for slot_a, slot_b in pair_candidates[:4]:
                    for question in self._single_doc_templates(subject, slot_a, slot_b):
                        if self._is_official_overlap(question):
                            continue
                        row = self._candidate_row(
                            qid=self._new_qid("v3_single", index),
                            question=question,
                            task_type="single_doc_factual",
                            source_docs=[doc.source_file_name],
                            target_slots=[slot_a, slot_b],
                            provenance="v3::single_doc_generated",
                            extra={"factual_primary_slot": slot_a},
                        )
                        signature = self._row_signature(row)
                        if signature in seen:
                            continue
                        seen.add(signature)
                        rows.append(row)
                        index += 1
                        if len(rows) >= self.SINGLE_DOC_TARGET:
                            break
                    if len(rows) >= self.SINGLE_DOC_TARGET:
                        break
            if len(rows) >= self.SINGLE_DOC_TARGET:
                break

        rows.sort(key=lambda item: item["qid"])
        _write_jsonl(self.single_doc_output_root / "single_doc_expansion_candidates.jsonl", rows)
        return rows

    def generate_table_body_candidates(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        seen = self._existing_signature_set()
        index = 1

        preferred_body_slots = ["purpose", "scope", "operations", "maintenance", "integration", "background", "deliverables"]
        preferred_table_slots = ["evaluation", "schedule", "budget", "modules", "integration"]

        for doc in self.question_generator.documents:
            table_slots = [slot for slot in preferred_table_slots if slot in doc.table_like_slots]
            body_slots = [slot for slot in preferred_body_slots if slot in doc.body_like_slots]
            if not table_slots or not body_slots:
                continue
            subject = self.question_generator._short_doc_subject(doc)
            for table_slot in table_slots[:3]:
                for body_slot in body_slots[:3]:
                    if table_slot == body_slot:
                        continue
                    for question in self._table_body_templates(subject, table_slot, body_slot):
                        if self._is_official_overlap(question):
                            continue
                        row = self._candidate_row(
                            qid=self._new_qid("v3_table_body", index),
                            question=question,
                            task_type="table_plus_body",
                            source_docs=[doc.source_file_name],
                            target_slots=[table_slot, body_slot],
                            provenance="v3::table_body_generated",
                            extra={"table_slot": table_slot, "body_slot": body_slot},
                        )
                        signature = self._row_signature(row)
                        if signature in seen:
                            continue
                        seen.add(signature)
                        rows.append(row)
                        index += 1
                        if len(rows) >= self.TABLE_BODY_TARGET:
                            break
                    if len(rows) >= self.TABLE_BODY_TARGET:
                        break
                if len(rows) >= self.TABLE_BODY_TARGET:
                    break
            if len(rows) >= self.TABLE_BODY_TARGET:
                break

        rows.sort(key=lambda item: item["qid"])
        _write_jsonl(self.table_body_output_root / "table_body_expansion_candidates.jsonl", rows)
        return rows

    def build_target_question_pool(self) -> list[dict[str, Any]]:
        single_doc_rows = self.generate_single_doc_candidates()
        table_body_rows = self.generate_table_body_candidates()
        pool = sorted(single_doc_rows + table_body_rows, key=lambda item: item["qid"])
        self.log_step(
            "build_target_question_pool",
            total=len(pool),
            single_doc_count=len(single_doc_rows),
            table_body_count=len(table_body_rows),
        )
        return pool

    def _slot_match(self, candidate: CandidateRow, slot: str) -> bool:
        text = " ".join(
            [
                str(candidate.text or ""),
                str(candidate.metadata.get("section_title", "")),
                str(candidate.metadata.get("chunk_role", "")),
                str(candidate.metadata.get("chunk_role_tags", "")),
                str(candidate.metadata.get("contextual_chunk_text", "")),
            ]
        )
        if slot == "budget" and int(candidate.metadata.get("has_budget_signal", 0) or 0):
            return True
        if slot in {"period", "schedule", "deadline"} and int(candidate.metadata.get("has_schedule_signal", 0) or 0):
            return True
        if slot == "contract" and int(candidate.metadata.get("has_contract_signal", 0) or 0):
            return True
        return any(alias in text for alias in _slot_aliases_v3(slot))

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
        source_doc = str((row.get("source_docs") or [""])[0]).strip()
        reranked = self._merge_and_rerank(row=row, query_text=row["question"], source_filter={source_doc} if source_doc else None)
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
                best_body = sorted(
                    body_candidates,
                    key=lambda c: (
                        self._same_section_bonus(best_table, c),
                        c.adjusted_score if c.adjusted_score is not None else c.fusion_score,
                    ),
                    reverse=True,
                )[0]
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

        selected = self.pipeline.rerank_with_profile(selected, self.pipeline.build_question_profile(row, row["question"]), row["question"])[:6]
        contexts = [self._candidate_to_context(candidate, rank=index + 1, route="v3_targeted") for index, candidate in enumerate(selected)]

        top2 = selected[:2]
        same_doc_top2_support = sum(1 for candidate in top2 if any(self._slot_match(candidate, slot) for slot in row.get("target_slots", [])))
        first_section = str(selected[0].metadata.get("section_title", "")) if selected else ""
        first_item = str(selected[0].metadata.get("chunk_role", "")) if selected else ""

        table_hit = any(int(candidate.metadata.get("has_table", 0) or 0) for candidate in selected) if task_type == "table_plus_body" else None
        nearby_body_hit = any(not int(candidate.metadata.get("has_table", 0) or 0) for candidate in selected) if task_type == "table_plus_body" else None
        parent_section_hit = False
        if task_type == "table_plus_body" and len(selected) >= 2:
            for left in selected:
                for right in selected:
                    if left.chunk_id == right.chunk_id:
                        continue
                    if int(left.metadata.get("has_table", 0) or 0) and not int(right.metadata.get("has_table", 0) or 0) and self._same_section_bonus(left, right) > 0:
                        parent_section_hit = True
                        break
                if parent_section_hit:
                    break

        weak_reasons: list[str] = []
        if task_type == "single_doc_factual":
            if same_doc_top2_support < 2:
                weak_reasons.append("same_doc_slot_support_low")
        elif task_type == "table_plus_body":
            if not table_hit or not nearby_body_hit:
                weak_reasons.append("table_body_coverage_low")
            elif not parent_section_hit:
                weak_reasons.append("parent_section_alignment_low")

        retrieval_row = {
            "id": f"{self.retrieval_profile}_qwen_v3_{row['qid']}",
            "qid": row["qid"],
            "question_id": row.get("question_id", row["qid"]),
            "source_docs": row.get("source_docs", []),
            "task_type": task_type,
            "answer_type": row["answer_type"],
            "question": row["question"],
            "expected_answerability": row["expected_answerability"],
            "target_slots": row.get("target_slots", []),
            "official_eval_overlap": bool(row.get("official_eval_overlap", False)),
            "requires_refusal": False,
            "retrieval_profile": self.retrieval_profile,
            "selected_pipeline": "v3_targeted",
            "contexts": contexts,
            "weak_context": bool(weak_reasons),
            "weak_context_reasons": weak_reasons,
            "metadata": {
                "question_source": row.get("provenance", ""),
                "table_related": task_type == "table_plus_body",
                "comparison": False,
                "follow_up": False,
                "primary_slots": row.get("target_slots", []),
                "factual_primary_slot": row.get("factual_primary_slot") or (row.get("target_slots") or [None])[0],
                "section_label": first_section,
                "item_title": first_item,
                "same_doc_top2_support": same_doc_top2_support,
                "table_hit": table_hit,
                "nearby_body_hit": nearby_body_hit,
                "parent_section_hit": parent_section_hit if task_type == "table_plus_body" else None,
                "table_body_coverage": {"table": bool(table_hit), "body": bool(nearby_body_hit)} if task_type == "table_plus_body" else None,
                "table_plus_body_strength": "strong" if table_hit and nearby_body_hit and parent_section_hit else "weak" if task_type == "table_plus_body" else None,
                "context_length": sum(len(ctx["text"]) for ctx in contexts),
            },
            "conversation_group_id": None,
            "depends_on_list": [],
        }
        fail_row = None
        if weak_reasons:
            fail_row = {
                "qid": row["qid"],
                "question": row["question"],
                "task_type": task_type,
                "selected_pipeline": "v3_targeted",
                "weak_context_reasons": weak_reasons,
                "provenance": row.get("provenance", ""),
            }
        return retrieval_row, fail_row

    def _post_filter(self, row: dict[str, Any]) -> dict[str, Any]:
        judge = dict(row.get("judge", {}) or {})
        task_type = str(row.get("task_type", ""))
        metadata = dict(row.get("metadata", {}) or {})
        accepted = True
        reasons: list[str] = []

        faithfulness = int(judge.get("faithfulness", 0) or 0)
        completeness = int(judge.get("completeness", 0) or 0)
        groundedness = int(judge.get("groundedness", 0) or 0)
        relevancy = int(judge.get("relevancy", 0) or 0)

        if row.get("official_eval_overlap"):
            accepted = False
            reasons.append("official_eval_overlap")
        if not str(row.get("target_answer", "")).strip():
            accepted = False
            reasons.append("empty_answer")
        if faithfulness < 4:
            accepted = False
            reasons.append("low_faithfulness")
        if groundedness < 4:
            accepted = False
            reasons.append("low_groundedness")
        if relevancy < 4:
            accepted = False
            reasons.append("low_relevancy")

        min_completeness = 4 if task_type == "single_doc_factual" else 3
        if completeness < min_completeness:
            accepted = False
            reasons.append("low_completeness")

        borderline_keep = False
        if row.get("weak_context"):
            weak_ok = False
            if task_type == "single_doc_factual":
                if int(metadata.get("same_doc_top2_support", 0) or 0) >= 1 and faithfulness >= 4 and groundedness >= 4 and relevancy >= 4 and completeness >= 4:
                    weak_ok = True
                    borderline_keep = True
            elif task_type == "table_plus_body":
                if metadata.get("table_hit") and metadata.get("nearby_body_hit") and faithfulness >= 4 and groundedness >= 4 and relevancy >= 4 and completeness >= 3:
                    weak_ok = True
                    borderline_keep = True
            if not weak_ok:
                accepted = False
                reasons.append("weak_context")

        if task_type == "table_plus_body":
            if not metadata.get("table_hit"):
                accepted = False
                reasons.append("table_hit_missing")
            if not metadata.get("nearby_body_hit"):
                accepted = False
                reasons.append("nearby_body_hit_missing")

        judge["accepted_for_sft"] = accepted
        judge["rejection_reason"] = "|".join(dict.fromkeys(reasons)) if reasons else str(judge.get("rejection_reason", "")).strip()
        row["judge"] = judge
        row["accepted_for_sft"] = accepted
        row["rejection_reason"] = judge["rejection_reason"]
        row.setdefault("metadata", {})
        row["metadata"]["answer_length"] = len(str(row.get("target_answer", "")))
        row["metadata"]["borderline_keep"] = borderline_keep
        return row

    def _prior_weak_context_by_type(self) -> Counter[str]:
        return Counter(
            str(row.get("task_type", ""))
            for row in self.prior_retrieval_rows
            if str(row.get("task_type", "")) in self.TARGET_TASK_TYPES and row.get("weak_context")
        )

    def _dedupe_against_prior(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prior_questions = [str(row.get("question", "")) for row in self.prior_raw_rows]
        filtered: list[dict[str, Any]] = []
        signatures = self._existing_signature_set()
        for row in rows:
            signature = self._row_signature(row)
            if signature in signatures:
                continue
            normalized = _normalize_question(row.get("question", ""))
            if any(_sequence_similarity(normalized, _normalize_question(question)) >= 0.97 for question in prior_questions):
                continue
            signatures.add(signature)
            filtered.append(row)
        return filtered

    def build_raw_dataset(self, judged_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        accepted_new = [row for row in judged_rows if row.get("accepted_for_sft")]
        accepted_new = self._dedupe_against_prior(accepted_new)

        raw_rows = list(self.prior_raw_rows)
        for row in accepted_new:
            raw_rows.append(
                {
                    "id": row["id"],
                    "source_docs": row["source_docs"],
                    "task_type": row["task_type"],
                    "question": row["question"],
                    "contexts": row["contexts"],
                    "target_answer": row["target_answer"],
                    "requires_refusal": row["requires_refusal"],
                    "teacher_model": row["teacher_model"],
                    "teacher_pipeline": row["teacher_pipeline"],
                    "retrieval_profile": row["retrieval_profile"],
                    "judge": row["judge"],
                    "official_eval_overlap": row["official_eval_overlap"],
                    "metadata": row["metadata"],
                    "accepted_for_sft": row["accepted_for_sft"],
                    "rejection_reason": row["rejection_reason"],
                    "conversation_group_id": row.get("conversation_group_id"),
                    "depends_on_list": row.get("depends_on_list", []),
                }
            )
        raw_rows.sort(key=lambda item: item["id"])
        _write_jsonl(self.data_root / f"raw_sft_dataset_{self.retrieval_profile}_qwen.jsonl", raw_rows)
        self.log_step("build_raw_dataset", previous_count=len(self.prior_raw_rows), added_count=len(accepted_new), merged_count=len(raw_rows))
        return raw_rows

    def write_compare_csvs(self, retrieval_rows: list[dict[str, Any]], judged_rows: list[dict[str, Any]]) -> None:
        import pandas as pd

        by_task = defaultdict(lambda: {"candidate_count": 0, "weak_context_count": 0, "accepted_count": 0})
        for row in retrieval_rows:
            task = str(row.get("task_type", ""))
            by_task[task]["candidate_count"] += 1
            if row.get("weak_context"):
                by_task[task]["weak_context_count"] += 1
        for row in judged_rows:
            task = str(row.get("task_type", ""))
            if row.get("accepted_for_sft"):
                by_task[task]["accepted_count"] += 1

        single_df = pd.DataFrame(
            [
                {
                    "task_type": "single_doc_factual",
                    **by_task.get("single_doc_factual", {}),
                }
            ]
        )
        table_df = pd.DataFrame(
            [
                {
                    "task_type": "table_plus_body",
                    **by_task.get("table_plus_body", {}),
                }
            ]
        )
        single_df.to_csv(self.single_doc_output_root / "single_doc_expansion_compare.csv", index=False, encoding="utf-8-sig")
        table_df.to_csv(self.table_body_output_root / "table_body_expansion_compare.csv", index=False, encoding="utf-8-sig")

    def build_v3_stats(
        self,
        *,
        retrieval_rows: list[dict[str, Any]],
        judged_rows: list[dict[str, Any]],
        raw_rows: list[dict[str, Any]],
        train_rows: list[dict[str, Any]],
        val_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        accepted_by_type = Counter(str(row.get("task_type", "")) for row in raw_rows)
        previous_by_type = dict(self.prior_stats.get("accepted_by_task_type", {}) or {})
        weak_context_by_type = Counter(
            str(row.get("task_type", "")) for row in retrieval_rows if row.get("weak_context")
        )
        prior_weak = self._prior_weak_context_by_type()

        stats = {
            "accepted_count": len(raw_rows),
            "accepted_count_previous": int(self.prior_stats.get("accepted_count", 0) or 0),
            "accepted_delta": len(raw_rows) - int(self.prior_stats.get("accepted_count", 0) or 0),
            "accepted_by_task_type": dict(accepted_by_type),
            "accepted_by_task_type_previous": previous_by_type,
            "single_doc_delta": accepted_by_type.get("single_doc_factual", 0) - int(previous_by_type.get("single_doc_factual", 0) or 0),
            "table_plus_body_delta": accepted_by_type.get("table_plus_body", 0) - int(previous_by_type.get("table_plus_body", 0) or 0),
            "weak_context_by_task_type": dict(weak_context_by_type),
            "weak_context_by_task_type_previous": dict(prior_weak),
            "train_count": len(train_rows),
            "val_count": len(val_rows),
            "ft_ready_minimum": accepted_by_type.get("single_doc_factual", 0) >= 1000 and accepted_by_type.get("table_plus_body", 0) >= 100,
            "ft_ready_overall": len(raw_rows) >= 2400 and accepted_by_type.get("single_doc_factual", 0) >= 1000 and accepted_by_type.get("table_plus_body", 0) >= 100,
        }
        _write_json(self.output_root / "qwen_sft_v3_refine_stats.json", stats)

        decision = "가능" if stats["ft_ready_overall"] else "보류"
        reason = (
            "single_doc_factual 1000+, table_plus_body 100+, accepted 2400+ 충족"
            if stats["ft_ready_overall"]
            else "single_doc_factual 또는 table_plus_body 최소 목표 미달"
        )
        lines = [
            "# Qwen SFT Targeted Expansion v3",
            "",
            f"- accepted: {stats['accepted_count_previous']} -> {stats['accepted_count']} (delta {stats['accepted_delta']})",
            f"- single_doc_factual: {previous_by_type.get('single_doc_factual', 0)} -> {accepted_by_type.get('single_doc_factual', 0)} (delta {stats['single_doc_delta']})",
            f"- table_plus_body: {previous_by_type.get('table_plus_body', 0)} -> {accepted_by_type.get('table_plus_body', 0)} (delta {stats['table_plus_body_delta']})",
            "",
            "## Final Distribution",
        ]
        for task_type in ["single_doc_factual", "comparison", "rejection", "table_plus_body", "follow_up"]:
            lines.append(f"- {task_type}: {accepted_by_type.get(task_type, 0)}")
        lines.extend(
            [
                "",
                "## Weak Context Trend",
                f"- single_doc_factual: {prior_weak.get('single_doc_factual', 0)} -> {weak_context_by_type.get('single_doc_factual', 0)}",
                f"- table_plus_body: {prior_weak.get('table_plus_body', 0)} -> {weak_context_by_type.get('table_plus_body', 0)}",
                "",
                "## FT Decision",
                f"- {decision}",
                f"- 이유: {reason}",
            ]
        )
        _write_markdown(self.output_root / "qwen_sft_v3_refine_report.md", lines)
        return stats

    def run(self) -> dict[str, Any]:
        pool_rows = self.build_target_question_pool()
        pool_rows, _ = self.build_overlap_blocklist(pool_rows)
        retrieval_rows = self.build_retrieval_candidates(pool_rows)
        judged_rows = self.generate_teacher_and_judge(retrieval_rows)
        self.write_compare_csvs(retrieval_rows, judged_rows)
        raw_rows = self.build_raw_dataset(judged_rows)
        train_rows, val_rows, split_manifest = self.split_raw_dataset(raw_rows)
        train_payload, val_payload = self.format_qwen_dataset(train_rows, val_rows)

        v3_train = self.data_root / "qwen_sft_train_v3.jsonl"
        v3_val = self.data_root / "qwen_sft_val_v3.jsonl"
        _write_jsonl(v3_train, train_payload)
        _write_jsonl(v3_val, val_payload)

        stats = self.build_v3_stats(
            retrieval_rows=retrieval_rows,
            judged_rows=judged_rows,
            raw_rows=raw_rows,
            train_rows=train_payload,
            val_rows=val_payload,
        )
        _write_jsonl(self.output_root / "qwen_sft_v3_build_log.jsonl", self.step_logs)
        return {
            "stats": stats,
            "split_manifest": split_manifest,
            "output_root": str(self.output_root),
        }

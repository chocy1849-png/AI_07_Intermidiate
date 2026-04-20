from __future__ import annotations

import json
import math
import random
import re
import sys
import time
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scenario_a.common_pipeline import CandidateRow, PipelinePaths, PipelineSettings
from scenario_a_qwen_ft.pipeline import QwenFTInstructionAPipeline
from scenario_a_qwen_ft.sft_pipeline import (
    QwenSFTDatasetBuilder,
    _contains_refusal,
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


def _slot_aliases(slot: str) -> list[str]:
    mapping = {
        "budget": ["예산", "금액", "사업비", "기초금액", "추정금액"],
        "period": ["기간", "사업 기간", "수행 기간", "일정", "개월", "일 이내"],
        "contract": ["계약 방식", "입찰 방식", "계약", "협상"],
        "purpose": ["목적", "배경", "필요성"],
        "scope": ["범위", "주요 범위", "구축 범위", "업무 범위"],
        "evaluation": ["평가 기준", "배점", "평가"],
        "eligibility": ["참가 자격", "자격 요건", "입찰 자격"],
        "deliverables": ["제출 조건", "제출물", "산출물", "납품"],
        "operations": ["운영 방식", "운영", "유지보수"],
        "integration": ["연계", "연동", "API"],
        "background": ["추진 배경", "배경"],
        "modules": ["기능 범위", "모듈", "기능"],
        "schedule": ["일정", "마감", "제출 기한"],
        "security": ["보안", "권한", "접근통제"],
    }
    return mapping.get(slot, [slot])


class QwenSFTExpansionBuilder(QwenSFTDatasetBuilder):
    def __init__(
        self,
        *,
        project_root: Path,
        question_root: Path,
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
            output_root=output_root or (project_root / "rag_outputs" / "qwen_ft_instruction_expansion_v1"),
            retrieval_profile=retrieval_profile,
            teacher_model=teacher_model,
            judge_model=judge_model,
            shard_count=shard_count,
            teacher_workers=teacher_workers,
            judge_workers=judge_workers,
            seed=seed,
        )
        self.pipeline = self.pipeline.__class__(
            PipelinePaths(project_root=self.project_root),
            PipelineSettings(embedding_backend_key=self.retrieval_profile, candidate_k=10, top_k=5),
        )
        self.question_generator = QwenFTInstructionAPipeline(project_root=self.project_root, output_dir=self.question_root)
        self.auto_vetted_rows = _read_jsonl(self.question_root / "question_auto_vetted.jsonl")
        self.auto_by_qid = {row["qid"]: row for row in self.auto_vetted_rows}
        self.candidate_rows = _read_jsonl(self.question_root / "question_candidates.jsonl")
        self.candidate_by_qid = {row["qid"]: row for row in self.candidate_rows}
        self.base_final_rows = _read_jsonl(self.question_root / "question_final_vetted.jsonl")
        self.base_output_root = self.project_root / "rag_outputs" / "qwen_ft_instruction"
        self.human_review_df = pd.read_csv(self.question_root / "question_human_review_sheet_filled.csv", dtype=str).fillna("")
        self.policy_targets = {
            "comparison": 1200,
            "rejection": 900,
            "table_plus_body": 600,
            "follow_up": 700,
        }

    @staticmethod
    def _answer_type_from_task(task_type: str) -> str:
        mapping = {
            "single_doc_factual": "factual",
            "comparison": "comparison",
            "rejection": "rejection",
            "table_plus_body": "factual",
            "follow_up": "follow_up",
        }
        return mapping.get(task_type, "factual")

    def _normalize_candidate(self, row: dict[str, Any], *, provenance: str) -> dict[str, Any]:
        normalized = dict(row)
        normalized["qid"] = str(row.get("qid", "")).strip()
        normalized["question"] = str(row.get("question", "")).strip()
        normalized["task_type"] = str(row.get("task_type", "")).strip()
        normalized["question_id"] = normalized["qid"]
        normalized["answer_type"] = self._answer_type_from_task(normalized["task_type"])
        normalized["source_docs"] = _split_pipe_list(row.get("source_docs"))
        normalized["target_slots"] = _split_pipe_list(row.get("target_slots"))
        depends_on = _split_pipe_list(row.get("depends_on_list"))
        if not depends_on and row.get("depends_on_qid"):
            depends_on = [str(row.get("depends_on_qid")).strip()]
        normalized["depends_on_list"] = depends_on
        normalized["official_eval_overlap"] = bool(row.get("official_eval_overlap", False))
        normalized["expected_answerability"] = str(row.get("expected_answerability", "answerable")).strip() or "answerable"
        normalized["requires_refusal"] = normalized["expected_answerability"] == "no_answer"
        normalized["provenance"] = provenance
        return normalized

    def _base_approved_rows(self) -> list[dict[str, Any]]:
        rows = [self._normalize_candidate(row, provenance="base_approved") for row in self.base_final_rows]
        rows.sort(key=lambda item: item["qid"])
        return rows

    def _revive_follow_up_rows(self, approved_qids: set[str]) -> list[dict[str, Any]]:
        revived: list[dict[str, Any]] = []
        for row in self.auto_vetted_rows:
            if row.get("task_type") != "follow_up" or row.get("auto_status") != "auto_pass_basic":
                continue
            qid = str(row.get("qid", "")).strip()
            if not qid or qid in approved_qids:
                continue
            parent_qid = str(row.get("depends_on_qid", "")).strip()
            if not parent_qid:
                continue
            parent = self.candidate_by_qid.get(parent_qid) or self.auto_by_qid.get(parent_qid)
            if not parent:
                continue
            normalized = self._normalize_candidate(row, provenance="policy_revived_follow_up")
            normalized["parent_qid"] = parent_qid
            normalized["parent_question"] = str(parent.get("question", "")).strip()
            normalized["parent_source_docs"] = _split_pipe_list(parent.get("source_docs"))
            normalized["source_state_anchor"] = "same_comparison_pair" if len(normalized["source_docs"]) >= 2 else "same_doc"
            normalized["followup_resolvable_with_context"] = True
            normalized["conversation_group_id"] = str(row.get("conversation_group_id", "")).strip() or f"revive::{parent_qid}"
            revived.append(normalized)
        revived.sort(key=lambda item: item["qid"])
        return revived

    def _load_existing_signatures(self, rows: list[dict[str, Any]]) -> set[str]:
        signatures: set[str] = set()
        for row in rows:
            signature = "::".join(
                [
                    str(row.get("task_type", "")),
                    "|".join(sorted(_split_pipe_list(row.get("source_docs")))),
                    "|".join(sorted(_split_pipe_list(row.get("target_slots")))),
                    _normalize_question(row.get("question", "")),
                ]
            )
            signatures.add(signature)
        return signatures

    def _official_question_bank(self) -> list[dict[str, Any]]:
        return self._load_official_eval_bank()

    def _is_official_overlap(self, question: str, source_docs: list[str]) -> bool:
        normalized = _normalize_question(question)
        for row in self._official_question_bank():
            official_question = str(row.get("question", "")).strip()
            if not official_question:
                continue
            if normalized == _normalize_question(official_question):
                return True
            similarity = _sequence_similarity(normalized, _normalize_question(official_question))
            if similarity >= 0.94:
                return True
        return False

    def _new_qid(self, prefix: str, index: int) -> str:
        return f"{prefix}_{index:06d}"

    def _comparison_templates(self, left: str, right: str, slot_a: str, slot_b: str) -> list[str]:
        a = self.question_generator._slot_label(slot_a)
        b = self.question_generator._slot_label(slot_b)
        return [
            f"{left}와 {right}를 {a}과 {b} 기준으로 비교해줘.",
            f"{left}와 {right}의 {a} 차이와 {b} 차이를 함께 정리해줘.",
            f"{left}와 {right}를 비교할 때 {a}, {b} 측면에서 각각 어떻게 다른지 설명해줘.",
        ]

    def _rejection_templates(self, subject: str, slot_alias: str) -> list[str]:
        return [
            f"{subject}의 {slot_alias}는 무엇이야?",
            f"{subject}에서 {slot_alias} 정보를 알려줘.",
        ]

    def _table_body_templates(self, subject: str, table_slot: str, body_slot: str) -> list[str]:
        a = self.question_generator._slot_label(table_slot)
        b = self.question_generator._slot_label(body_slot)
        return [
            f"{subject}에서 {a}과 {b}를 함께 정리해줘. 표와 본문 정보를 모두 반영해줘.",
            f"{subject}의 {a} 내용을 표 기준으로 보고, {b}는 본문 설명까지 포함해서 요약해줘.",
        ]

    def _follow_up_templates(self, parent: dict[str, Any]) -> list[str]:
        slots = list(parent.get("target_slots", []))
        slot_a = self.question_generator._slot_label(slots[0]) if slots else "핵심 조건"
        slot_b = self.question_generator._slot_label(slots[1]) if len(slots) > 1 else "추가 조건"
        if parent.get("task_type") == "comparison":
            return [
                f"그 비교 기준에서 {slot_a} 말고 {slot_b} 쪽 차이도 같이 설명해줘.",
                f"방금 비교한 두 사업 기준으로 {slot_a}보다 실무 영향이 큰 쪽이 어디인지 근거와 함께 말해줘.",
            ]
        return [
            f"그 사업 기준으로 {slot_a} 말고 추가로 확인해야 할 조건은 뭐가 있어?",
            f"그 문서에서 {slot_a}와 함께 보면 중요한 {slot_b} 조건도 정리해줘.",
        ]

    def _generate_targeted_candidates(self, existing_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        signature_seen = self._load_existing_signatures(existing_rows)

        comparison_index = 1
        comparison_slots = [
            ("budget", "period"),
            ("evaluation", "scope"),
            ("contract", "deliverables"),
            ("purpose", "scope"),
            ("operations", "integration"),
        ]
        for doc_a, doc_b in self.question_generator._build_comparison_pairs(limit_pairs=1400):
            left = self.question_generator._short_doc_subject(doc_a)
            right = self.question_generator._short_doc_subject(doc_b)
            for slot_a, slot_b in comparison_slots:
                if slot_a not in (doc_a.supported_slots | doc_b.supported_slots) and slot_b not in (doc_a.supported_slots | doc_b.supported_slots):
                    continue
                for question in self._comparison_templates(left, right, slot_a, slot_b):
                    signature = "::".join(["comparison", "|".join(sorted([doc_a.source_file_name, doc_b.source_file_name])), "|".join(sorted([slot_a, slot_b])), _normalize_question(question)])
                    if signature in signature_seen or self._is_official_overlap(question, [doc_a.source_file_name, doc_b.source_file_name]):
                        continue
                    rows.append(
                        self._normalize_candidate(
                            {
                                "qid": self._new_qid("exp_cmp", comparison_index),
                                "question": question,
                                "task_type": "comparison",
                                "expected_answerability": "answerable",
                                "source_docs": [doc_a.source_file_name, doc_b.source_file_name],
                                "target_slots": [slot_a, slot_b],
                                "comparison_axis": f"{slot_a}|{slot_b}",
                            },
                            provenance="generated_comparison",
                        )
                    )
                    signature_seen.add(signature)
                    comparison_index += 1
                    if comparison_index > self.policy_targets["comparison"]:
                        break
                if comparison_index > self.policy_targets["comparison"]:
                    break
            if comparison_index > self.policy_targets["comparison"]:
                break

        allowed_slots = list(self.question_generator.type4_lexicon.get("allowed_slots", {}).keys())
        rejection_index = 1
        for doc in self.question_generator.documents:
            subject = self.question_generator._short_doc_subject(doc)
            for slot in allowed_slots:
                slot_alias = self.question_generator._pick_slot_alias(slot)
                for question in self._rejection_templates(subject, slot_alias):
                    signature = "::".join(["rejection", doc.source_file_name, slot, _normalize_question(question)])
                    if signature in signature_seen or self._is_official_overlap(question, [doc.source_file_name]):
                        continue
                    rows.append(
                        self._normalize_candidate(
                            {
                                "qid": self._new_qid("exp_rej", rejection_index),
                                "question": question,
                                "task_type": "rejection",
                                "expected_answerability": "no_answer",
                                "source_docs": [doc.source_file_name],
                                "target_slots": [slot],
                            },
                            provenance="generated_rejection",
                        )
                    )
                    signature_seen.add(signature)
                    rejection_index += 1
                    if rejection_index > self.policy_targets["rejection"]:
                        break
                if rejection_index > self.policy_targets["rejection"]:
                    break
            if rejection_index > self.policy_targets["rejection"]:
                break

        table_index = 1
        for doc in self.question_generator.documents:
            if not doc.table_like_slots or not doc.body_like_slots:
                continue
            subject = self.question_generator._short_doc_subject(doc)
            for table_slot in sorted(doc.table_like_slots):
                for body_slot in sorted(doc.body_like_slots):
                    if table_slot == body_slot:
                        continue
                    for question in self._table_body_templates(subject, table_slot, body_slot):
                        signature = "::".join(["table_plus_body", doc.source_file_name, "|".join(sorted([table_slot, body_slot])), _normalize_question(question)])
                        if signature in signature_seen or self._is_official_overlap(question, [doc.source_file_name]):
                            continue
                        rows.append(
                            self._normalize_candidate(
                                {
                                    "qid": self._new_qid("exp_tbl", table_index),
                                    "question": question,
                                    "task_type": "table_plus_body",
                                    "expected_answerability": "answerable",
                                    "source_docs": [doc.source_file_name],
                                    "target_slots": [table_slot, body_slot],
                                },
                                provenance="generated_table_plus_body",
                            )
                        )
                        signature_seen.add(signature)
                        table_index += 1
                        if table_index > self.policy_targets["table_plus_body"]:
                            break
                    if table_index > self.policy_targets["table_plus_body"]:
                        break
                if table_index > self.policy_targets["table_plus_body"]:
                    break
            if table_index > self.policy_targets["table_plus_body"]:
                break

        followup_index = 1
        parent_pool = [row for row in existing_rows if row.get("task_type") in {"single_doc_factual", "comparison"}]
        random.shuffle(parent_pool)
        for parent in parent_pool:
            parent_qid = str(parent.get("qid", "")).strip()
            if not parent_qid:
                continue
            source_docs = list(parent.get("source_docs", []))
            if not source_docs:
                continue
            anchor = "same_comparison_pair" if len(source_docs) >= 2 else "same_doc"
            for question in self._follow_up_templates(parent):
                signature = "::".join(["follow_up", "|".join(sorted(source_docs)), "|".join(sorted(parent.get("target_slots", []))), _normalize_question(question)])
                if signature in signature_seen or self._is_official_overlap(question, source_docs):
                    continue
                row = self._normalize_candidate(
                    {
                        "qid": self._new_qid("exp_fup", followup_index),
                        "question": question,
                        "task_type": "follow_up",
                        "expected_answerability": "answerable",
                        "source_docs": source_docs,
                        "target_slots": list(parent.get("target_slots", [])),
                        "depends_on_qid": parent_qid,
                        "conversation_group_id": f"exp_followup::{parent_qid}",
                    },
                    provenance="generated_follow_up",
                )
                row["parent_qid"] = parent_qid
                row["parent_question"] = str(parent.get("question", "")).strip()
                row["parent_source_docs"] = source_docs
                row["source_state_anchor"] = anchor
                row["followup_resolvable_with_context"] = True
                rows.append(row)
                signature_seen.add(signature)
                followup_index += 1
                if followup_index > self.policy_targets["follow_up"]:
                    break
            if followup_index > self.policy_targets["follow_up"]:
                break

        return rows

    def build_expansion_question_pool(self) -> list[dict[str, Any]]:
        base_rows = self._base_approved_rows()
        approved_qids = {row["qid"] for row in base_rows}
        revived_followups = self._revive_follow_up_rows(approved_qids)
        generated_rows = self._generate_targeted_candidates(base_rows + revived_followups)

        merged: list[dict[str, Any]] = []
        signature_seen = set()
        for row in [*base_rows, *revived_followups, *generated_rows]:
            question = str(row.get("question", "")).strip()
            if not question:
                continue
            signature = "::".join(
                [
                    str(row.get("task_type", "")),
                    "|".join(sorted(row.get("source_docs", []))),
                    "|".join(sorted(row.get("target_slots", []))),
                    _normalize_question(question),
                ]
            )
            if signature in signature_seen:
                continue
            signature_seen.add(signature)
            merged.append(row)

        merged.sort(key=lambda item: item["qid"])
        _write_jsonl(self.output_root / "question_pool_expansion.jsonl", merged)
        _write_json(
            self.output_root / "question_pool_expansion_stats.json",
            {
                "base_approved_count": len(base_rows),
                "revived_follow_up_count": len(revived_followups),
                "generated_count": len(generated_rows),
                "merged_count": len(merged),
                "by_provenance": dict(Counter(row.get("provenance", "") for row in merged)),
                "by_task_type": dict(Counter(row.get("task_type", "") for row in merged)),
            },
        )
        self.log_step(
            "build_expansion_question_pool",
            base_approved=len(base_rows),
            revived_follow_up=len(revived_followups),
            generated=len(generated_rows),
            merged=len(merged),
        )
        return merged

    def _search_vector_filtered(self, query_embedding: list[float], *, source_docs: set[str], limit: int = 20) -> list[CandidateRow]:
        result = self.pipeline.chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            include=["documents", "metadatas", "distances"],
        )
        rows: list[CandidateRow] = []
        rank = 1
        for document, metadata, _ in zip(result["documents"][0], result["metadatas"][0], result["distances"][0], strict=False):
            source = str(metadata.get("source_file_name", ""))
            if source_docs and source not in source_docs:
                continue
            rows.append(
                CandidateRow(
                    chunk_id=str(metadata.get("chunk_id", "")),
                    text=str(document or ""),
                    metadata=dict(metadata),
                    fusion_score=(1.0 / (self.pipeline.settings.rrf_k + rank)) * self.pipeline.settings.vector_weight,
                )
            )
            rank += 1
        return rows

    def _search_bm25_filtered(self, query_text: str, *, source_docs: set[str], limit: int = 20) -> list[CandidateRow]:
        model = self.pipeline.bm25_index["model"]
        chunk_rows = self.pipeline.bm25_index["chunk_rows"]
        scores = model.get_scores(self.pipeline._bm25_tokenize(query_text))
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        rows: list[CandidateRow] = []
        rank = 1
        for row_index, _ in ranked:
            source = chunk_rows[row_index]
            if source_docs and str(source.get("source_file_name", "")) not in source_docs:
                continue
            rows.append(
                CandidateRow(
                    chunk_id=str(source.get("chunk_id", "")),
                    text=str(source.get("contextual_chunk_text", "")),
                    metadata=dict(source),
                    fusion_score=(1.0 / (self.pipeline.settings.rrf_k + rank)) * self.pipeline.settings.bm25_weight,
                )
            )
            rank += 1
            if len(rows) >= limit:
                break
        return rows

    def _merge_and_rerank(self, *, row: dict[str, Any], query_text: str, source_filter: set[str] | None = None) -> list[CandidateRow]:
        query_embedding = self.pipeline.embed_question(query_text)
        vector_rows = self._search_vector_filtered(query_embedding, source_docs=source_filter or set(), limit=24)
        bm25_rows = self._search_bm25_filtered(query_text, source_docs=source_filter or set(), limit=24)
        fused = self.pipeline.fuse_candidates(vector_rows, bm25_rows)
        reranked = self.pipeline.rerank_with_profile(fused, self.pipeline.build_question_profile(row, query_text), query_text)
        return reranked

    def _candidate_to_context(self, candidate: CandidateRow, rank: int, route: str) -> dict[str, Any]:
        return {
            "doc_id": str(candidate.metadata.get("document_id", "")),
            "chunk_id": str(candidate.metadata.get("chunk_id", "")),
            "file_name": str(candidate.metadata.get("source_file_name", "")),
            "section_label": str(candidate.metadata.get("section_title", "")),
            "parent_section_label": str(candidate.metadata.get("section_path", "")),
            "item_title": str(candidate.metadata.get("chunk_role", "")),
            "text": candidate.text,
            "rank": rank,
            "route": route,
        }

    def _pick_doc_anchored_candidates(self, row: dict[str, Any], query_text: str) -> list[CandidateRow]:
        expected_docs = set(row.get("source_docs", []))
        profile = self.pipeline.build_question_profile(row, query_text)
        selected: list[CandidateRow] = []
        seen: set[str] = set()
        for source_doc in expected_docs:
            reranked = self._merge_and_rerank(row=row, query_text=query_text, source_filter={source_doc})
            for candidate in reranked:
                if candidate.chunk_id in seen:
                    continue
                adjusted = (candidate.adjusted_score if candidate.adjusted_score is not None else candidate.fusion_score) + 0.0035
                selected.append(
                    CandidateRow(
                        chunk_id=candidate.chunk_id,
                        text=candidate.text,
                        metadata=dict(candidate.metadata),
                        fusion_score=candidate.fusion_score,
                        adjusted_score=adjusted,
                    )
                )
                seen.add(candidate.chunk_id)
                break
        return self.pipeline.rerank_with_profile(selected, profile, query_text)

    def _detect_axis_in_answer(self, answer: str, target_slots: list[str]) -> bool:
        text = str(answer or "")
        for slot in target_slots:
            if any(alias in text for alias in _slot_aliases(slot)):
                return True
        return False

    def _build_support_probe(self, row: dict[str, Any]) -> dict[str, Any]:
        source_docs = set(row.get("source_docs", []))
        slot_terms: list[str] = []
        slot_patterns: list[str] = []
        allowed_slots = self.question_generator.type4_lexicon.get("allowed_slots", {})
        for slot in row.get("target_slots", []):
            slot_terms.append(slot)
            slot_terms.extend(allowed_slots.get(slot, {}).get("aliases", []))
            slot_patterns.extend(allowed_slots.get(slot, {}).get("support_patterns", []))
        query_embedding = self.pipeline.embed_question(row["question"])

        def score_support_text(text: str) -> tuple[int, int]:
            normalized = str(text or "").lower()
            direct = 1 if any(re.search(pattern, str(text or ""), re.IGNORECASE) for pattern in slot_patterns) else 0
            indirect = 1 if (not direct and any(term and term.lower() in normalized for term in slot_terms)) else 0
            return direct, indirect

        dense_direct = 0
        dense_indirect = 0
        dense_result = self.pipeline.chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=12,
            include=["documents", "metadatas"],
        )
        for document, metadata in zip(dense_result["documents"][0], dense_result["metadatas"][0], strict=False):
            if str(metadata.get("source_file_name", "")) not in source_docs:
                continue
            direct, indirect = score_support_text(str(document or ""))
            dense_direct += direct
            dense_indirect += indirect

        bm25_direct = 0
        bm25_indirect = 0
        scores = self.pipeline.bm25_index["model"].get_scores(self.pipeline._bm25_tokenize(row["question"]))
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:12]
        for row_index, _ in ranked:
            chunk = self.pipeline.bm25_index["chunk_rows"][row_index]
            if str(chunk.get("source_file_name", "")) not in source_docs:
                continue
            direct, indirect = score_support_text(str(chunk.get("contextual_chunk_text", "")))
            bm25_direct += direct
            bm25_indirect += indirect

        metadata_direct = 0
        metadata_indirect = 0
        table_body_direct = 0
        table_body_indirect = 0
        for source_file_name in source_docs:
            doc = self.question_generator.doc_map.get(source_file_name)
            if doc is None:
                continue
            direct, indirect = score_support_text(json.dumps(doc.metadata, ensure_ascii=False))
            metadata_direct += direct
            metadata_indirect += indirect
            table_blob = " ".join(str(block.get("text", "")) for block in doc.table_blocks[:12])
            for text in [table_blob, doc.rag_ready_text[:3000]]:
                if not text:
                    continue
                direct, indirect = score_support_text(text)
                table_body_direct += direct
                table_body_indirect += indirect

        dense_hits = {"direct_hits": dense_direct, "indirect_hits": dense_indirect}
        bm25_hits = {"direct_hits": bm25_direct, "indirect_hits": bm25_indirect}
        norm_bm25_hits = {"direct_hits": 0, "indirect_hits": 0}
        metadata_hits = {"direct_hits": metadata_direct, "indirect_hits": metadata_indirect}
        table_body_hits = {"direct_hits": table_body_direct, "indirect_hits": table_body_indirect}
        direct_total = dense_hits["direct_hits"] + bm25_hits["direct_hits"] + norm_bm25_hits["direct_hits"] + metadata_hits["direct_hits"] + table_body_hits["direct_hits"]
        indirect_total = dense_hits["indirect_hits"] + bm25_hits["indirect_hits"] + norm_bm25_hits["indirect_hits"] + metadata_hits["indirect_hits"] + table_body_hits["indirect_hits"]
        ambiguity_flag = bool(indirect_total >= 3 and direct_total == 0)
        support_judge = "not_supported"
        confidence = 0.8
        if direct_total >= 1:
            support_judge = "supported"
            confidence = min(0.98, 0.65 + direct_total * 0.08)
        elif ambiguity_flag or indirect_total >= 4:
            support_judge = "ambiguous"
            confidence = 0.55
        return {
            "direct_support_flag": direct_total >= 1,
            "indirect_support_flag": indirect_total >= 1,
            "ambiguity_flag": ambiguity_flag,
            "support_confidence": round(confidence, 4),
            "support_judge": support_judge,
        }

    def _build_retrieval_candidate_one(self, row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
        task_type = str(row.get("task_type", ""))
        source_docs = list(row.get("source_docs", []))
        route = self.pipeline.decide_route(row)
        query_text = row["question"]
        if task_type == "follow_up" and row.get("parent_question"):
            query_text = f"{row['parent_question']}\n후속 질문: {row['question']}"

        reranked = self._merge_and_rerank(row=row, query_text=query_text)
        selected = list(reranked[:6])

        if source_docs:
            anchored = self._pick_doc_anchored_candidates(row, query_text)
            merged = self.pipeline.fuse_candidates(selected, anchored)
            selected = self.pipeline.rerank_with_profile(merged, self.pipeline.build_question_profile(row, query_text), query_text)[:6]

        if task_type == "comparison":
            candidates = list(reranked[:8])
            if route == "b03a":
                candidates = self.pipeline.apply_crag(row, query_text, candidates)
            selected = list(candidates[:6])
            selected_docs = {str(candidate.metadata.get("source_file_name", "")) for candidate in selected}
            for source_doc in source_docs[:2]:
                if source_doc in selected_docs:
                    continue
                doc_candidates = self._merge_and_rerank(row=row, query_text=query_text, source_filter={source_doc})
                if doc_candidates:
                    selected.append(doc_candidates[0])
            selected = self.pipeline.rerank_with_profile(selected, self.pipeline.build_question_profile(row, query_text), query_text)[:6]
        elif task_type == "table_plus_body":
            table_present = any(int(candidate.metadata.get("has_table", 0) or 0) for candidate in selected)
            body_present = any(not int(candidate.metadata.get("has_table", 0) or 0) for candidate in selected)
            if source_docs:
                extra = self._merge_and_rerank(row=row, query_text=query_text, source_filter={source_docs[0]})
                if not table_present:
                    for candidate in extra:
                        if int(candidate.metadata.get("has_table", 0) or 0):
                            selected.append(candidate)
                            table_present = True
                            break
                if not body_present:
                    for candidate in extra:
                        if not int(candidate.metadata.get("has_table", 0) or 0):
                            selected.append(candidate)
                            body_present = True
                            break
                selected = self.pipeline.rerank_with_profile(selected, self.pipeline.build_question_profile(row, query_text), query_text)[:6]

        contexts = [self._candidate_to_context(candidate, rank=index + 1, route=route) for index, candidate in enumerate(selected)]
        selected_doc_set = {ctx["file_name"] for ctx in contexts if ctx["file_name"]}
        dual_doc_coverage = len(set(source_docs) & selected_doc_set) if task_type == "comparison" else None
        table_hit = any(int(candidate.metadata.get("has_table", 0) or 0) for candidate in selected) if task_type == "table_plus_body" else None
        nearby_body_hit = any(not int(candidate.metadata.get("has_table", 0) or 0) for candidate in selected) if task_type == "table_plus_body" else None
        table_body_coverage = {"table": bool(table_hit), "body": bool(nearby_body_hit)} if task_type == "table_plus_body" else None

        support_probe = self._build_support_probe(row) if task_type == "rejection" else {}
        parent_qid = ""
        source_state_anchor = ""
        followup_resolvable = None
        if task_type == "follow_up":
            parent_qid = str(row.get("parent_qid") or (row.get("depends_on_list") or [""])[0]).strip()
            source_state_anchor = str(row.get("source_state_anchor", "")).strip() or ("same_comparison_pair" if len(source_docs) >= 2 else "same_doc")
            followup_resolvable = bool(parent_qid and set(source_docs) <= selected_doc_set)

        weak_reasons: list[str] = []
        if task_type == "comparison" and (dual_doc_coverage or 0) < min(2, len(set(source_docs)) or 2):
            weak_reasons.append("dual_doc_coverage_low")
        elif task_type == "table_plus_body" and not (table_hit and nearby_body_hit):
            weak_reasons.append("table_body_coverage_low")
        elif task_type == "rejection" and (support_probe.get("ambiguity_flag") or support_probe.get("support_judge") == "supported"):
            weak_reasons.append("rejection_support_ambiguous")
        elif task_type == "follow_up" and not followup_resolvable:
            weak_reasons.append("followup_anchor_resolution_low")
        elif len(contexts) < 2 and task_type != "rejection":
            weak_reasons.append("too_few_contexts")

        retrieval_row = {
            "id": f"{self.retrieval_profile}_qwen_exp_{row['qid']}",
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
            "selected_pipeline": route,
            "contexts": contexts,
            "weak_context": bool(weak_reasons),
            "weak_context_reasons": weak_reasons,
            "metadata": {
                "question_source": row.get("provenance", ""),
                "table_related": task_type == "table_plus_body" or any(slot in {"budget", "schedule", "evaluation", "modules"} for slot in row.get("target_slots", [])),
                "comparison": task_type == "comparison",
                "follow_up": task_type == "follow_up",
                "primary_slots": row.get("target_slots", []),
                "section_labels": sorted({ctx["section_label"] for ctx in contexts if ctx["section_label"]}),
                "doc_count": len(selected_doc_set),
                "dual_doc_coverage": dual_doc_coverage,
                "comparison_axis_detected": task_type == "comparison",
                "comparison_context_strength": "strong" if (dual_doc_coverage or 0) >= 2 else "weak",
                "table_hit": table_hit,
                "nearby_body_hit": nearby_body_hit,
                "table_body_coverage": table_body_coverage,
                "direct_support_flag": support_probe.get("direct_support_flag"),
                "indirect_support_flag": support_probe.get("indirect_support_flag"),
                "ambiguity_flag": support_probe.get("ambiguity_flag"),
                "support_confidence": support_probe.get("support_confidence"),
                "support_judge": support_probe.get("support_judge"),
                "parent_qid": parent_qid or None,
                "source_state_anchor": source_state_anchor or None,
                "followup_resolvable_with_context": followup_resolvable,
                "context_length": sum(len(ctx["text"]) for ctx in contexts),
            },
            "conversation_group_id": row.get("conversation_group_id"),
            "depends_on_list": row.get("depends_on_list", []),
            "parent_question": row.get("parent_question"),
            "parent_source_docs": row.get("parent_source_docs", []),
        }
        fail_row = None
        if weak_reasons:
            fail_row = {
                "qid": row["qid"],
                "question": row["question"],
                "task_type": task_type,
                "selected_pipeline": route,
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
        parts = []
        if row.get("task_type") == "follow_up" and row.get("parent_question"):
            parts.extend(["[이전 질문]", str(row.get("parent_question", ""))])
        parts.extend(["[질문]", row["question"], "[문맥]", "\n\n".join(context_blocks)])
        parts.extend(["[응답 지침]", "문맥에 근거해서만 답하세요. follow_up이면 이전 질문의 대상 사업/비교쌍을 유지하세요. rejection이면 문맥에 없을 때만 없다고 답하세요."])
        return "\n".join(parts)

    def _judge_prompt(self, row: dict[str, Any]) -> tuple[str, str]:
        context_text = "\n\n".join(
            [
                "\n".join(
                    [
                        f"[문서 {ctx['rank']} | {ctx['file_name']} | {ctx.get('section_label', '')}]",
                        ctx["text"],
                    ]
                )
                for ctx in row["contexts"]
            ]
        )
        extra = ""
        if row.get("task_type") == "follow_up" and row.get("parent_question"):
            extra += f"이전 질문:\n{row.get('parent_question', '')}\n\n"
        if row.get("task_type") == "rejection":
            md = row.get("metadata", {}) or {}
            extra += (
                "지원 판정 메타:\n"
                f"- support_judge: {md.get('support_judge')}\n"
                f"- direct_support_flag: {md.get('direct_support_flag')}\n"
                f"- ambiguity_flag: {md.get('ambiguity_flag')}\n\n"
            )
        system_prompt = (
            "You are the judge for a Korean RFP QA dataset. "
            "Score the answer on faithfulness, completeness, groundedness, and relevancy from 1 to 5. "
            "Return JSON only."
        )
        user_prompt = (
            f"{extra}"
            f"질문:\n{row['question']}\n\n"
            f"문맥:\n{context_text}\n\n"
            f"teacher 답변:\n{row.get('target_answer', '')}\n\n"
            "JSON only:\n"
            "{\n"
            '  "faithfulness": 1,\n'
            '  "completeness": 1,\n'
            '  "groundedness": 1,\n'
            '  "relevancy": 1,\n'
            '  "accepted_for_sft": false,\n'
            '  "rejection_reason": ""\n'
            "}"
        )
        return system_prompt, user_prompt

    def _post_filter(self, row: dict[str, Any]) -> dict[str, Any]:
        judge = dict(row.get("judge", {}) or {})
        task_type = str(row.get("task_type", ""))
        metadata = dict(row.get("metadata", {}) or {})
        accepted = True
        reasons: list[str] = []

        if row.get("official_eval_overlap"):
            accepted = False
            reasons.append("official_eval_overlap")
        if not str(row.get("target_answer", "")).strip():
            accepted = False
            reasons.append("empty_answer")
        if row.get("requires_refusal") and not _contains_refusal(row.get("target_answer", "")):
            accepted = False
            reasons.append("refusal_failure")

        faithfulness = int(judge.get("faithfulness", 0) or 0)
        completeness = int(judge.get("completeness", 0) or 0)
        groundedness = int(judge.get("groundedness", 0) or 0)
        relevancy = int(judge.get("relevancy", 0) or 0)

        if faithfulness < 4:
            accepted = False
            reasons.append("low_faithfulness")
        if groundedness < 4:
            accepted = False
            reasons.append("low_groundedness")
        if relevancy < 4:
            accepted = False
            reasons.append("low_relevancy")

        min_completeness = 4
        if task_type in {"comparison", "table_plus_body", "rejection", "follow_up"}:
            min_completeness = 3
        if completeness < min_completeness:
            accepted = False
            reasons.append("low_completeness")

        borderline_keep = False
        if row.get("weak_context"):
            weak_ok = False
            if task_type == "comparison":
                dual_ok = int(metadata.get("dual_doc_coverage", 0) or 0) >= 2
                if dual_ok and faithfulness >= 4 and groundedness >= 4 and relevancy >= 4:
                    weak_ok = True
                    borderline_keep = True
                else:
                    reasons.append("comparison_doc_coverage_low")
            elif task_type == "table_plus_body":
                if metadata.get("table_hit") and metadata.get("nearby_body_hit") and faithfulness >= 4 and groundedness >= 4 and relevancy >= 4 and completeness >= 3:
                    weak_ok = True
                    borderline_keep = True
            elif task_type == "rejection":
                if (
                    metadata.get("support_judge") == "not_supported"
                    and not metadata.get("direct_support_flag")
                    and not metadata.get("ambiguity_flag")
                    and faithfulness >= 4
                    and groundedness >= 4
                    and relevancy >= 4
                ):
                    weak_ok = True
                    borderline_keep = True
            elif task_type == "follow_up":
                if metadata.get("followup_resolvable_with_context") and metadata.get("source_state_anchor") and faithfulness >= 4 and groundedness >= 4 and relevancy >= 4:
                    weak_ok = True
                    borderline_keep = True
            if not weak_ok:
                accepted = False
                reasons.append("weak_context")

        if task_type == "comparison" and int(metadata.get("dual_doc_coverage", 0) or 0) < 2:
            accepted = False
            if "comparison_doc_coverage_low" not in reasons:
                reasons.append("comparison_doc_coverage_low")
        if task_type == "table_plus_body" and not (metadata.get("table_hit") and metadata.get("nearby_body_hit")):
            accepted = False
            if "table_body_coverage_low" not in reasons:
                reasons.append("table_body_coverage_low")
        if task_type == "follow_up" and not metadata.get("followup_resolvable_with_context"):
            accepted = False
            if "followup_unresolved" not in reasons:
                reasons.append("followup_unresolved")
        if task_type == "rejection" and metadata.get("ambiguity_flag"):
            accepted = False
            if "rejection_ambiguity" not in reasons:
                reasons.append("rejection_ambiguity")

        judge["accepted_for_sft"] = accepted
        judge["rejection_reason"] = "|".join(dict.fromkeys(reasons)) if reasons else str(judge.get("rejection_reason", "")).strip()
        row["judge"] = judge
        row["accepted_for_sft"] = accepted
        row["rejection_reason"] = judge["rejection_reason"]
        row.setdefault("metadata", {})
        row["metadata"]["answer_length"] = len(str(row.get("target_answer", "")))
        row["metadata"]["borderline_keep"] = borderline_keep
        row["metadata"]["comparison_axis_detected"] = self._detect_axis_in_answer(row.get("target_answer", ""), list(row.get("target_slots", []))) if task_type == "comparison" else metadata.get("comparison_axis_detected")
        return row

    def build_expansion_stats(
        self,
        *,
        pool_rows: list[dict[str, Any]],
        retrieval_rows: list[dict[str, Any]],
        judged_rows: list[dict[str, Any]],
        raw_rows: list[dict[str, Any]],
        train_rows: list[dict[str, Any]],
        val_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        old_stats_path = self.base_output_root / "qwen_sft_dataset_stats.json"
        prior_stats = json.loads(old_stats_path.read_text(encoding="utf-8")) if old_stats_path.exists() else {}
        accepted_by_type = Counter(row.get("task_type", "") for row in raw_rows)
        weak_context_by_type = Counter(row.get("task_type", "") for row in retrieval_rows if row.get("weak_context"))
        stats = {
            "question_pool_count": len(pool_rows),
            "retrieval_candidate_count": len(retrieval_rows),
            "weak_context_count": sum(1 for row in retrieval_rows if row.get("weak_context")),
            "weak_context_count_previous": int(prior_stats.get("weak_context_count", 0) or 0),
            "weak_context_delta": sum(1 for row in retrieval_rows if row.get("weak_context")) - int(prior_stats.get("weak_context_count", 0) or 0),
            "judged_count": len(judged_rows),
            "accepted_count": len(raw_rows),
            "accepted_count_previous": int(prior_stats.get("accepted_count", 0) or 0),
            "accepted_delta": len(raw_rows) - int(prior_stats.get("accepted_count", 0) or 0),
            "accepted_by_task_type": dict(accepted_by_type),
            "accepted_by_task_type_previous": dict(prior_stats.get("task_type_distribution", {}) or {}),
            "weak_context_by_task_type": dict(weak_context_by_type),
            "train_count": len(train_rows),
            "val_count": len(val_rows),
            "ft_ready_minimum": len(raw_rows) >= 2000
            and accepted_by_type.get("single_doc_factual", 0) >= 1000
            and accepted_by_type.get("comparison", 0) >= 300
            and accepted_by_type.get("rejection", 0) >= 200
            and accepted_by_type.get("table_plus_body", 0) >= 100
            and accepted_by_type.get("follow_up", 0) >= 100,
        }
        _write_json(self.output_root / "qwen_sft_expansion_stats.json", stats)

        lines = [
            "# Qwen SFT Expansion v1",
            "",
            "## Accepted Recovery",
            f"- previous accepted: {stats['accepted_count_previous']}",
            f"- current accepted: {stats['accepted_count']}",
            f"- delta: {stats['accepted_delta']}",
            "",
            "## Weak Context",
            f"- previous weak_context: {stats['weak_context_count_previous']}",
            f"- current weak_context: {stats['weak_context_count']}",
            f"- delta: {stats['weak_context_delta']}",
            "",
            "## Accepted by Task Type",
        ]
        for task_type, count in sorted(accepted_by_type.items()):
            previous = int((prior_stats.get("task_type_distribution", {}) or {}).get(task_type, 0) or 0)
            lines.append(f"- {task_type}: {previous} -> {count} (delta {count - previous})")
        lines.extend(
            [
                "",
                "## FT Readiness",
                f"- ft_ready_minimum: {stats['ft_ready_minimum']}",
                "- minimum target: accepted 2000+, single_doc 1000+, comparison 300+, rejection 200+, table_plus_body 100+, follow_up 100+",
            ]
        )
        _write_markdown(self.output_root / "qwen_sft_expansion_report.md", lines)
        self.log_step("build_expansion_stats", accepted=stats["accepted_count"], ft_ready=stats["ft_ready_minimum"])
        return stats

    def run(self) -> dict[str, Any]:
        pool_rows = self.build_expansion_question_pool()
        pool_rows, _ = self.build_overlap_blocklist(pool_rows)
        retrieval_rows = self.build_retrieval_candidates(pool_rows)
        judged_rows = self.generate_teacher_and_judge(retrieval_rows)
        raw_rows = self.build_raw_dataset(judged_rows)
        train_rows, val_rows, split_manifest = self.split_raw_dataset(raw_rows)
        train_payload, val_payload = self.format_qwen_dataset(train_rows=train_rows, val_rows=val_rows)
        stats = self.build_expansion_stats(
            pool_rows=pool_rows,
            retrieval_rows=retrieval_rows,
            judged_rows=judged_rows,
            raw_rows=raw_rows,
            train_rows=train_payload,
            val_rows=val_payload,
        )
        _write_jsonl(self.output_root / "qwen_sft_expansion_build_log.jsonl", self.step_logs)
        return {
            "stats": stats,
            "split_manifest": split_manifest,
            "output_root": str(self.output_root),
        }

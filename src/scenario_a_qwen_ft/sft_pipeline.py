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

from scenario_a.common_pipeline import PipelinePaths, PipelineSettings, ScenarioACommonPipeline


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _normalize_question(text: str) -> str:
    value = str(text or "").strip().lower()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^\w가-힣]", "", value)
    return value


def _sequence_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    from difflib import SequenceMatcher

    return SequenceMatcher(None, left, right).ratio()


def _contains_refusal(text: str) -> bool:
    value = str(text or "")
    patterns = [
        "제공된 문맥만으로는 확인되지 않습니다",
        "문맥만으로는 확인되지 않습니다",
        "확인되지 않습니다",
        "문서에 명시되어 있지 않습니다",
        "문서에 없습니다",
    ]
    return any(pattern in value for pattern in patterns)


class QwenSFTDatasetBuilder:
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
        teacher_workers: int = 4,
        judge_workers: int = 4,
        seed: int = 20260409,
    ) -> None:
        self.project_root = project_root.resolve()
        self.question_root = question_root.resolve()
        self.output_root = output_root or (self.project_root / "rag_outputs" / "qwen_ft_instruction")
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.data_root = self.output_root / "data" / "sft"
        self.shard_root = self.output_root / "shards"
        self.retrieval_profile = retrieval_profile
        self.teacher_model = teacher_model
        self.judge_model = judge_model
        self.shard_count = shard_count
        self.teacher_workers = teacher_workers
        self.judge_workers = judge_workers
        self.random = random.Random(seed)
        self.pipeline = ScenarioACommonPipeline(
            PipelinePaths(project_root=self.project_root),
            PipelineSettings(embedding_backend_key=self.retrieval_profile),
        )
        self.step_logs: list[dict[str, Any]] = []

    def log_step(self, step: str, **payload: Any) -> None:
        self.step_logs.append({"step": step, "ts": time.strftime("%Y-%m-%d %H:%M:%S"), **payload})

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

    @staticmethod
    def _split_pipe_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if value is None:
            return []
        text = str(value).strip()
        if not text:
            return []
        return [item.strip() for item in text.split("|") if item.strip()]

    def finalize_approved_questions(self) -> list[dict[str, Any]]:
        filled_jsonl = self.question_root / "question_final_vetted_filled.jsonl"
        approved_rows: list[dict[str, Any]] = []

        if filled_jsonl.exists():
            raw_rows = _read_jsonl(filled_jsonl)
            for row in raw_rows:
                approved_rows.append(
                    {
                        **row,
                        "question_id": str(row.get("qid", "")),
                        "answer_type": self._answer_type_from_task(str(row.get("task_type", ""))),
                        "source_docs": self._split_pipe_list(row.get("source_docs")),
                        "target_slots": self._split_pipe_list(row.get("target_slots")),
                        "depends_on_list": self._split_pipe_list(row.get("depends_on_qid")),
                        "official_eval_overlap": False,
                    }
                )
        else:
            auto_rows = {row["qid"]: row for row in _read_jsonl(self.question_root / "question_auto_vetted.jsonl")}
            filled_csv = self.question_root / "question_human_review_sheet_filled.csv"
            if not filled_csv.exists():
                raise FileNotFoundError(f"Filled review sheet not found: {filled_csv}")
            df = pd.read_csv(filled_csv, dtype=str).fillna("")
            for _, row in df.iterrows():
                decision = str(row.get("final_human_decision", "")).strip().lower()
                if decision not in {"approve", "approved", "human_approved", "yes", "y"}:
                    continue
                qid = str(row["qid"]).strip()
                source = auto_rows.get(qid)
                if source is None:
                    continue
                merged = dict(source)
                merged["human_review_status"] = "human_approved"
                merged["final_verdict"] = "approved_for_teacher"
                merged["question_id"] = qid
                merged["answer_type"] = self._answer_type_from_task(str(merged.get("task_type", "")))
                merged["depends_on_list"] = [merged["depends_on_qid"]] if merged.get("depends_on_qid") else []
                approved_rows.append(merged)

        approved_rows.sort(key=lambda item: item["qid"])
        _write_jsonl(self.question_root / "question_final_vetted.jsonl", approved_rows)
        self.log_step("finalize_approved_questions", approved_count=len(approved_rows))
        return approved_rows

    def _load_official_eval_bank(self) -> list[dict[str, Any]]:
        paths = [
            self.project_root / "docs" / "planning" / "pm" / "day3_partA_eval_questions_v1.txt",
            self.project_root / "docs" / "planning" / "pm" / "eval_questions_table_v1.txt",
        ]
        questions: list[dict[str, Any]] = []
        for path in paths:
            if not path.exists():
                continue
            lines = path.read_text(encoding="utf-8").splitlines()
            current: dict[str, Any] | None = None
            for raw_line in lines:
                head = raw_line.strip()
                if not head:
                    continue
                if re.fullmatch(r"(Q\d{2}|T[ABC]-\d{2})", head):
                    if current:
                        questions.append(current)
                    current = {"qid": head}
                    continue
                if current is None or ":" not in raw_line:
                    continue
                key, value = raw_line.split(":", 1)
                current[key.strip()] = value.strip()
            if current:
                questions.append(current)
        return questions

    def build_overlap_blocklist(self, approved_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        official_questions = self._load_official_eval_bank()
        blocklist = [
            {
                "qid": item.get("qid", ""),
                "question": item.get("question", ""),
                "normalized_question": _normalize_question(item.get("question", "")),
            }
            for item in official_questions
            if item.get("question")
        ]
        logs: list[dict[str, Any]] = []
        for row in approved_rows:
            question = str(row.get("question", ""))
            normalized = _normalize_question(question)
            best_match = {"qid": "", "question": "", "similarity": 0.0}
            overlap = False
            for item in blocklist:
                similarity = _sequence_similarity(normalized, str(item["normalized_question"]))
                if similarity > best_match["similarity"]:
                    best_match = {
                        "qid": str(item["qid"]),
                        "question": str(item["question"]),
                        "similarity": round(similarity, 4),
                    }
                if normalized and normalized == item["normalized_question"]:
                    overlap = True
                    break
                if similarity >= 0.94:
                    overlap = True
                    break
            row["official_eval_overlap"] = overlap
            logs.append(
                {
                    "qid": row["qid"],
                    "question": question,
                    "official_eval_overlap": overlap,
                    "best_match_qid": best_match["qid"],
                    "best_match_question": best_match["question"],
                    "best_similarity": best_match["similarity"],
                }
            )
        _write_json(self.output_root / "eval_overlap_blocklist.json", blocklist)
        _write_jsonl(self.output_root / "eval_overlap_check_log.jsonl", logs)
        self.log_step(
            "build_overlap_blocklist",
            approved_count=len(approved_rows),
            overlap_count=sum(1 for row in approved_rows if row.get("official_eval_overlap")),
        )
        return approved_rows, logs

    def _build_retrieval_candidate_one(self, row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
        retrieval = self.pipeline.retrieve(row, row["question"])
        contexts: list[dict[str, Any]] = []
        unique_docs: list[str] = []
        seen_docs: set[str] = set()
        has_table = False
        has_body = False
        for rank, candidate in enumerate(retrieval.candidates, start=1):
            file_name = str(candidate.metadata.get("source_file_name", ""))
            if file_name and file_name not in seen_docs:
                seen_docs.add(file_name)
                unique_docs.append(file_name)
            if int(candidate.metadata.get("has_table", 0) or 0):
                has_table = True
            else:
                has_body = True
            contexts.append(
                {
                    "doc_id": str(candidate.metadata.get("document_id", "")),
                    "chunk_id": str(candidate.metadata.get("chunk_id", "")),
                    "file_name": file_name,
                    "section_label": str(candidate.metadata.get("section_title", "")),
                    "parent_section_label": str(candidate.metadata.get("section_path", "")),
                    "item_title": str(candidate.metadata.get("chunk_role", "")),
                    "text": candidate.text,
                    "rank": rank,
                    "route": retrieval.route,
                }
            )

        weak_reasons: list[str] = []
        expected_doc_set = set(row.get("source_docs", []))
        dual_doc_coverage = len(set(unique_docs) & expected_doc_set) if row["task_type"] == "comparison" else None
        table_body_coverage = {"table": has_table, "body": has_body} if row["task_type"] == "table_plus_body" else None
        if len(contexts) < 2:
            weak_reasons.append("too_few_contexts")
        if row["task_type"] == "comparison" and (dual_doc_coverage or 0) < min(2, len(expected_doc_set) or 2):
            weak_reasons.append("dual_doc_coverage_low")
        if row["task_type"] == "table_plus_body" and not (has_table and has_body):
            weak_reasons.append("table_body_coverage_low")

        retrieval_row = {
            "id": f"{self.retrieval_profile}_qwen_{row['qid']}",
            "qid": row["qid"],
            "question_id": row.get("question_id", row["qid"]),
            "source_docs": row.get("source_docs", []),
            "task_type": row["task_type"],
            "answer_type": row["answer_type"],
            "question": row["question"],
            "expected_answerability": row["expected_answerability"],
            "target_slots": row.get("target_slots", []),
            "official_eval_overlap": bool(row.get("official_eval_overlap", False)),
            "requires_refusal": row["expected_answerability"] == "no_answer",
            "retrieval_profile": self.retrieval_profile,
            "selected_pipeline": retrieval.route,
            "contexts": contexts,
            "weak_context": bool(weak_reasons),
            "weak_context_reasons": weak_reasons,
            "metadata": {
                "table_related": row["task_type"] == "table_plus_body" or any(slot in {"budget", "schedule", "evaluation", "modules"} for slot in row.get("target_slots", [])),
                "comparison": row["task_type"] == "comparison",
                "follow_up": row["task_type"] == "follow_up",
                "primary_slots": row.get("target_slots", []),
                "section_labels": sorted({ctx["section_label"] for ctx in contexts if ctx["section_label"]}),
                "doc_count": len(unique_docs),
                "dual_doc_coverage": dual_doc_coverage,
                "table_body_coverage": table_body_coverage,
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
                "task_type": row["task_type"],
                "selected_pipeline": retrieval.route,
                "weak_context_reasons": weak_reasons,
            }
        return retrieval_row, fail_row

    def _process_retrieval_shard(self, shard_index: int, shard_rows: list[dict[str, Any]]) -> dict[str, Any]:
        shard_dir = self.shard_root / f"shard_{shard_index + 1:02d}_of_{self.shard_count:02d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        retrieval_path = shard_dir / f"retrieval_candidates_{self.retrieval_profile}.jsonl"
        fail_path = shard_dir / "retrieval_fail_log.jsonl"
        existing_retrieval = {row["qid"]: row for row in _read_jsonl(retrieval_path)}
        existing_fail = {row["qid"]: row for row in _read_jsonl(fail_path)}
        processed = 0

        for row in shard_rows:
            qid = row["qid"]
            if qid in existing_retrieval:
                continue
            retrieval_row, fail_row = self._build_retrieval_candidate_one(row)
            existing_retrieval[qid] = retrieval_row
            if fail_row is not None:
                existing_fail[qid] = fail_row
            _write_jsonl(retrieval_path, sorted(existing_retrieval.values(), key=lambda item: item["qid"]))
            _write_jsonl(fail_path, sorted(existing_fail.values(), key=lambda item: item["qid"]))
            processed += 1

        return {"shard_index": shard_index, "row_count": len(shard_rows), "processed": processed}

    def build_retrieval_candidates(self, approved_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out_path = self.output_root / f"retrieval_candidates_{self.retrieval_profile}.jsonl"
        fail_path = self.output_root / "retrieval_fail_log.jsonl"
        shards = self._assign_shards(approved_rows)
        with ThreadPoolExecutor(max_workers=min(self.shard_count, self.teacher_workers)) as executor:
            futures = [executor.submit(self._process_retrieval_shard, idx, shard_rows) for idx, shard_rows in enumerate(shards) if shard_rows]
            shard_reports = [future.result() for future in as_completed(futures)]

        retrieval_rows: list[dict[str, Any]] = []
        fail_rows: list[dict[str, Any]] = []
        for idx in range(self.shard_count):
            shard_dir = self.shard_root / f"shard_{idx + 1:02d}_of_{self.shard_count:02d}"
            retrieval_rows.extend(_read_jsonl(shard_dir / f"retrieval_candidates_{self.retrieval_profile}.jsonl"))
            fail_rows.extend(_read_jsonl(shard_dir / "retrieval_fail_log.jsonl"))

        retrieval_rows.sort(key=lambda item: item["qid"])
        fail_rows.sort(key=lambda item: item["qid"])
        _write_jsonl(out_path, retrieval_rows)
        _write_jsonl(fail_path, fail_rows)
        self.log_step(
            "build_retrieval_candidates",
            total=len(retrieval_rows),
            weak_context=len(fail_rows),
            shard_reports=sorted(shard_reports, key=lambda item: item["shard_index"]),
        )
        return retrieval_rows

    @staticmethod
    def _teacher_system_prompt() -> str:
        return (
            "You are the teacher model for a Korean RFP QA dataset. "
            "Answer only from the provided evidence. "
            "Do not invent unsupported facts. "
            "If the evidence is insufficient, answer in Korean with this exact style: "
            "'제공된 문맥만으로는 확인되지 않습니다.' "
            "For comparison questions, make the comparison axis explicit. "
            "For table-plus-body questions, use both table evidence and body evidence when available. "
            "Return only the final answer in Korean."
        )

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
        return "\n\n".join(
            [
                "[질문]",
                row["question"],
                "[문맥]",
                "\n\n".join(context_blocks),
                "[답변 지침]",
                "문맥에 근거해서만 한국어로 직접 답변하세요. 답변만 출력하세요.",
            ]
        )

    def _call_openai_text(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int,
        reasoning_effort: str | None = None,
        retries: int = 5,
    ) -> str:
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "input": [
                        {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                        {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
                    ],
                    "max_output_tokens": max_output_tokens,
                }
                if reasoning_effort:
                    kwargs["reasoning"] = {"effort": reasoning_effort}
                response = self.pipeline.openai_client.responses.create(
                    **kwargs,
                )
                return (response.output_text or "").strip()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt == retries:
                    break
                time.sleep(min(30.0, 2.0 * attempt))
        assert last_error is not None
        raise last_error

    def _teacher_generate_one(self, row: dict[str, Any]) -> dict[str, Any]:
        started = time.time()
        answer = self._call_openai_text(
            model=self.teacher_model,
            system_prompt=self._teacher_system_prompt(),
            user_prompt=self._teacher_user_prompt(row),
            max_output_tokens=700,
            reasoning_effort="low",
        )
        return {
            **row,
            "target_answer": answer,
            "teacher_model": self.teacher_model,
            "teacher_pipeline": f"scenario_a_{self.retrieval_profile}_teacher_v1",
            "teacher_elapsed_sec": round(time.time() - started, 3),
        }

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
        system_prompt = (
            "You are the judge for a Korean RFP QA dataset. "
            "Score the answer on faithfulness, completeness, groundedness, and relevancy from 1 to 5. "
            "Return JSON only."
        )
        user_prompt = (
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

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any]:
        cleaned = str(text).strip()
        cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("JSON object not found in judge response.")
        return json.loads(cleaned[start : end + 1])

    def _judge_one(self, row: dict[str, Any]) -> dict[str, Any]:
        started = time.time()
        system_prompt, user_prompt = self._judge_prompt(row)
        payload: dict[str, Any] | None = None
        last_raw = ""
        for _ in range(3):
            raw = self._call_openai_text(
                model=self.judge_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_output_tokens=500,
                reasoning_effort="minimal",
            )
            last_raw = raw
            try:
                payload = self._extract_json_object(raw)
                break
            except Exception:  # noqa: BLE001
                user_prompt = user_prompt + "\n\nReturn valid JSON only. Do not add any explanation."
        if payload is None:
            payload = {
                "faithfulness": 0,
                "completeness": 0,
                "groundedness": 0,
                "relevancy": 0,
                "accepted_for_sft": False,
                "rejection_reason": "judge_parse_error",
                "raw_judge_output": last_raw[:1000],
            }
        judge = {
            "faithfulness": int(payload["faithfulness"]),
            "completeness": int(payload["completeness"]),
            "groundedness": int(payload["groundedness"]),
            "relevancy": int(payload["relevancy"]),
            "accepted_for_sft": bool(payload["accepted_for_sft"]),
            "rejection_reason": str(payload.get("rejection_reason", "")).strip(),
        }
        return {**row, "judge": judge, "judge_elapsed_sec": round(time.time() - started, 3)}

    def _post_filter(self, row: dict[str, Any]) -> dict[str, Any]:
        judge = dict(row.get("judge", {}) or {})
        accepted = bool(judge.get("accepted_for_sft", False))
        reasons: list[str] = []

        if row.get("official_eval_overlap"):
            accepted = False
            reasons.append("official_eval_overlap")
        if row.get("weak_context"):
            accepted = False
            reasons.append("weak_context")
        if not str(row.get("target_answer", "")).strip():
            accepted = False
            reasons.append("empty_answer")
        if row.get("requires_refusal") and not _contains_refusal(row.get("target_answer", "")):
            accepted = False
            reasons.append("refusal_failure")

        min_completeness = 3 if row.get("task_type") in {"comparison", "table_plus_body"} else 4
        if int(judge.get("faithfulness", 0) or 0) < 4:
            accepted = False
            reasons.append("low_faithfulness")
        if int(judge.get("groundedness", 0) or 0) < 4:
            accepted = False
            reasons.append("low_groundedness")
        if int(judge.get("relevancy", 0) or 0) < 4:
            accepted = False
            reasons.append("low_relevancy")
        if int(judge.get("completeness", 0) or 0) < min_completeness:
            accepted = False
            reasons.append("low_completeness")
        if row.get("task_type") == "comparison" and (row.get("metadata", {}) or {}).get("dual_doc_coverage", 0) < 2:
            accepted = False
            reasons.append("comparison_doc_coverage_low")

        judge["accepted_for_sft"] = accepted
        judge["rejection_reason"] = "|".join(dict.fromkeys(reasons)) if reasons else str(judge.get("rejection_reason", "")).strip()
        row["judge"] = judge
        row["accepted_for_sft"] = accepted
        row["rejection_reason"] = judge["rejection_reason"]
        row.setdefault("metadata", {})
        row["metadata"]["answer_length"] = len(str(row.get("target_answer", "")))
        return row

    def _shard_index(self, qid: str) -> int:
        return zlib.crc32(qid.encode("utf-8")) % self.shard_count

    def _assign_shards(self, rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        shards: list[list[dict[str, Any]]] = [[] for _ in range(self.shard_count)]
        for row in rows:
            shards[self._shard_index(row["qid"])].append(row)
        for shard_rows in shards:
            shard_rows.sort(key=lambda item: item["qid"])
        return shards

    def _process_teacher_judge_shard(self, shard_index: int, shard_rows: list[dict[str, Any]]) -> dict[str, Any]:
        shard_dir = self.shard_root / f"shard_{shard_index + 1:02d}_of_{self.shard_count:02d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        teacher_path = shard_dir / f"teacher_answers_{self.retrieval_profile}.jsonl"
        judge_path = shard_dir / f"judged_sft_candidates_{self.retrieval_profile}.jsonl"
        teacher_log_path = shard_dir / "teacher_generation_log.jsonl"

        teacher_rows = {row["qid"]: row for row in _read_jsonl(teacher_path)}
        judged_rows = {row["qid"]: row for row in _read_jsonl(judge_path)}
        teacher_logs = {row["qid"]: row for row in _read_jsonl(teacher_log_path)}

        processed_teacher = 0
        processed_judge = 0
        for row in shard_rows:
            qid = row["qid"]
            if qid not in teacher_rows:
                answered = self._teacher_generate_one(row)
                teacher_rows[qid] = answered
                teacher_logs[qid] = {
                    "qid": qid,
                    "teacher_elapsed_sec": answered["teacher_elapsed_sec"],
                    "teacher_model": self.teacher_model,
                }
                processed_teacher += 1
                _write_jsonl(teacher_path, sorted(teacher_rows.values(), key=lambda item: item["qid"]))
                _write_jsonl(teacher_log_path, sorted(teacher_logs.values(), key=lambda item: item["qid"]))

            if qid not in judged_rows:
                judged = self._post_filter(self._judge_one(teacher_rows[qid]))
                judged_rows[qid] = judged
                processed_judge += 1
                _write_jsonl(judge_path, sorted(judged_rows.values(), key=lambda item: item["qid"]))

        return {
            "shard_index": shard_index,
            "row_count": len(shard_rows),
            "teacher_processed": processed_teacher,
            "judge_processed": processed_judge,
        }

    def generate_teacher_and_judge(self, retrieval_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        shards = self._assign_shards(retrieval_rows)
        with ThreadPoolExecutor(max_workers=min(self.shard_count, self.teacher_workers)) as executor:
            futures = [executor.submit(self._process_teacher_judge_shard, idx, shard_rows) for idx, shard_rows in enumerate(shards) if shard_rows]
            shard_reports = [future.result() for future in as_completed(futures)]

        teacher_rows: list[dict[str, Any]] = []
        teacher_logs: list[dict[str, Any]] = []
        judged_rows: list[dict[str, Any]] = []
        for idx in range(self.shard_count):
            shard_dir = self.shard_root / f"shard_{idx + 1:02d}_of_{self.shard_count:02d}"
            teacher_rows.extend(_read_jsonl(shard_dir / f"teacher_answers_{self.retrieval_profile}.jsonl"))
            teacher_logs.extend(_read_jsonl(shard_dir / "teacher_generation_log.jsonl"))
            judged_rows.extend(_read_jsonl(shard_dir / f"judged_sft_candidates_{self.retrieval_profile}.jsonl"))

        teacher_rows.sort(key=lambda item: item["qid"])
        teacher_logs.sort(key=lambda item: item["qid"])
        judged_rows.sort(key=lambda item: item["qid"])
        _write_jsonl(self.output_root / f"teacher_answers_{self.retrieval_profile}.jsonl", teacher_rows)
        _write_jsonl(self.output_root / "teacher_generation_log.jsonl", teacher_logs)
        _write_jsonl(self.output_root / f"judged_sft_candidates_{self.retrieval_profile}.jsonl", judged_rows)

        filter_report = {
            "total_rows": len(judged_rows),
            "accepted_rows": sum(1 for row in judged_rows if row.get("accepted_for_sft")),
            "rejected_rows": sum(1 for row in judged_rows if not row.get("accepted_for_sft")),
            "avg_teacher_latency_sec": round(sum(float(row.get("teacher_elapsed_sec", 0.0) or 0.0) for row in teacher_rows) / len(teacher_rows), 4) if teacher_rows else 0.0,
            "avg_judge_latency_sec": round(sum(float(row.get("judge_elapsed_sec", 0.0) or 0.0) for row in judged_rows) / len(judged_rows), 4) if judged_rows else 0.0,
            "rejection_reason_counts": dict(Counter(reason for row in judged_rows for reason in str(row.get("rejection_reason", "")).split("|") if reason)),
            "shard_reports": sorted(shard_reports, key=lambda item: item["shard_index"]),
        }
        _write_json(self.output_root / "qwen_sft_filter_report.json", filter_report)
        self.log_step("generate_teacher_and_judge", total=len(judged_rows), accepted=filter_report["accepted_rows"])
        return judged_rows

    def build_raw_dataset(self, judged_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        accepted_rows = [row for row in judged_rows if row.get("accepted_for_sft")]
        raw_rows: list[dict[str, Any]] = []
        for row in accepted_rows:
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
        _write_jsonl(self.data_root / f"raw_sft_dataset_{self.retrieval_profile}_qwen.jsonl", raw_rows)
        self.log_step("build_raw_dataset", accepted_count=len(raw_rows))
        return raw_rows

    @staticmethod
    def _group_key(row: dict[str, Any]) -> str:
        if row.get("conversation_group_id"):
            return f"conv::{row['conversation_group_id']}"
        duplicate_group = str(row.get("duplicate_group_id", "")).strip()
        if duplicate_group:
            return f"dup::{duplicate_group}"
        return str(row["id"])

    def split_raw_dataset(self, raw_rows: list[dict[str, Any]], val_ratio: float = 0.1) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in raw_rows:
            grouped[self._group_key(row)].append(row)

        groups_by_type: dict[str, list[tuple[str, list[dict[str, Any]]]]] = defaultdict(list)
        for group_key, rows in grouped.items():
            task_type = str(rows[0]["task_type"])
            groups_by_type[task_type].append((group_key, rows))

        train_rows: list[dict[str, Any]] = []
        val_rows: list[dict[str, Any]] = []
        split_manifest = {"val_ratio": val_ratio, "task_type_counts": {}}

        for task_type, groups in groups_by_type.items():
            self.random.shuffle(groups)
            total = sum(len(rows) for _, rows in groups)
            target_val = max(1, math.floor(total * val_ratio)) if total >= 10 else 0
            current_val = 0
            task_train: list[dict[str, Any]] = []
            task_val: list[dict[str, Any]] = []
            for _, rows in groups:
                if current_val < target_val:
                    task_val.extend(rows)
                    current_val += len(rows)
                else:
                    task_train.extend(rows)
            train_rows.extend(task_train)
            val_rows.extend(task_val)
            split_manifest["task_type_counts"][task_type] = {
                "total": total,
                "train": len(task_train),
                "val": len(task_val),
            }

        train_rows.sort(key=lambda item: item["id"])
        val_rows.sort(key=lambda item: item["id"])
        split_manifest["train_count"] = len(train_rows)
        split_manifest["val_count"] = len(val_rows)
        _write_json(self.data_root / "qwen_sft_split_manifest.json", split_manifest)
        self.log_step("split_raw_dataset", train_count=len(train_rows), val_count=len(val_rows))
        return train_rows, val_rows, split_manifest

    @staticmethod
    def _system_message() -> str:
        return (
            "당신은 한국어 RFP 문서를 근거로 답변하는 어시스턴트다. "
            "제공된 문맥에 근거해서만 답하고, 문서에 없는 정보는 추측하지 않는다."
        )

    @staticmethod
    def _format_context_blocks(row: dict[str, Any]) -> str:
        blocks = []
        for ctx in row["contexts"]:
            blocks.append(
                "\n".join(
                    [
                        f"[문서 {ctx['rank']} | {ctx['file_name']} | {ctx.get('section_label', '')}]",
                        ctx["text"],
                    ]
                )
            )
        return "\n\n".join(blocks)

    def format_qwen_dataset(self, train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        def convert(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            payload: list[dict[str, Any]] = []
            for row in rows:
                payload.append(
                    {
                        "messages": [
                            {"role": "system", "content": self._system_message()},
                            {
                                "role": "user",
                                "content": "\n\n".join(
                                    [
                                        "질문:",
                                        row["question"],
                                        "문맥:",
                                        self._format_context_blocks(row),
                                    ]
                                ),
                            },
                            {"role": "assistant", "content": row["target_answer"]},
                        ],
                        "metadata": {
                            "id": row["id"],
                            "task_type": row["task_type"],
                            "retrieval_profile": row["retrieval_profile"],
                            "source_docs": row["source_docs"],
                            "requires_refusal": row["requires_refusal"],
                        },
                    }
                )
            return payload

        train_payload = convert(train_rows)
        val_payload = convert(val_rows)
        _write_jsonl(self.data_root / "qwen_sft_train.jsonl", train_payload)
        _write_jsonl(self.data_root / "qwen_sft_val.jsonl", val_payload)
        self.log_step("format_qwen_dataset", train_rows=len(train_payload), val_rows=len(val_payload))
        return train_payload, val_payload

    def build_stats_reports(
        self,
        *,
        approved_rows: list[dict[str, Any]],
        retrieval_rows: list[dict[str, Any]],
        judged_rows: list[dict[str, Any]],
        raw_rows: list[dict[str, Any]],
        train_payload: list[dict[str, Any]],
        val_payload: list[dict[str, Any]],
    ) -> dict[str, Any]:
        top_docs = Counter(doc for row in raw_rows for doc in row.get("source_docs", []))
        rejected_rows = [row for row in judged_rows if not row.get("accepted_for_sft")]
        stats = {
            "approved_question_count": len(approved_rows),
            "retrieval_candidate_count": len(retrieval_rows),
            "weak_context_count": sum(1 for row in retrieval_rows if row.get("weak_context")),
            "judged_count": len(judged_rows),
            "accepted_count": len(raw_rows),
            "rejected_count": len(rejected_rows),
            "task_type_distribution": dict(Counter(row["task_type"] for row in raw_rows)),
            "refusal_count": sum(1 for row in raw_rows if row.get("requires_refusal")),
            "comparison_count": sum(1 for row in raw_rows if row.get("task_type") == "comparison"),
            "table_plus_body_count": sum(1 for row in raw_rows if row.get("task_type") == "table_plus_body"),
            "avg_context_length": round(sum(int((row.get("metadata", {}) or {}).get("context_length", 0) or 0) for row in raw_rows) / len(raw_rows), 2) if raw_rows else 0.0,
            "avg_answer_length": round(sum(int((row.get("metadata", {}) or {}).get("answer_length", 0) or 0) for row in raw_rows) / len(raw_rows), 2) if raw_rows else 0.0,
            "top_source_docs": [{"source_doc": doc, "count": count} for doc, count in top_docs.most_common(20)],
            "train_count": len(train_payload),
            "val_count": len(val_payload),
        }
        _write_json(self.output_root / "qwen_sft_dataset_stats.json", stats)

        lines = [
            "# Qwen SFT Distribution Report",
            "",
            f"- approved_question_count: {stats['approved_question_count']}",
            f"- retrieval_candidate_count: {stats['retrieval_candidate_count']}",
            f"- weak_context_count: {stats['weak_context_count']}",
            f"- judged_count: {stats['judged_count']}",
            f"- accepted_count: {stats['accepted_count']}",
            f"- rejected_count: {stats['rejected_count']}",
            f"- train_count: {stats['train_count']}",
            f"- val_count: {stats['val_count']}",
            "",
            "## Task Type Distribution",
        ]
        for task_type, count in stats["task_type_distribution"].items():
            lines.append(f"- {task_type}: {count}")
        lines.extend(["", "## Top Source Docs"])
        for item in stats["top_source_docs"]:
            lines.append(f"- {item['source_doc']}: {item['count']}")
        _write_markdown(self.output_root / "qwen_sft_distribution_report.md", lines)

        examples_lines = ["# Qwen SFT Examples", "", "## Accepted Examples"]
        for row in raw_rows[:3]:
            examples_lines.extend(
                [
                    f"- id: {row['id']}",
                    f"  - task_type: {row['task_type']}",
                    f"  - question: {row['question']}",
                    f"  - source_docs: {' | '.join(row['source_docs'])}",
                    f"  - answer: {row['target_answer'][:300]}",
                ]
            )
        examples_lines.extend(["", "## Rejected Examples"])
        for row in rejected_rows[:3]:
            examples_lines.extend(
                [
                    f"- id: {row['id']}",
                    f"  - task_type: {row['task_type']}",
                    f"  - question: {row['question']}",
                    f"  - rejection_reason: {row.get('rejection_reason', '')}",
                ]
            )
        _write_markdown(self.output_root / "qwen_sft_examples.md", examples_lines)

        _write_jsonl(self.output_root / "qwen_sft_build_log.jsonl", self.step_logs)
        return stats

    def run(self) -> dict[str, Any]:
        approved_rows = self.finalize_approved_questions()
        approved_rows, _ = self.build_overlap_blocklist(approved_rows)
        retrieval_rows = self.build_retrieval_candidates(approved_rows)
        judged_rows = self.generate_teacher_and_judge(retrieval_rows)
        raw_rows = self.build_raw_dataset(judged_rows)
        train_rows, val_rows, _ = self.split_raw_dataset(raw_rows)
        train_payload, val_payload = self.format_qwen_dataset(train_rows, val_rows)
        stats = self.build_stats_reports(
            approved_rows=approved_rows,
            retrieval_rows=retrieval_rows,
            judged_rows=judged_rows,
            raw_rows=raw_rows,
            train_payload=train_payload,
            val_payload=val_payload,
        )
        return stats

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scenario_a.common_pipeline import PipelinePaths, PipelineSettings, ScenarioACommonPipeline
from scenario_a.model_adapters import create_adapter


STOPWORDS = {
    "사업",
    "구축",
    "고도화",
    "용역",
    "시스템",
    "차세대",
    "통합",
    "정보",
    "운영",
    "기능",
    "개선",
    "플랫폼",
}

TASK_TYPE_TARGETS = {
    "single_doc_factual": 2400,
    "comparison": 1200,
    "rejection": 1200,
    "table_plus_body": 600,
    "follow_up": 600,
}

SINGLE_DOC_SLOT_PRIORITY = [
    ("budget", "period"),
    ("contract", "period"),
    ("budget", "purpose"),
    ("purpose", "scope"),
    ("security", "scope"),
    ("integration", "scope"),
    ("evaluation", "eligibility"),
    ("schedule", "deliverables"),
    ("ai_requirement", "scope"),
    ("operations", "maintenance"),
]

COMPARISON_SLOT_OPTIONS = [
    ("budget", "period"),
    ("purpose", "scope"),
    ("contract", "eligibility"),
    ("integration", "security"),
    ("operations", "deliverables"),
]


def _normalize_text(text: str) -> str:
    value = str(text or "").strip().lower()
    return re.sub(r"\s+", " ", value)


def _normalize_for_match(text: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", _normalize_text(text))


def _tokenize_koreanish(text: str) -> list[str]:
    return [token for token in re.findall(r"[0-9A-Za-z가-힣]+", str(text or "").lower()) if len(token) >= 2]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
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


def _load_yaml(path: Path) -> Any:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


@dataclass(slots=True)
class DocumentProfile:
    source_file_name: str
    document_id: str
    source_extension: str
    사업명: str
    발주기관: str
    사업금액: str
    공개일자: str
    입찰시작일: str
    입찰마감일: str
    사업요약: str
    text: str
    rag_ready_text: str
    table_blocks: list[dict[str, Any]]
    ocr_queue: list[dict[str, Any]]
    metadata: dict[str, Any]
    theme_tags: list[str]
    supported_slots: set[str]
    table_like_slots: set[str]
    body_like_slots: set[str]


class QwenFTInstructionAPipeline:
    def __init__(
        self,
        *,
        project_root: Path,
        output_dir: Path | None = None,
        seed: int = 20260409,
    ) -> None:
        self.project_root = project_root.resolve()
        self.output_dir = output_dir or (self.project_root / "rag_outputs" / "qwen_ft_instruction_a")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random = random.Random(seed)
        self.pipeline = ScenarioACommonPipeline(
            PipelinePaths(project_root=self.project_root),
            PipelineSettings(embedding_backend_key="openai_text_embedding_3_small", candidate_k=10, top_k=5),
        )
        self.documents = self._load_document_profiles()
        self.doc_map = {doc.source_file_name: doc for doc in self.documents}
        self.official_questions = self._load_official_eval_bank()
        self.type4_lexicon = _load_yaml(self.project_root / "config" / "type4_slot_lexicon.yaml")

    def _load_document_profiles(self) -> list[DocumentProfile]:
        processed_rows = _read_jsonl(self.project_root / "processed_data" / "processed_documents.jsonl")
        metadata_df = pd.read_csv(self.project_root / "data_list.csv")
        metadata_by_file = {str(row["파일명"]): row.to_dict() for _, row in metadata_df.iterrows()}
        profiles: list[DocumentProfile] = []
        for row in processed_rows:
            source_file_name = str(row.get("source_file_name", "")).strip()
            meta = metadata_by_file.get(source_file_name, {})
            artifacts = row.get("artifacts", {}) or {}
            text = str(row.get("text", "") or "")
            rag_ready_text = str(row.get("rag_ready_text", "") or text)
            theme_tags = self._extract_theme_tags(" ".join([str(meta.get("사업명", "")), str(meta.get("사업 요약", "")), source_file_name, rag_ready_text[:2000]]))
            supported_slots = self._detect_supported_slots(meta, rag_ready_text)
            table_like_slots = self._detect_table_like_slots(meta, rag_ready_text, artifacts)
            body_like_slots = {slot for slot in ["purpose", "scope", "security", "operations", "background", "integration"] if slot in supported_slots}
            profiles.append(
                DocumentProfile(
                    source_file_name=source_file_name,
                    document_id=str(row.get("document_id", "")),
                    source_extension=str(row.get("source_extension", "")),
                    사업명=str(meta.get("사업명", "")),
                    발주기관=str(meta.get("발주 기관", "")),
                    사업금액=str(meta.get("사업 금액", "")),
                    공개일자=str(meta.get("공개 일자", "")),
                    입찰시작일=str(meta.get("입찰 참여 시작일", "")),
                    입찰마감일=str(meta.get("입찰 참여 마감일", "")),
                    사업요약=str(meta.get("사업 요약", "")),
                    text=text,
                    rag_ready_text=rag_ready_text,
                    table_blocks=list(artifacts.get("table_blocks", []) or []),
                    ocr_queue=list(artifacts.get("ocr_queue", []) or []),
                    metadata={**(row.get("metadata", {}) or {}), **meta},
                    theme_tags=theme_tags,
                    supported_slots=supported_slots,
                    table_like_slots=table_like_slots,
                    body_like_slots=body_like_slots,
                )
            )
        return profiles

    def _extract_theme_tags(self, text: str) -> list[str]:
        tokens = []
        for token in _tokenize_koreanish(text):
            if token in STOPWORDS:
                continue
            tokens.append(token)
        counter = Counter(tokens)
        preferred = ["erp", "학사", "포털", "그룹웨어", "지도", "재난", "응급", "채팅", "상담", "이러닝", "선량", "극저온", "원자력", "보안", "플랫폼", "의료", "ai", "챗봇"]
        tags = [token for token in preferred if token in counter]
        tags.extend([token for token, _ in counter.most_common(12) if token not in tags])
        return tags[:12]

    def _detect_supported_slots(self, meta: dict[str, Any], text: str) -> set[str]:
        value = " ".join([str(meta.get("사업명", "")), str(meta.get("사업 요약", "")), str(meta.get("사업 금액", "")), str(meta.get("입찰 참여 마감일", "")), str(meta.get("공개 일자", "")), text[:5000]])
        rules = {
            "budget": r"(예산|금액|사업비|기초금액|추정금액|원\b)",
            "period": r"(사업기간|수행기간|계약체결일로부터|개월|일 이내|기간)",
            "contract": r"(계약방식|입찰방식|협상에 의한 계약|일반경쟁|제한경쟁|입찰)",
            "purpose": r"(목적|배경|필요성|추진 배경|추진 목적|사업 개요)",
            "scope": r"(업무 범위|주요 업무|요구사항|구축 범위|사업 범위|과업 범위)",
            "security": r"(보안|접근통제|암호화|개인정보|권한|계정 관리)",
            "integration": r"(연계|인터페이스|api|외부 시스템|타 시스템)",
            "evaluation": r"(평가기준|정량평가|정성평가|배점|평가 항목)",
            "eligibility": r"(참가자격|입찰참가자격|자격 요건|실적 요건)",
            "deliverables": r"(산출물|납품|검수|제출물)",
            "schedule": r"(일정|마감|공개일|개찰|제안서 제출|접수 기간)",
            "operations": r"(운영|유지보수|상주|장애 대응|모니터링)",
            "maintenance": r"(유지보수|하자보수|유지 관리)",
            "ai_requirement": r"(ai|인공지능|자연어|챗봇|예측|머신러닝)",
            "modules": r"(모듈|기능 목록|구성도|세부 기능|구축 대상 기능)",
            "background": r"(추진 배경|현황|문제점|개선 필요)",
        }
        supported = {slot for slot, pattern in rules.items() if re.search(pattern, value, re.IGNORECASE)}
        if str(meta.get("사업 금액", "")).strip():
            supported.add("budget")
        if str(meta.get("입찰 참여 마감일", "")).strip():
            supported.add("schedule")
        return supported

    def _detect_table_like_slots(self, meta: dict[str, Any], text: str, artifacts: dict[str, Any]) -> set[str]:
        tableish = set()
        combined = " ".join([str(meta.get("사업명", "")), str(meta.get("사업 요약", "")), text[:4000], " ".join(str(block.get("text", "")) for block in artifacts.get("table_blocks", [])[:6])])
        if int(meta.get("table_markers", 0) or 0) > 0 or artifacts.get("table_blocks"):
            tableish.update({"budget", "schedule"})
        if re.search(r"(모듈|기능 목록|도입 예정|구성표)", combined):
            tableish.add("modules")
        if re.search(r"(평가기준|배점|지표|가중치)", combined):
            tableish.add("evaluation")
        if re.search(r"(연계 현황|대상 지역|장비 목록|소프트웨어 내역)", combined):
            tableish.add("integration")
        return tableish

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
                line = raw_line.rstrip()
                head = line.strip()
                if re.fullmatch(r"(Q\d{2}|T[ABC]-\d{2})", head):
                    if current:
                        questions.append(current)
                    current = {"qid": head}
                    continue
                if current is None:
                    continue
                match = re.match(r"\s*([A-Za-z_]+)\s*:\s*(.+)$", line)
                if match:
                    current[match.group(1).strip()] = match.group(2).strip()
            if current:
                questions.append(current)
        return questions

    def _candidate_base(
        self,
        *,
        qid_index: int,
        source_docs: list[str],
        task_type: str,
        question: str,
        expected_answerability: str,
        target_slots: list[str],
        generation_source: str,
        conversation_group_id: str | None = None,
        depends_on_qid: str | None = None,
    ) -> dict[str, Any]:
        return {
            "qid": f"cand_{qid_index:06d}",
            "source_docs": source_docs,
            "task_type": task_type,
            "question": question.strip(),
            "expected_answerability": expected_answerability,
            "target_slots": target_slots,
            "generation_source": generation_source,
            "official_eval_overlap": False,
            "duplicate_group_id": None,
            "overlap_flag": False,
            "ambiguity_flag": False,
            "auto_status": "pending",
            "type4_status": None,
            "human_review_status": "pending",
            "final_verdict": None,
            "conversation_group_id": conversation_group_id,
            "depends_on_qid": depends_on_qid,
        }

    def _slot_label(self, slot: str) -> str:
        labels = {
            "budget": "예산",
            "period": "사업 기간",
            "contract": "계약 방식",
            "purpose": "사업 목적",
            "scope": "주요 범위",
            "security": "보안 요구사항",
            "integration": "연계 대상",
            "evaluation": "평가 기준",
            "eligibility": "참가 자격",
            "deliverables": "산출물과 납품 조건",
            "schedule": "일정",
            "operations": "운영 요구사항",
            "maintenance": "유지보수 요구사항",
            "ai_requirement": "AI 관련 요구사항",
            "modules": "기능 또는 모듈 구성",
            "background": "추진 배경",
        }
        return labels.get(slot, slot)

    def _short_doc_subject(self, doc: DocumentProfile) -> str:
        return re.sub(r"\s+", " ", doc.사업명 or doc.source_file_name).strip()

    def _pick_slot_alias(self, slot_name: str) -> str:
        aliases = self.type4_lexicon.get("allowed_slots", {}).get(slot_name, {}).get("aliases", [])
        return aliases[0] if aliases else slot_name

    def _build_follow_up_question(self, base: dict[str, Any]) -> str:
        slot_labels = [self._slot_label(slot) for slot in base["target_slots"]]
        if base["task_type"] == "comparison":
            return f"그렇다면 두 사업 중 {slot_labels[0]} 측면에서 더 복잡해 보이는 쪽은 어디야?"
        return f"그 사업 기준으로 {slot_labels[0]} 말고 추가로 확인해야 할 조건은 뭐가 있어?"

    def _build_comparison_pairs(self, limit_pairs: int) -> list[tuple[DocumentProfile, DocumentProfile]]:
        scored: list[tuple[int, tuple[str, ...], DocumentProfile, DocumentProfile]] = []
        for doc_a, doc_b in combinations(self.documents, 2):
            overlap = set(doc_a.theme_tags) & set(doc_b.theme_tags)
            score = len(overlap)
            if any(token in (doc_a.사업명 + doc_b.사업명).lower() for token in ["erp", "학사", "포털", "그룹웨어", "플랫폼"]):
                score += 1
            if score <= 0:
                continue
            scored.append((score, tuple(sorted(overlap))[:4], doc_a, doc_b))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [(doc_a, doc_b) for _, _, doc_a, doc_b in scored[:limit_pairs]]

    def _trim_to_targets(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            buckets[row["task_type"]].append(row)
        trimmed: list[dict[str, Any]] = []
        for task_type, target in TASK_TYPE_TARGETS.items():
            task_rows = buckets.get(task_type, [])
            self.random.shuffle(task_rows)
            trimmed.extend(task_rows[:target])
        trimmed.sort(key=lambda item: item["qid"])
        return trimmed

    def generate_candidates(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        rows: list[dict[str, Any]] = []
        log_rows: list[dict[str, Any]] = []
        qid_index = 1

        for doc in self.documents:
            generated_for_doc = 0
            for slot_a, slot_b in SINGLE_DOC_SLOT_PRIORITY:
                if slot_a not in doc.supported_slots and slot_b not in doc.supported_slots:
                    continue
                templates = [
                    f"{doc.사업명}에서 {self._slot_label(slot_a)}과 {self._slot_label(slot_b)}을 함께 정리해줘.",
                    f"{doc.발주기관}가 발주한 {self._short_doc_subject(doc)}의 {self._slot_label(slot_a)}과 {self._slot_label(slot_b)}를 요약해줘.",
                    f"{self._short_doc_subject(doc)}에서 {self._slot_label(slot_a)}이 어떻게 제시되는지, 그리고 {self._slot_label(slot_b)}은 어떤지 알려줘.",
                ]
                for question in templates:
                    rows.append(self._candidate_base(qid_index=qid_index, source_docs=[doc.source_file_name], task_type="single_doc_factual", question=question, expected_answerability="answerable", target_slots=[slot_a, slot_b], generation_source="synthetic_generator_v1::single_doc"))
                    log_rows.append({"qid": f"cand_{qid_index:06d}", "doc": doc.source_file_name, "template_family": "single_doc"})
                    qid_index += 1
                    generated_for_doc += 1
                    if generated_for_doc >= 24:
                        break
                if generated_for_doc >= 24:
                    break

        for doc_a, doc_b in self._build_comparison_pairs(limit_pairs=800):
            template_count = 0
            for slot_a, slot_b in COMPARISON_SLOT_OPTIONS:
                variants = [
                    f"{doc_a.사업명}과 {doc_b.사업명}을 {self._slot_label(slot_a)}과 {self._slot_label(slot_b)} 중심으로 비교해줘.",
                    f"{doc_a.발주기관} 사업과 {doc_b.발주기관} 사업의 {self._slot_label(slot_a)} 차이와 {self._slot_label(slot_b)} 차이를 정리해줘.",
                ]
                for question in variants:
                    rows.append(self._candidate_base(qid_index=qid_index, source_docs=[doc_a.source_file_name, doc_b.source_file_name], task_type="comparison", question=question, expected_answerability="answerable", target_slots=[slot_a, slot_b], generation_source="synthetic_generator_v1::comparison"))
                    log_rows.append({"qid": f"cand_{qid_index:06d}", "doc": f"{doc_a.source_file_name} | {doc_b.source_file_name}", "template_family": "comparison"})
                    qid_index += 1
                    template_count += 1
                    if template_count >= 4:
                        break
                if template_count >= 4:
                    break

        allowed_slots = list(self.type4_lexicon.get("allowed_slots", {}).keys())
        for doc in self.documents:
            for slot_name in allowed_slots[:12]:
                rows.append(self._candidate_base(qid_index=qid_index, source_docs=[doc.source_file_name], task_type="rejection", question=f"{doc.사업명}의 {self._pick_slot_alias(slot_name)}는 무엇이야?", expected_answerability="no_answer", target_slots=[slot_name], generation_source="synthetic_generator_v1::rejection"))
                log_rows.append({"qid": f"cand_{qid_index:06d}", "doc": doc.source_file_name, "template_family": "rejection"})
                qid_index += 1

        for doc in self.documents:
            if not doc.table_like_slots:
                continue
            generated = 0
            for table_slot in sorted(doc.table_like_slots):
                for body_slot in sorted(doc.body_like_slots or {"purpose", "scope"}):
                    question = f"{doc.사업명}에서 {self._slot_label(table_slot)}과 {self._slot_label(body_slot)}를 함께 정리해줘. 표 정보와 본문 설명을 모두 반영해줘."
                    rows.append(self._candidate_base(qid_index=qid_index, source_docs=[doc.source_file_name], task_type="table_plus_body", question=question, expected_answerability="answerable", target_slots=[table_slot, body_slot], generation_source="synthetic_generator_v1::table_plus_body"))
                    log_rows.append({"qid": f"cand_{qid_index:06d}", "doc": doc.source_file_name, "template_family": "table_plus_body"})
                    qid_index += 1
                    generated += 1
                    if generated >= 8:
                        break
                if generated >= 8:
                    break

        factual_rows = [row for row in rows if row["task_type"] == "single_doc_factual"][:400]
        comparison_rows = [row for row in rows if row["task_type"] == "comparison"][:200]
        conversation_index = 1
        for base in factual_rows + comparison_rows:
            convo_id = f"conv_{conversation_index:05d}"
            rows.append(self._candidate_base(qid_index=qid_index, source_docs=list(base["source_docs"]), task_type="follow_up", question=self._build_follow_up_question(base), expected_answerability="answerable", target_slots=list(base["target_slots"]), generation_source="synthetic_generator_v1::follow_up", conversation_group_id=convo_id, depends_on_qid=base["qid"]))
            log_rows.append({"qid": f"cand_{qid_index:06d}", "doc": " | ".join(base["source_docs"]), "template_family": "follow_up"})
            qid_index += 1
            conversation_index += 1

        rows = self._trim_to_targets(rows)
        _write_jsonl(self.output_dir / "question_candidates.jsonl", rows)
        _write_jsonl(self.output_dir / "question_generation_log.jsonl", log_rows)
        return rows, log_rows

    def _embed_texts(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors.extend(self.pipeline.embedding_backend.embed_queries(batch))
        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def auto_vet(self, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        official_questions = [item.get("question", "") for item in self.official_questions if item.get("question")]
        official_doc_map = [
            {
                "question": item.get("question", ""),
                "source_docs": [
                    part.strip()
                    for key, value in item.items()
                    if key in {"ground_truth_doc", "ground_truth_docs"} and value
                    for part in re.split(r"\s*\+\s*|\s*,\s*", value)
                    if part.strip()
                ],
            }
            for item in self.official_questions
            if item.get("question")
        ]
        official_norm_set = {_normalize_for_match(text) for text in official_questions}
        official_embed = self._embed_texts(official_questions)
        candidate_embed = self._embed_texts([row["question"] for row in rows])

        overlap_logs: list[dict[str, Any]] = []
        duplicate_logs: list[dict[str, Any]] = []
        signature_seen: dict[str, str] = {}
        duplicate_group_counter = 1

        for idx, row in enumerate(rows):
            question = row["question"]
            norm = _normalize_for_match(question)
            source_docs = set(row.get("source_docs", []))
            malformed_reasons: list[str] = []
            if len(question) < 12 or len(question) > 180:
                malformed_reasons.append("question_length")
            if not source_docs:
                malformed_reasons.append("missing_source_docs")
            if row["task_type"] == "comparison" and len(source_docs) < 2:
                malformed_reasons.append("comparison_missing_dual_docs")
            if row["task_type"] == "rejection" and row["expected_answerability"] != "no_answer":
                malformed_reasons.append("rejection_answerability_mismatch")
            if row["task_type"] == "follow_up" and not row.get("depends_on_qid"):
                malformed_reasons.append("follow_up_missing_dependency")
            if malformed_reasons:
                row["auto_status"] = "auto_rejected_malformed"
                row["auto_reject_reasons"] = malformed_reasons
                continue

            overlap_flag = False
            overlap_reason = ""
            if norm in official_norm_set:
                overlap_flag = True
                overlap_reason = "exact_or_normalized_match"
            elif len(official_questions) > 0:
                sims = candidate_embed[idx] @ official_embed.T
                best_idx = int(np.argmax(sims))
                best_score = float(sims[best_idx])
                official_doc_overlap = source_docs & set(official_doc_map[best_idx]["source_docs"])
                if best_score >= 0.88 or (best_score >= 0.84 and official_doc_overlap):
                    overlap_flag = True
                    overlap_reason = f"semantic_match:{best_score:.4f}"
                    overlap_logs.append({"qid": row["qid"], "question": question, "matched_official_question": official_questions[best_idx], "similarity": round(best_score, 4), "overlap_reason": overlap_reason})
            row["official_eval_overlap"] = overlap_flag
            row["overlap_flag"] = overlap_flag
            if overlap_flag:
                row["auto_status"] = "auto_rejected_overlap"
                continue

            signature = f"{row['task_type']}::{ '|'.join(sorted(source_docs)) }::{ '|'.join(sorted(row['target_slots'])) }"
            if signature in signature_seen:
                prior_qid = signature_seen[signature]
                prior_row = next(item for item in rows if item["qid"] == prior_qid)
                ratio = SequenceMatcher(None, _normalize_text(prior_row["question"]), _normalize_text(question)).ratio()
                if ratio >= 0.86:
                    row["auto_status"] = "auto_rejected_duplicate"
                    row["duplicate_group_id"] = f"dup_{duplicate_group_counter:05d}"
                    duplicate_logs.append({"qid": row["qid"], "question": question, "matched_qid": prior_qid, "matched_question": prior_row["question"], "duplicate_group_id": row["duplicate_group_id"], "similarity": round(ratio, 4)})
                    duplicate_group_counter += 1
                    continue
            else:
                signature_seen[signature] = row["qid"]

            row["auto_status"] = "auto_pass_basic"

        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row["auto_status"] == "auto_pass_basic":
                grouped[(row["task_type"], "|".join(sorted(row["source_docs"])), "|".join(sorted(row["target_slots"])))].append(row)
        for _, group_rows in grouped.items():
            if len(group_rows) < 2:
                continue
            matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform([row["question"] for row in group_rows])
            sims = cosine_similarity(matrix)
            seen: set[str] = set()
            for i in range(len(group_rows)):
                for j in range(i + 1, len(group_rows)):
                    if sims[i, j] < 0.92:
                        continue
                    row = group_rows[j]
                    if row["qid"] in seen:
                        continue
                    row["auto_status"] = "auto_rejected_duplicate"
                    row["duplicate_group_id"] = row["duplicate_group_id"] or f"dup_{duplicate_group_counter:05d}"
                    duplicate_logs.append({"qid": row["qid"], "question": row["question"], "matched_qid": group_rows[i]["qid"], "matched_question": group_rows[i]["question"], "duplicate_group_id": row["duplicate_group_id"], "similarity": round(float(sims[i, j]), 4)})
                    duplicate_group_counter += 1
                    seen.add(row["qid"])

        _write_jsonl(self.output_dir / "question_auto_vetted.jsonl", rows)
        _write_jsonl(self.output_dir / "overlap_check_log.jsonl", overlap_logs)
        _write_jsonl(self.output_dir / "duplicate_check_log.jsonl", duplicate_logs)
        return rows, overlap_logs, duplicate_logs

    def _score_support_text(self, text: str, slot_terms: list[str], slot_patterns: list[str]) -> tuple[int, int, list[str]]:
        normalized = _normalize_text(text)
        matches = [term for term in slot_terms if term and term.lower() in normalized]
        direct = 1 if any(re.search(pattern, text, re.IGNORECASE) for pattern in slot_patterns) else 0
        indirect = 1 if (matches and not direct) else 0
        return direct, indirect, matches[:8]

    def _dense_support_check(self, query_embedding: list[float], source_docs: set[str], slot_terms: list[str], slot_patterns: list[str]) -> dict[str, Any]:
        result = self.pipeline.chroma_collection.query(query_embeddings=[query_embedding], n_results=10, include=["documents", "metadatas"])
        direct_hits = 0
        indirect_hits = 0
        matches: list[str] = []
        support_candidates_topk: list[dict[str, Any]] = []
        for document, metadata in zip(result["documents"][0], result["metadatas"][0], strict=False):
            if str(metadata.get("source_file_name", "")) not in source_docs:
                continue
            direct, indirect, local_matches = self._score_support_text(str(document or ""), slot_terms, slot_patterns)
            direct_hits += direct
            indirect_hits += indirect
            matches.extend(local_matches)
            support_candidates_topk.append({"chunk_id": metadata.get("chunk_id", ""), "source_file_name": metadata.get("source_file_name", ""), "direct": direct, "indirect": indirect})
        return {"direct_hits": direct_hits, "indirect_hits": indirect_hits, "slot_keyword_matches": matches, "support_candidates_topk": support_candidates_topk}

    def _bm25_support_check(self, question: str, source_docs: set[str], slot_terms: list[str], slot_patterns: list[str]) -> dict[str, Any]:
        model = self.pipeline.bm25_index["model"]
        chunk_rows = self.pipeline.bm25_index["chunk_rows"]
        scores = model.get_scores(self.pipeline._bm25_tokenize(question))
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:10]
        direct_hits = 0
        indirect_hits = 0
        matches: list[str] = []
        for row_index, _ in ranked:
            row = chunk_rows[row_index]
            if str(row.get("source_file_name", "")) not in source_docs:
                continue
            direct, indirect, local_matches = self._score_support_text(str(row.get("contextual_chunk_text", "")), slot_terms, slot_patterns)
            direct_hits += direct
            indirect_hits += indirect
            matches.extend(local_matches)
        return {"direct_hits": direct_hits, "indirect_hits": indirect_hits, "slot_keyword_matches": matches}

    def _metadata_support_check(self, source_docs: set[str], slot_terms: list[str], slot_patterns: list[str]) -> dict[str, Any]:
        direct_hits = 0
        indirect_hits = 0
        matches: list[str] = []
        for source_file_name in source_docs:
            doc = self.doc_map.get(source_file_name)
            if doc is None:
                continue
            direct, indirect, local_matches = self._score_support_text(json.dumps(doc.metadata, ensure_ascii=False), slot_terms, slot_patterns)
            direct_hits += direct
            indirect_hits += indirect
            matches.extend(local_matches)
        return {"direct_hits": direct_hits, "indirect_hits": indirect_hits, "slot_keyword_matches": matches}

    def _table_body_support_check(self, source_docs: set[str], slot_terms: list[str], slot_patterns: list[str]) -> dict[str, Any]:
        direct_hits = 0
        indirect_hits = 0
        matches: list[str] = []
        for source_file_name in source_docs:
            doc = self.doc_map.get(source_file_name)
            if doc is None:
                continue
            texts = [
                " ".join(str(block.get("text", "")) for block in doc.table_blocks[:10]),
                doc.rag_ready_text[:2000],
            ]
            for text in texts:
                if not text:
                    continue
                direct, indirect, local_matches = self._score_support_text(text, slot_terms, slot_patterns)
                direct_hits += direct
                indirect_hits += indirect
                matches.extend(local_matches)
        return {"direct_hits": direct_hits, "indirect_hits": indirect_hits, "slot_keyword_matches": matches}

    def run_type4_detection(self, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        allowed_slots = self.type4_lexicon.get("allowed_slots", {})
        type4_candidates = [row for row in rows if row["task_type"] == "rejection" and row["auto_status"] == "auto_pass_basic"]
        slot_mapping_rows: list[dict[str, Any]] = []
        detection_rows: list[dict[str, Any]] = []
        embeddings = self._embed_texts([row["question"] for row in type4_candidates], batch_size=128)
        for idx, row in enumerate(type4_candidates):
            slot_terms: list[str] = []
            slot_patterns: list[str] = []
            for slot in row["target_slots"]:
                slot_terms.append(slot)
                slot_terms.extend(allowed_slots.get(slot, {}).get("aliases", []))
                slot_patterns.extend(allowed_slots.get(slot, {}).get("support_patterns", []))
            slot_mapping_rows.append({"qid": row["qid"], "question": row["question"], "target_slots": row["target_slots"], "normalized_slot_terms": slot_terms})
            source_docs = set(row["source_docs"])
            dense_hits = self._dense_support_check(embeddings[idx].tolist(), source_docs, slot_terms, slot_patterns)
            bm25_hits = self._bm25_support_check(row["question"], source_docs, slot_terms, slot_patterns)
            norm_bm25_hits = self._bm25_support_check(_normalize_text(row["question"]), source_docs, slot_terms, slot_patterns)
            metadata_hits = self._metadata_support_check(source_docs, slot_terms, slot_patterns)
            table_body_hits = self._table_body_support_check(source_docs, slot_terms, slot_patterns)
            direct_total = dense_hits["direct_hits"] + bm25_hits["direct_hits"] + norm_bm25_hits["direct_hits"] + metadata_hits["direct_hits"] + table_body_hits["direct_hits"]
            indirect_total = dense_hits["indirect_hits"] + bm25_hits["indirect_hits"] + norm_bm25_hits["indirect_hits"] + metadata_hits["indirect_hits"] + table_body_hits["indirect_hits"]
            ambiguity_flag = bool(indirect_total >= 3 and direct_total == 0)
            if direct_total >= 1:
                support_judge = "supported"
                confidence = min(0.98, 0.65 + direct_total * 0.08)
                reason = "retrieval_or_pattern_found_direct_support"
                type4_status = "type4_rejected_supported"
            elif ambiguity_flag or indirect_total >= 3:
                support_judge = "ambiguous"
                confidence = 0.55
                reason = "indirect_or_partial_signals_present"
                type4_status = "type4_ambiguous"
            else:
                support_judge = "not_supported"
                confidence = 0.85 if indirect_total == 0 else 0.72
                reason = "no_direct_support_detected_across_retrieval_paths"
                type4_status = "type4_accepted"
                row["auto_status"] = "auto_pass_type4"
            row["type4_status"] = type4_status
            row["ambiguity_flag"] = ambiguity_flag
            row["auto_checks"] = {
                "dense_hit_count": dense_hits["direct_hits"],
                "bm25_hit_count": bm25_hits["direct_hits"],
                "normalized_bm25_hit_count": norm_bm25_hits["direct_hits"],
                "metadata_hit_count": metadata_hits["direct_hits"],
                "table_body_hit_count": table_body_hits["direct_hits"],
                "slot_keyword_matches": sorted(set(dense_hits["slot_keyword_matches"] + bm25_hits["slot_keyword_matches"] + metadata_hits["slot_keyword_matches"] + table_body_hits["slot_keyword_matches"])),
                "support_candidates_topk": dense_hits["support_candidates_topk"][:5],
                "support_judge": support_judge,
                "support_confidence": round(confidence, 4),
                "reason": reason,
                "ambiguity_flag": ambiguity_flag,
            }
            detection_rows.append({"qid": row["qid"], "question": row["question"], "task_type": row["task_type"], "expected_answerability": row["expected_answerability"], "target_slots": row["target_slots"], "auto_checks": row["auto_checks"], "auto_status": row["auto_status"], "type4_status": row["type4_status"]})
        _write_json(self.output_dir / "type4_slot_lexicon.yaml", self.type4_lexicon)
        _write_jsonl(self.output_dir / "question_slot_mapping.jsonl", slot_mapping_rows)
        _write_jsonl(self.output_dir / "type4_detection_results.jsonl", detection_rows)
        report = {"candidate_count": len(type4_candidates), "accepted_count": sum(1 for row in detection_rows if row["type4_status"] == "type4_accepted"), "ambiguous_count": sum(1 for row in detection_rows if row["type4_status"] == "type4_ambiguous"), "rejected_supported_count": sum(1 for row in detection_rows if row["type4_status"] == "type4_rejected_supported")}
        _write_json(self.output_dir / "type4_detection_report.json", report)
        return rows, detection_rows, report

    def apply_status_transitions(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for row in rows:
            if row["auto_status"] in {"auto_pass_basic", "auto_pass_type4"}:
                row["human_review_status"] = "needs_human_review"
                row["final_verdict"] = None
            else:
                row["final_verdict"] = None
        return rows

    def _load_existing_human_decisions(self) -> dict[str, str]:
        csv_path = self.output_dir / "question_human_review_sheet.csv"
        if not csv_path.exists():
            return {}
        df = pd.read_csv(csv_path, dtype=str).fillna("")
        return {
            str(row["qid"]).strip(): str(row["final_human_decision"]).strip()
            for _, row in df.iterrows()
            if str(row.get("final_human_decision", "")).strip()
        }

    def _recommended_action(self, row: dict[str, Any]) -> str:
        if row.get("overlap_flag") or row.get("duplicate_group_id"):
            return "manual_check_required"
        if row.get("ambiguity_flag"):
            return "manual_check_required"
        if row["task_type"] == "comparison" and len(row.get("source_docs", [])) < 2:
            return "manual_check_required"
        if row["task_type"] == "table_plus_body":
            slots = set(row.get("target_slots", []))
            if not (slots & {"budget", "schedule", "modules", "integration", "evaluation"}) or not (slots & {"purpose", "scope", "security", "operations", "background"}):
                return "manual_check_required"
        if row["task_type"] == "rejection":
            support_confidence = float((row.get("auto_checks", {}) or {}).get("support_confidence", 0.0) or 0.0)
            return "review" if support_confidence >= 0.8 else "manual_check_required"
        return "review"

    def build_human_review_sheet(self, rows: list[dict[str, Any]]) -> pd.DataFrame:
        existing_decisions = self._load_existing_human_decisions()
        review_rows = []
        for row in rows:
            if row["auto_status"].startswith("auto_rejected"):
                continue
            auto_checks = row.get("auto_checks", {}) or {}
            review_rows.append(
                {
                    "qid": row["qid"],
                    "question": row["question"],
                    "task_type": row["task_type"],
                    "expected_answerability": row["expected_answerability"],
                    "source_docs": " | ".join(row["source_docs"]),
                    "target_slots": " | ".join(row["target_slots"]),
                    "auto_status": row["auto_status"],
                    "type4_status": row.get("type4_status"),
                    "overlap_flag": row.get("overlap_flag", False),
                    "ambiguity_flag": row.get("ambiguity_flag", False),
                    "support_judge": auto_checks.get("support_judge", ""),
                    "support_confidence": auto_checks.get("support_confidence", ""),
                    "recommended_action": self._recommended_action(row),
                    "final_human_decision": existing_decisions.get(row["qid"], ""),
                }
            )
        df = pd.DataFrame(review_rows)
        df.to_csv(self.output_dir / "question_human_review_sheet.csv", index=False, encoding="utf-8-sig")
        df.to_excel(self.output_dir / "question_human_review_sheet.xlsx", index=False)
        return df

    def finalize(self, rows: list[dict[str, Any]], review_sheet_df: pd.DataFrame | None = None) -> list[dict[str, Any]]:
        decision_map: dict[str, str] = {}
        if review_sheet_df is not None and not review_sheet_df.empty:
            decision_map = {
                str(row["qid"]).strip(): str(row["final_human_decision"]).strip().lower()
                for _, row in review_sheet_df.iterrows()
                if str(row.get("final_human_decision", "")).strip()
            }
        final_rows: list[dict[str, Any]] = []
        for row in rows:
            decision = decision_map.get(row["qid"], "")
            if decision in {"approve", "approved", "human_approved", "y", "yes"} and row["auto_status"] in {"auto_pass_basic", "auto_pass_type4"}:
                row["human_review_status"] = "human_approved"
                row["final_verdict"] = "approved_for_teacher"
                final_rows.append(row)
            elif decision in {"reject", "rejected", "human_rejected", "n", "no"}:
                row["human_review_status"] = "human_rejected"
                row["final_verdict"] = None
        _write_jsonl(self.output_dir / "question_final_vetted.jsonl", final_rows)
        return final_rows

    def _extract_json_list(self, text: str) -> list[dict[str, Any]]:
        cleaned = str(text).strip()
        cleaned = re.sub(r"^```json\s*", "", cleaned)
        cleaned = re.sub(r"^```\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if not match:
            return []
        try:
            payload = json.loads(match.group(0))
        except Exception:
            return []
        return payload if isinstance(payload, list) else []

    def _build_gemma_review_prompt(self, batch: list[dict[str, Any]]) -> str:
        payload = []
        for row in batch:
            payload.append(
                {
                    "qid": row["qid"],
                    "task_type": row["task_type"],
                    "question": row["question"],
                    "expected_answerability": row["expected_answerability"],
                    "source_docs": row["source_docs"],
                    "target_slots": row["target_slots"],
                    "type4_status": row.get("type4_status"),
                    "auto_checks": row.get("auto_checks"),
                }
            )
        return json.dumps(
            {
                "rules": [
                    "질문의 명확성, task_type 적합성, answerability 적합성, evidence expectation, ambiguity risk, type4 validity를 1~5로 평가한다.",
                    "반드시 JSON 배열만 반환한다.",
                ],
                "output_format": [
                    {
                        "qid": "cand_000001",
                        "question_clarity": 1,
                        "task_type_fit": 1,
                        "answerability_fit": 1,
                        "evidence_expectation": 1,
                        "ambiguity_risk": 1,
                        "type4_validity": 1,
                        "recommended_action": "accept|review|reject",
                        "reason": "간단한 근거",
                    }
                ],
                "items": payload,
            },
            ensure_ascii=False,
            indent=2,
        )

    def build_stats_and_examples(self, rows: list[dict[str, Any]], *, type4_detection_rows: list[dict[str, Any]], final_rows: list[dict[str, Any]]) -> dict[str, Any]:
        stats = {
            "total_candidate_count": len(rows),
            "overlap_removed_count": sum(1 for row in rows if row["auto_status"] == "auto_rejected_overlap"),
            "duplicate_removed_count": sum(1 for row in rows if row["auto_status"] == "auto_rejected_duplicate"),
            "type4_candidate_count": sum(1 for row in rows if row["task_type"] == "rejection"),
            "type4_accepted_count": sum(1 for row in type4_detection_rows if row["type4_status"] == "type4_accepted"),
            "needs_human_review_count": sum(1 for row in rows if row["human_review_status"] == "needs_human_review"),
            "final_human_approved_count": sum(1 for row in rows if row["human_review_status"] == "human_approved"),
            "approved_for_teacher_count": len(final_rows),
            "task_type_distribution": dict(Counter(row["task_type"] for row in rows)),
            "answerability_distribution": dict(Counter(row["expected_answerability"] for row in rows)),
            "ambiguity_warning_count": sum(1 for row in rows if row.get("ambiguity_flag")),
        }
        _write_json(self.output_dir / "question_vetting_stats.json", stats)
        accepted = [row for row in rows if row["auto_status"] in {"auto_pass_basic", "auto_pass_type4"}][:5]
        rejected = [row for row in rows if row["auto_status"].startswith("auto_rejected")][:5]
        type4_accept = [row for row in rows if row.get("type4_status") == "type4_accepted"][:3]
        type4_reject = [row for row in rows if row.get("type4_status") in {"type4_rejected_supported", "type4_ambiguous"}][:3]
        lines = ["# Question Vetting Examples", "", "## Accepted Questions"]
        lines.extend(self._format_examples(accepted))
        lines.extend(["", "## Rejected Questions"])
        lines.extend(self._format_examples(rejected))
        lines.extend(["", "## Type4 Accepted Examples"])
        lines.extend(self._format_examples(type4_accept))
        lines.extend(["", "## Type4 Rejected Examples"])
        lines.extend(self._format_examples(type4_reject))
        (self.output_dir / "question_vetting_examples.md").write_text("\n".join(lines), encoding="utf-8")
        return stats

    def _format_examples(self, rows: list[dict[str, Any]]) -> list[str]:
        lines: list[str] = []
        for row in rows:
            lines.extend([f"- qid: {row['qid']}", f"  - task_type: {row['task_type']}", f"  - question: {row['question']}", f"  - source_docs: {' | '.join(row['source_docs'])}", f"  - auto_status: {row['auto_status']}", f"  - type4_status: {row.get('type4_status')}", f"  - human_review_status: {row.get('human_review_status')}"])
        return lines

    def run(self) -> dict[str, Any]:
        candidates, _ = self.generate_candidates()
        rows, _, _ = self.auto_vet(candidates)
        rows, type4_detection_rows, _ = self.run_type4_detection(rows)
        rows = self.apply_status_transitions(rows)
        review_sheet_df = self.build_human_review_sheet(rows)
        final_rows = self.finalize(rows, review_sheet_df=review_sheet_df)
        return self.build_stats_and_examples(rows, type4_detection_rows=type4_detection_rows, final_rows=final_rows)

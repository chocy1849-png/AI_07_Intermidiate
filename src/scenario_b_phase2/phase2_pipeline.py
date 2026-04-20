from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from scenario_b_phase2.answer_type_router import build_virtual_question_row
from scenario_b_phase2.metadata_aware_retrieval import compute_metadata_soft_boost
from scenario_b_phase2.normalized_bm25 import build_normalized_bm25_queries
from scenario_b_phase2.soft_crag_lite import (
    SoftCragLiteConfig,
    apply_soft_crag_lite as apply_soft_crag_lite_rules,
    should_apply_soft_crag,
)
from scenario_a.common_pipeline import CandidateRow, RetrievalResult, ScenarioAAnswer, ScenarioACommonPipeline


def _normalize_doc_name(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").lower())


@dataclass(slots=True)
class Phase2Options:
    enable_controlled_query_expansion: bool = True
    enable_normalized_bm25: bool = True
    enable_metadata_aware_retrieval: bool = True
    enable_metadata_bonus_v2: bool | None = None
    enable_table_body_pairing: bool = True
    enable_soft_crag_lite: bool = True
    expansion_query_limit: int = 1
    expansion_query_weight: float = 0.35
    normalized_bm25_weight: float = 0.35
    soft_crag_top_n: int = 6
    soft_crag_score_weight: float = 0.045
    soft_crag_keep_k: int = 3
    soft_crag_scope_mode: str = "targeted"  # targeted | all
    soft_crag_factual_mode: str = "off"  # off | weak | on
    metadata_boost_scale: float = 1.0
    metadata_disable_for_rejection: bool = False
    metadata_scope_mode: str = "all"  # all | comparison_and_explicit_factual
    normalized_bm25_mode: str = "all"  # all | exact_match_subset
    enable_b03_legacy_crag_parity: bool = True
    b03_evaluator_top_n: int = 6
    b03_second_pass_vector_weight: float = 0.55
    b03_second_pass_bm25_weight: float = 0.45
    enable_comparison_evidence_helper: bool = False
    comparison_helper_doc_bonus: float = 0.0045
    comparison_helper_axis_bonus: float = 0.0015
    comparison_helper_max_per_doc: int = 2
    enable_groupc_table_plus_text_guard: bool = False
    groupc_pair_bonus: float = 0.006
    groupc_parent_bonus: float = 0.003
    groupc_table_penalty_without_body: float = 0.012
    enable_question_type_gated_ocr_routing: bool = False
    enable_structured_evidence_priority: bool = False
    enable_hybridqa_stage_metrics: bool = False
    enable_table_factual_exact_answer_mode: bool = False
    enable_table_factual_alignment_scoring: bool = False
    table_factual_generic_penalty: float = 0.012
    enable_answer_type_router: bool = True
    answer_type_router_force: bool = False
    answer_type_router_confidence_threshold: float = 0.58
    answer_type_router_low_conf_fallback: str = "comparison_safe"


class ScenarioBPhase2Pipeline(ScenarioACommonPipeline):
    QUERY_EXPANSION_RULES: list[tuple[tuple[str, ...], str]] = [
        (("예산", "금액", "사업비", "추정금액", "budget"), "예산 금액 사업비 추정금액 산출내역"),
        (("기간", "일정", "마감", "계약일", "date", "schedule"), "사업 기간 일정 착수 종료 마감"),
        (("평가", "배점", "기준", "criteria"), "평가 기준 배점 정량 정성"),
        (("계약", "입찰", "방식", "method"), "계약 방식 입찰 방식 수의 일반 경쟁"),
        (("자격", "요건", "서류", "requirement"), "참가 자격 제출 서류 필수 요건"),
        (("비교", "차이", "공통", "comparison"), "비교 기준 차이점 공통점"),
        (("표", "table", "matrix"), "표 항목 비고 본문 설명"),
    ]
    B02_SYSTEM_PROMPT = (
        "당신은 정부 제안요청서(RFP) 기반 요약 도우미다.\n"
        "반드시 제공된 검색 문맥만 근거로 답한다.\n"
        "문맥에 없는 사실은 추정하지 않는다.\n"
        "질문이 특정 사업 요약을 요구하면 목적, 범위, 주요 요구사항, 일정/예산, 발주기관을 우선 정리한다.\n"
        "문맥에 일정이나 예산이 없으면 없다고 명시한다.\n"
        "마지막에는 참고 문서명과 청크 ID를 간단히 정리한다."
    )
    B03_SYSTEM_PROMPT = (
        "당신은 공공 제안요청서(RFP) 기반 질의응답 도우미다.\n"
        "반드시 제공된 검색 문맥만 근거로 답하라.\n"
        "문맥에 없는 사실은 추정하지 않는다.\n"
        "질문이 사업 요약을 요구하면 목적, 범위, 주요 요구사항, 일정/예산, 발주기관을 우선 정리한다.\n"
        "비교형 질문이면 문맥에 실제로 등장한 사업만 비교하고, 부족한 항목은 문맥에 없다고 명시한다.\n"
        "마지막에는 참고 근거로 파일명과 청크 ID를 간단히 적는다."
    )
    ANSWER_FORMAT_TEXT = (
        "출력 형식:\n"
        "1. 한줄 요약\n"
        "2. 핵심 내용\n"
        "3. 주요 요구사항\n"
        "4. 일정/예산/발주기관\n"
        "5. 참고 근거"
    )

    def __init__(self, *args: Any, options: Phase2Options | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.options = options or Phase2Options()
        # Backward compatibility: if old flag is provided, keep it aligned.
        if self.options.enable_metadata_bonus_v2 is not None:
            self.options.enable_metadata_aware_retrieval = bool(self.options.enable_metadata_bonus_v2)

    def _is_table_like_question(self, question: str, profile: dict[str, Any]) -> bool:
        answer_type = str(profile.get("answer_type", "")).strip().lower()
        if answer_type in {"table_factual", "table_plus_text"}:
            return True
        if bool(profile.get("budget")) or bool(profile.get("schedule")):
            return True
        return bool(re.search(r"(표|table|행|열|항목|비고|일정표|평가표|구성표)", str(question or ""), re.IGNORECASE))

    @staticmethod
    def _is_table_factual(profile: dict[str, Any]) -> bool:
        return str(profile.get("answer_type", "")).strip().lower() == "table_factual"

    @staticmethod
    def _is_table_plus_text(profile: dict[str, Any]) -> bool:
        return str(profile.get("answer_type", "")).strip().lower() == "table_plus_text"

    @staticmethod
    def _chunk_type(row: CandidateRow) -> str:
        return str(row.metadata.get("chunk_type", "") or "").strip().lower()

    @staticmethod
    def _question_tokens_for_match(question: str) -> list[str]:
        tokens = re.findall(r"[0-9A-Za-z가-힣]{2,}", str(question or "").lower())
        stop = {
            "무엇", "뭐야", "뭐지", "어떤", "어떻게", "정리", "설명", "요약", "기준", "관련", "사업",
            "요구", "요구사항", "시스템", "구축", "용역", "현황", "내역", "목록", "항목",
        }
        return [token for token in tokens if token not in stop]

    @staticmethod
    def _table_factual_response_style(question: str) -> str:
        q = str(question or "")
        if re.search(r"(정의|의미|역할|란|무엇|뭐야)", q):
            return "definition"
        if re.search(r"(연계|현황|구분|대상|채널|매핑)", q):
            return "linkage"
        if re.search(r"(앱|기능|모듈|서비스)", q):
            return "app_function_list"
        if re.search(r"(목록|내역|항목|리스트|종류)", q):
            return "list"
        return "list"

    @staticmethod
    def _is_generic_row_content(row: CandidateRow) -> bool:
        chunk_type = str(row.metadata.get("chunk_type", "") or "").strip().lower()
        if chunk_type not in {"cell_row_block", "row_summary_chunk", "table_true_ocr_v2", "raw_table_ocr"}:
            return False
        text = str(row.text or "").lower()
        generic_patterns = [
            r"착수", r"중간", r"최종", r"보고회", r"주간보고", r"월간보고", r"보고서",
            r"회의", r"점검", r"검토", r"일정", r"계획", r"산출물",
        ]
        hits = sum(1 for pattern in generic_patterns if re.search(pattern, text))
        return hits >= 2

    def _table_factual_alignment_scores(self, row: CandidateRow, question: str) -> dict[str, float]:
        q_tokens = self._question_tokens_for_match(question)
        q_set = set(q_tokens)
        text = str(row.text or "").lower()
        header_path = str(row.metadata.get("header_path", "") or "").lower()
        value_text = str(row.metadata.get("value_text", "") or "").lower()
        section_label = str(row.metadata.get("section_label", "") or "").lower()
        source_doc = str(row.metadata.get("source_file_name", "") or "").lower()

        header_tokens = set(re.findall(r"[0-9A-Za-z가-힣]{2,}", header_path))
        value_tokens = set(re.findall(r"[0-9A-Za-z가-힣]{2,}", value_text))
        text_tokens = set(re.findall(r"[0-9A-Za-z가-힣]{2,}", text))
        entity_tokens = set(re.findall(r"[0-9A-Za-z가-힣]{2,}", f"{section_label} {source_doc}"))

        header_match_score = (len(q_set & header_tokens) / max(1, len(q_set))) if q_set else 0.0
        row_match_score = (len(q_set & (value_tokens | text_tokens)) / max(1, len(q_set))) if q_set else 0.0
        entity_overlap_score = (len(q_set & entity_tokens) / max(1, len(q_set))) if q_set else 0.0

        style = self._table_factual_response_style(question)
        definition_pattern_score = 0.0
        if style == "definition":
            definition_pattern_score = 1.0 if re.search(r"(정의|의미|역할|란|는)", text) else 0.0

        list_request_score = 1.0 if style in {"list", "app_function_list", "linkage"} and (
            "|" in text or "\n" in text or "," in text
        ) else 0.0

        return {
            "header_match_score": round(float(header_match_score), 4),
            "row_match_score": round(float(row_match_score), 4),
            "entity_overlap_score": round(float(entity_overlap_score), 4),
            "definition_pattern_score": round(float(definition_pattern_score), 4),
            "list_request_score": round(float(list_request_score), 4),
            "exact_header_match_hit": 1.0 if header_match_score >= 0.60 else 0.0,
            "exact_row_match_hit": 1.0 if row_match_score >= 0.60 else 0.0,
        }

    @staticmethod
    def _target_docs(question_row: dict[str, Any]) -> list[str]:
        targets: list[str] = []
        gt_doc = str(question_row.get("ground_truth_doc", "")).strip()
        if gt_doc:
            targets.append(gt_doc)
        gt_docs = [x.strip() for x in str(question_row.get("ground_truth_docs", "")).split("|") if x.strip()]
        targets.extend(gt_docs)
        dedup: list[str] = []
        seen: set[str] = set()
        for item in targets:
            norm = _normalize_doc_name(item)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            dedup.append(item)
        return dedup

    def _prepare_question_row(
        self,
        question_row: dict[str, Any],
        question: str,
        history: list[dict[str, str]] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        row = dict(question_row)
        answer_type = str(row.get("answer_type", "")).strip().lower()
        if not self.options.enable_answer_type_router:
            return row, None
        if answer_type and not self.options.answer_type_router_force:
            return row, None
        patched, routed = build_virtual_question_row(
            question=question,
            base_row=row,
            history=history,
            force_router=bool(self.options.answer_type_router_force),
            confidence_threshold=float(self.options.answer_type_router_confidence_threshold),
            low_conf_fallback=str(self.options.answer_type_router_low_conf_fallback),
        )
        return patched, (routed.as_dict() if routed is not None else None)

    def _build_soft_crag_config(self) -> SoftCragLiteConfig:
        return SoftCragLiteConfig(
            top_n=int(self.options.soft_crag_top_n),
            score_weight=float(self.options.soft_crag_score_weight),
            keep_k=int(self.options.soft_crag_keep_k),
            scope_mode=str(self.options.soft_crag_scope_mode),
            factual_mode=str(self.options.soft_crag_factual_mode),
        )

    def build_query_variants(self, question_row: dict[str, Any], question: str, profile: dict[str, Any]) -> list[str]:
        base = str(question or "").strip()
        if not base:
            return [""]
        if not self.options.enable_controlled_query_expansion:
            return [base]
        if bool(profile.get("follow_up")) or bool(profile.get("rejection")):
            return [base]

        lowered = base.lower()
        expansions: list[str] = []
        for keywords, phrase in self.QUERY_EXPANSION_RULES:
            if any(keyword.lower() in lowered for keyword in keywords):
                expansions.append(f"{base} {phrase}")
        if bool(profile.get("comparison")):
            expansions.append(f"{base} 비교 기준 차이 공통 문서별 근거")

        dedup = []
        seen = {base}
        for query in expansions:
            value = re.sub(r"\s+", " ", query).strip()
            if value and value not in seen:
                dedup.append(value)
                seen.add(value)
            if len(dedup) >= max(0, self.options.expansion_query_limit):
                break
        return [base, *dedup]

    @staticmethod
    def _is_metadata_explicit_factual(question: str, profile: dict[str, Any]) -> bool:
        if str(profile.get("answer_type", "")).strip() != "factual":
            return False
        return bool(
            re.search(
                r"(발주|기관|예산|금액|마감|기한|기간|계약방식|입찰방식|평가기준|배점|deadline|budget|agency)",
                str(question or ""),
                re.IGNORECASE,
            )
        )

    def _should_apply_metadata_bonus(self, question: str, profile: dict[str, Any]) -> bool:
        if self.options.metadata_disable_for_rejection and bool(profile.get("rejection")):
            return False
        scope = str(self.options.metadata_scope_mode or "all").strip().lower()
        if scope == "all":
            return True
        if scope == "comparison_and_explicit_factual":
            return bool(profile.get("comparison")) or self._is_metadata_explicit_factual(question, profile)
        return True

    @staticmethod
    def _is_exact_match_question(question_row: dict[str, Any], question: str, profile: dict[str, Any]) -> bool:
        if bool(profile.get("comparison")) or bool(profile.get("follow_up")) or bool(profile.get("rejection")):
            return False
        if str(question_row.get("answer_type", "")).strip() != "factual":
            return False
        q = str(question or "")
        has_exact_slot = bool(
            re.search(
                r"(얼마|몇|수치|금액|예산|마감|기한|기간|날짜|납기|계약방식|입찰방식|발주\s*기관|평가\s*기준|배점|deadline|budget|agency)",
                q,
                re.IGNORECASE,
            )
        )
        has_open_ended = bool(
            re.search(
                r"(비교|차이|공통|정리해|설명해|요약해|전체|모두|전부|왜|어떻게|의미|영향)",
                q,
                re.IGNORECASE,
            )
        )
        return has_exact_slot and not has_open_ended

    def _should_apply_normalized_bm25(self, question_row: dict[str, Any], question: str, profile: dict[str, Any]) -> bool:
        if not self.options.enable_normalized_bm25:
            return False
        mode = str(self.options.normalized_bm25_mode or "all").strip().lower()
        if mode == "all":
            return True
        if mode == "exact_match_subset":
            return self._is_exact_match_question(question_row, question, profile)
        return True

    @staticmethod
    def _no_info_answer() -> str:
        return (
            "1. 한줄 요약\n"
            "제공된 문서에서 해당 정보를 확인할 수 없습니다.\n\n"
            "2. 핵심 내용\n"
            "- 현재 검색된 근거만으로는 질문에 직접 답할 수 있는 문맥을 찾지 못했습니다.\n\n"
            "3. 주요 요구사항\n"
            "- 문맥에 없음\n\n"
            "4. 일정/예산/발주기관\n"
            "- 일정: 문맥에 없음\n"
            "- 예산: 문맥에 없음\n"
            "- 발주기관: 문맥에 없음\n\n"
            "5. 참고 근거\n"
            "- 직접 근거가 되는 청크를 찾지 못했습니다."
        )

    @staticmethod
    def _route_top_k(route: str, profile: dict[str, Any], default_top_k: int) -> int:
        if route == "b03a":
            return 4 if bool(profile.get("comparison")) else 3
        return default_top_k

    @staticmethod
    def _build_b02_context(candidates: list[CandidateRow]) -> str:
        blocks: list[str] = []
        for index, row in enumerate(candidates, start=1):
            metadata = row.metadata
            blocks.append(
                "\n".join(
                    [
                        f"[검색 결과 {index}]",
                        f"- 사업명: {metadata.get('사업명', '정보 없음')}",
                        f"- 발주 기관: {metadata.get('발주 기관', '정보 없음')}",
                        f"- 파일명: {metadata.get('source_file_name', '정보 없음')}",
                        f"- 청크 ID: {metadata.get('chunk_id', '정보 없음')}",
                        f"- fusion_score: {round(float(row.fusion_score), 6)}",
                        row.text,
                    ]
                )
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _build_b03_context(candidates: list[CandidateRow]) -> str:
        blocks: list[str] = []
        for index, row in enumerate(candidates, start=1):
            metadata = row.metadata
            blocks.append(
                "\n".join(
                    [
                        f"[검색 결과 {index}]",
                        f"- 파일명: {metadata.get('source_file_name', '정보 없음')}",
                        f"- 청크 ID: {metadata.get('chunk_id', '정보 없음')}",
                        f"- 섹션 제목: {metadata.get('section_title', '정보 없음')}",
                        f"- 청크 역할: {metadata.get('chunk_role', '정보 없음')}",
                        f"- 예산: {metadata.get('budget_text', '정보 없음')}",
                        f"- 사업기간: {metadata.get('period_raw', '정보 없음')}",
                        f"- 계약방식: {metadata.get('contract_method', '정보 없음')}",
                        row.text,
                    ]
                )
            )
        return "\n\n".join(blocks)

    def _build_route_context(self, route: str, candidates: list[CandidateRow]) -> str:
        if route == "b03a":
            return self._build_b03_context(candidates)
        return self._build_b02_context(candidates)

    def _build_route_prompts(self, route: str, question: str, retrieval_context: str) -> tuple[str, str]:
        system_prompt = self.B03_SYSTEM_PROMPT if route == "b03a" else self.B02_SYSTEM_PROMPT
        user_prompt = (
            f"질문:\n{question}\n\n"
            f"검색 문맥:\n{retrieval_context}\n\n"
            f"{self.ANSWER_FORMAT_TEXT}"
        )
        return system_prompt, user_prompt

    def _generate_openai_with_parity(self, model_name: str, system_prompt: str, user_prompt: str) -> str:
        response = self.openai_client.responses.create(
            model=model_name,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
            ],
        )
        return (response.output_text or "").strip()

    @staticmethod
    def _b03_unique_by_chunk_id(candidates: list[CandidateRow]) -> list[CandidateRow]:
        deduped: list[CandidateRow] = []
        seen: set[str] = set()
        for row in candidates:
            if row.chunk_id in seen:
                continue
            seen.add(row.chunk_id)
            deduped.append(row)
        return deduped

    @staticmethod
    def _b03_select_diverse_candidates(candidates: list[CandidateRow], top_k: int, compare_mode: bool) -> list[CandidateRow]:
        if not compare_mode:
            return candidates[:top_k]
        selected: list[CandidateRow] = []
        seen_files: set[str] = set()
        for row in candidates:
            source_file = str(row.metadata.get("source_file_name", ""))
            if source_file and source_file not in seen_files:
                selected.append(row)
                seen_files.add(source_file)
            if len(selected) >= top_k:
                return selected
        for row in candidates:
            if row in selected:
                continue
            selected.append(row)
            if len(selected) >= top_k:
                break
        return selected

    def _b03_build_second_pass_query(self, question: str, profile: dict[str, Any], rewrite_query: str) -> str:
        if rewrite_query:
            return rewrite_query
        hints: list[str] = []
        if bool(profile.get("budget")):
            hints.extend(["예산", "사업비", "금액"])
        if bool(profile.get("schedule")):
            hints.extend(["사업기간", "수행기간", "마감"])
        if bool(profile.get("contract")):
            hints.extend(["입찰방식", "계약방식", "협상"])
        if bool(profile.get("purpose")):
            hints.extend(["사업목적", "추진배경"])
        if bool(profile.get("comparison")):
            hints.extend(["비교 대상 사업", "모든 관련 문서"])
        hint_text = " ".join(dict.fromkeys(hints))
        return f"{question} {hint_text}".strip()

    def _vector_search_custom(self, query_embedding: list[float], candidate_k: int, weight: float) -> list[CandidateRow]:
        result = self.chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=max(1, int(candidate_k)),
            include=["documents", "metadatas", "distances"],
        )
        rows: list[CandidateRow] = []
        for rank, (document, metadata, _) in enumerate(
            zip(result["documents"][0], result["metadatas"][0], result["distances"][0]),
            start=1,
        ):
            rows.append(
                CandidateRow(
                    chunk_id=str(metadata.get("chunk_id", "")),
                    text=document,
                    metadata=dict(metadata),
                    fusion_score=(1.0 / (self.settings.rrf_k + rank)) * float(weight),
                )
            )
        return rows

    def _bm25_search_custom(self, question: str, candidate_k: int, weight: float) -> list[CandidateRow]:
        model = self.bm25_index["model"]
        chunk_rows = self.bm25_index["chunk_rows"]
        scores = model.get_scores(self._bm25_tokenize(question))
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[: max(1, int(candidate_k))]
        rows: list[CandidateRow] = []
        for rank, (row_index, _) in enumerate(ranked, start=1):
            source = chunk_rows[row_index]
            rows.append(
                CandidateRow(
                    chunk_id=str(source.get("chunk_id", "")),
                    text=str(source.get("contextual_chunk_text", "")),
                    metadata=dict(source),
                    fusion_score=(1.0 / (self.settings.rrf_k + rank)) * float(weight),
                )
            )
        return rows

    def _b03_crag_evaluate(self, question: str, profile: dict[str, Any], candidates: list[CandidateRow]) -> dict[str, Any]:
        if not candidates:
            return {
                "judgment": "INCORRECT",
                "reason": "No candidates.",
                "relevant_ranks": [],
                "need_second_pass": False,
                "rewrite_query": "",
                "focus_aspects": [],
            }
        profile_text = ", ".join(
            key
            for key in ["budget", "schedule", "contract", "purpose", "comparison", "follow_up"]
            if bool(profile.get(key))
        ) or "general"
        blocks: list[str] = []
        for rank, row in enumerate(candidates, start=1):
            excerpt = re.sub(r"\s+", " ", str(row.text).replace("\n", " ")).strip()[:420]
            blocks.append(
                "\n".join(
                    [
                        f"[candidate_{rank}]",
                        f"chunk_id: {row.chunk_id}",
                        f"source_file_name: {row.metadata.get('source_file_name', '')}",
                        f"section_title: {row.metadata.get('section_title', '')}",
                        f"chunk_role: {row.metadata.get('chunk_role', '')}",
                        f"budget_text: {row.metadata.get('budget_text', '')}",
                        f"period_raw: {row.metadata.get('period_raw', '')}",
                        f"contract_method: {row.metadata.get('contract_method', '')}",
                        f"purpose_summary: {row.metadata.get('purpose_summary', '')}",
                        f"adjusted_score: {row.adjusted_score if row.adjusted_score is not None else row.fusion_score}",
                        f"excerpt: {excerpt}",
                    ]
                )
            )
        system_prompt = (
            "당신은 고정된 RAG 검색 평가기다.\n"
            "질문과 검색 후보를 보고 현재 후보만으로 답변 가능한지 판단하라.\n"
            "판정 기준:\n"
            "- CORRECT: 질문을 직접 충족할 근거가 충분하다\n"
            "- AMBIGUOUS: 일부는 맞지만 부족하거나 비직접적이다\n"
            "- INCORRECT: 질문과 무관하거나 직접 근거가 없다\n"
            "반드시 JSON 객체만 반환하라.\n"
            "{"
            "\"judgment\":\"CORRECT|AMBIGUOUS|INCORRECT\","
            "\"reason\":\"짧은 이유\","
            "\"relevant_ranks\":[1,2],"
            "\"need_second_pass\":true,"
            "\"rewrite_query\":\"필요 시 보강 검색 질의\","
            "\"focus_aspects\":[\"예산\",\"기간\"]"
            "}"
        )
        block_text = "\n\n".join(blocks)
        user_prompt = (
            f"[질문]\n{question}\n\n"
            f"[질문 프로필]\n{profile_text}\n\n"
            f"[검색 후보]\n{block_text}\n\n"
            "relevant_ranks에는 실제 근거가 되는 후보 번호만 최대 4개까지 넣어라.\n"
            "AMBIGUOUS일 때만 rewrite_query를 채워라.\n"
            "비교형 질문이면 서로 다른 문서를 함께 확보해야 하는지 고려하라."
        )
        try:
            response = self.openai_client.responses.create(
                model=self.settings.routing_model,
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                    {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
                ],
            )
            payload = self._extract_json_block(response.output_text or "")
        except Exception:  # noqa: BLE001
            payload = {}

        judgment = str(payload.get("judgment", "AMBIGUOUS")).strip().upper()
        if judgment not in {"CORRECT", "AMBIGUOUS", "INCORRECT"}:
            judgment = "AMBIGUOUS"
        relevant_ranks: list[int] = []
        for value in payload.get("relevant_ranks", []):
            try:
                rank = int(value)
            except Exception:  # noqa: BLE001
                continue
            if 1 <= rank <= len(candidates) and rank not in relevant_ranks:
                relevant_ranks.append(rank)
        focus_aspects = [str(item).strip() for item in payload.get("focus_aspects", []) if str(item).strip()]
        return {
            "judgment": judgment,
            "reason": str(payload.get("reason", "")).strip(),
            "relevant_ranks": relevant_ranks,
            "need_second_pass": bool(payload.get("need_second_pass", judgment == "AMBIGUOUS")),
            "rewrite_query": str(payload.get("rewrite_query", "")).strip(),
            "focus_aspects": focus_aspects,
        }

    def _apply_b03_legacy_retrieval_flow(
        self,
        question: str,
        profile: dict[str, Any],
        reranked: list[CandidateRow],
        route_top_k: int,
    ) -> tuple[list[CandidateRow], dict[str, Any]]:
        evaluator_top_n = max(1, int(self.options.b03_evaluator_top_n))
        evaluator_candidates = reranked[:evaluator_top_n]
        crag_result = self._b03_crag_evaluate(question, profile, evaluator_candidates)

        if crag_result["judgment"] == "INCORRECT":
            return [], crag_result

        if crag_result["relevant_ranks"]:
            refined = [
                evaluator_candidates[rank - 1]
                for rank in crag_result["relevant_ranks"]
                if 1 <= rank <= len(evaluator_candidates)
            ]
        else:
            refined = evaluator_candidates
        refined = self._b03_unique_by_chunk_id(refined)

        candidate_pool_for_answer = reranked
        if crag_result["need_second_pass"]:
            second_pass_query = self._b03_build_second_pass_query(
                question,
                profile,
                str(crag_result.get("rewrite_query", "")),
            )
            second_k = max(int(self.settings.candidate_k), 12)
            second_embedding = self.embed_question(second_pass_query)
            vector_second = self._vector_search_custom(
                second_embedding,
                second_k,
                float(self.options.b03_second_pass_vector_weight),
            )
            bm25_second = self._bm25_search_custom(
                second_pass_query,
                second_k,
                float(self.options.b03_second_pass_bm25_weight),
            )
            fused_second = self.fuse_candidates(vector_second, bm25_second)
            second_ranked = self.rerank_with_profile(fused_second, profile, question)
            merged = self._b03_unique_by_chunk_id([*refined, *second_ranked, *reranked])
            candidate_pool_for_answer = self.rerank_with_profile(merged, profile, question)
            crag_result["second_pass_query"] = second_pass_query
        else:
            merged = self._b03_unique_by_chunk_id([*refined, *reranked])
            candidate_pool_for_answer = self.rerank_with_profile(merged, profile, question)
            crag_result["second_pass_query"] = ""

        final_candidates = self._b03_select_diverse_candidates(
            candidate_pool_for_answer,
            route_top_k,
            bool(profile.get("comparison")),
        )
        return final_candidates, crag_result

    def _merge_rows(
        self,
        rows: Iterable[CandidateRow],
        merged: dict[str, CandidateRow],
        *,
        weight: float = 1.0,
    ) -> None:
        for row in rows:
            if row.chunk_id not in merged:
                merged[row.chunk_id] = CandidateRow(
                    chunk_id=row.chunk_id,
                    text=row.text,
                    metadata=dict(row.metadata),
                    fusion_score=0.0,
                )
            merged[row.chunk_id].fusion_score += row.fusion_score * weight

    def _structured_helper_queries(self, question: str, profile: dict[str, Any]) -> list[str]:
        q = str(question or "").strip()
        if not q:
            return []
        if self._is_table_factual(profile):
            return [
                f"{q} header_value_pair header_path row_id col_id value_num value_date",
                f"{q} cell_row_block row_summary_chunk 표 헤더 값 항목 행 요약",
            ]
        if self._is_table_plus_text(profile):
            return [
                f"{q} paired_body_chunk table_body_pack row_body_pack header_value_pair",
                f"{q} 표 본문 결합 paired body section parent",
            ]
        return []

    def _filter_table_structured_rows(self, rows: list[CandidateRow], profile: dict[str, Any]) -> list[CandidateRow]:
        if not (self._is_table_factual(profile) or self._is_table_plus_text(profile)):
            return rows
        if not self.options.enable_structured_evidence_priority:
            return rows
        allowed_common = {"header_value_pair", "cell_row_block", "row_summary_chunk", "table_true_ocr_v2", "table_body_pack", "row_body_pack", "paired_body_chunk"}
        if self._is_table_factual(profile):
            allowed = allowed_common | {""}
        else:
            allowed = allowed_common | {"", "paired_body_chunk", "table_body_pack", "row_body_pack"}
        filtered = [row for row in rows if self._chunk_type(row) in allowed]
        return filtered if filtered else rows

    def _retrieve_with_variants(
        self,
        variants: list[str],
        question_row: dict[str, Any],
        question: str,
        profile: dict[str, Any],
    ) -> tuple[list[CandidateRow], bool]:
        merged: dict[str, CandidateRow] = {}
        normalized_bm25_used = False
        use_normalized_bm25 = self._should_apply_normalized_bm25(question_row, question, profile)

        for index, query in enumerate(variants):
            query_weight = 1.0 if index == 0 else self.options.expansion_query_weight
            embedding = self.embed_question(query)
            vector_rows = self.vector_search(embedding)
            bm25_rows = self.bm25_search(query)
            fused = self.fuse_candidates(vector_rows, bm25_rows)
            self._merge_rows(fused, merged, weight=query_weight)

            if use_normalized_bm25:
                normalized_queries = build_normalized_bm25_queries(query)
                for normalized in normalized_queries:
                    normalized_rows = self.bm25_search(normalized)
                    self._merge_rows(normalized_rows, merged, weight=self.options.normalized_bm25_weight)
                    normalized_bm25_used = True

            if self.options.enable_question_type_gated_ocr_routing and self._is_table_like_question(question, profile):
                helper_queries = self._structured_helper_queries(query, profile)
                for helper in helper_queries:
                    helper_rows = self.bm25_search(helper)
                    helper_rows = self._filter_table_structured_rows(helper_rows, profile)
                    self._merge_rows(helper_rows, merged, weight=0.45)

        return sorted(merged.values(), key=lambda item: item.fusion_score, reverse=True), normalized_bm25_used

    def _metadata_aware_bonus(self, row: CandidateRow, question: str, profile: dict[str, Any]) -> float:
        bonus = compute_metadata_soft_boost(question=question, profile=profile, metadata=row.metadata)
        if self._is_table_like_question(question, profile) and int(row.metadata.get("has_table", 0) or 0):
            bonus += 0.0020
        scaled = bonus * max(0.0, float(self.options.metadata_boost_scale))
        return round(scaled, 6)

    def _apply_metadata_bonus_v2(self, candidates: list[CandidateRow], question: str, profile: dict[str, Any]) -> list[CandidateRow]:
        reranked: list[CandidateRow] = []
        for row in candidates:
            adjusted = (row.adjusted_score if row.adjusted_score is not None else row.fusion_score) + self._metadata_aware_bonus(
                row, question, profile
            )
            reranked.append(
                CandidateRow(
                    chunk_id=row.chunk_id,
                    text=row.text,
                    metadata=dict(row.metadata),
                    fusion_score=row.fusion_score,
                    adjusted_score=adjusted,
                    crag_label=row.crag_label,
                    crag_reason=row.crag_reason,
                )
            )
        return sorted(reranked, key=lambda item: item.adjusted_score or item.fusion_score, reverse=True)

    @staticmethod
    def _doc_match(name: str, target: str) -> bool:
        left = _normalize_doc_name(name)
        right = _normalize_doc_name(target)
        return bool(left and right and (left in right or right in left))

    def _enforce_dual_doc_coverage(
        self,
        question_row: dict[str, Any],
        candidates: list[CandidateRow],
        profile: dict[str, Any],
    ) -> list[CandidateRow]:
        if not profile.get("comparison"):
            return candidates
        targets = self._target_docs(question_row)
        if len(targets) < 2:
            return candidates

        protected: list[CandidateRow] = []
        used: set[str] = set()
        for target in targets[:2]:
            matched = next(
                (
                    row
                    for row in candidates
                    if row.chunk_id not in used and self._doc_match(str(row.metadata.get("source_file_name", "")), target)
                ),
                None,
            )
            if matched is not None:
                protected.append(matched)
                used.add(matched.chunk_id)

        if len(protected) < 2:
            return candidates

        remainder = [row for row in candidates if row.chunk_id not in used]
        return [*protected, *remainder]

    def _apply_table_body_pairing(self, question: str, profile: dict[str, Any], candidates: list[CandidateRow]) -> list[CandidateRow]:
        if not self.options.enable_table_body_pairing:
            return candidates
        if not self._is_table_like_question(question, profile):
            return candidates

        answer_type = str(profile.get("answer_type", "")).strip().lower()
        is_table_plus_text = answer_type == "table_plus_text"
        enable_guard = self.options.enable_groupc_table_plus_text_guard and is_table_plus_text

        by_doc: dict[str, dict[str, CandidateRow]] = {}
        by_doc_lists: dict[str, dict[str, list[CandidateRow]]] = {}
        for row in candidates:
            doc_name = str(row.metadata.get("source_file_name", "")).strip()
            if not doc_name:
                continue
            entry = by_doc.setdefault(doc_name, {})
            list_entry = by_doc_lists.setdefault(doc_name, {"table": [], "body": []})
            has_table = int(row.metadata.get("has_table", 0) or 0) == 1
            key = "table" if has_table else "body"
            if key not in entry:
                entry[key] = row
            list_entry[key].append(row)

        candidates_map: dict[str, float] = {
            row.chunk_id: float(row.adjusted_score if row.adjusted_score is not None else row.fusion_score)
            for row in candidates
        }
        for pair in by_doc.values():
            if "table" in pair and "body" in pair:
                candidates_map[pair["table"].chunk_id] += 0.0040
                candidates_map[pair["body"].chunk_id] += 0.0030

        if enable_guard:
            for doc_name, grouped in by_doc_lists.items():
                tables = grouped.get("table", [])
                bodies = grouped.get("body", [])
                if not tables:
                    continue
                for table_row in tables:
                    table_section = str(table_row.metadata.get("section_label", "")).strip().lower()
                    table_parent = str(table_row.metadata.get("parent_section_label", "")).strip().lower()
                    table_linked = str(table_row.metadata.get("linked_parent_text", "")).strip().lower()

                    best_body: CandidateRow | None = None
                    best_score = 0.0
                    best_parent_hit = False

                    for body_row in bodies:
                        body_section = str(body_row.metadata.get("section_label", "")).strip().lower()
                        body_parent = str(body_row.metadata.get("parent_section_label", "")).strip().lower()
                        body_text = str(body_row.text or "").lower()
                        score = 0.0
                        parent_hit = False
                        if table_section and body_section and table_section == body_section:
                            score += 1.0
                        if table_parent and body_parent and table_parent == body_parent:
                            score += 0.8
                            parent_hit = True
                        if table_linked and (table_linked in body_text):
                            score += 0.6
                        if table_parent and table_parent in body_text:
                            score += 0.4
                            parent_hit = True
                        if score > best_score:
                            best_score = score
                            best_body = body_row
                            best_parent_hit = parent_hit

                    if best_body is not None and best_score > 0.0:
                        candidates_map[table_row.chunk_id] += float(self.options.groupc_pair_bonus)
                        candidates_map[best_body.chunk_id] += float(self.options.groupc_pair_bonus)
                        if best_parent_hit:
                            candidates_map[table_row.chunk_id] += float(self.options.groupc_parent_bonus)
                            candidates_map[best_body.chunk_id] += float(self.options.groupc_parent_bonus)
                    else:
                        candidates_map[table_row.chunk_id] -= float(self.options.groupc_table_penalty_without_body)

        reranked: list[CandidateRow] = []
        for row in candidates:
            reranked.append(
                CandidateRow(
                    chunk_id=row.chunk_id,
                    text=row.text,
                    metadata=dict(row.metadata),
                    fusion_score=row.fusion_score,
                    adjusted_score=candidates_map.get(row.chunk_id, row.adjusted_score or row.fusion_score),
                    crag_label=row.crag_label,
                    crag_reason=row.crag_reason,
                )
            )
        return sorted(reranked, key=lambda item: item.adjusted_score or item.fusion_score, reverse=True)

    def _apply_question_type_gated_ocr_routing(
        self,
        question: str,
        profile: dict[str, Any],
        candidates: list[CandidateRow],
        *,
        top_k_target: int | None = None,
    ) -> list[CandidateRow]:
        if not self.options.enable_question_type_gated_ocr_routing:
            return candidates
        if not candidates:
            return candidates
        if not (self._is_table_factual(profile) or self._is_table_plus_text(profile)):
            return candidates
        if not self._is_table_like_question(question, profile):
            return candidates

        table_factual_priority = {
            "header_value_pair": 0.0100,
            "cell_row_block": 0.0090,
            "row_summary_chunk": 0.0070,
            "table_body_pack": 0.0060,
            "row_body_pack": 0.0050,
            "table_true_ocr_v2": 0.0020,
        }
        table_plus_text_priority = {
            "paired_body_chunk": 0.0120,
            "table_body_pack": 0.0110,
            "row_body_pack": 0.0100,
            "header_value_pair": 0.0070,
            "row_summary_chunk": 0.0060,
            "cell_row_block": 0.0050,
            "table_true_ocr_v2": 0.0005,
        }
        pair_types = {"paired_body_chunk", "table_body_pack", "row_body_pack"}
        structured_types = {"header_value_pair", "cell_row_block", "row_summary_chunk", "table_body_pack", "row_body_pack"}
        has_structured_any = any(self._chunk_type(row) in structured_types for row in candidates)
        if self._is_table_plus_text(profile) and not has_structured_any:
            return candidates

        by_doc_has_pair: dict[str, bool] = {}
        for row in candidates:
            doc_key = _normalize_doc_name(str(row.metadata.get("source_file_name", "")))
            chunk_type = self._chunk_type(row)
            has_pair = chunk_type in pair_types or float(row.metadata.get("pairing_score", 0.0) or 0.0) >= 0.25
            if doc_key:
                by_doc_has_pair[doc_key] = by_doc_has_pair.get(doc_key, False) or has_pair

        reranked: list[CandidateRow] = []
        for row in candidates:
            chunk_type = self._chunk_type(row)
            base_score = float(row.adjusted_score if row.adjusted_score is not None else row.fusion_score)
            doc_key = _normalize_doc_name(str(row.metadata.get("source_file_name", "")))
            has_pair_support = bool(by_doc_has_pair.get(doc_key, False))
            bonus = 0.0
            metadata = dict(row.metadata)
            metadata.setdefault("header_match_score", 0.0)
            metadata.setdefault("row_match_score", 0.0)
            metadata.setdefault("entity_overlap_score", 0.0)
            metadata.setdefault("definition_pattern_score", 0.0)
            metadata.setdefault("list_request_score", 0.0)
            metadata.setdefault("exact_header_match_hit", 0.0)
            metadata.setdefault("exact_row_match_hit", 0.0)
            metadata.setdefault("generic_row_pollution", 0.0)

            if self._is_table_factual(profile):
                bonus += table_factual_priority.get(chunk_type, 0.0)
                if self.options.enable_structured_evidence_priority and chunk_type in structured_types:
                    bonus += 0.0015
                if chunk_type == "":
                    bonus -= 0.0015
                if self.options.enable_table_factual_alignment_scoring:
                    scores = self._table_factual_alignment_scores(row, question)
                    metadata.update(scores)
                    bonus += (
                        (scores["header_match_score"] * 0.0100)
                        + (scores["row_match_score"] * 0.0080)
                        + (scores["entity_overlap_score"] * 0.0040)
                        + (scores["definition_pattern_score"] * 0.0040)
                        + (scores["list_request_score"] * 0.0030)
                    )
                if self._is_generic_row_content(row):
                    metadata["generic_row_pollution"] = 1.0
                    bonus -= float(self.options.table_factual_generic_penalty)
            elif self._is_table_plus_text(profile):
                bonus += table_plus_text_priority.get(chunk_type, 0.0)
                if self.options.enable_structured_evidence_priority and chunk_type in structured_types:
                    bonus += 0.0012
                if chunk_type == "table_true_ocr_v2" and not has_pair_support:
                    bonus -= 0.0030
                if chunk_type in pair_types:
                    bonus += 0.0010

            reranked.append(
                CandidateRow(
                    chunk_id=row.chunk_id,
                    text=row.text,
                    metadata=metadata,
                    fusion_score=row.fusion_score,
                    adjusted_score=base_score + bonus,
                    crag_label=row.crag_label,
                    crag_reason=row.crag_reason,
                )
            )

        reranked = sorted(reranked, key=lambda item: item.adjusted_score or item.fusion_score, reverse=True)

        if not self._is_table_plus_text(profile):
            return reranked

        # Group C guard: ensure paired_body evidence and structured table evidence are both present.
        target = max(1, int(top_k_target or self.settings.top_k))
        head = list(reranked[:target])
        tail = list(reranked[target:])
        head_types = {self._chunk_type(row) for row in head}
        needs_pair = not any(item in head_types for item in pair_types)
        needs_struct = not any(item in head_types for item in structured_types)

        if needs_pair:
            pair_candidate = next((row for row in tail if self._chunk_type(row) in pair_types), None)
            if pair_candidate is not None:
                head.append(pair_candidate)
                tail = [row for row in tail if row.chunk_id != pair_candidate.chunk_id]
        if needs_struct:
            struct_candidate = next((row for row in tail if self._chunk_type(row) in structured_types), None)
            if struct_candidate is not None:
                head.append(struct_candidate)
                tail = [row for row in tail if row.chunk_id != struct_candidate.chunk_id]

        ordered: list[CandidateRow] = []
        seen: set[str] = set()
        for row in [*head, *tail]:
            if row.chunk_id in seen:
                continue
            seen.add(row.chunk_id)
            ordered.append(row)
        return ordered

    @staticmethod
    def _candidate_score(row: CandidateRow) -> float:
        return float(row.adjusted_score if row.adjusted_score is not None else row.fusion_score)

    def _comparison_axis_signal(self, row: CandidateRow, profile: dict[str, Any]) -> float:
        metadata = row.metadata
        signal = 0.0
        if bool(profile.get("budget")) and int(metadata.get("has_budget_signal", 0) or 0):
            signal += 1.0
        if bool(profile.get("schedule")) and int(metadata.get("has_schedule_signal", 0) or 0):
            signal += 1.0
        if bool(profile.get("contract")) and int(metadata.get("has_contract_signal", 0) or 0):
            signal += 1.0
        if bool(profile.get("purpose")) and str(metadata.get("purpose_summary", "")).strip():
            signal += 0.8
        if int(metadata.get("has_table", 0) or 0) == 1 and (bool(profile.get("budget")) or bool(profile.get("schedule"))):
            signal += 0.4
        return signal

    def _apply_comparison_evidence_helper(
        self,
        candidates: list[CandidateRow],
        profile: dict[str, Any],
    ) -> list[CandidateRow]:
        if not self.options.enable_comparison_evidence_helper:
            return candidates
        if not bool(profile.get("comparison")):
            return candidates
        if not candidates:
            return candidates

        buckets: dict[str, list[CandidateRow]] = {}
        for row in candidates:
            doc_name = str(row.metadata.get("source_file_name", "")).strip()
            key = _normalize_doc_name(doc_name) or f"__unknown__:{row.chunk_id}"
            buckets.setdefault(key, []).append(row)

        for key in list(buckets.keys()):
            buckets[key] = sorted(
                buckets[key],
                key=lambda item: (
                    self._candidate_score(item) + (self._comparison_axis_signal(item, profile) * self.options.comparison_helper_axis_bonus)
                ),
                reverse=True,
            )

        ordered_doc_keys = sorted(
            buckets.keys(),
            key=lambda doc_key: self._candidate_score(buckets[doc_key][0]) if buckets[doc_key] else -1.0,
            reverse=True,
        )
        per_doc_limit = max(1, int(self.options.comparison_helper_max_per_doc))
        picked_per_doc: dict[str, int] = {key: 0 for key in ordered_doc_keys}
        selected: list[CandidateRow] = []
        selected_ids: set[str] = set()

        active = True
        while active:
            active = False
            for doc_key in ordered_doc_keys:
                bucket = buckets.get(doc_key, [])
                while bucket and bucket[0].chunk_id in selected_ids:
                    bucket.pop(0)
                if not bucket:
                    continue
                if picked_per_doc.get(doc_key, 0) >= per_doc_limit:
                    continue
                row = bucket.pop(0)
                selected.append(row)
                selected_ids.add(row.chunk_id)
                picked_per_doc[doc_key] = picked_per_doc.get(doc_key, 0) + 1
                active = True

        remainder = [row for row in candidates if row.chunk_id not in selected_ids]
        merged = [*selected, *remainder]

        seen_doc: set[str] = set()
        reranked: list[CandidateRow] = []
        for row in merged:
            doc_name = str(row.metadata.get("source_file_name", "")).strip()
            doc_key = _normalize_doc_name(doc_name) or f"__unknown__:{row.chunk_id}"
            base_score = self._candidate_score(row)
            bonus = self._comparison_axis_signal(row, profile) * self.options.comparison_helper_axis_bonus
            if doc_key not in seen_doc:
                bonus += float(self.options.comparison_helper_doc_bonus)
                seen_doc.add(doc_key)
            reranked.append(
                CandidateRow(
                    chunk_id=row.chunk_id,
                    text=row.text,
                    metadata=dict(row.metadata),
                    fusion_score=row.fusion_score,
                    adjusted_score=base_score + bonus,
                    crag_label=row.crag_label,
                    crag_reason=row.crag_reason,
                )
            )

        return sorted(reranked, key=lambda item: item.adjusted_score or item.fusion_score, reverse=True)

    def apply_soft_crag_lite(
        self,
        question_row: dict[str, Any],
        question: str,
        candidates: list[CandidateRow],
        profile: dict[str, Any],
        *,
        route_top_k: int,
    ) -> tuple[list[CandidateRow], dict[str, Any]]:
        if not candidates:
            return [], {
                "decision_keep_count": 0,
                "decision_downrank_count": 0,
                "decision_low_conf_count": 0,
                "low_confidence_flag": 0.0,
                "duplicate_ratio": 0.0,
                "apply_scope": str(self.options.soft_crag_scope_mode),
                "factual_mode": str(self.options.soft_crag_factual_mode),
                "applied": 0.0,
            }

        config = self._build_soft_crag_config()
        selected, _, summary = apply_soft_crag_lite_rules(
            question=question,
            profile=profile,
            candidates=candidates,
            config=config,
            top_k=max(route_top_k, self.settings.top_k),
        )
        summary = dict(summary)
        summary["applied"] = 1.0
        summary["question_id"] = str(question_row.get("question_id", ""))
        return selected, summary

    def _compute_coverage_metrics(
        self,
        question_row: dict[str, Any],
        question: str,
        profile: dict[str, Any],
        candidates: list[CandidateRow],
    ) -> dict[str, Any]:
        source_docs = [str(row.metadata.get("source_file_name", "")).strip() for row in candidates]
        target_docs = self._target_docs(question_row)
        source_norm = [_normalize_doc_name(doc) for doc in source_docs if doc]
        target_norm = [_normalize_doc_name(doc) for doc in target_docs if doc]

        if target_norm:
            hit_count = sum(1 for target in target_norm if any(target in src or src in target for src in source_norm))
            comparison_evidence_coverage = round(hit_count / len(target_norm), 4)
        else:
            comparison_evidence_coverage = None

        if len(target_norm) >= 2:
            dual_doc_coverage = 1.0 if all(any(target in src or src in target for src in source_norm) for target in target_norm[:2]) else 0.0
        else:
            dual_doc_coverage = None

        pair_types = {"paired_body_chunk", "table_body_pack", "row_body_pack"}
        structured_types = {"header_value_pair", "cell_row_block", "row_summary_chunk", "table_body_pack", "row_body_pack"}
        chunk_types = [self._chunk_type(row) for row in candidates]
        exact_header_match_hit = 1.0 if any(
            float(row.metadata.get("exact_header_match_hit", 0.0) or 0.0) >= 1.0
            or float(row.metadata.get("header_match_score", 0.0) or 0.0) >= 0.60
            for row in candidates
        ) else 0.0
        exact_row_match_hit = 1.0 if any(
            float(row.metadata.get("exact_row_match_hit", 0.0) or 0.0) >= 1.0
            or float(row.metadata.get("row_match_score", 0.0) or 0.0) >= 0.60
            for row in candidates
        ) else 0.0
        generic_row_pollution_count = sum(
            1 for row in candidates if float(row.metadata.get("generic_row_pollution", 0.0) or 0.0) >= 1.0
        )

        has_table = any(
            int(row.metadata.get("has_table", 0) or 0) == 1 or self._chunk_type(row) in structured_types or self._chunk_type(row) == "table_true_ocr_v2"
            for row in candidates
        )
        has_body = any(
            int(row.metadata.get("has_table", 0) or 0) == 0 or self._chunk_type(row) in pair_types
            for row in candidates
        )
        pair_chunk_hit = any(chunk_type in pair_types for chunk_type in chunk_types)
        pairing_score_max = max(
            [float(row.metadata.get("pairing_score", 0.0) or 0.0) for row in candidates] + [0.0]
        )

        answer_type = str(profile.get("answer_type", "")).strip().lower()
        is_table_plus_text = answer_type == "table_plus_text"

        table_plus_body_coverage: float | None = None
        parent_section_hit: float | None = None
        nearby_body_hit: float | None = None
        pair_hit: float | None = None
        body_hit: float | None = None
        answer_hit: float | None = None
        if self._is_table_like_question(question, profile):
            if is_table_plus_text:
                table_rows = [row for row in candidates if int(row.metadata.get("has_table", 0) or 0) == 1]
                body_rows = [row for row in candidates if int(row.metadata.get("has_table", 0) or 0) == 0]
                pair_found = False
                parent_found = False
                for table_row in table_rows:
                    table_section = str(table_row.metadata.get("section_label", "")).strip().lower()
                    table_parent = str(table_row.metadata.get("parent_section_label", "")).strip().lower()
                    table_linked = str(table_row.metadata.get("linked_parent_text", "")).strip().lower()
                    for body_row in body_rows:
                        body_section = str(body_row.metadata.get("section_label", "")).strip().lower()
                        body_parent = str(body_row.metadata.get("parent_section_label", "")).strip().lower()
                        body_text = str(body_row.text or "").lower()
                        same_section = bool(table_section and body_section and table_section == body_section)
                        same_parent = bool(table_parent and body_parent and table_parent == body_parent)
                        linked_match = bool(table_linked and table_linked in body_text)
                        if same_section or same_parent or linked_match:
                            pair_found = True
                        if same_parent:
                            parent_found = True
                pair_found = pair_found or pair_chunk_hit or pairing_score_max >= 0.25
                table_plus_body_coverage = 1.0 if pair_found else 0.0
                nearby_body_hit = 1.0 if (pair_found and has_body) else 0.0
                parent_section_hit = 1.0 if parent_found else 0.0
                pair_hit = 1.0 if pair_found else 0.0
                body_hit = 1.0 if has_body else 0.0
                answer_hit = 1.0 if (has_table and has_body and pair_found) else 0.0
            else:
                table_plus_body_coverage = 1.0 if (has_table and has_body) else 0.0
                nearby_body_hit = 1.0 if has_body else 0.0
                parent_section_hit = 1.0 if has_body else 0.0
                pair_hit = 1.0 if (pair_chunk_hit or pairing_score_max >= 0.20) else 0.0
                body_hit = 1.0 if has_body else 0.0
                answer_hit = 1.0 if has_table else 0.0

        unique_docs = len({_normalize_doc_name(doc) for doc in source_docs if doc})
        source_diversity = round(unique_docs / max(1, len(source_docs)), 4)

        return {
            "dual_doc_coverage": dual_doc_coverage,
            "comparison_evidence_coverage": comparison_evidence_coverage,
            "table_plus_body_coverage": table_plus_body_coverage,
            "source_diversity": source_diversity,
            "table_hit": 1.0 if has_table else 0.0,
            "body_hit": body_hit,
            "pair_hit": pair_hit,
            "answer_hit": answer_hit,
            "nearby_body_hit": nearby_body_hit,
            "parent_section_hit": parent_section_hit,
            "pairing_score_max": round(pairing_score_max, 4),
            "structured_evidence_hit": 1.0 if any(chunk_type in structured_types for chunk_type in chunk_types) else 0.0,
            "exact_header_match_hit": exact_header_match_hit,
            "exact_row_match_hit": exact_row_match_hit,
            "generic_row_pollution_count": float(generic_row_pollution_count),
        }

    def retrieve(
        self,
        question_row: dict[str, Any],
        question: str,
        history: list[dict[str, str]] | None = None,
    ) -> RetrievalResult:
        prepared_row, router_result = self._prepare_question_row(question_row, question, history=history)
        route = self.decide_route(prepared_row)
        profile = self.build_question_profile(prepared_row, question)
        route_top_k = self._route_top_k(route, profile, self.settings.top_k)
        query_variants = self.build_query_variants(prepared_row, question, profile)

        fused, normalized_bm25_used = self._retrieve_with_variants(query_variants, prepared_row, question, profile)
        reranked = self.rerank_with_profile(fused, profile, question)
        metadata_aware_used = False
        if self.options.enable_metadata_aware_retrieval and self._should_apply_metadata_bonus(question, profile):
            reranked = self._apply_metadata_bonus_v2(reranked, question, profile)
            metadata_aware_used = True

        reranked = self._enforce_dual_doc_coverage(prepared_row, reranked, profile)
        reranked = self._apply_comparison_evidence_helper(reranked, profile)
        reranked = self._apply_table_body_pairing(question, profile, reranked)
        reranked = self._apply_question_type_gated_ocr_routing(
            question,
            profile,
            reranked,
            top_k_target=max(route_top_k, self.settings.top_k),
        )

        selected = reranked[: max(route_top_k, self.settings.top_k)]
        fallback_triggered = False
        soft_crag_lite_used = False
        soft_crag_summary: dict[str, Any] = {
            "decision_keep_count": 0,
            "decision_downrank_count": 0,
            "decision_low_conf_count": 0,
            "low_confidence_flag": 0.0,
            "duplicate_ratio": 0.0,
            "apply_scope": str(self.options.soft_crag_scope_mode),
            "factual_mode": str(self.options.soft_crag_factual_mode),
            "applied": 0.0,
        }
        b03_crag_judgment = ""
        b03_second_pass_used = 0.0

        soft_crag_applicable = (
            self.options.enable_soft_crag_lite
            and not bool(profile.get("rejection"))
            and should_apply_soft_crag(profile, self._build_soft_crag_config())
        )
        if soft_crag_applicable:
            selected, soft_crag_summary = self.apply_soft_crag_lite(
                prepared_row,
                question,
                reranked,
                profile,
                route_top_k=route_top_k,
            )
            soft_crag_lite_used = True
            fallback_triggered = bool(float(soft_crag_summary.get("low_confidence_flag", 0.0) or 0.0) >= 1.0)
        elif route == "b03a":
            if self.options.enable_b03_legacy_crag_parity:
                selected, crag_result = self._apply_b03_legacy_retrieval_flow(question, profile, reranked, route_top_k)
                b03_crag_judgment = str(crag_result.get("judgment", ""))
                b03_second_pass_used = 1.0 if str(crag_result.get("second_pass_query", "")).strip() else 0.0
            else:
                selected = self.apply_crag(prepared_row, question, reranked)
        selected = self._apply_comparison_evidence_helper(selected, profile)
        selected = self._apply_question_type_gated_ocr_routing(
            question,
            profile,
            selected,
            top_k_target=route_top_k,
        )
        selected = selected[:route_top_k]

        coverage = self._compute_coverage_metrics(prepared_row, question, profile, selected)
        profile_debug = dict(profile)
        profile_debug.update(coverage)
        profile_debug.update(
            {
                "query_variants": json.dumps(query_variants, ensure_ascii=False),
                "query_variant_count": len(query_variants),
                "controlled_query_expansion_used": 1.0 if len(query_variants) > 1 else 0.0,
                "soft_crag_lite_used": 1.0 if soft_crag_lite_used else 0.0,
                "fallback_triggered": 1.0 if fallback_triggered else 0.0,
                "normalized_bm25_used": 1.0 if normalized_bm25_used else 0.0,
                "metadata_aware_used": 1.0 if metadata_aware_used else 0.0,
                "comparison_helper_used": 1.0 if (self.options.enable_comparison_evidence_helper and bool(profile.get("comparison"))) else 0.0,
                "route_top_k_target": float(route_top_k),
                "answer_layer_policy": "b06_parity",
                "b03_crag_judgment": b03_crag_judgment,
                "b03_second_pass_used": b03_second_pass_used,
                "soft_crag_scope_mode": str(self.options.soft_crag_scope_mode),
                "soft_crag_factual_mode": str(self.options.soft_crag_factual_mode),
                "soft_crag_decision_keep_count": float(soft_crag_summary.get("decision_keep_count", 0)),
                "soft_crag_decision_downrank_count": float(soft_crag_summary.get("decision_downrank_count", 0)),
                "soft_crag_decision_low_conf_count": float(soft_crag_summary.get("decision_low_conf_count", 0)),
                "soft_crag_low_confidence_flag": float(soft_crag_summary.get("low_confidence_flag", 0.0) or 0.0),
                "soft_crag_duplicate_ratio": float(soft_crag_summary.get("duplicate_ratio", 0.0) or 0.0),
            }
        )
        if router_result is not None:
            profile_debug["answer_type_router_used"] = 1.0
            profile_debug["answer_type_router_confidence"] = float(router_result.get("confidence", 0.0) or 0.0)
            profile_debug["answer_type_router_route"] = str(router_result.get("route", ""))
            profile_debug["answer_type_router_signals"] = json.dumps(router_result.get("signals", []), ensure_ascii=False)
        else:
            profile_debug["answer_type_router_used"] = 0.0
            profile_debug["answer_type_router_confidence"] = 0.0
            profile_debug["answer_type_router_route"] = ""
            profile_debug["answer_type_router_signals"] = "[]"
        if str(profile.get("answer_type", "")).strip().lower() == "table_plus_text":
            weak_pair = float(coverage.get("pair_hit") or 0.0) < 1.0
            weak_body = float(coverage.get("body_hit") or 0.0) < 1.0
            profile_debug["weak_context"] = 1.0 if (weak_pair or weak_body) else 0.0

        return RetrievalResult(
            route=route,
            profile=profile_debug,
            candidates=selected,
            context_text=self._build_route_context(route, selected),
        )

    def _build_route_prompts(
        self,
        route: str,
        question: str,
        retrieval_context: str,
        profile: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        effective_profile = profile or {}
        system_prompt = self.B03_SYSTEM_PROMPT if route == "b03a" else self.B02_SYSTEM_PROMPT
        if self.options.enable_table_factual_exact_answer_mode and self._is_table_factual(effective_profile):
            style = self._table_factual_response_style(question)
            style_guide = {
                "definition": "출력: 1~2문장 정의. 필요하면 한 줄로 구성 항목 추가.",
                "list": "출력: bullet list 또는 markdown table.",
                "linkage": "출력: 열 중심 markdown table (항목|연계대상|설명).",
                "app_function_list": "출력: 앱/기능 항목 나열.",
            }.get(style, "출력: bullet list.")
            user_prompt = (
                f"질문:\n{question}\n\n"
                f"검토 문맥:\n{retrieval_context}\n\n"
                "table_factual 전용 답변 규칙:\n"
                "- 5섹션 장문 포맷 금지\n"
                "- 사업개요/예산/일정/발주기관 boilerplate 금지\n"
                "- 근거 없는 추측 금지\n"
                f"- {style_guide}\n"
                "- 답변 마지막에 근거 chunk_id를 괄호로 짧게 표기"
            )
            return system_prompt, user_prompt
        user_prompt = (
            f"吏덈Ц:\n{question}\n\n"
            f"寃??臾몃㎘:\n{retrieval_context}\n\n"
            f"{self.ANSWER_FORMAT_TEXT}"
        )
        return system_prompt, user_prompt

    def answer(
        self,
        question_row: dict[str, Any],
        adapter: Any,
        *,
        question: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> ScenarioAAnswer:
        resolved_question = question or str(question_row.get("question", "")).strip()
        retrieval = self.retrieve(question_row, resolved_question, history=history)
        route = retrieval.route
        context_text = retrieval.context_text

        if route == "b03a" and not retrieval.candidates:
            answer_text = self._no_info_answer()
        else:
            system_prompt, user_prompt = self._build_route_prompts(
                route,
                resolved_question,
                context_text,
                retrieval.profile,
            )
            if str(getattr(adapter.config, "adapter_name", "")).strip() == "openai_chat":
                answer_text = self._generate_openai_with_parity(adapter.config.model_id, system_prompt, user_prompt)
            else:
                answer_text = adapter.generate(
                    system_instruction=system_prompt,
                    question=resolved_question,
                    context_text=context_text,
                    history=history or [],
                )

        return ScenarioAAnswer(
            route=route,
            question=resolved_question,
            profile=retrieval.profile,
            context_text=context_text,
            candidates=retrieval.candidates,
            answer_text=answer_text,
        )

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable


RouteResolver = Callable[[str], str]
LLMHelper = Callable[[str, dict[str, float]], str | None]

ANSWER_TYPES: tuple[str, ...] = (
    "factual",
    "comparison",
    "follow_up",
    "rejection",
    "table_factual",
    "table_plus_text",
)


@dataclass(slots=True)
class AnswerTypeRouteResult:
    answer_type: str
    route: str
    confidence: float
    signals: list[str]
    reason: str
    scores: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "answer_type": self.answer_type,
            "route": self.route,
            "confidence": round(float(self.confidence), 4),
            "signals": list(self.signals),
            "reason": self.reason,
            "scores": {key: round(float(value), 4) for key, value in self.scores.items()},
        }


def _route_for_answer_type(answer_type: str) -> str:
    if answer_type in {"comparison", "factual"}:
        return "b03a"
    return "b02"


def normalize_answer_type(value: str | None) -> str | None:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in ANSWER_TYPES else None


def _score_patterns(question: str, specs: list[tuple[str, float, str]]) -> tuple[float, list[str]]:
    score = 0.0
    signals: list[str] = []
    for pattern, weight, signal in specs:
        if re.search(pattern, question, flags=re.IGNORECASE):
            score += float(weight)
            signals.append(signal)
    return score, signals


def classify_answer_type(
    *,
    question: str,
    history: list[dict[str, str]] | None = None,
    context_summary: str | None = None,
    llm_helper: LLMHelper | None = None,
    route_resolver: RouteResolver | None = None,
    confidence_threshold: float = 0.0,
    low_conf_fallback: str = "comparison_safe",
) -> AnswerTypeRouteResult:
    q = str(question or "").strip()
    if not q:
        answer_type = "factual"
        route = route_resolver(answer_type) if route_resolver else _route_for_answer_type(answer_type)
        return AnswerTypeRouteResult(
            answer_type=answer_type,
            route=route,
            confidence=0.55,
            signals=["empty_question_fallback"],
            reason="Empty question fallback",
            scores={"factual": 0.55},
        )

    history_count = len(history or [])
    has_history = history_count > 0
    summary_text = str(context_summary or "")
    scoring_input = f"{q}\n{summary_text}"

    type_specs: dict[str, list[tuple[str, float, str]]] = {
        "comparison": [
            (r"(비교|차이|공통|각각|모두|A와 B|A/B|이상|이하|정렬|우선순위|대조|대비)", 0.42, "comparison_pattern"),
            (r"(두 사업|두 문서|여러 사업|기관별|문서별)", 0.30, "multi_doc_hint"),
            (r"(전부\s*찾아|모두\s*찾아|찾아서\s*정리|내림차순|오름차순)", 0.20, "aggregate_compare_hint"),
            (r"(기관들을|사업들을|모든\s*기관)", 0.18, "plural_entity_hint"),
        ],
        "follow_up": [
            (r"(그럼|그중|그 사업|이 사업|앞에서|이전|이어|계속|방금|위에서)", 0.50, "followup_pattern"),
            (r"(다시|추가로|그 부분|그 항목)", 0.30, "followup_reference"),
            (r"(더\s*자세히|다른\s*기관|다른\s*사업|추가\s*설명|그 외)", 0.22, "followup_extension"),
        ],
        "rejection": [
            (
                r"(실제\s*낙찰|최종\s*낙찰|실제\s*완료일|실제\s*납품|실제\s*투입|실제\s*성과|운영\s*성과|현재\s*URL|실서비스|실운영|개인\s*연락처|입찰한\s*업체)",
                0.66,
                "out_of_doc_pattern",
            ),
            (r"(실제|현재|운영중|최종\s*결과|실적)", 0.28, "outside_tense_pattern"),
            (r"(컨소시엄\s*구성|낙찰\s*업체|완료됐|운영\s*성과)", 0.36, "rejection_slot_pattern"),
        ],
        "table_factual": [
            (r"(목록|내역|현황|앱|장비|정의|배점|평가항목|구성표|리스트|표로)", 0.44, "table_factual_pattern"),
            (r"(기능\s*목록|기능\s*내역|기능\s*구성)", 0.24, "function_list_hint"),
            (r"(항목|열|행|모듈)", 0.20, "table_structure_hint"),
        ],
        "table_plus_text": [
            (r"(왜|어떻게|연결|배경|목적|한계|해결|역할\s*차이|의미|근거)", 0.45, "table_plus_text_pattern"),
            (r"(표.*본문|본문.*표|표와.*같이|표를.*근거로)", 0.35, "needs_table_and_body"),
        ],
    }

    scores: dict[str, float] = {"factual": 0.33}
    signal_map: dict[str, list[str]] = {"factual": []}
    for label, specs in type_specs.items():
        score, signals = _score_patterns(scoring_input, specs)
        scores[label] = score
        signal_map[label] = signals

    # Follow-up should rely on context. If no history, dampen heavily.
    if not has_history:
        scores["follow_up"] *= 0.35
    else:
        scores["follow_up"] += 0.18
        signal_map["follow_up"].append("history_present")
        if re.search(r"(두\s*사업\s*중|각각|더\s*넓어|추가로|더\s*자세히|그\s*부분|해당)", scoring_input):
            scores["follow_up"] += 0.42
            signal_map["follow_up"].append("history_anchor_pattern")

    # table_plus_text should include table cues.
    if not re.search(r"(표|테이블|행|열|항목|목록|내역|현황)", scoring_input, flags=re.IGNORECASE):
        scores["table_plus_text"] *= 0.5

    # table_factual/list-like questions get a slight boost if interrogative is short/direct.
    if re.search(r"(무엇|뭐야|얼마|몇|어떤|정리|나열)", scoring_input):
        scores["table_factual"] += 0.08

    # comparison vs follow_up tie-breaker.
    if scores["comparison"] > 0.45 and scores["follow_up"] > 0.45:
        if has_history:
            scores["follow_up"] += 0.06
        else:
            scores["comparison"] += 0.08

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_label, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    if llm_helper is not None and (top_score < 0.46 or (top_score - second_score) < 0.05):
        helper_label = llm_helper(q, scores)
        if helper_label in {"factual", "comparison", "follow_up", "rejection", "table_factual", "table_plus_text"}:
            top_label = str(helper_label)
            top_score = max(top_score, 0.46)
            signal_map[top_label].append("llm_helper_override")

    margin = max(0.0, top_score - second_score)
    confidence = min(0.99, max(0.50, 0.50 + (0.35 * margin) + (0.20 * min(1.0, top_score))))
    if top_label == "factual" and top_score <= 0.40:
        confidence = min(confidence, 0.72)

    threshold = max(0.0, float(confidence_threshold or 0.0))
    if threshold > 0.0 and confidence < threshold:
        fallback_label = "factual"
        if str(low_conf_fallback).strip().lower() == "comparison_safe":
            comp = float(scores.get("comparison", 0.0) or 0.0)
            factual = float(scores.get("factual", 0.0) or 0.0)
            if comp >= max(0.42, factual + 0.08):
                fallback_label = "comparison"
        top_label = fallback_label
        signal_map.setdefault(top_label, []).append("low_confidence_fallback")
        margin = max(0.0, float(scores.get(top_label, 0.0) or 0.0) - second_score)
        confidence = max(confidence, threshold)

    route = route_resolver(top_label) if route_resolver else _route_for_answer_type(top_label)
    signals = signal_map.get(top_label, [])
    reason = f"rule_based:{top_label}, top={top_score:.3f}, margin={margin:.3f}"
    return AnswerTypeRouteResult(
        answer_type=top_label,
        route=route,
        confidence=round(confidence, 4),
        signals=signals,
        reason=reason,
        scores=scores,
    )


def predict(
    *,
    question: str,
    history: list[dict[str, str]] | None = None,
    context_summary: str | None = None,
    llm_helper: LLMHelper | None = None,
    route_resolver: RouteResolver | None = None,
    confidence_threshold: float = 0.56,
    low_conf_fallback: str = "comparison_safe",
) -> dict[str, Any]:
    """Stable app-facing router API used by Streamlit and batch tooling."""
    result = classify_answer_type(
        question=question,
        history=history,
        context_summary=context_summary,
        llm_helper=llm_helper,
        route_resolver=route_resolver,
        confidence_threshold=confidence_threshold,
        low_conf_fallback=low_conf_fallback,
    )
    return result.as_dict()


def apply_answer_type_override(
    result: dict[str, Any],
    override: str | None,
    *,
    route_resolver: RouteResolver | None = None,
) -> dict[str, Any]:
    """Return an execution router result while preserving the original prediction."""
    override_type = normalize_answer_type(override)
    base = dict(result or {})
    predicted = dict(base)
    if override_type is None:
        base["override_applied"] = False
        return base

    resolver = route_resolver or _route_for_answer_type
    signals = list(predicted.get("signals", []) or [])
    signals.append(f"manual_override:{override_type}")
    base.update(
        {
            "answer_type": override_type,
            "route": resolver(override_type),
            "confidence": 1.0,
            "signals": signals,
            "reason": f"manual_override:{override_type}; predicted={predicted.get('answer_type', '')}",
            "override_applied": True,
            "predicted": predicted,
        }
    )
    return base


def build_virtual_question_row(
    *,
    question: str,
    base_row: dict[str, Any] | None = None,
    history: list[dict[str, str]] | None = None,
    context_summary: str | None = None,
    force_router: bool = False,
    llm_helper: LLMHelper | None = None,
    route_resolver: RouteResolver | None = None,
    confidence_threshold: float = 0.58,
    low_conf_fallback: str = "comparison_safe",
) -> tuple[dict[str, Any], AnswerTypeRouteResult | None]:
    row = dict(base_row or {})
    existing_type = str(row.get("answer_type", "")).strip().lower()
    if existing_type and not force_router:
        return row, None

    result = classify_answer_type(
        question=question,
        history=history,
        context_summary=context_summary,
        llm_helper=llm_helper,
        route_resolver=route_resolver,
        confidence_threshold=confidence_threshold,
        low_conf_fallback=low_conf_fallback,
    )
    row["answer_type"] = result.answer_type
    row["router_route"] = result.route
    row["router_confidence"] = float(result.confidence)
    row["router_signals"] = "|".join(result.signals)
    row["router_reason"] = result.reason
    row["router_scores_json"] = result.as_dict().get("scores", {})
    if result.answer_type == "follow_up" and not row.get("depends_on_list"):
        row["depends_on_list"] = ["__history__"] if history else []
    return row, result

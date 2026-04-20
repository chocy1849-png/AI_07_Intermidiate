from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


_AMBIGUOUS_TOKENS = ("약", "대략", "이상", "이하", "초과", "미만", "전후", "내외")


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _compact_text(value: str) -> str:
    text = _safe_text(value).lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[\"'`“”‘’\(\)\[\]\{\}<>]", "", text)
    return text


def _normalize_for_contains(value: str) -> str:
    text = _compact_text(value)
    text = re.sub(r"[.,:;!?/\\|·•\-_=+~]", "", text)
    return text


def _split_first_sentence(value: str) -> str:
    text = _safe_text(value)
    if not text:
        return ""
    for sep in ("\n", "。", ".", "!", "?"):
        idx = text.find(sep)
        if idx > 0:
            return text[:idx].strip()
    return text


def _looks_like_generic_long_form(value: str) -> bool:
    text = _safe_text(value)
    if len(text) < 520:
        return False
    markers = [
        "1.",
        "2.",
        "3.",
        "한줄 요약",
        "핵심 내용",
        "주요 요구사항",
        "일정/예산/발주기관",
        "참고 근거",
    ]
    hit = sum(1 for marker in markers if marker in text)
    return hit >= 2


def _normalize_phone(value: str) -> str:
    digits = re.sub(r"\D+", "", value or "")
    if digits.startswith("82") and len(digits) >= 11:
        digits = "0" + digits[2:]
    return digits


def _normalize_email(value: str) -> str:
    return _safe_text(value).lower().replace(" ", "")


def _normalize_url(value: str) -> str:
    text = _safe_text(value).lower()
    if not text:
        return ""
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", text):
        text = "https://" + text
    try:
        parsed = urlparse(text)
    except ValueError:
        return _compact_text(value)
    host = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    return f"{host}{path}"


def _normalize_date(value: str) -> str:
    text = _safe_text(value)
    text = text.replace("년", "-").replace("월", "-").replace("일", "")
    text = text.replace(".", "-").replace("/", "-")
    match = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", text)
    if not match:
        return ""
    year = int(match.group(1))
    month = int(match.group(2))
    day = int(match.group(3))
    if month < 1 or month > 12 or day < 1 or day > 31:
        return ""
    return f"{year:04d}-{month:02d}-{day:02d}"


def _normalize_duration(value: str) -> str:
    text = _safe_text(value).replace(" ", "")
    unit_map = {"달": "개월", "개월": "개월", "일": "일", "주": "주", "년": "년", "시간": "시간", "분": "분"}
    match = re.search(r"(\d+(?:\.\d+)?)\s*(일|개월|달|주|년|시간|분)", text)
    if not match:
        return ""
    number = float(match.group(1))
    number_text = str(int(number)) if number.is_integer() else str(number)
    unit = unit_map.get(match.group(2), match.group(2))
    if "계약일로부터" in text:
        return f"계약일로부터{number_text}{unit}"
    return f"{number_text}{unit}"


def _korean_amount_to_won(value: str) -> int | None:
    text = _safe_text(value).replace(",", "").replace(" ", "")
    if not text:
        return None
    if any(token in text for token in _AMBIGUOUS_TOKENS):
        return None

    total = 0.0
    consumed = False

    patterns = [
        (r"(\d+(?:\.\d+)?)억", 100_000_000),
        (r"(\d+(?:\.\d+)?)천만", 10_000_000),
        (r"(\d+(?:\.\d+)?)백만", 1_000_000),
        (r"(\d+(?:\.\d+)?)만", 10_000),
        (r"(\d+(?:\.\d+)?)천원", 1_000),
    ]
    for pattern, unit in patterns:
        for match in re.finditer(pattern, text):
            total += float(match.group(1)) * unit
            consumed = True

    if consumed:
        return int(round(total))

    if "원" in text or re.fullmatch(r"\d+(?:\.\d+)?", text):
        match = re.search(r"(\d+(?:\.\d+)?)", text)
        if match:
            return int(round(float(match.group(1))))
    return None


def _extract_first_number(value: str) -> float | None:
    text = _safe_text(value).replace(",", "")
    if any(token in text for token in _AMBIGUOUS_TOKENS):
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _split_list_items(value: str) -> list[str]:
    text = _safe_text(value)
    if not text:
        return []
    text = re.sub(r"[\r\n]+", "\n", text)
    text = re.sub(r"[•·]", ",", text)
    text = re.sub(r"\b및\b", ",", text)
    text = re.sub(r"\b그리고\b", ",", text)
    text = re.sub(r"\s*/\s*", ",", text)
    text = re.sub(r"\s*;\s*", ",", text)
    chunks = re.split(r"[,\n|]+", text)
    items: list[str] = []
    for chunk in chunks:
        candidate = re.sub(r"^\s*[-*]\s*", "", chunk).strip()
        candidate = re.sub(r"^\d+[.)]\s*", "", candidate).strip()
        if candidate:
            items.append(candidate)
    return items


def _parse_number_pair(value: str) -> tuple[float, ...]:
    nums = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", _safe_text(value).replace(",", ""))]
    return tuple(nums[:2])


def _choice_text(choice: str) -> str:
    return re.sub(r"^\s*\d+\s*[.)]\s*", "", _safe_text(choice))


def _is_negated_context(text: str, start: int, end: int) -> bool:
    window = text[max(0, start - 6) : min(len(text), end + 10)]
    return any(token in window for token in ("아닌", "아니", "말고", "제외", "아님"))


def _extract_choice_signal(answer: str, choices: list[str]) -> tuple[int | None, str, int]:
    text = _safe_text(answer)
    cleaned = text.lower()

    # numeric index: only allow explicit answer patterns (start-anchored or with answer cue)
    index_value: int | None = None
    match = re.search(r"^\s*(?:정답\s*[:：]?\s*|답\s*[:：]?\s*)?([1-9])\s*(?:번|\.|\)|$)", cleaned)
    if not match:
        match = re.search(r"(?:정답|답|보기)\s*[:：]?\s*([1-9])\s*(?:번|\.|\)|$)", cleaned)
    if match:
        index_value = int(match.group(1))
    else:
        match = re.match(r"^\s*([1-9])\s*$", cleaned)
        if match:
            index_value = int(match.group(1))

    choice_text_hits: list[tuple[int, str]] = []
    for idx, choice in enumerate(choices, start=1):
        candidate = _choice_text(choice)
        if not candidate:
            continue
        for found in re.finditer(re.escape(candidate), text):
            if _is_negated_context(text, found.start(), found.end()):
                continue
            choice_text_hits.append((idx, candidate))

    # de-duplicate hits by index
    seen_idx: set[int] = set()
    dedup_hits: list[tuple[int, str]] = []
    for idx, candidate in choice_text_hits:
        if idx in seen_idx:
            continue
        seen_idx.add(idx)
        dedup_hits.append((idx, candidate))

    choice_text = ""
    hit_count = len(dedup_hits)
    if hit_count == 1:
        idx, candidate = dedup_hits[0]
        choice_text = candidate
        if index_value is None:
            index_value = idx
    elif hit_count == 0 and index_value and 1 <= index_value <= len(choices):
        choice_text = _choice_text(choices[index_value - 1])

    if not choice_text:
        stripped = re.sub(r"^\s*\d+\s*[.)]\s*", "", text).strip()
        choice_text = stripped or text
    return index_value, choice_text, hit_count


@dataclass
class NormalizedQBankItem:
    item_id: str
    question: str
    question_type: str
    answer_format: str
    answer_eval_type: str
    canonical_answer: str
    answer_aliases: list[str]
    choice_index: int | None
    scoring_mode: str
    choices: list[str]
    difficulty: str
    category: str


@dataclass
class GradingResult:
    predicted_raw: str
    extracted_answer: str
    canonical_expected: str
    matched_mode: str
    strict_correct: bool
    tolerant_correct: bool
    partial_score: float
    error_reason: str

    def to_row(self) -> dict[str, Any]:
        return {
            "predicted_raw": self.predicted_raw,
            "extracted_answer": self.extracted_answer,
            "canonical_expected": self.canonical_expected,
            "matched_mode": self.matched_mode,
            "strict_correct": self.strict_correct,
            "tolerant_correct": self.tolerant_correct,
            "partial_score": self.partial_score,
            "error_reason": self.error_reason,
        }


def load_qbank_v4(path: Path) -> dict[str, NormalizedQBankItem]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    raw_items = payload.get("items", [])
    if not raw_items:
        raise RuntimeError(f"qbank v4 is empty: {path}")
    items: dict[str, NormalizedQBankItem] = {}
    for raw in raw_items:
        item_id = _safe_text(raw.get("id"))
        if not item_id:
            continue
        answer_eval_type = _safe_text(raw.get("answer_eval_type")).lower()
        if answer_eval_type == "list":
            answer_eval_type = "list_set"
        if answer_eval_type == "number_pair":
            answer_eval_type = "slot_pair"
        if answer_eval_type == "string":
            answer_eval_type = "free_string"
        aliases_raw = raw.get("answer_aliases") or []
        aliases = [_safe_text(x) for x in aliases_raw if _safe_text(x)]
        canonical_answer = _safe_text(raw.get("canonical_answer")) or _safe_text(raw.get("answer"))
        choice_index = raw.get("choice_index")
        if choice_index is not None:
            try:
                choice_index = int(choice_index)
            except (TypeError, ValueError):
                choice_index = None
        normalized = NormalizedQBankItem(
            item_id=item_id,
            question=_safe_text(raw.get("question")),
            question_type=_safe_text(raw.get("question_type")).lower(),
            answer_format=_safe_text(raw.get("answer_format")).lower(),
            answer_eval_type=answer_eval_type,
            canonical_answer=canonical_answer,
            answer_aliases=aliases,
            choice_index=choice_index,
            scoring_mode=_safe_text(raw.get("scoring_mode")).lower(),
            choices=[_safe_text(c) for c in (raw.get("choices") or []) if _safe_text(c)],
            difficulty=_safe_text(raw.get("difficulty")).lower(),
            category=_safe_text(raw.get("category")).lower(),
        )
        items[item_id] = normalized
    return items


def _expected_text_candidates(item: NormalizedQBankItem) -> list[str]:
    candidates = [item.canonical_answer, *item.answer_aliases]
    dedup: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = _compact_text(candidate)
        if key and key not in seen:
            seen.add(key)
            dedup.append(candidate)
    return dedup


def _is_close_float(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)


def grade_answer(model_answer: str, item: NormalizedQBankItem) -> GradingResult:
    raw = _safe_text(model_answer)
    canonical_expected = item.canonical_answer
    eval_type = item.answer_eval_type
    candidates = _expected_text_candidates(item)
    extracted = ""
    matched_mode = ""
    strict = False
    tolerant = False
    partial = 0.0
    error_reason = "wrong_value"

    if eval_type == "choice":
        idx, choice_text, hit_count = _extract_choice_signal(raw, item.choices)
        extracted = f"index={idx}, text={choice_text}" if idx else choice_text
        expected_idx = item.choice_index
        expected_text = ""
        if expected_idx and 1 <= expected_idx <= len(item.choices):
            expected_text = _choice_text(item.choices[expected_idx - 1])
        if not expected_text:
            expected_text = item.canonical_answer

        expected_text_norm = _normalize_for_contains(expected_text)
        choice_text_norm = _normalize_for_contains(choice_text)
        index_exact = bool(expected_idx and idx == expected_idx)
        text_exact = bool(expected_text_norm and choice_text_norm == expected_text_norm)
        strict = bool((index_exact or text_exact) and hit_count <= 1)
        if strict:
            matched_mode = "choice_exact"
            tolerant = True
        else:
            candidate_norms = {_normalize_for_contains(x) for x in candidates if x}
            tolerant = bool(choice_text_norm and choice_text_norm in candidate_norms and hit_count <= 1)
            if not tolerant and expected_idx and idx == expected_idx:
                tolerant = True
            if tolerant:
                matched_mode = "choice_alias_or_index"
                error_reason = "formatting_only"

    elif eval_type in {"currency", "number"}:
        if eval_type == "currency":
            predicted_value = _korean_amount_to_won(raw)
            expected_values = {_korean_amount_to_won(x) for x in candidates}
        else:
            predicted_value = _extract_first_number(raw)
            expected_values = {_extract_first_number(x) for x in candidates}
        expected_values = {x for x in expected_values if x is not None}
        extracted = "" if predicted_value is None else str(predicted_value)
        strict = False
        if predicted_value is not None:
            for expected in expected_values:
                if isinstance(predicted_value, float) or isinstance(expected, float):
                    if _is_close_float(float(predicted_value), float(expected)):
                        strict = True
                        break
                elif predicted_value == expected:
                    strict = True
                    break
        if strict:
            matched_mode = "normalized_numeric_exact"
            tolerant = True
        else:
            full_norm = _normalize_for_contains(raw)
            tolerant = any(_normalize_for_contains(candidate) in full_norm for candidate in candidates if candidate)
            if tolerant:
                matched_mode = "numeric_alias_contains"
                error_reason = "formatting_only"
            elif predicted_value is None:
                error_reason = "extraction_fail"

    elif eval_type == "date":
        predicted_date = _normalize_date(raw)
        expected_dates = {_normalize_date(x) for x in candidates}
        expected_dates = {x for x in expected_dates if x}
        extracted = predicted_date
        strict = bool(predicted_date and predicted_date in expected_dates)
        if strict:
            matched_mode = "date_normalized_exact"
            tolerant = True
        else:
            tolerant = any(_normalize_for_contains(candidate) in _normalize_for_contains(raw) for candidate in candidates if candidate)
            if tolerant:
                matched_mode = "date_contains"
                error_reason = "formatting_only"
            elif not predicted_date:
                error_reason = "extraction_fail"

    elif eval_type == "duration":
        predicted_duration = _normalize_duration(raw)
        expected_durations = {_normalize_duration(x) for x in candidates}
        expected_durations = {x for x in expected_durations if x}
        extracted = predicted_duration
        strict = bool(predicted_duration and predicted_duration in expected_durations)
        if strict:
            matched_mode = "duration_normalized_exact"
            tolerant = True
        else:
            if predicted_duration and predicted_duration.startswith("계약일로부터"):
                stripped = predicted_duration.replace("계약일로부터", "", 1)
                tolerant = stripped in {x.replace("계약일로부터", "", 1) for x in expected_durations}
            else:
                tolerant = any(_normalize_for_contains(candidate) in _normalize_for_contains(raw) for candidate in candidates if candidate)
            if tolerant:
                matched_mode = "duration_tolerant"
                error_reason = "formatting_only"
            elif not predicted_duration:
                error_reason = "extraction_fail"

    elif eval_type == "email":
        predicted_email = ""
        match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", raw)
        if match:
            predicted_email = _normalize_email(match.group(0))
        extracted = predicted_email
        expected_emails = {_normalize_email(x) for x in candidates if x}
        strict = bool(predicted_email and predicted_email in expected_emails)
        tolerant = strict or any(expected in _normalize_email(raw) for expected in expected_emails if expected)
        if strict:
            matched_mode = "email_exact"
        elif tolerant:
            matched_mode = "email_contains"
            error_reason = "formatting_only"
        else:
            error_reason = "extraction_fail" if not predicted_email else "wrong_value"

    elif eval_type == "phone":
        digits = _normalize_phone(raw)
        extracted = digits
        expected_phones = {_normalize_phone(x) for x in candidates if x}
        strict = bool(digits and digits in expected_phones)
        tolerant = strict or any(expected and expected in digits for expected in expected_phones)
        if strict:
            matched_mode = "phone_exact"
        elif tolerant:
            matched_mode = "phone_contains"
            error_reason = "formatting_only"
        else:
            error_reason = "extraction_fail" if not digits else "wrong_value"

    elif eval_type == "url":
        match = re.search(r"(https?://[^\s]+|www\.[^\s]+)", raw, flags=re.IGNORECASE)
        predicted_url = _normalize_url(match.group(0) if match else raw)
        extracted = predicted_url
        expected_urls = {_normalize_url(x) for x in candidates if x}
        strict = bool(predicted_url and predicted_url in expected_urls)
        tolerant = strict or any(expected and expected in predicted_url for expected in expected_urls)
        if strict:
            matched_mode = "url_exact"
        elif tolerant:
            matched_mode = "url_contains"
            error_reason = "formatting_only"
        else:
            error_reason = "extraction_fail" if not match else "wrong_value"

    elif eval_type in {"list_set", "slot_pair"}:
        if eval_type == "slot_pair":
            expected_pair = _parse_number_pair(item.canonical_answer)
            if len(expected_pair) < 2:
                for alias in item.answer_aliases:
                    alias_pair = _parse_number_pair(alias)
                    if len(alias_pair) >= 2:
                        expected_pair = alias_pair
                        break
            predicted_pair = _parse_number_pair(raw)
            extracted = ",".join(str(x) for x in predicted_pair)
            strict = len(expected_pair) >= 2 and len(predicted_pair) >= 2 and _is_close_float(predicted_pair[0], expected_pair[0]) and _is_close_float(predicted_pair[1], expected_pair[1])
            if strict:
                tolerant = True
                partial = 1.0
                matched_mode = "slot_pair_exact"
            else:
                if len(expected_pair) >= 2 and len(predicted_pair) >= 2:
                    expected_set = {expected_pair[0], expected_pair[1]}
                    predicted_set = {predicted_pair[0], predicted_pair[1]}
                    hit = sum(1 for x in expected_set if any(_is_close_float(x, y) for y in predicted_set))
                    partial = round(hit / 2.0, 4)
                    tolerant = hit == 2
                    if tolerant:
                        matched_mode = "slot_pair_unordered_match"
                        error_reason = "formatting_only"
                    elif partial > 0:
                        error_reason = "partial_list"
                else:
                    error_reason = "extraction_fail"
        else:
            expected_items = _split_list_items(item.canonical_answer)
            if len(expected_items) <= 1:
                for alias in item.answer_aliases:
                    alt = _split_list_items(alias)
                    if len(alt) > len(expected_items):
                        expected_items = alt
            expected_norm = {_normalize_for_contains(x) for x in expected_items if x}
            predicted_items = _split_list_items(raw)
            predicted_norm = {_normalize_for_contains(x) for x in predicted_items if x}
            extracted = ", ".join(predicted_items)
            if expected_norm:
                hit = len(expected_norm.intersection(predicted_norm))
                partial = round(hit / max(1, len(expected_norm)), 4)
                strict = hit == len(expected_norm) and len(predicted_norm) == len(expected_norm)
                tolerant = hit == len(expected_norm)
                if strict:
                    matched_mode = "list_set_exact"
                elif tolerant:
                    matched_mode = "list_set_superset_ok"
                    error_reason = "extra_explanation"
                elif partial > 0:
                    error_reason = "partial_list"
                else:
                    error_reason = "wrong_value"
            else:
                error_reason = "extraction_fail"

    else:
        first_sentence = _split_first_sentence(raw)
        extracted = first_sentence
        expected_norms = {_normalize_for_contains(x) for x in candidates if x}
        sentence_norm = _normalize_for_contains(first_sentence)
        full_norm = _normalize_for_contains(raw)
        strict = bool(sentence_norm and sentence_norm in expected_norms)
        if strict:
            matched_mode = "string_first_sentence_exact"
            tolerant = True
        else:
            tolerant = any(expected and expected in full_norm for expected in expected_norms)
            if tolerant:
                matched_mode = "string_contains_alias"
                error_reason = "extra_explanation"
            else:
                error_reason = "wrong_value" if first_sentence else "extraction_fail"

    # scoring_mode-aware guard: for normalized_exact targets, overly long boilerplate
    # should not get tolerant pass unless strict extraction already succeeded.
    if (
        not strict
        and tolerant
        and item.scoring_mode == "normalized_exact"
        and item.answer_eval_type in {"free_string", "number", "currency", "date", "duration", "email", "phone", "url"}
        and _looks_like_generic_long_form(raw)
    ):
        tolerant = False
        partial = 0.0
        matched_mode = "verbose_guard_reject"
        error_reason = "extra_explanation"

    if strict:
        partial = 1.0
        error_reason = "ok"
    elif tolerant and partial <= 0:
        partial = 1.0
    if tolerant and error_reason == "wrong_value":
        error_reason = "formatting_only"
    if not tolerant and partial == 0 and not extracted:
        error_reason = "extraction_fail"

    return GradingResult(
        predicted_raw=raw,
        extracted_answer=extracted,
        canonical_expected=canonical_expected,
        matched_mode=matched_mode or ("no_match" if not tolerant else "tolerant_match"),
        strict_correct=bool(strict),
        tolerant_correct=bool(tolerant),
        partial_score=float(partial),
        error_reason=error_reason,
    )

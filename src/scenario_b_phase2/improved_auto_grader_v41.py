from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


_AMBIGUOUS_TOKENS = ("약", "대략", "이상", "이하", "초과", "미만", "전후", "내외")
_HEADING_TOKENS = ("한줄 요약", "핵심 내용", "주요 요구사항", "일정/예산/발주기관", "참고 근거")


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _compact_text(value: str) -> str:
    text = _safe_text(value).lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[\"'`“”‘’\(\)\[\]\{\}<>]", "", text)
    return text


def _normalize_for_compare(value: str) -> str:
    text = _compact_text(value)
    text = re.sub(r"[.,:;!?/\\|·•\-_=+~]", "", text)
    return text


def _remove_structural_heading_lines(answer: str) -> str:
    lines = [line.strip() for line in _safe_text(answer).splitlines()]
    kept: list[str] = []
    for line in lines:
        if not line:
            continue
        if any(token in line for token in _HEADING_TOKENS):
            continue
        if re.match(r"^\s*\d+\.\s*(?:[가-힣A-Za-z ]{0,20})?$", line):
            continue
        if re.match(r"^\s*\[(질문|응답|참고|검색 결과)\]", line):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def _split_sentences(text: str) -> list[str]:
    if not text:
        return []
    normalized = re.sub(r"[\r\n]+", ". ", text)
    chunks = re.split(r"[.!?。]\s*", normalized)
    out: list[str] = []
    for chunk in chunks:
        c = chunk.strip()
        if c:
            out.append(c)
    return out


def _question_mode(question: str) -> str:
    q = _safe_text(question)
    if re.search(r"(각각|비율|정량|정성|기술.*가격|가격.*기술)", q):
        return "slot_pair"
    if re.search(r"(목록|국가|앱|기능|내역|현황|항목|종류)", q):
        return "list_set"
    if re.search(r"(몇\s*(개|명|부|건|곳|종|월|일|년)|얼마|최대|총)", q):
        return "numeric_count"
    if re.search(r"(무엇|어디|누구|어느|어떤)", q):
        return "noun_phrase"
    return "general"


def _extract_question_keywords(question: str) -> list[str]:
    tokens = re.findall(r"[가-힣A-Za-z0-9]{2,}", _safe_text(question))
    stop = {
        "해당",
        "사업",
        "무엇",
        "어디",
        "누구",
        "어느",
        "어떤",
        "인가",
        "인가요",
        "얼마",
        "몇개",
        "몇명",
        "각각",
        "목록",
    }
    out: list[str] = []
    for token in tokens:
        key = token.lower().replace(" ", "")
        if key in stop:
            continue
        out.append(token.lower())
    return out[:8]


def _score_sentence(sentence: str, keywords: list[str]) -> float:
    s = sentence.lower()
    overlap = sum(1 for key in keywords if key in s)
    cue_bonus = 0.0
    if re.search(r"(정답|은|는|이|가).*(입니다|이다|임)", sentence):
        cue_bonus += 1.5
    if re.search(r"(총|최대|사업기간|예산|발주기관|제출|평가|계약방식)", sentence):
        cue_bonus += 1.0
    length_penalty = 0.0
    if len(sentence) > 180:
        length_penalty = -0.6
    return float(overlap) + cue_bonus + length_penalty


def _select_relevant_text(question: str, answer: str, max_sentences: int = 3) -> str:
    cleaned = _remove_structural_heading_lines(answer)
    if not cleaned:
        cleaned = _safe_text(answer)
    keywords = _extract_question_keywords(question)
    sentences = _split_sentences(cleaned)
    if not sentences:
        return cleaned
    ranked = sorted(sentences, key=lambda x: _score_sentence(x, keywords), reverse=True)
    return ". ".join(ranked[:max_sentences]).strip()


def _mine_answer_phrase(question: str, answer: str, prefer_numeric: bool = False) -> str:
    cleaned = _remove_structural_heading_lines(answer)
    if not cleaned:
        cleaned = _safe_text(answer)

    mode = _question_mode(question)
    keywords = _extract_question_keywords(question)
    sentences = _split_sentences(cleaned)

    # cue first
    cue_patterns = [
        r"정답(?:은|:)\s*([^.\n]+)",
        r"(?:사업기간|기간|예산|총 예산|발주기관|제출 장소|평가방식)\s*(?:은|는|이|가)?\s*([^.\n]+?)\s*(?:입니다|이다|임)",
        r"(?:최대|총)\s*([^.\n]+?)\s*(?:입니다|이다|임)",
    ]
    for pattern in cue_patterns:
        match = re.search(pattern, cleaned)
        if match:
            phrase = _safe_text(match.group(1))
            if phrase:
                return phrase

    if not sentences:
        return cleaned

    # choose best sentence by question overlap
    ranked = sorted(sentences, key=lambda x: _score_sentence(x, keywords), reverse=True)
    best = ranked[0]

    if mode == "numeric_count" or prefer_numeric:
        number_match = re.search(r"([-+]?\d+(?:[.,]\d+)?)\s*(개|명|부|건|곳|종|월|일|년|회|%)?", best)
        if number_match:
            num = number_match.group(1).replace(",", "")
            unit = number_match.group(2) or ""
            return f"{num}{unit}".strip()

    if mode == "noun_phrase":
        noun_match = re.search(r"(?:은|는|이|가)\s*([^,.;\n]{2,60}?)(?:입니다|이다|임|$)", best)
        if noun_match:
            return _safe_text(noun_match.group(1))

    generic = re.search(r"(?:은|는|이|가)\s*([^,.;\n]{2,80}?)(?:입니다|이다|임|$)", best)
    if generic:
        return _safe_text(generic.group(1))
    return _safe_text(best)


def _candidate_in_text(candidates: list[str], text: str) -> bool:
    target = _normalize_for_compare(text)
    if not target:
        return False
    for candidate in candidates:
        norm = _normalize_for_compare(candidate)
        if norm and norm in target:
            return True
    return False


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
    y, m, d = int(match.group(1)), int(match.group(2)), int(match.group(3))
    if not (1 <= m <= 12 and 1 <= d <= 31):
        return ""
    return f"{y:04d}-{m:02d}-{d:02d}"


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


def _choice_text(choice: str) -> str:
    return re.sub(r"^\s*\d+\s*[.)]\s*", "", _safe_text(choice))


def _is_negated_context(text: str, start: int, end: int) -> bool:
    window = text[max(0, start - 6) : min(len(text), end + 10)]
    return any(token in window for token in ("아닌", "아니", "말고", "제외", "아님"))


def _extract_choice_signal(answer: str, choices: list[str]) -> tuple[int | None, str, int]:
    text = _safe_text(answer)
    cleaned = text.lower()

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

    hits: list[tuple[int, str]] = []
    for idx, choice in enumerate(choices, start=1):
        candidate = _choice_text(choice)
        if not candidate:
            continue
        for found in re.finditer(re.escape(candidate), text):
            if _is_negated_context(text, found.start(), found.end()):
                continue
            hits.append((idx, candidate))

    seen_idx: set[int] = set()
    dedup: list[tuple[int, str]] = []
    for idx, candidate in hits:
        if idx in seen_idx:
            continue
        seen_idx.add(idx)
        dedup.append((idx, candidate))

    choice_text = ""
    hit_count = len(dedup)
    if hit_count == 1:
        idx, candidate = dedup[0]
        choice_text = candidate
        if index_value is None:
            index_value = idx
    elif hit_count == 0 and index_value and 1 <= index_value <= len(choices):
        choice_text = _choice_text(choices[index_value - 1])

    if not choice_text:
        stripped = re.sub(r"^\s*\d+\s*[.)]\s*", "", text).strip()
        choice_text = stripped or text
    return index_value, choice_text, hit_count


def _normalize_list_item(value: str) -> str:
    item = _safe_text(value)
    item = re.sub(r"^\s*[-*]\s*", "", item)
    item = re.sub(r"^\s*\d+[.)]\s*", "", item)
    item = re.sub(r"^(해당|본)?\s*(사업\s*)?(대상\s*)?(국가|지역|항목|목록|내역|기능)\s*(은|는|이|가|:)\s*", "", item)
    item = re.sub(r"\s*(입니다|이다|임)\s*$", "", item)
    item = re.sub(r"\(.*?\)", "", item).strip()
    base = _normalize_for_compare(item)
    alias_map = {
        "키르기즈스탄": "키르기스스탄",
        "키르기스": "키르기스스탄",
        "아이오에스": "ios",
        "ios": "ios",
        "안드로이드": "android",
        "android": "android",
        "파워포인트": "powerpoint",
        "ppt": "powerpoint",
        "한글": "hwp",
        "hwp": "hwp",
    }
    if base in alias_map:
        return alias_map[base]
    return base


def _split_list_items(value: str) -> list[str]:
    text = _safe_text(value)
    if not text:
        return []
    text = re.sub(r"[\r\n]+", "\n", text)
    text = re.sub(r"[•·]", ",", text)
    text = re.sub(r"\b및\b", ",", text)
    text = re.sub(r"\b그리고\b", ",", text)
    text = re.sub(r"\s*/\s*", ",", text)
    chunks = re.split(r"[,\n|;]+", text)
    items: list[str] = []
    for chunk in chunks:
        # If list candidates are space-separated without clear delimiters, split by location-like nouns.
        if "," not in chunk and "\n" not in chunk and len(chunk) > 18:
            entities = re.findall(r"[가-힣]{2,}(?:시|군|구|도|읍|면|리|국|권)", chunk)
            if len(entities) >= 2:
                for ent in entities:
                    normalized = _normalize_list_item(ent)
                    if normalized:
                        items.append(normalized)
                continue
        normalized = _normalize_list_item(chunk)
        if normalized:
            items.append(normalized)
    return items


def _extract_slot_pair(question: str, answer: str) -> tuple[tuple[float, float] | None, str]:
    text = _remove_structural_heading_lines(answer)
    if not text:
        text = _safe_text(answer)
    # remove heading-like numbers
    text = re.sub(r"(?m)^\s*\d+\.\s*", "", text)

    q = _safe_text(question)
    labels: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
        ("정량/정성", ("정량", "정량평가"), ("정성", "정성평가")),
        ("기술/가격", ("기술", "기술평가"), ("가격", "가격평가")),
    ]
    selected = None
    for name, left_labels, right_labels in labels:
        if any(label in q for label in left_labels + right_labels):
            selected = (name, left_labels, right_labels)
            break
    if selected is None:
        if "각각" in q:
            selected = ("각각", ("첫", "전"), ("두", "후"))
        else:
            selected = ("generic", ("",), ("",))

    _, left_labels, right_labels = selected

    def find_value(label_group: tuple[str, ...]) -> float | None:
        if label_group == ("",):
            return None
        for label in label_group:
            pattern = rf"{re.escape(label)}[^\d\n]{{0,10}}([-+]?\d+(?:\.\d+)?)"
            match = re.search(pattern, text)
            if match:
                return float(match.group(1))
        return None

    left = find_value(left_labels)
    right = find_value(right_labels)
    if left is not None and right is not None:
        return (left, right), "label_aware_pair"

    # fallback: explicit pair pattern only (avoid noisy headings)
    pair_match = re.search(
        r"([-+]?\d+(?:\.\d+)?)\s*(?:점|%|원|일|개|명|부|건|곳|종)?\s*[/,]\s*([-+]?\d+(?:\.\d+)?)",
        text,
    )
    if pair_match:
        return (float(pair_match.group(1)), float(pair_match.group(2))), "explicit_pair_pattern"

    return None, "slot_pair_extract_fail"


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
        aliases = [_safe_text(x) for x in (raw.get("answer_aliases") or []) if _safe_text(x)]
        canonical_answer = _safe_text(raw.get("canonical_answer")) or _safe_text(raw.get("answer"))
        choice_index = raw.get("choice_index")
        if choice_index is not None:
            try:
                choice_index = int(choice_index)
            except (TypeError, ValueError):
                choice_index = None
        items[item_id] = NormalizedQBankItem(
            item_id=item_id,
            question=_safe_text(raw.get("question")),
            question_type=_safe_text(raw.get("question_type")).lower(),
            answer_format=_safe_text(raw.get("answer_format")).lower(),
            answer_eval_type=answer_eval_type,
            canonical_answer=canonical_answer,
            answer_aliases=aliases,
            choice_index=choice_index,
            scoring_mode=_safe_text(raw.get("scoring_mode")).lower(),
            choices=[_safe_text(x) for x in (raw.get("choices") or []) if _safe_text(x)],
            difficulty=_safe_text(raw.get("difficulty")).lower(),
            category=_safe_text(raw.get("category")).lower(),
        )
    return items


def _expected_candidates(item: NormalizedQBankItem) -> list[str]:
    values = [item.canonical_answer, *item.answer_aliases]
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        norm = _normalize_for_compare(value)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(value)
    return out


def _is_close_float(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)


def grade_answer(model_answer: str, item: NormalizedQBankItem) -> GradingResult:
    raw = _safe_text(model_answer)
    eval_type = item.answer_eval_type
    candidates = _expected_candidates(item)
    canonical_expected = item.canonical_answer
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
        expected_text = _choice_text(item.choices[expected_idx - 1]) if expected_idx and 1 <= expected_idx <= len(item.choices) else item.canonical_answer
        expected_norm = _normalize_for_compare(expected_text)
        choice_norm = _normalize_for_compare(choice_text)
        index_exact = bool(expected_idx and idx == expected_idx)
        text_exact = bool(expected_norm and expected_norm == choice_norm)
        strict = bool((index_exact or text_exact) and hit_count <= 1)
        if strict:
            tolerant = True
            matched_mode = "choice_exact"
        else:
            candidate_norms = {_normalize_for_compare(x) for x in candidates if x}
            tolerant = bool(choice_norm and choice_norm in candidate_norms and hit_count <= 1)
            if not tolerant and index_exact:
                tolerant = True
            if tolerant:
                matched_mode = "choice_alias_or_index"
                error_reason = "formatting_only"

    elif eval_type in {"currency", "number"}:
        mined = _mine_answer_phrase(item.question, raw, prefer_numeric=True)
        if eval_type == "currency":
            predicted = _korean_amount_to_won(mined or raw)
            expected_values = {_korean_amount_to_won(x) for x in candidates}
        else:
            predicted = _extract_first_number(mined or raw)
            expected_values = {_extract_first_number(x) for x in candidates}
        expected_values = {x for x in expected_values if x is not None}
        extracted = "" if predicted is None else str(predicted)
        if predicted is not None:
            for expected in expected_values:
                if isinstance(predicted, float) or isinstance(expected, float):
                    if _is_close_float(float(predicted), float(expected)):
                        strict = True
                        break
                elif predicted == expected:
                    strict = True
                    break
        if strict:
            tolerant = True
            matched_mode = "numeric_exact"
        else:
            tolerant = _candidate_in_text(candidates, mined)
            if tolerant:
                matched_mode = "numeric_mined_contains"
                error_reason = "formatting_only"
            elif predicted is None:
                error_reason = "extraction_fail"

    elif eval_type == "date":
        mined = _mine_answer_phrase(item.question, raw, prefer_numeric=True)
        predicted = _normalize_date(mined or raw)
        expected = {_normalize_date(x) for x in candidates}
        expected = {x for x in expected if x}
        extracted = predicted
        strict = bool(predicted and predicted in expected)
        if strict:
            tolerant = True
            matched_mode = "date_exact"
        else:
            tolerant = _candidate_in_text(candidates, mined)
            if tolerant:
                matched_mode = "date_mined_contains"
                error_reason = "formatting_only"
            elif not predicted:
                error_reason = "extraction_fail"

    elif eval_type == "duration":
        mined = _mine_answer_phrase(item.question, raw, prefer_numeric=True)
        predicted = _normalize_duration(mined or raw)
        if not predicted and mined:
            predicted = _normalize_duration(raw)
        expected = {_normalize_duration(x) for x in candidates}
        expected = {x for x in expected if x}
        extracted = predicted
        strict = bool(predicted and predicted in expected)
        if strict:
            tolerant = True
            matched_mode = "duration_exact"
        else:
            if predicted and predicted.startswith("계약일로부터"):
                stripped = predicted.replace("계약일로부터", "", 1)
                tolerant = stripped in {x.replace("계약일로부터", "", 1) for x in expected}
            else:
                tolerant = _candidate_in_text(candidates, mined)
            if tolerant:
                matched_mode = "duration_tolerant"
                error_reason = "formatting_only"
            elif not predicted:
                error_reason = "extraction_fail"

    elif eval_type == "email":
        match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", raw)
        predicted = _normalize_email(match.group(0) if match else "")
        expected = {_normalize_email(x) for x in candidates if x}
        extracted = predicted
        strict = bool(predicted and predicted in expected)
        tolerant = strict or any(x and x in _normalize_email(raw) for x in expected)
        if strict:
            matched_mode = "email_exact"
        elif tolerant:
            matched_mode = "email_contains"
            error_reason = "formatting_only"
        else:
            error_reason = "extraction_fail" if not predicted else "wrong_value"

    elif eval_type == "phone":
        predicted = _normalize_phone(raw)
        expected = {_normalize_phone(x) for x in candidates if x}
        extracted = predicted
        strict = bool(predicted and predicted in expected)
        tolerant = strict or any(x and x in predicted for x in expected)
        if strict:
            matched_mode = "phone_exact"
        elif tolerant:
            matched_mode = "phone_contains"
            error_reason = "formatting_only"
        else:
            error_reason = "extraction_fail" if not predicted else "wrong_value"

    elif eval_type == "url":
        match = re.search(r"(https?://[^\s]+|www\.[^\s]+)", raw, flags=re.IGNORECASE)
        predicted = _normalize_url(match.group(0) if match else raw)
        expected = {_normalize_url(x) for x in candidates if x}
        extracted = predicted
        strict = bool(predicted and predicted in expected)
        tolerant = strict or any(x and x in predicted for x in expected)
        if strict:
            matched_mode = "url_exact"
        elif tolerant:
            matched_mode = "url_contains"
            error_reason = "formatting_only"
        else:
            error_reason = "extraction_fail" if not match else "wrong_value"

    elif eval_type == "slot_pair":
        expected_pair = None
        for cand in [item.canonical_answer, *item.answer_aliases]:
            nums = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", _safe_text(cand).replace(",", ""))]
            if len(nums) >= 2:
                expected_pair = (nums[0], nums[1])
                break
        predicted_pair, pair_mode = _extract_slot_pair(item.question, raw)
        extracted = "" if not predicted_pair else f"{predicted_pair[0]},{predicted_pair[1]}"
        if expected_pair and predicted_pair:
            strict = _is_close_float(predicted_pair[0], expected_pair[0]) and _is_close_float(predicted_pair[1], expected_pair[1])
            tolerant = strict
            if strict:
                matched_mode = f"slot_pair_{pair_mode}"
            else:
                # unordered fallback
                expected_set = {expected_pair[0], expected_pair[1]}
                predicted_set = {predicted_pair[0], predicted_pair[1]}
                hit = sum(1 for x in expected_set if any(_is_close_float(x, y) for y in predicted_set))
                partial = round(hit / 2.0, 4)
                tolerant = hit == 2
                if tolerant:
                    matched_mode = f"slot_pair_unordered_{pair_mode}"
                    error_reason = "formatting_only"
                elif partial > 0:
                    error_reason = "partial_list"
                else:
                    error_reason = "wrong_value"
        else:
            error_reason = "extraction_fail"
            matched_mode = pair_mode

    elif eval_type == "list_set":
        expected_items = _split_list_items(item.canonical_answer)
        if len(expected_items) <= 1:
            for alias in item.answer_aliases:
                alt = _split_list_items(alias)
                if len(alt) > len(expected_items):
                    expected_items = alt
        relevant_text = _select_relevant_text(item.question, raw, max_sentences=3)
        predicted_items = _split_list_items(relevant_text if relevant_text else raw)
        expected_set = {x for x in expected_items if x}
        predicted_set = {x for x in predicted_items if x}
        extracted = ", ".join(sorted(predicted_set))
        if expected_set:
            hit = len(expected_set.intersection(predicted_set))
            partial = round(hit / max(1, len(expected_set)), 4)
            strict = hit == len(expected_set) and len(predicted_set) == len(expected_set)
            tolerant = hit == len(expected_set)
            if strict:
                matched_mode = "list_set_exact"
            elif tolerant:
                matched_mode = "list_set_coverage_ok"
                error_reason = "extra_explanation"
            elif partial > 0:
                error_reason = "partial_list"
            else:
                error_reason = "wrong_value"
        else:
            error_reason = "extraction_fail"

    else:
        phrase = _mine_answer_phrase(item.question, raw, prefer_numeric=False)
        relevant_text = _select_relevant_text(item.question, raw, max_sentences=2)
        extracted = phrase
        phrase_norm = _normalize_for_compare(phrase)
        candidate_norms = {_normalize_for_compare(x) for x in candidates if x}
        strict = bool(phrase_norm and phrase_norm in candidate_norms)
        if strict:
            tolerant = True
            matched_mode = "free_string_phrase_exact"
        else:
            tolerant = _candidate_in_text(candidates, phrase + "\n" + relevant_text)
            if tolerant:
                matched_mode = "free_string_mined_contains"
                error_reason = "extra_explanation"
            else:
                error_reason = "extraction_fail" if not phrase else "wrong_value"

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

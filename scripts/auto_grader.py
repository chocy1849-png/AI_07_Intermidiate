"""
자동평가 채점 엔진 프로토타입
- 윤민님 문제은행(JSON) 로드
- RAG 파이프라인 호출 → 모델 응답 생성
- 정답 비교 → 채점 (공백제거 + 객관식매핑 + 숫자정규화)
- 결과 CSV + 집계 리포트 출력

협의 완료 (4/14):
  - 공백: 채점 시 제거
  - 객관식: 텍스트 유도 + 번호 폴백 매핑
  - 숫자: 정확한 금액 표현 차이 허용, "약/대략" 오답
  - 베이스: 04_베이스라인_평가.py 재량껏 수정
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import warnings
from pathlib import Path
from typing import Any


# ============================================================
# 1. JSON 로더
# ============================================================

def load_question_bank(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    items = data.get("items", [])
    if not items:
        raise RuntimeError(f"문항이 비어있습니다: {path}")
    validate_question_bank(items, path=path)
    print(f"[로드] {len(items)}문항 로드 완료 (v={data.get('version', '?')})")
    return items


def validate_question_bank(items: list[dict[str, Any]], path: Path) -> None:
    """문제은행 구조 경고 출력.
    치명적 오류로 중단하지는 않지만, 채점 결과 왜곡 가능성이 큰 항목을 알려준다."""
    warnings_found: list[str] = []

    for item in items:
        qid = str(item.get("id", "<unknown>"))
        qtype = item.get("question_type")

        if qtype == "multiple_choice":
            choices = item.get("choices")
            if not isinstance(choices, list) or not choices:
                warnings_found.append(
                    f"{qid}: multiple_choice인데 choices 필드가 없거나 비어 있습니다. "
                    "현재 채점에서는 번호 매핑이 불가능합니다."
                )
            else:
                choice_texts = [_choice_text(str(choice)) for choice in choices]
                if str(item.get("answer", "")).strip() not in choice_texts and str(item.get("answer", "")).strip() not in [str(c).strip() for c in choices]:
                    warnings_found.append(
                        f"{qid}: answer가 choices 텍스트와 정확히 일치하지 않습니다. "
                        "객관식 정합성 확인이 필요합니다."
                    )

        document_id = str(item.get("document_id", "")).strip()
        if qid and document_id and not qid.startswith(f"{document_id}_"):
            warnings_found.append(
                f"{qid}: id와 document_id({document_id})가 불일치합니다."
            )

    if warnings_found:
        print(f"[경고] 문제은행 구조 이슈 {len(warnings_found)}건 발견: {path}")
        for message in warnings_found[:20]:
            print(f"  - {message}")
        if len(warnings_found) > 20:
            print(f"  ... 외 {len(warnings_found) - 20}건")
        for message in warnings_found:
            warnings.warn(message, stacklevel=2)


# ============================================================
# 2. 채점 함수
# ============================================================

def strip_spaces(text: str) -> str:
    """양쪽 strip + 모든 공백(스페이스, 탭, 개행) 제거"""
    return re.sub(r"\s+", "", text)


def _parse_korean_sub(s: str) -> int:
    """한국어 하위 단위 파싱. '5천2백' → 5200, '5200' → 5200, '7천' → 7000"""
    if s.isdigit():
        return int(s)
    value = 0
    천_m = re.search(r"(\d+)천", s)
    백_m = re.search(r"(\d+)백", s)
    십_m = re.search(r"(\d+)십", s)
    if 천_m:
        value += int(천_m.group(1)) * 1000
    if 백_m:
        value += int(백_m.group(1)) * 100
    if 십_m:
        value += int(십_m.group(1)) * 10
    return value


def _extract_money(text: str) -> str | None:
    """문장에서 금액 부분만 추출. '총 예산은 352,000,000원입니다.' → '352,000,000원'"""
    # "N억N천만원" 패턴
    m = re.search(r"[\d.]+억[^\s,.]*?원", text)
    if m:
        return m.group(0)
    # "N천만원" 패턴
    m = re.search(r"[\d천백십]+만[^\s,]*?원", text)
    if m:
        return m.group(0)
    # "000,000,000원" 패턴
    m = re.search(r"[\d,]+원", text)
    if m:
        return m.group(0)
    return None


def normalize_number(text: str) -> int | None:
    """
    금액 문자열을 정수로 정규화.
    "352,000,000원" → 352000000
    "3억 5,200만원" → 352000000
    "3억5천2백만원" → 352000000
    "1억5천만원"    → 150000000
    "9.8억원"       → 980000000
    "약 3억" → None (약/대략 포함 시 거부 → 오답)
    """
    if re.search(r"약|대략|대충|정도|추정|이상|이하|내외|최대|최소|미만|초과|전후", text):
        return None

    # 문장에서 금액 부분 추출 시도
    extracted = _extract_money(text)
    if extracted:
        text = extracted

    cleaned = text.replace(",", "").replace(" ", "").replace("원", "")

    # 순수 숫자인 경우
    if cleaned.isdigit():
        return int(cleaned)

    total = 0

    # 소수점 억 단위: "9.8억" → 9.8 * 1억
    소수억_match = re.search(r"(\d+\.\d+)억", cleaned)
    억_match = re.search(r"(\d+)억", cleaned)

    if 소수억_match:
        total += int(float(소수억_match.group(1)) * 100_000_000)
        after_억 = cleaned[소수억_match.end():]
    elif 억_match:
        total += int(억_match.group(1)) * 100_000_000
        after_억 = cleaned[억_match.end():]
    else:
        after_억 = cleaned

    # 만 단위: "5천2백만", "5200만", "7천만" 등 복합 표현 처리
    만_match = re.search(r"([\d천백십]+)만", after_억)
    if 만_match:
        만_value = _parse_korean_sub(만_match.group(1))
        total += 만_value * 10_000

    # "N,N천원" 또는 "N천원" 패턴 (억/만 없이 천 단위만 있는 경우)
    if total == 0:
        천_match = re.search(r"([\d,]+)천", cleaned)
        if 천_match:
            천_val = int(천_match.group(1).replace(",", ""))
            total = 천_val * 1_000

    return total if total > 0 else None


def normalize_date(text: str) -> str | None:
    """
    날짜 문자열을 YYYY-MM-DD로 정규화.
    "2024-07-10" → "2024-07-10"
    "2024.07.10" → "2024-07-10"
    "2024년 7월 10일" → "2024-07-10"
    """
    cleaned = text.strip().replace(" ", "")

    # YYYY-MM-DD 또는 YYYY.MM.DD
    m = re.search(r"(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})", cleaned)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

    # YYYY년 M월 D일
    m = re.search(r"(\d{4})년(\d{1,2})월(\d{1,2})일", cleaned)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

    return None


def _choice_text(choice: str) -> str:
    """'1. 일반경쟁입찰' → '일반경쟁입찰'"""
    return re.sub(r"^\d+\.\s*", "", choice).strip()


def map_choice_number(model_answer: str, choices: list[str]) -> str:
    """
    객관식 번호 답변 → 보기 텍스트 매핑
    "2번" → choices[1]의 텍스트
    "2. 제한경쟁입찰" → "제한경쟁입찰"
    "보기 2번" → choices[1]의 텍스트
    """
    번호맵 = {
        "1": 0, "2": 1, "3": 2, "4": 3,
        "1번": 0, "2번": 1, "3번": 2, "4번": 3,
        "①": 0, "②": 1, "③": 2, "④": 3,
        "첫번째": 0, "두번째": 1, "세번째": 2, "네번째": 3,
        "첫 번째": 0, "두 번째": 1, "세 번째": 2, "네 번째": 3,
    }

    cleaned = model_answer.strip()

    # "보기 2번" → "2번"
    cleaned_no_prefix = re.sub(r"^(보기|정답[은:]?|답[은:]?)\s*", "", cleaned).strip()
    if cleaned_no_prefix in 번호맵:
        idx = 번호맵[cleaned_no_prefix]
        if idx < len(choices):
            return _choice_text(choices[idx])

    # 원본이 번호맵에 있으면
    if cleaned in 번호맵:
        idx = 번호맵[cleaned]
        if idx < len(choices):
            return _choice_text(choices[idx])

    # 선택지 텍스트와 정확히 일치하면 우선 반환 ("M+5" ⊂ "M" 오매핑 방지)
    for choice in choices:
        ct = _choice_text(choice)
        if strip_spaces(ct) == strip_spaces(cleaned):
            return ct

    # "2. 제한경쟁입찰" → "제한경쟁입찰" (번호. 텍스트 형식 제거)
    # 날짜("2025.1.1")가 잘리지 않도록 1~2자리 숫자 + 뒤에 공백 필수
    stripped = re.sub(r"^\d{1,2}\.\s+", "", cleaned).strip()
    if stripped != cleaned:
        return stripped

    # 문장형: "답은 제한경쟁입찰입니다." → 보기 텍스트가 포함되어있는지 확인
    # 단, "X이 아닌 Y" 패턴에서 X를 정답으로 오인하지 않도록 부정어 체크
    _NEGATIONS = ("아닌", "아니라", "아니고", "않고", "말고", "제외", "아닙니다")

    def _norm_sep(s: str) -> str:
        """구분자(, / ·) 를 통일하여 비교 (예: '90%,10%' ↔ '90%/10%')"""
        return re.sub(r"[,/·%]", lambda m: {",%": ",", "/": ",", "·": ","}.get(m.group(0), m.group(0)), s)

    ps = strip_spaces(cleaned)
    ps_norm = _norm_sep(ps)
    matched_choices = []  # (text_position, choice_text)
    for choice in choices:
        ct = _choice_text(choice)
        if not ct:
            continue
        cs = strip_spaces(ct)
        cs_norm = _norm_sep(cs)
        # 원본 비교 또는 구분자 정규화 후 비교
        match_str = ps if cs in ps else (ps_norm if cs_norm in ps_norm else None)
        if match_str is not None:
            search_cs = cs if cs in ps else cs_norm
            idx = match_str.index(search_cs)
            after_text = match_str[idx + len(search_cs):idx + len(search_cs) + 10]
            has_negation = any(neg in after_text for neg in _NEGATIONS)
            if not has_negation:
                matched_choices.append((idx, ct))
    if len(matched_choices) == 1:
        return matched_choices[0][1]
    elif len(matched_choices) > 1:
        # 여러 보기가 포함된 경우 텍스트에서 가장 먼저 등장한 것 반환 (모델은 정답을 먼저 언급)
        matched_choices.sort(key=lambda x: x[0])
        return matched_choices[0][1]

    return cleaned


def grade(model_answer: str, item: dict[str, Any]) -> bool:
    """
    채점 메인 함수.
    1. 객관식 번호 폴백
    2. 공백 제거 후 비교
    3. 숫자 정규화 비교
    """
    truth = item["answer"]
    qtype = item.get("question_type", "short_answer")

    # 1) 객관식 번호 폴백
    if qtype == "multiple_choice" and "choices" in item:
        model_answer = map_choice_number(model_answer, item["choices"])
        # truth에도 "N. 텍스트" 형식이면 번호 제거 (문제은행 작성 방식 차이 대응)
        truth = _choice_text(truth)

    # 2) 공백 제거 후 Exact Match
    if strip_spaces(model_answer) == strip_spaces(truth):
        return True

    # 3) 정답이 모델 답변에 포함되어 있는지 (문장형 답변 대응)
    #    - 정답이 4글자 이상일 때만 (짧으면 오탐 위험)
    #    - 앞뒤 글자가 숫자면 부분 매칭이므로 제외 ("10개" in "100개" 방지)
    #    - 정답 뒤에 부정어가 오면 제외 ("제한경쟁입찰이 아닌" 방지)
    _NEGATIONS = ("아닌", "아니라", "아니고", "않고", "말고", "제외", "아닙니다")
    truth_clean = strip_spaces(truth)
    pred_clean = strip_spaces(model_answer)
    if len(truth_clean) >= 4 and truth_clean in pred_clean:
        idx = pred_clean.index(truth_clean)
        before = pred_clean[idx - 1] if idx > 0 else ""
        after_idx = idx + len(truth_clean)
        after = pred_clean[after_idx] if after_idx < len(pred_clean) else ""
        after_text = pred_clean[after_idx:after_idx + 10]
        has_negation = any(neg in after_text for neg in _NEGATIONS)
        # 앞뒤가 숫자가 아니고, 부정어가 없을 때만 정답
        if not (before.isdigit() or after.isdigit()) and not has_negation:
            return True

    # 4) 숫자 정규화 비교 (금액 표현 차이 허용)
    pred_num = normalize_number(model_answer)
    truth_num = normalize_number(truth)
    if pred_num is not None and truth_num is not None:
        return pred_num == truth_num

    # 5) 날짜 정규화 비교 (YYYY-MM-DD, YYYY.MM.DD, YYYY년 M월 D일)
    pred_date = normalize_date(model_answer)
    truth_date = normalize_date(truth)
    if pred_date is not None and truth_date is not None:
        return pred_date == truth_date

    # 6) 쉼표 구분 숫자 목록 비교 (예: GT="20, 70" → 답변에 20과 70이 모두 포함)
    truth_nums = re.findall(r"\d+", truth_clean)
    if len(truth_nums) >= 2:
        pred_nums = re.findall(r"\d+", pred_clean)
        truth_counts = {n: truth_nums.count(n) for n in set(truth_nums)}
        pred_counts = {n: pred_nums.count(n) for n in set(pred_nums)}
        if all(pred_counts.get(n, 0) >= count for n, count in truth_counts.items()):
            return True

    return False


# ============================================================
# 3. RAG 파이프라인 호출 (모드 선택)
# ============================================================

def ask_dummy(question: str, **kwargs) -> str:
    """테스트용 더미 — 정답을 그대로 반환하지 않음"""
    return f"[더미 응답] {question}"


_FILE_NAME_CACHE: dict[str, str] = {}


def _normalize_fname(name: str) -> str:
    """파일명 정규화: 연속 공백 → 단일 공백, 확장자 앞 공백 제거, strip."""
    import re as _re
    name = _re.sub(r"\s+", " ", name).strip()
    name = _re.sub(r"\s+\.(hwp|pdf|xlsx?)$", r".\1", name, flags=_re.I)
    return name


def _nospace(name: str) -> str:
    """공백 완전 제거 (부분 포함 비교용)."""
    return re.sub(r"\s+", "", name)


def _resolve_file_name(document_name: str) -> str:
    """JSON의 document_name을 실제 저장된 file_name으로 변환.
    일치하는 file_name이 없으면 빈 문자열 반환 (필터 미사용 신호)."""
    if not document_name:
        return ""
    if document_name in _FILE_NAME_CACHE:
        return _FILE_NAME_CACHE[document_name]

    from src.db.parsed_store import load_parsed_documents
    stored_names = [d["metadata"].get("file_name", "") for d in load_parsed_documents()]

    # 1) 완전 일치
    if document_name in stored_names:
        _FILE_NAME_CACHE[document_name] = document_name
        return document_name

    # 2) 정규화 후 일치 (확장자 앞 공백 등)
    doc_norm = _normalize_fname(document_name)
    for sn in stored_names:
        if _normalize_fname(sn) == doc_norm:
            _FILE_NAME_CACHE[document_name] = sn
            return sn

    # 3) 공백 제거 후 부분 포함 (축약 파일명 대응)
    doc_ns = _nospace(document_name)
    for sn in stored_names:
        sn_ns = _nospace(sn)
        if doc_ns in sn_ns or sn_ns in doc_ns:
            _FILE_NAME_CACHE[document_name] = sn
            return sn

    # 4) 못 찾으면 빈 문자열 → 필터 없이 실행
    _FILE_NAME_CACHE[document_name] = ""
    return ""


def ask_rag(question: str, document_name: str = "", choices: list | None = None, **kwargs) -> str:
    """하은 파이프라인으로 질문 → 응답 생성.
    document_name이 있으면 해당 문서 내에서만 검색(file_name 필터).
    choices가 있으면 질문에 선택지를 포함하여 객관식 답변을 유도."""
    import sys
    from pathlib import Path as _Path
    _root = _Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from dotenv import load_dotenv
    load_dotenv(_root / ".env")
    from rag_pipeline import answer_query

    resolved_name = _resolve_file_name(document_name)
    if document_name:
        query = f"[{document_name}] {question}"
    else:
        query = question
    if choices:
        choices_text = "\n".join(f"  {c}" for c in choices)
        query = f"{query}\n\n[선택지]\n{choices_text}\n\n반드시 위 선택지 중 하나만 골라 번호와 텍스트를 함께 답하세요."
    filters = {"file_name": resolved_name} if resolved_name else None
    result = answer_query(query=query, search_mode="하이브리드", top_k=10, filters=filters, eval_mode=True)
    return result.get("answer", "")


def ask_with_ground_truth(question: str, answer: str, **kwargs) -> str:
    """채점 로직 검증용 — 정답을 그대로 반환"""
    return answer


# ============================================================
# 4. 평가 실행
# ============================================================

def run_evaluation(
    items: list[dict[str, Any]],
    ask_fn,
    limit: int = 0,
) -> list[dict[str, Any]]:
    """전체 문항 평가 실행"""
    if limit > 0:
        items = items[:limit]

    results = []
    for item in items:
        started = time.time()

        try:
            model_answer = ask_fn(
                question=item["question"],
                answer=item["answer"],  # 검증 모드용
                document_name=item.get("document_name", ""),
                choices=item.get("choices", []),
            )
        except Exception as exc:
            print(f"  [오류] {item['id']}: {exc}")
            model_answer = f"[오류: {exc}]"
        elapsed = round(time.time() - started, 2)

        is_correct = grade(model_answer, item)

        results.append({
            "id": item["id"],
            "document_id": item.get("document_id", ""),
            "document_name": item.get("document_name", ""),
            "question_type": item.get("question_type", ""),
            "difficulty": item.get("difficulty", ""),
            "category": item.get("category", ""),
            "question": item["question"],
            "ground_truth": item["answer"],
            "model_answer": model_answer,
            "is_correct": is_correct,
            "elapsed_sec": elapsed,
        })

        status = "✓" if is_correct else "✗"
        print(f"  [{status}] {item['id']} | {item['question_type']} | {item['difficulty']}")

    return results


# ============================================================
# 5. 결과 집계
# ============================================================

def aggregate(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """전체/유형별/난이도별/카테고리별 집계"""
    groups = [("전체", results)]

    # question_type별
    for qtype in sorted({r["question_type"] for r in results}):
        rows = [r for r in results if r["question_type"] == qtype]
        groups.append((f"type:{qtype}", rows))

    # difficulty별
    for diff in ["basic", "intermediate", "advanced"]:
        rows = [r for r in results if r["difficulty"] == diff]
        if rows:
            groups.append((f"difficulty:{diff}", rows))

    # category별
    for cat in sorted({r["category"] for r in results}):
        rows = [r for r in results if r["category"] == cat]
        groups.append((f"category:{cat}", rows))

    summary = []
    for label, rows in groups:
        correct = sum(1 for r in rows if r["is_correct"])
        total = len(rows)
        summary.append({
            "group": label,
            "total": total,
            "correct": correct,
            "accuracy": round(correct / total, 4) if total > 0 else 0,
        })

    return summary


# ============================================================
# 6. CSV 저장
# ============================================================

def save_csv(path: Path, rows: list[dict[str, Any]]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[저장] {path}")


# ============================================================
# 7. 메인
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="자동평가 채점 엔진")
    parser.add_argument(
        "--문제은행",
        default="data/PartA_RFP_AutoGrading_QBank_sample.json",
        help="문제은행 JSON 경로",
    )
    parser.add_argument(
        "--모드",
        choices=["검증", "더미", "rag"],
        default="검증",
        help="검증=정답반환(채점로직테스트), 더미=더미응답, rag=실제파이프라인",
    )
    parser.add_argument(
        "--제한",
        type=int,
        default=0,
        help="0이면 전체, 양수면 앞에서부터 N문항만",
    )
    parser.add_argument(
        "--출력경로",
        default="data/eval_results",
        help="결과 CSV 저장 디렉토리",
    )
    args = parser.parse_args()

    # 경로 설정
    base_dir = Path(__file__).resolve().parent.parent
    qbank_path = base_dir / args.문제은행
    output_dir = base_dir / args.출력경로

    # 문제은행 로드
    items = load_question_bank(qbank_path)

    # 모드 선택
    ask_fn = {
        "검증": ask_with_ground_truth,
        "더미": ask_dummy,
        "rag": ask_rag,
    }[args.모드]

    print(f"\n[실행] 모드={args.모드}, 문항수={len(items)}")
    print("=" * 60)

    # 평가 실행
    results = run_evaluation(items, ask_fn, limit=args.제한)

    # 집계
    summary = aggregate(results)

    print("\n" + "=" * 60)
    print("[집계 결과]")
    for row in summary:
        print(f"  {row['group']:30s} | {row['correct']}/{row['total']} = {row['accuracy']:.1%}")

    # CSV 저장
    save_csv(output_dir / "eval_detail.csv", results)
    save_csv(output_dir / "eval_summary.csv", summary)

    print(f"\n[완료] 총 {len(results)}문항 평가")


if __name__ == "__main__":
    main()

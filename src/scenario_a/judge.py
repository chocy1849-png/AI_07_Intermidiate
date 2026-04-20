from __future__ import annotations

import json
import re
from typing import Any


def extract_json(text: str) -> dict[str, Any]:
    cleaned = str(text).strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError("JSON object not found in judge response.")
    return json.loads(match.group(0))


def build_judge_prompts(row: dict[str, Any]) -> tuple[str, str]:
    system_prompt = (
        "당신은 공공 제안요청서 기반 RAG 답변 평가자다.\n"
        "질문, 검색 문맥, 답변, 정답 힌트를 보고 아래 4개 항목을 1~5점으로 채점하라.\n"
        "- faithfulness_score: 문맥 밖 내용이나 과장이 없는가\n"
        "- completeness_score: 질문이 요구한 핵심 요소를 빠뜨리지 않았는가\n"
        "- groundedness_score: 답변이 검색 문맥에 직접 근거하는가\n"
        "- relevancy_score: 질문에 직접적으로 답하고 있는가\n"
        "반드시 JSON 객체만 반환하라."
    )
    user_prompt = (
        f"question_id: {row.get('question_id', '')}\n"
        f"type_group: {row.get('type_group', '')}\n"
        f"answer_type: {row.get('answer_type', '')}\n"
        f"question: {row.get('question', '')}\n"
        f"ground_truth_hint: {row.get('ground_truth_hint', '')}\n"
        f"expected: {row.get('expected', '')}\n\n"
        f"[검색 문맥]\n{row.get('retrieval_context', '')}\n\n"
        f"[답변]\n{row.get('answer_text', '')}\n\n"
        "아래 형식의 JSON만 반환하라.\n"
        "{\n"
        '  "faithfulness_score": 1,\n'
        '  "completeness_score": 1,\n'
        '  "groundedness_score": 1,\n'
        '  "relevancy_score": 1,\n'
        '  "evaluator_note": "간단한 평가 메모"\n'
        "}"
    )
    return system_prompt, user_prompt


def judge_row(openai_client: Any, judge_model: str, row: dict[str, Any]) -> dict[str, Any]:
    system_prompt, user_prompt = build_judge_prompts(row)
    response = openai_client.responses.create(
        model=judge_model,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
        ],
    )
    payload = extract_json(response.output_text)
    return {
        "faithfulness_score": int(payload["faithfulness_score"]),
        "completeness_score": int(payload["completeness_score"]),
        "groundedness_score": int(payload["groundedness_score"]),
        "relevancy_score": int(payload["relevancy_score"]),
        "evaluator_note": str(payload.get("evaluator_note", "")).strip(),
    }

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parsed_dir = ROOT / "data" / "parsed"
    chroma_dir = ROOT / "data" / "chroma_db"
    metadata_db = ROOT / "data" / "metadata.db"
    qbank = ROOT / "(260417) PartA_RFP_AutoGrading_QBank_v3_fixed.json"
    judge_input = ROOT / "judge_subset_20.json"

    parsed_count = len(list(parsed_dir.glob("*.json"))) if parsed_dir.exists() else 0
    chroma_exists = chroma_dir.exists()
    metadata_exists = metadata_db.exists()
    openai_key = bool(os.getenv("OPENAI_API_KEY"))

    status = {
        "root": str(ROOT),
        "openai_api_key": openai_key,
        "question_bank_exists": qbank.exists(),
        "judge_input_exists": judge_input.exists(),
        "parsed_dir_exists": parsed_dir.exists(),
        "parsed_file_count": parsed_count,
        "chroma_dir_exists": chroma_exists,
        "metadata_db_exists": metadata_exists,
        "ready_for_verify_mode": qbank.exists(),
        "ready_for_rag_mode": openai_key and parsed_count > 0 and chroma_exists and metadata_exists,
        "ready_for_llm_judge": openai_key and judge_input.exists() and parsed_count > 0 and chroma_exists and metadata_exists,
    }
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

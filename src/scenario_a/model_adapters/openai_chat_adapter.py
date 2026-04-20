from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from scenario_a.model_adapters.base import BaseModelAdapter


class OpenAIChatAdapter(BaseModelAdapter):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.client: Any | None = None

    def load(self) -> None:
        if self.client is not None:
            return
        load_dotenv(Path.cwd() / ".env", override=False)
        api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI chat adapter.")
        base_url = (os.getenv("OPENAI_BASE_URL") or "").strip() or "https://api.openai.com/v1"
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise RuntimeError("openai package is required for OpenAI chat adapter.") from exc
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(
        self,
        *,
        system_instruction: str,
        question: str,
        context_text: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        self.load()
        assert self.client is not None
        messages = self.build_messages(
            system_instruction=system_instruction,
            question=question,
            context_text=context_text,
            history=history or [],
        )
        response = self.client.responses.create(
            model=self.config.model_id,
            input=[
                {
                    "role": message["role"],
                    "content": [{"type": "input_text", "text": message["content"]}],
                }
                for message in messages
            ],
            max_output_tokens=self.config.max_new_tokens,
        )
        return self._trim_stop_sequences(response.output_text or "")

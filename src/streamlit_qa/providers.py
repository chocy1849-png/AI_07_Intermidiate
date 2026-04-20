from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scenario_a.model_adapters import create_adapter
from scenario_a.model_adapters.base import AdapterConfig, BaseModelAdapter

from streamlit_qa.config import LocalModelSpec


def _require_openai() -> Any:
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError("openai package is required for OpenAI/OpenAI-compatible providers.") from exc
    return OpenAI


@dataclass(slots=True)
class ProviderSelection:
    provider: str
    model_name: str
    api_key: str = ""
    base_url: str = ""
    max_new_tokens: int = 768
    temperature: float = 0.0
    top_p: float = 1.0
    local_model_ui_key: str = ""

    def label(self) -> str:
        return f"{self.provider}:{self.model_name}"


class OpenAICompatibleAdapter:
    def __init__(self, selection: ProviderSelection) -> None:
        if not selection.api_key.strip():
            raise RuntimeError("API key is required for OpenAI/OpenAI-compatible provider.")
        base_url = selection.base_url.strip() or "https://api.openai.com/v1"
        self._selection = selection
        self._base_url = base_url
        self._client: Any | None = None
        self.config = AdapterConfig(
            model_key=f"{selection.provider}:{selection.model_name}",
            adapter_name="openai_compatible",
            model_id=selection.model_name,
            runtime="openai",
            max_new_tokens=max(1, int(selection.max_new_tokens)),
            temperature=float(selection.temperature),
            top_p=float(selection.top_p),
            stop_sequences=(),
            use_chat_template=False,
        )

    @property
    def client(self) -> Any:
        if self._client is None:
            OpenAI = _require_openai()
            self._client = OpenAI(api_key=self._selection.api_key.strip(), base_url=self._base_url)
        return self._client

    @staticmethod
    def _build_messages(
        *,
        system_instruction: str,
        question: str,
        context_text: str,
        history: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [{"role": "system", "content": system_instruction}]
        for turn in history:
            role = str(turn.get("role", "")).strip() or "user"
            content = str(turn.get("content", "")).strip()
            if role not in {"user", "assistant", "system"}:
                role = "user"
            if content:
                messages.append({"role": role, "content": content})
        user_prompt = "\n\n".join(
            [
                "[QUESTION]",
                question,
                "[EVIDENCE]",
                context_text or "(no evidence)",
                "[ANSWER FORMAT]",
                "Answer in Korean. Keep the answer factual, direct, and grounded in the evidence.",
            ]
        )
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _trim_stop_sequences(self, text: str) -> str:
        trimmed = text
        for stop in self.config.stop_sequences:
            if stop and stop in trimmed:
                trimmed = trimmed.split(stop, 1)[0]
        return trimmed.strip()

    def generate(
        self,
        *,
        system_instruction: str,
        question: str,
        context_text: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        messages = self._build_messages(
            system_instruction=system_instruction,
            question=question,
            context_text=context_text,
            history=history or [],
        )
        kwargs: dict[str, Any] = {
            "model": self.config.model_id,
            "input": [
                {
                    "role": message["role"],
                    "content": [{"type": "input_text", "text": message["content"]}],
                }
                for message in messages
            ],
            "max_output_tokens": self.config.max_new_tokens,
        }
        if self.config.temperature > 0:
            kwargs["temperature"] = self.config.temperature
            kwargs["top_p"] = self.config.top_p
        response = self.client.responses.create(**kwargs)
        return self._trim_stop_sequences(response.output_text or "")


def create_local_adapter(selection: ProviderSelection, local_spec: LocalModelSpec) -> BaseModelAdapter:
    config = AdapterConfig(
        model_key=local_spec.key,
        adapter_name=local_spec.adapter_name,
        model_id=local_spec.model_id,
        tokenizer_id=local_spec.tokenizer_id,
        runtime=local_spec.runtime,
        device_map=local_spec.device_map,
        torch_dtype=local_spec.torch_dtype,
        trust_remote_code=local_spec.trust_remote_code,
        max_new_tokens=max(1, int(selection.max_new_tokens)),
        temperature=float(selection.temperature),
        top_p=float(selection.top_p),
        stop_sequences=local_spec.stop_sequences,
    )
    return create_adapter(config)


def create_provider_adapter(
    selection: ProviderSelection,
    local_model_specs: dict[str, LocalModelSpec],
) -> BaseModelAdapter | OpenAICompatibleAdapter:
    if selection.provider in {"openai_api", "openai_compatible"}:
        return OpenAICompatibleAdapter(selection)
    if selection.provider != "local":
        raise KeyError(f"Unsupported provider: {selection.provider}")
    spec_key = selection.local_model_ui_key or selection.model_name
    if spec_key not in local_model_specs:
        raise KeyError(f"Unknown local model key: {spec_key}")
    return create_local_adapter(selection, local_model_specs[spec_key])

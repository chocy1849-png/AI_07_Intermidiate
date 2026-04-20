from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class AdapterConfig:
    model_key: str
    adapter_name: str
    model_id: str
    tokenizer_id: str | None = None
    runtime: str = "transformers"
    ollama_model_name: str | None = None
    device_map: str = "auto"
    torch_dtype: str = "auto"
    trust_remote_code: bool = False
    max_new_tokens: int = 768
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    stop_sequences: tuple[str, ...] = field(default_factory=tuple)
    use_chat_template: bool = True
    extra_generate_kwargs: dict[str, Any] = field(default_factory=dict)


class BaseModelAdapter(ABC):
    def __init__(self, config: AdapterConfig) -> None:
        self.config = config
        self.model: Any | None = None
        self.tokenizer: Any | None = None

    @staticmethod
    def _require_transformers() -> tuple[Any, Any]:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Scenario A local model adapters require torch and transformers."
            ) from exc
        return torch, (AutoModelForCausalLM, AutoTokenizer)

    def _resolve_dtype(self, torch_module: Any) -> Any:
        mapping = {
            "auto": "auto",
            "float16": torch_module.float16,
            "bfloat16": getattr(torch_module, "bfloat16", torch_module.float16),
            "float32": torch_module.float32,
        }
        if self.config.torch_dtype not in mapping:
            raise ValueError(f"Unsupported torch_dtype: {self.config.torch_dtype}")
        return mapping[self.config.torch_dtype]

    def load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        torch, (AutoModelForCausalLM, AutoTokenizer) = self._require_transformers()
        tokenizer_id = self.config.tokenizer_id or self.config.model_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
            trust_remote_code=self.config.trust_remote_code,
        )
        model_kwargs: dict[str, Any] = {
            "device_map": self.config.device_map,
            "trust_remote_code": self.config.trust_remote_code,
        }
        dtype = self._resolve_dtype(torch)
        if dtype != "auto":
            model_kwargs["torch_dtype"] = dtype
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            **model_kwargs,
        )

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None

    def build_messages(
        self,
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

    def render_prompt(
        self,
        *,
        system_instruction: str,
        question: str,
        context_text: str,
        history: list[dict[str, str]],
    ) -> str:
        if self.tokenizer is None:
            self.load()
        assert self.tokenizer is not None
        messages = self.build_messages(
            system_instruction=system_instruction,
            question=question,
            context_text=context_text,
            history=history,
        )
        if self.config.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return self.build_plain_prompt(
            system_instruction=system_instruction,
            question=question,
            context_text=context_text,
            history=history,
        )

    def build_plain_prompt(
        self,
        *,
        system_instruction: str,
        question: str,
        context_text: str,
        history: list[dict[str, str]],
    ) -> str:
        messages = self.build_messages(
            system_instruction=system_instruction,
            question=question,
            context_text=context_text,
            history=history,
        )
        return "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages) + "\nASSISTANT:"

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
        if self.model is None or self.tokenizer is None:
            self.load()
        assert self.model is not None
        assert self.tokenizer is not None

        prompt = self.render_prompt(
            system_instruction=system_instruction,
            question=question,
            context_text=context_text,
            history=history or [],
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if hasattr(self.model, "device"):
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        do_sample = self.config.temperature > 0
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": self.config.repetition_penalty,
        }
        if do_sample:
            generate_kwargs["temperature"] = self.config.temperature
            generate_kwargs["top_p"] = self.config.top_p
        generate_kwargs.update(self.config.extra_generate_kwargs)

        output = self.model.generate(**inputs, **generate_kwargs)
        generated_ids = output[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return self._trim_stop_sequences(text)

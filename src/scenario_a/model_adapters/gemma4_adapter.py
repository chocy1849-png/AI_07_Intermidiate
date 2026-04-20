from __future__ import annotations

from scenario_a.model_adapters.base import BaseModelAdapter


class Gemma4Adapter(BaseModelAdapter):
    def load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        torch, (AutoModelForCausalLM, AutoTokenizer) = self._require_transformers()
        tokenizer_id = self.config.tokenizer_id or self.config.model_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
            trust_remote_code=self.config.trust_remote_code,
            extra_special_tokens={},
        )
        model_kwargs: dict[str, object] = {
            "device_map": self.config.device_map,
            "trust_remote_code": self.config.trust_remote_code,
            "attn_implementation": "eager",
        }
        dtype = self._resolve_dtype(torch)
        if dtype != "auto":
            model_kwargs["torch_dtype"] = dtype
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            **model_kwargs,
        )

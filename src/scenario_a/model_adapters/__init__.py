from .base import AdapterConfig, BaseModelAdapter
from .exaone_adapter import ExaoneAdapter
from .gemma4_adapter import Gemma4Adapter
from .openai_chat_adapter import OpenAIChatAdapter
from .qwen_adapter import QwenAdapter


def create_adapter(config: AdapterConfig) -> BaseModelAdapter:
    registry = {
        "qwen": QwenAdapter,
        "gemma4": Gemma4Adapter,
        "exaone": ExaoneAdapter,
        "openai_chat": OpenAIChatAdapter,
    }
    if config.adapter_name not in registry:
        raise KeyError(f"Unsupported adapter: {config.adapter_name}")
    return registry[config.adapter_name](config)


__all__ = [
    "AdapterConfig",
    "BaseModelAdapter",
    "QwenAdapter",
    "Gemma4Adapter",
    "ExaoneAdapter",
    "OpenAIChatAdapter",
    "create_adapter",
]

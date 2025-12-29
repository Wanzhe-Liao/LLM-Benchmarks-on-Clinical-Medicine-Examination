from typing import Dict, Any
from .base import (
    BaseLLMAdapter,
    OpenAICompatibleAdapter,
    AnthropicAdapter,
    GoogleAdapter,
    DashScopeAdapter,
    MiniMaxAdapter
)


class ModelFactory:
    PROVIDER_MAP = {
        'openai': OpenAICompatibleAdapter,
        'anthropic': AnthropicAdapter,
        'google': GoogleAdapter,
        'dashscope': DashScopeAdapter,
        'minimax': MiniMaxAdapter
    }

    @classmethod
    def create(cls, model_id: str, config: Dict[str, Any]) -> BaseLLMAdapter:
        provider = config.get('provider', 'openai')
        adapter_class = cls.PROVIDER_MAP.get(provider)

        if not adapter_class:
            raise ValueError(f"Unknown provider: {provider}")

        return adapter_class(config)

    @classmethod
    def create_all(cls, models_config: Dict[str, Dict[str, Any]]) -> Dict[str, BaseLLMAdapter]:
        adapters = {}
        for model_id, config in models_config.items():
            if config.get('api_key') and config.get('base_url'):
                adapters[model_id] = cls.create(model_id, config)
        return adapters
